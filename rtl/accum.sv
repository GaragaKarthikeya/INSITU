// The value-side accumulator, one query head, 64 channels.
//     acc[ch] += (v_norm * p * centroid_v[v_code[ch]]) >> ACC_SHIFT
//     acc[ch]  = (acc[ch] * factor) >> PROB_FRAC        on a rescale
//
// Four multipliers for sixty-four channels
// ----------------------------------------
// `centroid_v` has only 2**VAL_BITS = 4 possible values, so the per-channel
// product has only four possible values too.  Compute `w = v_norm * p` once,
// then `w * c` for each of the four constants, and every channel is a 4:1 mux
// and an add.  Five multiplies where the obvious form needs sixty-five, and it
// falls out of attending on codes rather than on reconstructed vectors -- the
// same property that empties `score_lane`'s inner loop.
//
// Each term is truncated before it joins the sum
// ----------------------------------------------
// `_accumulate_terms` shifts every term individually and says why: truncating
// the sum instead would depend on the order the terms arrived in, and a
// vectorised sum would stop equalling a sequential one.  So the >> is on the
// muxed product, ahead of the adder, not after it.
//
// The rescale is folded 8 channels wide
// -------------------------------------
// 64 permanent 32x16 multipliers to serve an event that happens about ln T
// times per scan -- eleven times in 32,768 tokens -- would be most of the DSPs
// in the datapath sitting idle 99.7% of the time.  Folded 8 wide it is 8
// multipliers and 8 cycles, and the scan stalls for those cycles.  The stall
// is the design, not a limitation of it.
//
// A scale waits for the add pipe to drain
// ---------------------------------------
// The model rescales an accumulator that every earlier token has already
// reached.  Terms are three stages deep here, so accepting a scale while any
// are in flight would scale a partial accumulator and then add un-scaled terms
// on top of it -- an error proportional to how full the pipe happened to be,
// which is the kind that shows up as a small accuracy loss and nothing else.
`timescale 1ns/1ps
module accum #(
    parameter int D          = 64,
    parameter int VAL_BITS   = 2,
    parameter int NORM_BITS  = 16,
    parameter int PROB_BITS  = 16,
    parameter int PROB_FRAC  = 15,
    parameter int CENT_BITS  = 20,
    parameter int ACC_WIDTH  = 32,   // FixedFormat.acc_width
    // centroid_frac + norm_frac + prob_frac - acc_frac.  `acc_shift_for`.
    parameter int ACC_SHIFT  = 23,
    parameter int LANES      = 8,    // channels rescaled per cycle
    // The value codebook, ascending, index 0 in the low bits.  A format
    // artifact of `Codebook.build`, like every other table in this datapath.
    parameter logic [(2**VAL_BITS)*CENT_BITS-1:0] CENTROIDS =
        {20'h0c152, 20'h039f3, 20'hfc60d, 20'hf3eae}
) (
    input  logic clk,
    input  logic rstn,
    input  logic start,                  // clear the accumulator for a new scan

    input  logic op_valid,
    output logic op_ready,
    input  logic op_scale,
    input  logic [PROB_BITS-1:0] op_factor,
    input  logic [PROB_BITS-1:0] op_p,
    input  logic [D*VAL_BITS-1:0] op_vcodes,
    input  logic signed [NORM_BITS-1:0] op_vnorm,

    output logic tok_done,               // one ADD retired into acc
    output logic scale_done,
    output logic [D*ACC_WIDTH-1:0] out_acc,
    output logic [31:0] out_overflows,   // the model counts these; so does this
    output logic [31:0] out_scale_cycles // measured stall, for plan.MD
);
    localparam int NC     = 2**VAL_BITS;
    localparam int W_W    = NORM_BITS + PROB_BITS;            // v_norm * p
    localparam int T_W    = W_W + CENT_BITS;                  // ... * centroid
    localparam int TERM_W = T_W - ACC_SHIFT;
    localparam int GROUPS = D / LANES;
    // accept -> w, w -> the products, products -> the four terms, terms -> the
    // Selected term, that -> acc.
    //
    // `a_w * centroid` is 32x20 and `acc * factor` is 48x16: both are wider
    // than one DSP48 and map to a cascaded pair, which does not reach 250 MHz
    // without a register between the multiply and the cascade adder. That
    // register is the DSP's own MREG and Vivado infers it only if the RTL has
    // one there -- so the product is registered separately from the shift.
    // This is why the block's paths kept reappearing at ~4.0 ns however often
    // the surrounding logic was split: the limit was inside the multiplier,
    // not around it.
    // The valid that gates a stage is `avld[stage-1]`: `a_w` and `a_term` are
    // single registers overwritten every cycle, and gating one stage too late
    // reads a later token's term.
    //
    // The select is its own stage.  Folding the 64 4:1 muxes into the same
    // cycle as the 33-bit saturating add put 13 endpoints at -0.2 ns once four
    // lanes were placed, and the mux is not part of the loop -- only the add
    // is -- so splitting it costs a cycle of latency and keeps II=1.
    localparam int ADD_LAT = 5;

    localparam logic signed [ACC_WIDTH-1:0] A_HI = {1'b0, {(ACC_WIDTH-1){1'b1}}};
    localparam logic signed [ACC_WIDTH-1:0] A_LO = {1'b1, {(ACC_WIDTH-1){1'b0}}};

    generate
        if (D % LANES != 0) initial $fatal(1, "LANES must divide D");
        if (TERM_W < 1) initial $fatal(1, "ACC_SHIFT discards the whole term");
    endgenerate

    logic signed [ACC_WIDTH-1:0] acc [0:D-1];

    // Loop temporaries. Declared at module scope because Icarus does not
    // support procedural `automatic`; they are written and read within one
    // iteration and never carry state between them.
    logic [VAL_BITS-1:0] code;
    logic signed [ACC_WIDTH:0] sum;
    integer ch;

    // -- the add pipe -------------------------------------------------------
    logic [ADD_LAT-1:0] avld;
    // Same argument: one code word selects all 64 muxes.
    (* max_fanout = 8 *)
    logic [D*VAL_BITS-1:0] a_codes [0:ADD_LAT-1];
    logic signed [W_W-1:0] a_w, a_w_q;
    // Overflow is counted a cycle later, as a popcount.
    //
    // Each of the 64 channels incrementing one counter directly makes 64
    // conditions fan into that counter's clock enable, and it showed up on the
    // critical path -- diagnostic logic pacing a datapath. The count is only
    // read at the end of a scan, so a cycle of lag costs nothing and the total
    // is identical.
    logic [D-1:0] ovf_bits;
    logic [$clog2(D+1)-1:0] ovf_n;
    // The four possible products fan out to all 64 channel muxes, which is 64
    // loads on every bit of every term and the reason this was the critical
    // path at +0.008 ns. Synthesis is told to replicate the registers rather
    // than route one copy across the block; the arithmetic is untouched.
    (* max_fanout = 8 *)
    logic signed [TERM_W-1:0] a_term [0:NC-1];
    logic signed [T_W-1:0]    a_prod [0:NC-1];      // the raw product, MREG
    logic signed [TERM_W-1:0] a_mterm [0:D-1];

    // -- the scale sequencer, in two stages ---------------------------------
    //
    // Stage A selects a group of `LANES` accumulator words; stage B multiplies
    // them by the factor and writes them back. One cycle where there was one
    // multiply, and the reason is placed timing, not arithmetic: `sgrp` fans
    // out across all 64 accumulators, and reading through that mux and then
    // through a 48x16 product -- two cascaded DSPs -- in a single cycle routed
    // to -0.550 ns inside `attn_top`, against +0.675 ns for this block alone.
    // Splitting it puts a register directly on the DSP's A input, which is
    // where the cascade wants one.
    //
    // The arithmetic is untouched: the same operand, the same factor, the same
    // truncating shift, so every golden in `tb_softmax_accum` holds. What
    // changes is the cost -- 9 cycles per event where it was 8 -- and that is
    // a measured number, reported by `out_scale_cycles` and carried in
    // `AttentionConfig.rescale_cycles`, never one this file asserts.
    logic scaling, s_wb, s_wb2;
    logic [$clog2(GROUPS+1)-1:0] sgrp, sgrp_b, sgrp_c;
    logic [PROB_BITS-1:0] sfactor;
    logic signed [ACC_WIDTH-1:0] s_operand [0:LANES-1];
    logic signed [ACC_WIDTH+PROB_BITS-1:0] s_prod [0:LANES-1];

    wire pipe_busy = (avld != '0);
    // A scale is accepted only into an empty pipe; an add is accepted whenever
    // no rescale is running -- and the writeback stage counts as running, or
    // an add would reach `acc` in the same cycle the fold writes it back.
    assign op_ready = !scaling && !s_wb && !s_wb2 && (!op_scale || !pipe_busy);

    wire take = op_valid && op_ready;

    always_ff @(posedge clk) begin
        if (!rstn || start) begin
            for (int i = 0; i < D; i++) acc[i] <= '0;
            avld <= '0; scaling <= 1'b0; s_wb <= 1'b0; s_wb2 <= 1'b0; sgrp <= '0;
            ovf_bits <= '0;
            out_overflows <= '0; out_scale_cycles <= '0;
            tok_done <= 1'b0; scale_done <= 1'b0;
        end else begin
            tok_done <= 1'b0;
            scale_done <= 1'b0;

            // -- stage 0: accept --------------------------------------------
            avld <= {avld[ADD_LAT-2:0], (take && !op_scale)};
            a_codes[0] <= op_vcodes;
            for (int i = 1; i < ADD_LAT; i++) a_codes[i] <= a_codes[i-1];

            // -- stage 1: w = v_norm * p ------------------------------------
            //
            // Registered again before the term multiply: `a_w` is a DSP output
            // feeding four more DSPs, and driving them directly put two
            // multipliers in series in one cycle.
            a_w   <= $signed(op_vnorm) * $signed({1'b0, op_p});
            a_w_q <= a_w;

            // -- stage 2: the four products, registered before the shift -----
            for (int c = 0; c < NC; c++)
                a_prod[c] <= $signed(T_W'(a_w_q)) *
                    $signed(T_W'($signed(CENTROIDS[c*CENT_BITS +: CENT_BITS])));

            // -- stage 3: truncate to the accumulator's format ---------------
            for (int c = 0; c < NC; c++)
                a_term[c] <= TERM_W'(a_prod[c] >>> ACC_SHIFT);

            // -- stage 4: 64 muxes ------------------------------------------
            if (avld[ADD_LAT-2])
                for (int ch = 0; ch < D; ch++) begin
                    code = a_codes[ADD_LAT-2][ch*VAL_BITS +: VAL_BITS];
                    a_mterm[ch] <= a_term[code];
                end

            // -- stage 5: 64 saturating adds --------------------------------
            ovf_bits <= '0;
            if (avld[ADD_LAT-1]) begin
                for (int ch = 0; ch < D; ch++) begin
                    sum  = $signed({acc[ch][ACC_WIDTH-1], acc[ch]}) +
                           (ACC_WIDTH+1)'($signed(a_mterm[ch]));
                    if (sum > (ACC_WIDTH+1)'(A_HI)) begin
                        acc[ch] <= A_HI; ovf_bits[ch] <= 1'b1;
                    end else if (sum < (ACC_WIDTH+1)'(A_LO)) begin
                        acc[ch] <= A_LO; ovf_bits[ch] <= 1'b1;
                    end else
                        acc[ch] <= ACC_WIDTH'(sum);
                end
                tok_done <= 1'b1;
            end

            // -- stage 6: fold the saturation flags into the counter ---------
            ovf_n = '0;
            for (int ch = 0; ch < D; ch++) ovf_n = ovf_n + {{($clog2(D+1)-1){1'b0}}, ovf_bits[ch]};
            if (|ovf_bits) out_overflows <= out_overflows + 32'(ovf_n);

            // -- the rescale, stage A: select ---------------------------------
            s_wb <= 1'b0;
            if (take && op_scale) begin
                scaling <= 1'b1; sgrp <= '0; sfactor <= op_factor;
            end else if (scaling) begin
                for (int l = 0; l < LANES; l++)
                    s_operand[l] <= acc[sgrp * LANES + l];
                sgrp_b <= sgrp;
                s_wb   <= 1'b1;
                if (sgrp == GROUPS-1) scaling <= 1'b0;
                sgrp <= sgrp + 1'b1;
            end

            // -- the rescale, stage B: multiply (registered) -------------------
            //
            // 48x16 is a cascaded DSP pair, so the product gets its own cycle
            // exactly as the term products above do.
            if (s_wb)
                for (int l = 0; l < LANES; l++)
                    s_prod[l] <= $signed({{PROB_BITS{s_operand[l][ACC_WIDTH-1]}},
                                          s_operand[l]}) * $signed({1'b0, sfactor});
            s_wb2  <= s_wb;
            sgrp_c <= sgrp_b;

            // -- the rescale, stage C: shift and write back --------------------
            //
            // Stage A reads the group after next while this writes the previous
            // one; they are disjoint by construction, so no channel is read and
            // written in the same cycle.
            if (s_wb2)
                for (int l = 0; l < LANES; l++) begin
                    ch = sgrp_c * LANES + l;
                    acc[ch] <= ACC_WIDTH'(s_prod[l] >>> PROB_FRAC);
                end

            if (scaling || s_wb || s_wb2) out_scale_cycles <= out_scale_cycles + 1'b1;
            if (s_wb2 && sgrp_c == GROUPS-1) scale_done <= 1'b1;
        end
    end

    generate
        genvar g;
        for (g = 0; g < D; g = g + 1) begin : pack
            assign out_acc[g*ACC_WIDTH +: ACC_WIDTH] = acc[g];
        end
    endgenerate
endmodule
