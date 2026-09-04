// Rotated vector + wire norm -> codes.  `ops/quantize.py::_encode_plane`.
//
// WHY THERE IS NO DIVIDER HERE
// ----------------------------
// The obvious encoder normalizes (`x / norm`) and then compares against the
// codebook boundaries, which costs a divide per channel.  Comparing `x / norm`
// against `b` is the same test as comparing `x` against `b * norm`, so the
// boundaries are scaled by the norm ONCE per vector -- 2**BITS - 1 multiplies
// -- and every one of the D channels is then a plain comparison.
//
// TIES FALL TO THE LOWER BIN
// --------------------------
// The comparison is `>`, never `>=`.  A channel sitting exactly on a boundary
// belongs to the bin below it, which is the convention `Codebook.gaussian`
// built the boundaries under.  The two must agree or every tie shifts by one
// code; `tb_rot_encode` is fed constructed on-boundary vectors because random
// stimulus hits an exact tie with probability about 2**-32.
//
// BINARY SEARCH, NOT A COMPARATOR ARRAY
// -------------------------------------
// BITS stages of one comparison each, rather than 2**BITS - 1 comparisons and
// a popcount.  At BITS=4 the two are close; the sweep in step 15 raises
// KEY_BITS, and there the array is 255 comparators per lane against 8 stages.
`timescale 1ns/1ps
module rot_encode #(
    parameter int D          = 64,
    parameter int IN_BITS    = 24,
    parameter int BITS       = 4,    // KEY_BITS or VAL_BITS
    parameter int NORM_BITS  = 16,
    parameter int BOUND_BITS = 20,   // widest gaussian boundary is 78,586
    // centroid_frac + norm_frac - qk_frac.  The comparison happens in
    // Q(threshold_frac), and `KVQuantizer.__init__` refuses to build a format
    // where this is negative, because scaling x up would have to truncate it.
    parameter int THRESH_SHIFT = 8,
    parameter int LANES        = 8,  // channels compared per cycle
    // Ascending decision boundaries in Q(centroid_frac), index 0 in the low
    // bits.  The default is QuantConfig(key_bits=4)'s gaussian codebook; a
    // build at another width supplies the generated word.  This is a FORMAT
    // artifact, like the sign diagonal -- never a second Lloyd-Max solve.
    parameter logic [(2**BITS-1)*BOUND_BITS-1:0] BOUNDS =
        {20'h132fa, 20'h0eb9b, 20'h0b794, 20'h08c5b, 20'h0660a,
         20'h042a3, 20'h020ec, 20'h00000, 20'hfdf14, 20'hfbd5d,
         20'hf99f5, 20'hf73a4, 20'hf486c, 20'hf1465, 20'hecd06}
) (
    input  logic clk,
    input  logic rstn,
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [D*IN_BITS-1:0] in_vec,
    input  logic [NORM_BITS-1:0] in_norm,
    output logic out_valid,
    input  logic out_ready,
    // Lane i in [BITS*i +: BITS] -- LSB-first, which is byte for byte the
    // layout `ops/quantize.py::_pack_codes` writes into a cache row.
    output logic [D*BITS-1:0] out_codes
);
    localparam int NB      = 2**BITS - 1;
    // Wide enough for BOTH operands: norm*boundary needs
    // (NORM_BITS-1) + BOUND_BITS bits, x<<THRESH_SHIFT needs
    // IN_BITS + THRESH_SHIFT + 1.  Sized to the larger, not to x.
    localparam int THR_W   = (NORM_BITS + BOUND_BITS > IN_BITS + THRESH_SHIFT + 1)
                             ? NORM_BITS + BOUND_BITS : IN_BITS + THRESH_SHIFT + 1;
    localparam int STEPS   = D/LANES;

    generate
        if (D % LANES != 0) initial $fatal(1, "LANES must divide D");
        if (THRESH_SHIFT < 0)
            initial $fatal(1, "a negative THRESH_SHIFT would truncate x and move bins");
    endgenerate

    wire advance = !out_valid || out_ready;

    // -- per-vector state --------------------------------------------------
    logic signed [D*IN_BITS-1:0] held;
    logic signed [THR_W-1:0] thr [0:NB-1];
    logic busy, done;
    logic [$clog2(STEPS):0] step;
    logic [D*BITS-1:0] codes;

    // The thresholds are per-vector state that the in-flight levels are
    // still reading, so acceptance waits for the pipeline to DRAIN, not
    // just for the last group to be issued.
    assign in_ready = advance && !busy && !(|pvalid) && !out_valid;

    // Thresholds are built the cycle the vector is accepted, from the norm at
    // its WIRE precision -- the same integer the cache row carries.
    generate
        genvar b;
        for (b = 0; b < NB; b = b + 1) begin : threshold
            wire signed [BOUND_BITS-1:0] bound = BOUNDS[b*BOUND_BITS +: BOUND_BITS];
            wire signed [THR_W-1:0] product =
                $signed({1'b0, in_norm}) * bound;
            always_ff @(posedge clk)
                if (advance && in_ready && in_valid) thr[b] <= product;
        end
    endgenerate

    // -- BITS-deep binary search, LANES channels wide ----------------------
    // r = #{i : x > B[i]}.  r >= t exactly when x > B[t-1], so a candidate is
    // formed one bit at a time from the top and kept when the comparison holds.
    //
    // One REGISTERED level per bit, not four chained comparisons in a cycle:
    // the combinational form measured -0.641 ns at 250 MHz, because a 36-bit
    // compare against a mux-selected threshold is not cheap and four of them
    // in series is most of a clock period.  The pipeline costs BITS cycles of
    // latency per vector on a block that runs once per token.
    logic signed [THR_W-1:0] xs [1:BITS][0:LANES-1];
    logic [BITS-1:0]         rr [1:BITS][0:LANES-1];
    logic [$clog2(STEPS):0]  wstep [1:BITS];
    logic [BITS-1:0]         pvalid;

    generate
        genvar j, l;
        for (j = 0; j < BITS; j = j + 1) begin : level
            // PREV keeps the index inside the arrays on level 0, whose inputs
            // are the seed expressions below rather than a previous level.
            localparam int PREV = (j == 0) ? 1 : j;
            for (l = 0; l < LANES; l = l + 1) begin : lane
                wire signed [THR_W-1:0] x_in = (j == 0)
                    ? (THR_W'($signed(held[(step*LANES + l)*IN_BITS +: IN_BITS]))
                       <<< THRESH_SHIFT)
                    : xs[PREV][l];
                wire [BITS-1:0] r_in  = (j == 0) ? {BITS{1'b0}} : rr[PREV][l];
                wire [BITS-1:0] cand  = r_in | (BITS'(1) << (BITS-1-j));
                // `>` and never `>=`: a channel exactly on a boundary belongs
                // to the bin BELOW it.
                wire take = x_in > thr[cand - 1];
                always_ff @(posedge clk) if (advance) begin
                    xs[j+1][l] <= x_in;
                    rr[j+1][l] <= take ? cand : r_in;
                end
            end
            always_ff @(posedge clk) if (advance)
                wstep[j+1] <= (j == 0) ? step : wstep[PREV];
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rstn) begin
            busy <= 1'b0; done <= 1'b0; step <= '0; pvalid <= '0;
        end else if (advance) begin
            done <= 1'b0;
            if (!busy) begin
                busy <= in_valid && in_ready;
                step <= '0;
                if (in_valid && in_ready) held <= in_vec;
            end else begin
                step <= step + 1'b1;
                if (step == STEPS-1) busy <= 1'b0;
            end
            pvalid <= {pvalid[BITS-2:0], busy};
            if (pvalid[BITS-1]) begin
                for (int c = 0; c < LANES; c++)
                    codes[(wstep[BITS]*LANES + c)*BITS +: BITS] <= rr[BITS][c];
                if (wstep[BITS] == STEPS-1) done <= 1'b1;
            end
        end
    end

    assign out_valid = done;
    assign out_codes = codes;
endmodule
