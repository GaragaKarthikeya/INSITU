// The online softmax recurrence, one query head.  `ops/attention.py::attend_online`.
//
//     grew = s_t > m
//     if grew:  factor = exp(s_t - m);  l *= factor >> 15;  acc *= factor >> 15;  m = s_t
//     p = exp(m - s_t);  l += p
//
// One pass, because the cache is OFF-DIE
// --------------------------------------
// The two-pass form needs the global maximum before it can accumulate, which
// means reading the cache twice.  DDR is the wall this whole design is built
// around, so the second read is not affordable and the running maximum is.
// `attend_two_pass` and this are not bit-identical -- the online form truncates
// the accumulator once per rescale -- and the golden here is `attend_online`.
//
// Speculate "no new max", because new maxima get rare logarithmically
// -------------------------------------------------------------------
// Between rescales `m` is a constant, so `exp`, `l +=` and `acc +=` are pure
// feed-forward and pipeline at II=1.  The expected number of new maxima in T
// samples is the harmonic number, about ln T + 0.577 -- eleven at a context of
// 32,768 -- so the rare path can afford to stall the pipe entirely rather than
// carry permanent hardware.  The hazard shrinks as context grows, which is the
// regime this design targets.
//
// What a rescale actually costs here
// ----------------------------------
// The maximum must not move while tokens computed against the old one are
// still in flight, so a rescale drains the exp pipe, spends three more cycles
// computing the factor, issues the scale, and waits for `accum` to fold 64
// channels through 8 multipliers.  `in_ready` drops for all of it; the input
// FIFO is what keeps `score_lane` retiring one row per cycle through the
// stall, and `hw_max_fill` reports how deep it actually had to be.
//
// The FIFO carries the value plane too
// ------------------------------------
// `plan.MD` sized this as 16 x 48 b, counting only the score.  `accum` needs
// `v_idx` and `v_norm` for the same token whose probability is emerging, and
// the only other way to get them is a second reader on the DDR stream.  So the
// entry is 48 + D*VAL_BITS + 16 bits wide, not 48.
`timescale 1ns/1ps
module softmax_online #(
    parameter int D           = 64,
    parameter int VAL_BITS    = 2,
    parameter int SCORE_WIDTH = 48,
    parameter int NORM_BITS   = 16,
    parameter int PROB_BITS   = 16,
    parameter int PROB_FRAC   = 15,
    parameter int L_WIDTH     = 48,
    parameter int FIFO_DEPTH  = 32,
    parameter int PAY_BITS    = D*VAL_BITS + NORM_BITS
) (
    input  logic clk,
    input  logic rstn,
    input  logic start,                     // begin a scan: m = -inf, l = 0

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [SCORE_WIDTH-1:0] in_score,
    input  logic [D*VAL_BITS-1:0] in_vcodes,
    input  logic signed [NORM_BITS-1:0] in_vnorm,

    // The op stream to `accum.sv`, in order: a scale always precedes the add
    // of the token that caused it, exactly as the model rescales before adding.
    output logic op_valid,
    input  logic op_ready,
    output logic op_scale,
    output logic [PROB_BITS-1:0] op_factor,
    output logic [PROB_BITS-1:0] op_p,
    output logic [D*VAL_BITS-1:0] op_vcodes,
    output logic signed [NORM_BITS-1:0] op_vnorm,

    output logic signed [SCORE_WIDTH-1:0] out_m,
    output logic [L_WIDTH-1:0] out_l,
    output logic [$clog2(FIFO_DEPTH+1)-1:0] hw_max_fill,   // measured, not assumed
    output logic l_overflow
);
    localparam int LAT = 3;                                  // exp_lut's depth
    localparam logic signed [SCORE_WIDTH-1:0] SCORE_LO =
        {1'b1, {(SCORE_WIDTH-1){1'b0}}};                     // Q.lo, the model's m0
    localparam logic [PROB_BITS-1:0] UNITY = PROB_BITS'(1) << PROB_FRAC;

    // -- input FIFO ---------------------------------------------------------
    localparam int ENTRY = SCORE_WIDTH + PAY_BITS;
    localparam int AW    = $clog2(FIFO_DEPTH);

    logic [ENTRY-1:0] fifo [0:FIFO_DEPTH-1];
    logic [AW:0] wptr, rptr;
    wire [AW:0] fill = wptr - rptr;
    wire full  = fill == FIFO_DEPTH[AW:0];
    wire empty = fill == 0;

    assign in_ready = !full;

    wire [ENTRY-1:0] head = fifo[rptr[AW-1:0]];
    wire signed [SCORE_WIDTH-1:0] h_score = head[ENTRY-1 -: SCORE_WIDTH];
    wire [PAY_BITS-1:0] h_pay = head[PAY_BITS-1:0];

    // -- state --------------------------------------------------------------
    logic signed [SCORE_WIDTH-1:0] m;
    logic [L_WIDTH-1:0] l;
    logic growing;                       // a new maximum is being folded in
    logic signed [SCORE_WIDTH-1:0] g_score;

    // in-flight tags, gated by `adv` so a stalled consumer freezes everything
    logic [LAT-1:0] ivld, ifac;
    logic [PAY_BITS-1:0] ipay [0:LAT-1];

    wire adv = !op_valid || op_ready;
    wire drained = (ivld == '0);

    // A new maximum is detected against the current m, and m only moves once
    // the pipe is drained, so every in-flight token was compared against the
    // same m. That is what makes the speculation safe rather than merely rare.
    wire grew_now = !empty && !growing && (h_score > m);
    wire pop = adv && !empty && !growing && !grew_now;

    // Unsigned, and SCORE_WIDTH is exactly enough: both operands live in
    // Q(score_width) so the widest difference is (2**47-1) - (-2**47), which
    // is 2**48-1. One bit narrower would wrap a huge distance into a small one
    // and turn a probability of zero into one of 1.0.
    wire [SCORE_WIDTH-1:0] delta = growing
        ? SCORE_WIDTH'(g_score - m)      // the factor: exp(new_max - old_max)
        : SCORE_WIDTH'(m - h_score);     // the probability: exp(m - s)

    wire [PROB_BITS-1:0] exp_p;

    exp_lut #(.DELTA_BITS(SCORE_WIDTH)) xp (
        .clk, .rstn, .en(adv), .in_delta(delta), .out_p(exp_p));

    // -- the loop -----------------------------------------------------------
    logic fac_pushed;
    wire push_fac = growing && drained && !fac_pushed;
    wire push_any = pop || push_fac;

    // 48 x 16 needs a 64-bit product; leaving it to the self-determined width
    // of `l` would truncate the rescale to 48 bits and quietly lose the top of
    // every running sum.
    wire [L_WIDTH+PROB_BITS-1:0] l_scaled = ({{PROB_BITS{1'b0}}, l} * exp_p);

    // The 48x16 rescale of `l` is two cycles: the product is registered, and
    // the shift into `l` happens the cycle after. Multiply, shift and the mux
    // into `l` in one cycle was the block's critical path standalone -- as this
    // file's own header predicted -- and the last one left in `attn_top` at
    // +0.043 ns.
    //
    // The factor does not exist any earlier than the cycle it is used: it is
    // the exp pipe's output, and `push_fac` only starts the lookup. So the
    // extra cycle goes after. It is free: the scale op reaches `accum` on the
    // original cycle, `accum` spends nine cycles folding 64 channels, and the
    // next add cannot retire for LAT cycles after pops resume -- so nothing
    // reads `l` in between.
    logic l_pend;
    logic [L_WIDTH+PROB_BITS-1:0] l_prod;

    always_ff @(posedge clk) begin
        if (!rstn || start) begin
            wptr <= '0; rptr <= '0;
            m <= SCORE_LO; l <= '0;
            l_pend <= 1'b0; l_prod <= '0;
            growing <= 1'b0; fac_pushed <= 1'b0;
            ivld <= '0; ifac <= '0;
            hw_max_fill <= '0; l_overflow <= 1'b0;
        end else begin
            if (in_valid && in_ready) begin
                fifo[wptr[AW-1:0]] <= {in_score, in_vcodes, in_vnorm};
                wptr <= wptr + 1'b1;
            end
            if (fill > hw_max_fill) hw_max_fill <= fill;

            if (adv) begin
                ivld <= {ivld[LAT-2:0], push_any};
                ifac <= {ifac[LAT-2:0], push_fac};
                for (int i = LAT-1; i > 0; i--) ipay[i] <= ipay[i-1];
                ipay[0] <= h_pay;

                if (pop) rptr <= rptr + 1'b1;
                if (grew_now && !growing) begin
                    growing <= 1'b1;
                    g_score <= h_score;
                    fac_pushed <= 1'b0;
                end
                if (push_fac) fac_pushed <= 1'b1;

                // The op retiring this cycle updates `l` in the same order it
                // updates `acc`, so a scale lands on the l that all earlier
                // ADDs have already reached.
                // The pending product lands first: an add cannot retire in the
                // cycle after a scale (the pipe was drained to issue it), so
                // these never both write `l`.
                l_pend <= 1'b0;
                if (l_pend) l <= L_WIDTH'(l_prod >> PROB_FRAC);

                if (ivld[LAT-1]) begin
                    if (ifac[LAT-1]) begin
                        l_prod <= l_scaled;
                        l_pend <= 1'b1;
                        m <= g_score;
                        growing <= 1'b0;
                        fac_pushed <= 1'b0;
                    end else begin
                        l <= l + L_WIDTH'(exp_p);
                        // Sticky: a running sum that overflowed once is wrong
                        // for the rest of the scan, so the flag must not clear.
                        if (&l[L_WIDTH-1 -: 2]) l_overflow <= 1'b1;
                    end
                end
            end
        end
    end

    assign op_valid  = ivld[LAT-1];
    assign op_scale  = ifac[LAT-1];
    assign op_factor = exp_p;
    assign op_p      = ifac[LAT-1] ? UNITY : exp_p;
    assign op_vcodes = ipay[LAT-1][PAY_BITS-1 -: D*VAL_BITS];
    assign op_vnorm  = ipay[LAT-1][NORM_BITS-1:0];
    assign out_m = m;
    assign out_l = l;
endmodule
