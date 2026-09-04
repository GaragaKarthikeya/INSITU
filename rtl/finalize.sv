// Divide by the softmax denominator.  `ops/attention.py::_finalize`.
//
//     recip   = (1 << (recip_frac + prob_frac)) / l        ONE integer divide
//     out[ch] = (acc[ch] * recip) >> recip_frac            D multiplies
//
// ONE DIVIDE PER HEAD, NOT PER CHANNEL AND NOT PER TOKEN
// ------------------------------------------------------
// A divider is the most expensive thing in this datapath.  Inverting `l` once
// and multiplying gives the same integer, so the cost is one divide per query
// head per step against 64 x T if the normalisation were folded into the scan.
// It also CANNOT be folded into the scan: `l` is not final until the last
// cached token has been scored, which is the defining property of the online
// softmax and the reason `accum` carries an unnormalised sum at all.
//
// THIS IS THE SEAM
// ----------------
// The output is `merge_heads(out)` in Q16 from `kernel.py:236` -- exactly what
// the FPGA returns, just before `to_float` and the `W_o'` projection.  Nothing
// downstream of this block exists on the PL, because `R^-1` is folded into
// `W_o` offline.
//
// l = 0 IS A REAL INPUT
// ---------------------
// A masked or padded position scores nothing, so its denominator is zero.
// `reciprocal` answers with zero rather than dividing, and the result is the
// zero vector.  It is an input to accept, not an error to raise.
//
// THE RESULT IS NOT CLAMPED, AND THAT IS DELIBERATE
// -------------------------------------------------
// `_finalize` does not clamp, so neither does this: a clamp that fired would
// make the block disagree with the model it is checked against.  The output is
// carried at OUT_BITS and `range_error` reports a value the 24-bit seam could
// not hold -- the same convention `rot_fwht` uses, and for the same reason.
// In practice the output is a convex combination of decoded value centroids
// and sits far inside Q8.16; the 32 real heads are checked against that.
//
// SLOW ON PURPOSE
// ---------------
// 32 cycles of restoring division and 16 of folded multiply, once per head at
// the END of a T-cycle scan.  Widening either buys back cycles nothing is
// waiting on, and takes DSPs from the score lanes, which are the thing that is
// actually starved.  The same argument that sized `rot_norm`'s fold.
`timescale 1ns/1ps
module finalize #(
    parameter int D          = 64,
    parameter int ACC_WIDTH  = 32,    // FixedFormat.acc_width
    parameter int L_WIDTH    = 48,
    parameter int PROB_FRAC  = 15,
    parameter int RECIP_FRAC = 16,
    parameter int OUT_BITS   = 48,
    parameter int SEAM_BITS  = 24,    // FixedFormat.qk_width
    parameter int LANES      = 4,     // channels multiplied per cycle
    // The numerator is a compile-time power of two, so the divider's
    // dividend is constant and only the divisor varies. 2**31 needs 32 b.
    // A parameter and not a localparam only because a port is this wide.
    parameter int RECIP_BITS = RECIP_FRAC + PROB_FRAC + 1
) (
    input  logic clk,
    input  logic rstn,

    input  logic in_valid,
    output logic in_ready,
    input  logic [D*ACC_WIDTH-1:0] in_acc,
    input  logic [L_WIDTH-1:0] in_l,

    output logic out_valid,
    input  logic out_ready,
    output logic [D*OUT_BITS-1:0] out_vec,
    output logic [RECIP_BITS-1:0] out_recip,
    // A channel the seam cannot carry. Never a silent wrap, never a clamp.
    output logic range_error
);
    localparam int NUM_BIT    = RECIP_FRAC + PROB_FRAC;       // the set bit of NUM
    localparam int REM_W      = RECIP_BITS + 1;
    localparam int GROUPS     = D / LANES;
    localparam int PROD_W     = ACC_WIDTH + RECIP_BITS + 1;

    generate
        if (D % LANES != 0) initial $fatal(1, "LANES must divide D");
        // The bound is on the reciprocal's VALUE, not on its register width:
        // `l` is at least 1 wherever the divide runs at all, so `recip` is at
        // most 2**(RECIP_FRAC+PROB_FRAC) and the product is at most
        // 2**(ACC_WIDTH-1) * 2**(RECIP_FRAC+PROB_FRAC). After the shift that
        // needs ACC_WIDTH + PROB_FRAC + 1 bits and never more.
        if (OUT_BITS < ACC_WIDTH + PROB_FRAC + 1)
            initial $fatal(1, "OUT_BITS cannot hold the widest quotient product");
    endgenerate

    typedef enum logic [1:0] {IDLE, DIV, MUL, DONE} state_t;
    state_t st;

    logic [D*ACC_WIDTH-1:0] acc_h;
    logic [L_WIDTH-1:0]     l_h;
    logic [RECIP_BITS-1:0]  q;
    logic [REM_W-1:0]       rem;
    logic [$clog2(RECIP_BITS):0] dstep;
    logic [$clog2(GROUPS)+1:0]   mstep;
    logic mvld;

    // `l` at or above 2**RECIP_BITS makes the quotient zero, and a divisor that
    // wide would need a remainder wider than the dividend can ever reach. The
    // guard is exact, not a range check: 2**31 / l is 0 for every l > 2**31.
    wire l_too_big = |l_h[L_WIDTH-1:RECIP_BITS];
    wire [RECIP_BITS-1:0] divisor = l_h[RECIP_BITS-1:0];

    // one restoring step: shift the next dividend bit in, subtract if it fits
    wire [REM_W-1:0] rem_shifted =
        {rem[REM_W-2:0], (dstep == (RECIP_BITS-1-NUM_BIT)) ? 1'b1 : 1'b0};
    wire fits = rem_shifted >= {1'b0, divisor};

    assign in_ready = (st == IDLE);
    assign out_valid = (st == DONE);
    assign out_recip = q;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= IDLE; q <= '0; rem <= '0; dstep <= '0; mstep <= '0;
            mvld <= 1'b0;
        end else begin
            case (st)
                IDLE: if (in_valid) begin
                    acc_h <= in_acc; l_h <= in_l;
                    q <= '0; rem <= '0; dstep <= '0;
                    st <= DIV;
                end
                DIV: begin
                    // A zero denominator and one too wide both answer zero,
                    // and both skip the 32 iterations rather than producing
                    // that answer the slow way.
                    if (l_h == '0 || l_too_big) begin
                        q <= '0;
                        mstep <= '0; mvld <= 1'b0; st <= MUL;
                    end else begin
                        rem <= fits ? (rem_shifted - {1'b0, divisor}) : rem_shifted;
                        q   <= {q[RECIP_BITS-2:0], fits};
                        dstep <= dstep + 1'b1;
                        if (dstep == RECIP_BITS-1) begin
                            mstep <= '0; mvld <= 1'b0; st <= MUL;
                        end
                    end
                end
                MUL: begin
                    mvld <= (mstep < GROUPS);
                    mstep <= mstep + 1'b1;
                    if (mstep == GROUPS + 1) st <= DONE;
                end
                DONE: if (out_ready) st <= IDLE;
            endcase
        end
    end

    // -- the folded multiply ------------------------------------------------
    // `acc` is signed, `recip` is not; the product is signed and the shift is
    // arithmetic, because `rshift` truncates toward minus infinity and `acc`
    // is routinely negative.
    //
    // At THESE widths `>>` and `>>>` happen to agree: the product needs at most
    // ACC_WIDTH + PROB_FRAC + 1 = OUT_BITS bits, so the only place the two
    // differ is bit OUT_BITS, which the truncation discards. Written
    // arithmetically anyway -- it is what the model does, and the equivalence
    // is a property of one parameter set, not of the operation.
    logic [$clog2(GROUPS)+1:0] mstep_d;
    logic signed [PROD_W-1:0] prod [0:LANES-1];
    // `mstep` runs two past the last group so the pipe drains; clamping the
    // read index keeps those cycles from part-selecting off the end of `acc_h`
    // and pushing X into a product nothing reads.
    wire [$clog2(GROUPS)-1:0] rgrp =
        (mstep < GROUPS) ? mstep[$clog2(GROUPS)-1:0] : '0;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            range_error <= 1'b0;
            mstep_d <= '0;
        end else begin
        if (st == IDLE && in_valid) range_error <= 1'b0;
        mstep_d <= mstep;
        for (int i = 0; i < LANES; i++)
            prod[i] <= $signed(PROD_W'($signed(
                           acc_h[(rgrp*LANES + i)*ACC_WIDTH +: ACC_WIDTH]))) *
                       $signed({1'b0, q});
        if (mvld) begin
            for (int i = 0; i < LANES; i++) begin
                out_vec[(mstep_d*LANES + i)*OUT_BITS +: OUT_BITS] <=
                    OUT_BITS'(prod[i] >>> RECIP_FRAC);
                if (!((prod[i] >>> RECIP_FRAC) >= -(PROD_W'(1) << (SEAM_BITS-1)) &&
                      (prod[i] >>> RECIP_FRAC) < (PROD_W'(1) << (SEAM_BITS-1))))
                    range_error <= 1'b1;
            end
        end
        end
    end
endmodule
