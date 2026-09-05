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
    // The normalised divide's second half. Exposed for the same reason
    // `out_recip` is: a wrong shift and a wrong quotient both come out as a
    // wrong vector, and a bench that can only see the vector cannot say which.
    output logic [$clog2(L_WIDTH)+1:0] out_shift,
    // A channel the seam cannot carry. Never a silent wrap, never a clamp.
    output logic range_error
);
    // THE DIVIDE IS NORMALISED, AND THAT IS A NUMERIC CHANGE
    // ------------------------------------------------------
    // This used to be `floor(2**31 / l)` at a fixed `RECIP_FRAC`, which loses
    // precision linearly in context: the online denominator is `l = T*2**15`,
    // so the quotient was `2**16/T` -- 64 levels at ctx 1,024, 2 at 32,768, 1
    // at 65,536, and 0 at 131,072, where `l_too_big` fired and every channel
    // came out zero. The error was the fractional part of `2**31/l`, so it was
    // a LOTTERY: 0.2% at powers of two, 25% at ctx 24,576. Every context this
    // project had tested is a power of two.
    //
    // So `l` is normalised INTO the divisor's width instead, and the numerator
    // is the constant `2**(2*RECIP_BITS-2)`. The quotient then always lands in
    // (2**30, 2**31] -- 31 significant bits at EVERY context, with no ceiling
    // -- and the caller shifts by `16 + s` rather than by `RECIP_FRAC`, where
    // `s` is the position of `l`'s leading one. `ops/attention.py::reciprocal`
    // is the same arithmetic and the goldens come from it.
    localparam int NUM_SHIFT  = 2 * RECIP_BITS - 2;           // 62
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

    typedef enum logic [2:0] {IDLE, NORM, DIV, MUL, DONE} state_t;
    state_t st;

    logic [D*ACC_WIDTH-1:0] acc_h;
    logic [L_WIDTH-1:0]     l_h;
    logic [RECIP_BITS-1:0]  q;
    logic [RECIP_BITS-1:0]  divisor_r;
    logic [REM_W-1:0]       rem;
    logic [$clog2(RECIP_BITS):0] dstep;
    logic [$clog2(GROUPS)+1:0]   mstep;
    logic mvld;

    // `l`'s leading one. A priority encoder, and the only new logic the
    // normalisation costs on the divisor side. There is no `l_too_big` any
    // more: normalising removed the ceiling that made it necessary.
    //
    // IT GETS ITS OWN STATE, BECAUSE IT WAS THE CRITICAL PATH.
    // `l_h` is constant for the whole divide, so this encoder and the shifter
    // under it were being recomputed on all 32 iterations to produce the same
    // number -- and in series with the remainder's compare-and-subtract, which
    // routed to 13 logic levels and WNS +0.024 ns. Synthesis will not hoist it
    // out: `divisor` feeds `rem`, so it is inside the loop as written. `NORM`
    // takes it out by hand, at one cycle per query head at the end of a
    // T-cycle scan. Read from the routed report, not guessed -- the first
    // attempt split the output barrel shifter instead and bought 0.002 ns.
    logic [$clog2(L_WIDTH)-1:0] lead;
    always_comb begin
        lead = '0;
        for (int b = 1; b < L_WIDTH; b++)
            if (l_h[b]) lead = ($clog2(L_WIDTH))'(b);
    end

    // Shift `l` so its leading one sits at bit RECIP_BITS-1: right when it is
    // large, left when it is small. Exactly one of the two is ever non-zero.
    wire [$clog2(L_WIDTH)-1:0] shr = (lead >= RECIP_BITS-1)
                                   ? (lead - ($clog2(L_WIDTH))'(RECIP_BITS-1)) : '0;
    wire [$clog2(L_WIDTH)-1:0] shl = (lead >= RECIP_BITS-1)
                                   ? '0 : (($clog2(L_WIDTH))'(RECIP_BITS-1) - lead);
    wire [L_WIDTH-1:0] l_shifted = (l_h >> shr) << shl;
    wire [RECIP_BITS-1:0] divisor = l_shifted[RECIP_BITS-1:0];

    // The total right shift the product takes: NUM_SHIFT - PROB_FRAC + (lead -
    // (RECIP_BITS-1)), which is 16 + lead and therefore never negative.
    logic [$clog2(L_WIDTH)+1:0] oshift, oshift_h;
    assign oshift = ($clog2(L_WIDTH)+2)'(NUM_SHIFT - PROB_FRAC - (RECIP_BITS-1)) +
                    ($clog2(L_WIDTH)+2)'(lead);

    // One restoring step. The dividend is a single set bit far above the
    // divisor, so every bit shifted in from here is zero and the remainder is
    // SEEDED at 2**(RECIP_BITS-1) rather than at 0 -- which is the alignment
    // step, skipped rather than iterated.
    wire [REM_W-1:0] rem_shifted = {rem[REM_W-2:0], 1'b0};
    wire fits = rem_shifted >= {1'b0, divisor_r};

    assign in_ready = (st == IDLE);
    assign out_valid = (st == DONE);
    assign out_recip = q;
    assign out_shift = oshift_h;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= IDLE; q <= '0; rem <= '0; dstep <= '0; mstep <= '0;
            mvld <= 1'b0;
        end else begin
            case (st)
                IDLE: if (in_valid) begin
                    acc_h <= in_acc; l_h <= in_l;
                    st <= NORM;
                end

                // The encoder, the normaliser and the output shift, once.
                NORM: begin
                    divisor_r <= divisor;
                    oshift_h  <= oshift;
                    // Seeded, not zeroed: a restoring divide producing an
                    // n-bit quotient starts with the dividend's bits ABOVE the
                    // low n, and this dividend is the single bit 2**NUM_SHIFT.
                    // So the seed is `NUM >> RECIP_BITS`, and the first
                    // RECIP_BITS steps of the textbook loop -- which would
                    // produce nothing but that alignment -- are skipped.
                    q <= '0; rem <= REM_W'(1) << (NUM_SHIFT - RECIP_BITS);
                    dstep <= '0;
                    // A zero denominator answers zero and skips the 32
                    // iterations rather than producing that answer the slow
                    // way. There is no "too wide" case any more: `divisor` is
                    // `l` normalised, so it is always exactly RECIP_BITS.
                    if (l_h == '0) begin
                        mstep <= '0; mvld <= 1'b0; st <= MUL;
                    end else
                        st <= DIV;
                end

                // One restoring step per cycle, on the REGISTERED divisor:
                // nothing but the remainder's compare-and-subtract is in this
                // loop now.
                DIV: begin
                    rem <= fits ? (rem_shifted - {1'b0, divisor_r}) : rem_shifted;
                    q   <= {q[RECIP_BITS-2:0], fits};
                    dstep <= dstep + 1'b1;
                    if (dstep == RECIP_BITS-1) begin
                        mstep <= '0; mvld <= 1'b0; st <= MUL;
                    end
                end
                MUL: begin
                    mvld <= (mstep < GROUPS);
                    mstep <= mstep + 1'b1;
                    // Four drain cycles: select, multiply, shift coarse,
                    // shift fine, write back. The shift became two stages
                    // when the divide was normalised -- see below.
                    if (mstep == GROUPS + 4) st <= DONE;
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
    // The SELECT is its own stage, for the same reason `accum`'s fold has one:
    // `mstep` picks 4 of 64 accumulator words through a wide mux, and reading
    // through that mux and a 32x32 product -- two cascaded DSPs -- in one cycle
    // routed to -0.157 ns at 250 MHz once four of these were on the die,
    // against +0.863 ns for the block alone. Registering the operand puts the
    // flop on the DSP's A input, where the cascade wants it.
    //
    // The arithmetic is untouched, so every golden holds; the divide costs one
    // more cycle, once per query head at the END of a T-cycle scan.
    logic [$clog2(GROUPS)+1:0] mstep_d, mstep_d2, mstep_d3, mstep_d4;
    logic mvld_d, mvld_d2, mvld_d3;
    logic signed [ACC_WIDTH-1:0] sel [0:LANES-1];
    logic signed [PROD_W-1:0] prod [0:LANES-1];
    logic signed [PROD_W-1:0] shc  [0:LANES-1];
    logic signed [PROD_W-1:0] shp  [0:LANES-1];
    // `mstep` runs two past the last group so the pipe drains; clamping the
    // read index keeps those cycles from part-selecting off the end of `acc_h`
    // and pushing X into a product nothing reads.
    wire [$clog2(GROUPS)-1:0] rgrp =
        (mstep < GROUPS) ? mstep[$clog2(GROUPS)-1:0] : '0;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            range_error <= 1'b0;
            mstep_d <= '0; mstep_d2 <= '0; mstep_d3 <= '0; mstep_d4 <= '0;
            mvld_d <= 1'b0; mvld_d2 <= 1'b0; mvld_d3 <= 1'b0;
        end else begin
        if (st == IDLE && in_valid) range_error <= 1'b0;
        mstep_d  <= mstep;
        mstep_d2 <= mstep_d;
        mstep_d3 <= mstep_d2;
        mstep_d4 <= mstep_d3;
        mvld_d   <= mvld;
        mvld_d2  <= mvld_d;
        mvld_d3  <= mvld_d2;
        for (int i = 0; i < LANES; i++)
            sel[i] <= $signed(acc_h[(rgrp*LANES + i)*ACC_WIDTH +: ACC_WIDTH]);
        for (int i = 0; i < LANES; i++)
            prod[i] <= $signed(PROD_W'(sel[i])) * $signed({1'b0, q});
        // THE SHIFT IS TWO STAGES, AND THAT IS THE NORMALISATION'S BILL.
        //
        // It used to be `>>> RECIP_FRAC`, a constant, which is free wiring.
        // Normalising made it `>>> oshift_h` -- a 65-bit variable arithmetic
        // shift over a range of 47, so a six-level barrel -- and in the same
        // cycle as the range compare and the output write it took this block
        // from the +0.863 ns step 9 recorded to **-0.028 ns**. Giving it a
        // register of its own recovered only +0.022: still the critical path,
        // and thin enough that `attn_top`, which closed at +0.082 ns in the
        // full system, would very likely have gone negative on it.
        //
        // The barrel is also SPLIT here -- by eights, then by the remainder.
        // `(x >>> a) >>> b == x >>> (a+b)` for an arithmetic shift, because
        // sign extension is idempotent, which is the only reason splitting is
        // legal at all.
        //
        // BUT IT WAS NOT THE FIX, AND THE RECORD SHOULD SAY SO. Splitting it
        // bought 0.002 ns: +0.022 to +0.024. The critical path was somewhere
        // else entirely -- the priority encoder inside the divide loop, see
        // `NORM` -- and taking that out took the block to +1.443 ns, past the
        // +0.863 it had before any of this. The split is KEPT because it costs
        // one cycle per query head at the end of a T-cycle scan and removes a
        // six-level barrel from the output path, where the full system's
        // routing is worse than out of context. It is insurance, not the
        // repair.
        //
        // It also stops the range check from being a second and third shifter:
        // the comparison now reads the registered result rather than
        // recomputing it twice more.
        for (int i = 0; i < LANES; i++)
            shc[i] <= prod[i] >>> {oshift_h[$clog2(L_WIDTH)+1:3], 3'b000};
        for (int i = 0; i < LANES; i++)
            shp[i] <= shc[i] >>> oshift_h[2:0];
        if (mvld_d3) begin
            for (int i = 0; i < LANES; i++) begin
                out_vec[(mstep_d4*LANES + i)*OUT_BITS +: OUT_BITS] <=
                    OUT_BITS'(shp[i]);
                if (!(shp[i] >= -(PROD_W'(1) << (SEAM_BITS-1)) &&
                      shp[i] < (PROD_W'(1) << (SEAM_BITS-1))))
                    range_error <= 1'b1;
            end
        end
        end
    end
endmodule
