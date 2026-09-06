// `exp(-delta)` as `2^-i * 2^-f`: a 256-entry table on the fraction and a
// shift on the integer part.  `ops/attention.py::ExpLut`.
//
// Why a table on [0,1) and not on the whole range
// -----------------------------------------
// `delta` is a distance below a running maximum, in Q(acc_frac), and it is
// unbounded above.  Converting to base 2 -- `exp(-d) = 2^(-d*log2e)` -- splits
// the exponent into an integer part, which is a right shift, and a fraction in
// [0,1), which is all the table has to cover.  So the table is 2**LUT_BITS
// entries no matter how large delta gets.
//
// The two-step form, not the flat one
// -----------------------------------
// `ExpLut.__call__` collapses table-then-shift into one gather over 4,353
// entries.  That is a numpy optimisation -- it was 28% of the model's runtime
// as two passes -- and `check_exp_lut_flat_matches_two_step` asserts the two
// agree on every reachable input.  In hardware the collapse is a pessimisation:
// 4,353 x 16 b is 68 Kb of block RAM per lane, replicated four times, against
// 256 x 16 b of distributed ROM and a barrel shift.  `ExpLut.two_step` is the
// hardware form and the flat one is the model's.
//
// Clamping delta is exact, not a safety net
// -----------------------------------------
// Every delta at or above `DELTA_MAX` maps to the table's terminal zero, so
// the clamp cannot change a result -- it only keeps the multiply at 20x17
// bits and therefore inside a single DSP.  `ExpLut._delta_max` is the same
// number and `tb_exp_lut` checks the parameter against it.
`timescale 1ns/1ps
module exp_lut #(
    parameter int DELTA_BITS   = 48,   // a difference of two Q(acc_frac) scores
    parameter int ACC_FRAC     = 16,
    parameter int LUT_BITS     = 8,    // FixedFormat.exp_lut_bits
    parameter int PROB_BITS    = 16,   // FixedFormat.prob_width
    parameter int MAX_INT_PART = 16,   // prob_frac + 1; beyond it the result is 0
    parameter int LOG2E        = 94548,   // log2(e) in Q(ACC_FRAC)
    parameter int DELTA_MAX    = 772248   // ExpLut._delta_max
) (
    input  logic clk,
    input  logic rstn,
    input  logic en,
    input  logic [DELTA_BITS-1:0] in_delta,     // non-negative by construction
    output logic [PROB_BITS-1:0]  out_p         // Q(prob_frac)
);
    localparam int SAT_W = $clog2(DELTA_MAX + 1);
    localparam int T_W   = SAT_W + $clog2(LOG2E + 1) - ACC_FRAC + 1;
    localparam int I_W   = $clog2(MAX_INT_PART + 2);

    // The fraction table, a format artifact of `ExpLut.__init__` -- never a
    // second evaluation of 2**-f here.  Packed LSB-first -- entry `i` at
    // [PROB_BITS*i +: PROB_BITS] -- the same convention as `rot_encode`'s
    // Bounds and `qtab_build`'s centroids.
    localparam logic [(1<<LUT_BITS)*PROB_BITS-1:0] TABLE = {
        16'h402c, 16'h4059, 16'h4086, 16'h40b2, 16'h40df, 16'h410c, 16'h4139, 16'h4167,
        16'h4194, 16'h41c2, 16'h41ef, 16'h421d, 16'h424b, 16'h4279, 16'h42a7, 16'h42d5,
        16'h4304, 16'h4332, 16'h4361, 16'h4390, 16'h43bf, 16'h43ee, 16'h441d, 16'h444c,
        16'h447b, 16'h44ab, 16'h44db, 16'h450a, 16'h453a, 16'h456a, 16'h459b, 16'h45cb,
        16'h45fb, 16'h462c, 16'h465d, 16'h468d, 16'h46be, 16'h46f0, 16'h4721, 16'h4752,
        16'h4784, 16'h47b5, 16'h47e7, 16'h4819, 16'h484b, 16'h487d, 16'h48af, 16'h48e2,
        16'h4914, 16'h4947, 16'h497a, 16'h49ad, 16'h49e0, 16'h4a13, 16'h4a47, 16'h4a7a,
        16'h4aae, 16'h4ae2, 16'h4b16, 16'h4b4a, 16'h4b7e, 16'h4bb3, 16'h4be7, 16'h4c1c,
        16'h4c51, 16'h4c86, 16'h4cbb, 16'h4cf0, 16'h4d26, 16'h4d5b, 16'h4d91, 16'h4dc7,
        16'h4dfd, 16'h4e33, 16'h4e69, 16'h4e9f, 16'h4ed6, 16'h4f0d, 16'h4f44, 16'h4f7b,
        16'h4fb2, 16'h4fe9, 16'h5021, 16'h5058, 16'h5090, 16'h50c8, 16'h5100, 16'h5138,
        16'h5171, 16'h51a9, 16'h51e2, 16'h521b, 16'h5254, 16'h528d, 16'h52c6, 16'h52ff,
        16'h5339, 16'h5373, 16'h53ad, 16'h53e7, 16'h5421, 16'h545b, 16'h5496, 16'h54d1,
        16'h550c, 16'h5547, 16'h5582, 16'h55bd, 16'h55f9, 16'h5634, 16'h5670, 16'h56ac,
        16'h56e8, 16'h5725, 16'h5761, 16'h579e, 16'h57db, 16'h5818, 16'h5855, 16'h5892,
        16'h58cf, 16'h590d, 16'h594b, 16'h5989, 16'h59c7, 16'h5a05, 16'h5a44, 16'h5a82,
        16'h5ac1, 16'h5b00, 16'h5b3f, 16'h5b7f, 16'h5bbe, 16'h5bfe, 16'h5c3e, 16'h5c7e,
        16'h5cbe, 16'h5cfe, 16'h5d3f, 16'h5d80, 16'h5dc1, 16'h5e02, 16'h5e43, 16'h5e84,
        16'h5ec6, 16'h5f08, 16'h5f4a, 16'h5f8c, 16'h5fce, 16'h6011, 16'h6053, 16'h6096,
        16'h60d9, 16'h611c, 16'h6160, 16'h61a3, 16'h61e7, 16'h622b, 16'h626f, 16'h62b4,
        16'h62f8, 16'h633d, 16'h6382, 16'h63c7, 16'h640c, 16'h6451, 16'h6497, 16'h64dd,
        16'h6523, 16'h6569, 16'h65af, 16'h65f6, 16'h663d, 16'h6684, 16'h66cb, 16'h6712,
        16'h675a, 16'h67a2, 16'h67e9, 16'h6832, 16'h687a, 16'h68c2, 16'h690b, 16'h6954,
        16'h699d, 16'h69e6, 16'h6a30, 16'h6a7a, 16'h6ac4, 16'h6b0e, 16'h6b58, 16'h6ba2,
        16'h6bed, 16'h6c38, 16'h6c83, 16'h6ccf, 16'h6d1a, 16'h6d66, 16'h6db2, 16'h6dfe,
        16'h6e4a, 16'h6e97, 16'h6ee4, 16'h6f30, 16'h6f7e, 16'h6fcb, 16'h7019, 16'h7066,
        16'h70b4, 16'h7103, 16'h7151, 16'h71a0, 16'h71ef, 16'h723e, 16'h728d, 16'h72dd,
        16'h732c, 16'h737c, 16'h73cc, 16'h741d, 16'h746d, 16'h74be, 16'h750f, 16'h7560,
        16'h75b2, 16'h7604, 16'h7655, 16'h76a8, 16'h76fa, 16'h774d, 16'h779f, 16'h77f2,
        16'h7846, 16'h7899, 16'h78ed, 16'h7941, 16'h7995, 16'h79e9, 16'h7a3e, 16'h7a93,
        16'h7ae8, 16'h7b3d, 16'h7b93, 16'h7be8, 16'h7c3e, 16'h7c95, 16'h7ceb, 16'h7d42,
        16'h7d99, 16'h7df0, 16'h7e47, 16'h7e9f, 16'h7ef7, 16'h7f4f, 16'h7fa7, 16'h8000
    };

    // S0 -- clamp.  Exact: everything at or above DELTA_MAX is already zero.
    logic [SAT_W-1:0] s0_d;
    // S1 -- delta * log2(e), Q(ACC_FRAC) after the shift.  One DSP.
    logic [T_W-1:0] s1_t;
    // S2 -- split, gather, shift.
    logic [PROB_BITS-1:0] s2_p;

    wire [SAT_W-1:0] clamped =
        (in_delta > DELTA_BITS'(DELTA_MAX)) ? SAT_W'(DELTA_MAX) : SAT_W'(in_delta);
    wire [SAT_W+$clog2(LOG2E+1):0] scaled = s0_d * LOG2E;
    wire [I_W-1:0] i_part = s1_t[T_W-1:ACC_FRAC];
    wire [LUT_BITS-1:0] addr = s1_t[ACC_FRAC-1 -: LUT_BITS];

    always_ff @(posedge clk) begin
        if (!rstn) begin
            s0_d <= '0; s1_t <= '0; s2_p <= '0;
        end else if (en) begin
            s0_d <= clamped;
            s1_t <= T_W'(scaled >> ACC_FRAC);
            s2_p <= (i_part > I_W'(MAX_INT_PART))
                    ? '0 : (TABLE[addr*PROB_BITS +: PROB_BITS] >> i_part);
        end
    end

    assign out_p = s2_p;
endmodule
