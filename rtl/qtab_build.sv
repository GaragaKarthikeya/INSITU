// Query vector -> the 64 x 2**BITS table of `q_rot[ch] * centroid[i]`.
//
// Why this block exists at all
// ----------------------------
// A cached key is `norm * centroid[code]`, so
//
//     <q, k>  =  norm * sum_ch  q_rot[ch] * centroid[ code[ch] ]
//
// and the only distinct values the inner multiply can ever produce are the
// 2**BITS entries of the codebook.  Precomputing them once per query vector
// turns every cached token afterwards into D table gathers and an adder tree.
// That is the whole reason `score_lane.sv` has no multiplier in its inner
// loop, and it is what makes DRAM -- never arithmetic -- the constraint.
//
// The build is amortised over the entire scan: 64 cycles here against T
// cycles there, and T is the context.  At the shortest context this plan
// targets that is already 13x.
//
// One memory per channel, not one memory
// --------------------------------------
// `score_lane` gathers all D channels in the same cycle, each at its own
// address, so this cannot be one wide memory -- it is D independent
// 2*2**BITS-deep, PROD_W-wide memories (the extra bit of depth is the
// ping-pong bank).  At D=64, BITS=4 that is a 32x43 distributed RAM per
// channel, which is LUTs; a single 44 Kb block would serve one channel per
// cycle and starve the tree 64:1.
//
// The fold is over codes, not channels
// ------------------------------------
// Every cycle writes LANES channels at one code index, so each channel's RAM
// takes one write per cycle and its single write port is enough.  Folding the
// other way -- all codes of one channel per cycle -- would need 2**BITS write
// ports on each RAM and 2**BITS times the multipliers to feed them.
//
// PING-PONG, and the commit is explicit
// -------------------------------------
// Group h+1's table is built while group h is still being scanned out of the
// one already committed.  The swap is an input rather than automatic on
// completion, because the builder cannot know when the reader's scan ends and
// swapping mid-scan would rewrite the query under a half-finished sum.
`timescale 1ns/1ps
module qtab_build #(
    parameter int D         = 64,
    parameter int IN_BITS   = 24,   // Q8.16 lane, the seam's width
    parameter int BITS      = 4,    // KEY_BITS
    parameter int CENT_BITS = 20,   // widest gaussian centroid is 89,465
    parameter int LANES     = 16,   // channels multiplied per cycle
    // Ascending centroids in Q(centroid_frac), index 0 in the low bits.  A
    // Format artifact of `Codebook.build`, exactly as `rot_encode`'s bounds
    // and `rot_fwht`'s SIGN_NEG are -- never a second Lloyd-Max solve.
    parameter logic [(2**BITS)*CENT_BITS-1:0] CENTROIDS =
        {20'h15d79, 20'h1087b, 20'h0cebb, 20'h0a06d,
         20'h0784a, 20'h053cb, 20'h0317b, 20'h0105d,
         20'hfefa3, 20'hfce85, 20'hfac35, 20'hf87b6,
         20'hf5f93, 20'hf3145, 20'hef785, 20'hea287},
    // Structural, not measured: a signed IN_BITS by signed CENT_BITS product
    // needs IN_BITS + CENT_BITS - 1 bits and never more.  `hw/vectors.py`
    // asserts every emitted golden fits it, so a width change fails there
    // rather than wrapping here.  A parameter and not a localparam only
    // because the gather port is this wide.
    parameter int PROD_W = IN_BITS + CENT_BITS - 1
) (
    input  logic clk,
    input  logic rstn,

    // -- build side ----------------------------------------------------
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [D*IN_BITS-1:0] in_q,
    output logic out_valid,             // a table is built and awaiting commit
    input  logic commit,                // swap banks; the reader's scan has ended

    // -- gather side ---------------------------------------------------
    // Combinational address in, registered products out, one cycle later.
    input  logic gat_en,
    input  logic [D*BITS-1:0] gat_codes,
    output logic [D*PROD_W-1:0] gat_prod
);
    localparam int NB     = 2**BITS;
    localparam int GROUPS = D / LANES;
    localparam int STEPS  = NB * GROUPS;

    generate
        if (D % LANES != 0) initial $fatal(1, "LANES must divide D");
    endgenerate

    logic bank;                 // the bank the GATHER side reads
    wire  wbank = ~bank;

    // -- build sequencer ---------------------------------------------------
    logic busy, ready_flag;
    logic [$clog2(STEPS)-1:0] step;
    logic signed [D*IN_BITS-1:0] held;

    wire [$clog2(NB)-1:0]     code_s = step[$clog2(STEPS)-1 -: $clog2(NB)];
    wire [$clog2(GROUPS)-1:0] grp_s  = step[$clog2(GROUPS)-1:0];

    // A table under construction is being written; accepting a second query
    // would rewrite it half-built.  A table already built and not yet
    // committed is the spare, so it must not be overwritten either.
    assign in_ready = !busy && !ready_flag;

    // stage 1 -- operands registered
    logic s1_valid;
    logic signed [IN_BITS-1:0]   s1_x    [0:LANES-1];
    logic signed [CENT_BITS-1:0] s1_c;
    logic [$clog2(NB)-1:0]       s1_code;
    logic [$clog2(GROUPS)-1:0]   s1_grp;

    // stage 2 -- products registered, then written
    logic s2_valid;
    logic signed [PROD_W-1:0]  s2_p   [0:LANES-1];
    logic [$clog2(NB)-1:0]     s2_code;
    logic [$clog2(GROUPS)-1:0] s2_grp;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            busy <= 1'b0; ready_flag <= 1'b0; step <= '0;
            s1_valid <= 1'b0; s2_valid <= 1'b0; bank <= 1'b0;
        end else begin
            if (!busy) begin
                if (in_valid && in_ready) begin
                    held <= in_q;
                    busy <= 1'b1;
                    step <= '0;
                end
            end else begin
                step <= step + 1'b1;
                if (step == STEPS-1) busy <= 1'b0;
            end

            s1_valid <= busy;
            s1_c     <= CENTROIDS[code_s*CENT_BITS +: CENT_BITS];
            s1_code  <= code_s;
            s1_grp   <= grp_s;
            for (int l = 0; l < LANES; l++)
                s1_x[l] <= held[(grp_s*LANES + l)*IN_BITS +: IN_BITS];

            s2_valid <= s1_valid;
            s2_code  <= s1_code;
            s2_grp   <= s1_grp;
            for (int l = 0; l < LANES; l++)
                s2_p[l] <= PROD_W'($signed(s1_x[l]) * PROD_W'($signed(s1_c)));

            // The last write retires one cycle after the last multiply.
            if (s2_valid && s2_code == NB-1 && s2_grp == GROUPS-1)
                ready_flag <= 1'b1;
            if (commit && ready_flag) begin
                bank       <= wbank;
                ready_flag <= 1'b0;
            end
        end
    end

    assign out_valid = ready_flag;

    // -- the tables, one independent memory per channel ---------------------
    //
    // D separate arrays and not one two-dimensional one.  The first attempt
    // declared `tab [0:D-1][0:2*NB-1]` and synthesis inferred registers:
    // 93,486 FF and zero LUTRAM for a single lane, which is 374k FF across
    // four and does not fit the part.  Vivado infers distributed RAM from a
    // one-dimensional array with a scalar write address and an asynchronous
    // read, and from very little else; the shape below is that, replicated.
    //
    // Each channel's write enable is a comparison against the group being
    // built, so the address into any one memory is `{bank, code}` alone --
    // 2**BITS deep per bank, which is a RAM32X1D at D=64, BITS=4.
    generate
        genvar ch;
        for (ch = 0; ch < D; ch = ch + 1) begin : gather
            localparam int GRP = ch / LANES;
            localparam int LN  = ch % LANES;

            logic [PROD_W-1:0] tab [0:2*NB-1];
            wire we = s2_valid && (s2_grp == GRP[$clog2(GROUPS)-1:0]);
            wire [$clog2(NB)-1:0] code = gat_codes[ch*BITS +: BITS];

            always_ff @(posedge clk) begin
                if (we) tab[{wbank, s2_code}] <= s2_p[LN];
                if (gat_en) gat_prod[ch*PROD_W +: PROD_W] <= tab[{bank, code}];
            end
        end
    endgenerate
endmodule
