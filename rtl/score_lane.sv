// One cached row in, one score out.  `ops/attention.py::CompressedAttention.scores`.
//
//     s_t = clamp( (sum_ch qtab[ch][kcode[ch]]) * k_norm  >>  SCORE_SHIFT )
//
// THE INNER LOOP HAS NO MULTIPLIER IN IT
// --------------------------------------
// The per-channel product `q_rot[ch] * centroid[kcode[ch]]` was already
// computed by `qtab_build.sv`, once per query vector, because the codebook has
// only 2**KEY_BITS distinct values.  So a cached token costs D gathers, a
// six-level adder tree, and ONE multiply by the row's norm -- two DSPs, not
// sixty-four.  Four of these lanes cost ~2% of the DSPs a dense attention unit
// would need for the same work, which is the whole reason this design is
// bound by DRAM and not by arithmetic.
//
// THE LANE SPLITS THE ROW ITSELF
// ------------------------------
// `row_data` is the 52 bytes `ops/quantize.py::pack` wrote, not pre-separated
// fields.  A bench that passes has therefore agreed with `_pack_codes`'s bit
// layout rather than with a second decoder written to match it, which is the
// same reason `tb_rot_encode` checks its goldens against `kv_rows.hex`.
//
// The value plane rides through untouched.  `accum.sv` needs `v_idx` and
// `v_norm` for the SAME token whose score is emerging here, and re-reading the
// row downstream would mean two readers of one DDR stream.  Carrying them is a
// shift register; splitting the stream would be a second prefetcher.
//
// THE NORM IS SIGNED
// ------------------
// `_unpack_word` sign-extends, so the model multiplies by a SIGNED 16-bit
// norm.  Norms come out of `isqrt` and are non-negative in every reachable
// case, but "unreachable" is not "impossible" and the arithmetic here has to
// be the arithmetic the golden performs, not the arithmetic its inputs happen
// to make equivalent.
//
// WHY THE MULTIPLY IS SPLIT IN TWO
// --------------------------------
// A 49x16 product is not one DSP48: the primitive is 27x18.  Splitting the
// dot product at bit 25 makes it two independent DSP-sized multiplies plus a
// shifted add, which is what the device has, rather than leaving synthesis to
// build a cascade whose depth lands in one clock period.
`timescale 1ns/1ps
module score_lane #(
    parameter int D            = 64,
    parameter int IN_BITS      = 24,   // Q8.16 lane
    parameter int KEY_BITS     = 4,
    parameter int VAL_BITS     = 2,
    parameter int NORM_BITS    = 16,
    parameter int CENT_BITS    = 20,
    // qk_frac + centroid_frac + norm_frac - acc_frac.  `FixedFormat.score_shift_for`.
    parameter int SCORE_SHIFT  = 24,
    parameter int SCORE_WIDTH  = 48,   // FixedFormat.score_width
    parameter int PROD_W       = IN_BITS + CENT_BITS - 1,
    // `pack`'s row, byte for byte: k codes, v codes, k_norm, v_norm.
    parameter int ROW_BITS     = D*KEY_BITS + D*VAL_BITS + 2*NORM_BITS,
    parameter logic [(2**KEY_BITS)*CENT_BITS-1:0] CENTROIDS =
        {20'h15d79, 20'h1087b, 20'h0cebb, 20'h0a06d,
         20'h0784a, 20'h053cb, 20'h0317b, 20'h0105d,
         20'hfefa3, 20'hfce85, 20'hfac35, 20'hf87b6,
         20'hf5f93, 20'hf3145, 20'hef785, 20'hea287}
) (
    input  logic clk,
    input  logic rstn,

    // -- the query: one table build per query vector --------------------
    input  logic q_valid,
    output logic q_ready,
    input  logic signed [D*IN_BITS-1:0] q_vec,
    output logic tab_ready,             // a table is built, awaiting commit
    input  logic tab_commit,            // swap it in; the previous scan has ended

    // -- the scan: one cached row per cycle -----------------------------
    input  logic row_valid,
    output logic row_ready,
    input  logic [ROW_BITS-1:0] row_data,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [SCORE_WIDTH-1:0] out_score,
    output logic [D*VAL_BITS-1:0] out_vcodes,
    output logic signed [NORM_BITS-1:0] out_vnorm,
    // The golden clamps and counts; so does this.  A score that saturated is
    // a numeric event worth a signal, never something to discover as a
    // plateau in an accuracy plot.
    output logic out_clip
);
    localparam int KCODE_BITS = D * KEY_BITS;
    localparam int VCODE_BITS = D * VAL_BITS;
    localparam int DOT_W      = PROD_W + $clog2(D);
    localparam int LO_W       = 25;                       // DSP-sized split
    localparam int HI_W       = DOT_W - LO_W;
    localparam int RAW_W      = DOT_W + NORM_BITS;
    localparam int STAGES     = 8;                        // accept -> out_valid

    generate
        if (SCORE_SHIFT < 0)
            initial $fatal(1, "a negative SCORE_SHIFT is a left shift; say so");
        if (HI_W < 1) initial $fatal(1, "DOT_W must exceed the low split");
    endgenerate

    wire advance = !out_valid || out_ready;
    assign row_ready = advance;

    logic [STAGES-1:0] vld;
    assign out_valid = vld[STAGES-1];

    // -- P0: accept the row and split it -----------------------------------
    // Offsets are `pack`'s, in its order: k codes, v codes, k_norm, v_norm.
    logic [KCODE_BITS-1:0] p0_kcodes;
    logic [VCODE_BITS-1:0] vcodes [0:STAGES-2];
    // Deep enough to reach the multiply, which reads it at P5 -- five
    // stages after the row was accepted, not two.
    logic signed [NORM_BITS-1:0] knorm [0:4];
    logic signed [NORM_BITS-1:0] vnorm [0:STAGES-2];

    wire [KCODE_BITS-1:0] w_kcodes = row_data[0 +: KCODE_BITS];
    wire [VCODE_BITS-1:0] w_vcodes = row_data[KCODE_BITS +: VCODE_BITS];
    wire signed [NORM_BITS-1:0] w_knorm =
        row_data[KCODE_BITS + VCODE_BITS +: NORM_BITS];
    wire signed [NORM_BITS-1:0] w_vnorm =
        row_data[KCODE_BITS + VCODE_BITS + NORM_BITS +: NORM_BITS];

    // -- P1: the gather ----------------------------------------------------
    logic [D*PROD_W-1:0] prod;

    qtab_build #(.D(D), .IN_BITS(IN_BITS), .BITS(KEY_BITS),
                 .CENT_BITS(CENT_BITS), .CENTROIDS(CENTROIDS), .PROD_W(PROD_W))
    tab (
        .clk, .rstn,
        .in_valid(q_valid), .in_ready(q_ready), .in_q(q_vec),
        .out_valid(tab_ready), .commit(tab_commit),
        .gat_en(advance), .gat_codes(p0_kcodes), .gat_prod(prod));

    // -- P2..P4: 64 -> 1 in six levels, registered every two ---------------
    // Two levels per cycle and not six: six chained adders at 43+ bits is most
    // of a 4 ns period, and this pipe must retire one row EVERY cycle for the
    // whole scan.  The latency is paid once per token; the throughput is paid
    // T times.
    localparam int L2_W = PROD_W + 2;
    localparam int L4_W = PROD_W + 4;
    logic signed [L2_W-1:0] l2 [0:D/4-1];
    logic signed [L4_W-1:0] l4 [0:D/16-1];
    logic signed [DOT_W-1:0] dot;

    // -- P5..P6: one multiply by the norm, split DSP-sized -----------------
    logic signed [HI_W+NORM_BITS-1:0] p_hi;
    logic signed [LO_W+NORM_BITS:0]   p_lo;   // {1'b0, lo} is LO_W+1 bits signed
    logic signed [RAW_W-1:0] raw;

    localparam logic signed [SCORE_WIDTH-1:0] S_HI = {1'b0, {(SCORE_WIDTH-1){1'b1}}};
    localparam logic signed [SCORE_WIDTH-1:0] S_LO = {1'b1, {(SCORE_WIDTH-1){1'b0}}};

    always_ff @(posedge clk) begin
        if (!rstn) begin
            vld <= '0;
        end else if (advance) begin
            vld <= {vld[STAGES-2:0], (row_valid && row_ready)};

            // P0
            if (row_valid && row_ready) begin
                p0_kcodes <= w_kcodes;
                vcodes[0] <= w_vcodes;
                knorm[0]  <= w_knorm;
                vnorm[0]  <= w_vnorm;
            end
            // P1 -- `prod` is registered inside qtab_build.
            for (int i = 1; i <= 4; i++) knorm[i] <= knorm[i-1];
            for (int i = 1; i <= STAGES-2; i++) begin
                vcodes[i] <= vcodes[i-1];
                vnorm[i]  <= vnorm[i-1];
            end

            // P2: levels 1-2, 64 -> 16
            for (int i = 0; i < D/4; i++)
                l2[i] <= L2_W'($signed(prod[(4*i+0)*PROD_W +: PROD_W]))
                       + L2_W'($signed(prod[(4*i+1)*PROD_W +: PROD_W]))
                       + L2_W'($signed(prod[(4*i+2)*PROD_W +: PROD_W]))
                       + L2_W'($signed(prod[(4*i+3)*PROD_W +: PROD_W]));
            // P3: levels 3-4, 16 -> 4
            for (int i = 0; i < D/16; i++)
                l4[i] <= L4_W'(l2[4*i+0]) + L4_W'(l2[4*i+1])
                       + L4_W'(l2[4*i+2]) + L4_W'(l2[4*i+3]);
            // P4: levels 5-6, 4 -> 1
            dot <= DOT_W'(l4[0]) + DOT_W'(l4[1]) + DOT_W'(l4[2]) + DOT_W'(l4[3]);

            // P5: two DSP-sized products of the same 49x16 multiply
            p_hi <= $signed(dot[DOT_W-1 -: HI_W]) * knorm[4];
            p_lo <= $signed({1'b0, dot[LO_W-1:0]}) * knorm[4];
            // P6
            raw <= (RAW_W'(p_hi) <<< LO_W) + RAW_W'(p_lo);
            // P7: floor shift then clamp, in that order -- `rshift` truncates
            // toward minus infinity and the model clamps what it produced.
            out_clip <= 1'b0;
            if ((raw >>> SCORE_SHIFT) > S_HI) begin
                out_score <= S_HI; out_clip <= 1'b1;
            end else if ((raw >>> SCORE_SHIFT) < S_LO) begin
                out_score <= S_LO; out_clip <= 1'b1;
            end else begin
                out_score <= SCORE_WIDTH'(raw >>> SCORE_SHIFT);
            end
            out_vcodes <= vcodes[STAGES-2];
            out_vnorm  <= vnorm[STAGES-2];
        end
    end
endmodule
