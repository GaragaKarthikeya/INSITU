// The block: 512-bit stream in, 32 attention heads out, cache in DDR.
//
// What this file is
// -----------------
// Every block below it has been checked on its own against a golden the
// unmodified kernel produced. This one is the control that makes them one
// datapath, and the only thing it can get wrong is order: which vector is a
// key, when a table may be swapped, and whether the token being scored is
// already in the cache. So the FSM is written as one sequence per group and
// the interesting order is a state name, not a counter comparison.
//
// The order inside a group is causality
// -------------------------------------
// `[k][v][q0..q3]`, and the k and v are encoded and written TO DDR before the
// scan starts, because the current token attends to itself: `cache.append_rotated`
// runs before attend in `kernel.py:187`, and the scan then covers
// `slice(0, ctx+1)`. Getting this backwards produces an off-by-one-token
// attention that still runs and still looks plausible.
//
// Groups run one at a time
// ------------------------
// `PARALLEL_KV = 1`: DDR cannot feed a second KV head, so the eight groups are
// sequential and only four query heads' softmax state is ever live. Phase A of
// group h+1 overlapping Phase B of group h is a throughput optimisation on
// ~150 cycles against a T-cycle scan (0.5% at ctx 32k); it is deliberately not
// done here, because it would put ingress and the scan in the same state and
// this block's whole job is that the order is readable.
//
// The scan IS fed BY `kv_store_ddr`, which IS the only cache
// ----------------------------------------------------------
// One row per cycle, four planes, four AXI-HP masters. The four score lanes
// share the row -- that is the GQA win -- so `row_ready` is the and of theirs:
// one lane stalling stalls the row, which keeps the four softmaxes on the same
// token without a skid buffer per lane.
`timescale 1ns/1ps
module attn_top #(
    parameter int D          = 64,
    parameter int LANE_BITS  = 24,      // FixedFormat.qk_width, the seam
    parameter int BEAT_BITS  = 512,
    parameter int KV_HEADS   = 8,
    parameter int GROUPS     = 4,       // query heads per KV head
    parameter int KEY_BITS   = 4,
    parameter int VAL_BITS   = 2,
    parameter int NORM_BITS  = 16,
    parameter int SCORE_WIDTH= 48,
    parameter int ACC_WIDTH  = 32,
    parameter int L_WIDTH    = 48,
    parameter int OUT_BITS   = 48,
    parameter int DW         = 128,
    parameter int AW         = 49,
    parameter int IDW        = 6,
    parameter int N_PORTS    = 4,
    parameter int ROW_BYTES  = 52,
    parameter logic [N_PORTS*8-1:0] PLANE_W = {8'd4, 8'd16, 8'd16, 8'd16},
    // The value plane's boundaries and centroids.  A format artifact of
    // `Codebook.build`, exactly as `rot_encode`'s key default is -- never a
    // second Lloyd-Max solve.
    parameter logic [(2**VAL_BITS-1)*20-1:0] VAL_BOUNDS =
        {20'h07da2, 20'h00000, 20'hf825d},
    parameter int VEC_BITS   = D * LANE_BITS,
    parameter int ROW_BITS   = D*KEY_BITS + D*VAL_BITS + 2*NORM_BITS,
    parameter int EGR_BITS   = GROUPS * D * LANE_BITS
) (
    input  logic clk,
    input  logic rstn,

    // -- run control -----------------------------------------------------
    input  logic start,                       // one decode step
    input  logic [AW-1:0] cache_base,
    input  logic [31:0] head_stride,
    input  logic [31:0] plane_span,
    input  logic [31:0] n_tokens,             // ctx + 1, this token included
    output logic busy,
    output logic done,                        // one pulse, the step is retired

    // -- ingress: 8 groups of 6 vectors ----------------------------------
    input  logic s_valid,
    output logic s_ready,
    input  logic [BEAT_BITS-1:0] s_data,

    // -- egress: 4 heads per group, as they finalize ----------------------
    output logic m_valid,
    input  logic m_ready,
    output logic [BEAT_BITS-1:0] m_data,
    output logic m_last,

    // -- AXI4 read, one master per plane ---------------------------------
    output logic [N_PORTS*IDW-1:0] arid,
    output logic [N_PORTS*AW-1:0]  araddr,
    output logic [N_PORTS*8-1:0]   arlen,
    output logic [N_PORTS*3-1:0]   arsize,
    output logic [N_PORTS*2-1:0]   arburst,
    output logic [N_PORTS-1:0]     arlock,
    output logic [N_PORTS*4-1:0]   arcache,
    output logic [N_PORTS*3-1:0]   arprot,
    output logic [N_PORTS*4-1:0]   arqos,
    output logic [N_PORTS-1:0]     arvalid,
    input  logic [N_PORTS-1:0]     arready,
    input  logic [N_PORTS*IDW-1:0] rid,
    input  logic [N_PORTS*DW-1:0]  rdata,
    input  logic [N_PORTS*2-1:0]   rresp,
    input  logic [N_PORTS-1:0]     rlast,
    input  logic [N_PORTS-1:0]     rvalid,
    output logic [N_PORTS-1:0]     rready,

    // -- AXI4 write, the new token's 8 rows -------------------------------
    output logic [IDW-1:0]  awid,
    output logic [AW-1:0]   awaddr,
    output logic [7:0]      awlen,
    output logic [2:0]      awsize,
    output logic [1:0]      awburst,
    output logic [3:0]      awcache,
    output logic [2:0]      awprot,
    output logic            awvalid,
    input  logic            awready,
    output logic [DW-1:0]   wdata,
    output logic [DW/8-1:0] wstrb,
    output logic            wlast,
    output logic            wvalid,
    input  logic            wready,
    input  logic [1:0]      bresp,
    input  logic            bvalid,
    output logic            bready,

    // -- what the run measured, never assumed -----------------------------
    output logic [31:0] scan_cycles,          // cycles spent in S_SCAN, all 8 groups
    output logic [31:0] busy_cycles,          // cycles from `start` to `done`
    output logic [31:0] starve_cycles,        // rows the scan waited on DDR
    output logic [31:0] clip_count,           // scores that saturated
    output logic [31:0] overflow_count,       // accumulator channels that wrapped
    output logic range_error,                 // a result the 24-bit seam lost
    output logic norm_saturated               // a Q7.9 norm hit its clamp
);
    localparam int NBEAT_EGR = EGR_BITS / BEAT_BITS;

    initial if (EGR_BITS % BEAT_BITS != 0)
        $fatal(1, "a group's egress must be a whole number of beats");

    // ---------------------------------------------------------------- FSM
    typedef enum logic [3:0] {
        S_IDLE, S_RECV, S_ROT, S_NORM, S_ENC, S_WRITE, S_WRWAIT,
        S_QLOAD, S_COMMIT, S_SCAN, S_FIN, S_EGRESS, S_DONE
    } state_t;
    state_t st;

    logic [$clog2(KV_HEADS)-1:0] grp;
    logic [2:0] vidx;                          // which of the six, latched
    logic [VEC_BITS-1:0] cur, rvec;
    logic [D*KEY_BITS-1:0] k_codes_r;
    logic [D*VAL_BITS-1:0] v_codes_r;
    logic [NORM_BITS-1:0] k_norm_r, v_norm_r, norm_r;

    wire [ROW_BITS-1:0] row_out = {v_norm_r, k_norm_r, v_codes_r, k_codes_r};

    // -------------------------------------------------------- ingress
    logic ing_ready;
    logic ing_valid;
    logic [VEC_BITS-1:0] ing_vec;
    logic [2:0] ing_idx;
    logic [2:0] ing_grp;

    attn_ingress #(.D(D), .LANE_BITS(LANE_BITS), .BEAT_BITS(BEAT_BITS),
                   .PER_GROUP(2+GROUPS), .GROUPS(KV_HEADS))
    ing (.clk, .rstn, .start(start), .s_valid, .s_ready, .s_data,
         .out_valid(ing_valid), .out_ready(ing_ready), .out_vec(ing_vec),
         .out_idx(ing_idx), .out_group(ing_grp));

    // -------------------------------------------------------- rotate
    logic rot_in_valid, rot_in_ready, rot_out_valid, rot_out_ready, rot_range;
    logic [VEC_BITS-1:0] rot_out;

    rot_fwht rot (.clk, .rstn, .in_valid(rot_in_valid), .in_ready(rot_in_ready),
                  .in_vec(cur), .out_valid(rot_out_valid), .out_ready(rot_out_ready),
                  .out_vec(rot_out), .range_error(rot_range));

    // -------------------------------------------------------- norm, encode
    logic nrm_in_valid, nrm_in_ready, nrm_out_valid, nrm_out_ready, nrm_sat;
    logic [NORM_BITS-1:0] nrm_out;

    rot_norm #(.D(D), .IN_BITS(LANE_BITS), .NORM_BITS(NORM_BITS))
    nrm (.clk, .rstn, .in_valid(nrm_in_valid), .in_ready(nrm_in_ready),
         .in_vec(rvec), .out_valid(nrm_out_valid), .out_ready(nrm_out_ready),
         .out_norm(nrm_out), .out_saturated(nrm_sat));

    logic enck_valid, enck_ready, enck_out, encv_valid, encv_ready, encv_out;
    logic [D*KEY_BITS-1:0] enck_codes;
    logic [D*VAL_BITS-1:0] encv_codes;

    rot_encode #(.D(D), .IN_BITS(LANE_BITS), .BITS(KEY_BITS), .NORM_BITS(NORM_BITS))
    enc_k (.clk, .rstn, .in_valid(enck_valid), .in_ready(enck_ready),
           .in_vec(rvec), .in_norm(norm_r), .out_valid(enck_out),
           .out_ready(1'b1), .out_codes(enck_codes));

    rot_encode #(.D(D), .IN_BITS(LANE_BITS), .BITS(VAL_BITS), .NORM_BITS(NORM_BITS),
                 .BOUNDS(VAL_BOUNDS))
    enc_v (.clk, .rstn, .in_valid(encv_valid), .in_ready(encv_ready),
           .in_vec(rvec), .in_norm(norm_r), .out_valid(encv_out),
           .out_ready(1'b1), .out_codes(encv_codes));

    // -------------------------------------------------------- the cache
    // Incremental, not a multiply, and registered.
    //
    // `cache_base + grp*head_stride + p*plane_span` is a 49-bit product and two
    // 49-bit adds. Registering it kept that arithmetic between two flops and it
    // stayed on the critical path -- 16 levels of carry from a counter that
    // changes once per group. There is no reason to multiply: the group index
    // only ever advances by one, so the base only ever advances by
    // `head_stride`. One 49-bit add per plane, once per group, hundreds of
    // cycles before anything reads it.
    logic [N_PORTS*AW-1:0] head_base;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            head_base <= '0;
        end else if (start) begin
            for (int p = 0; p < N_PORTS; p++)
                head_base[p*AW +: AW] <= cache_base + AW'(p) * AW'(plane_span);
        end else if (st == S_EGRESS && m_valid && m_ready
                     && egr_beat == NBEAT_EGR-1 && grp != KV_HEADS-1) begin
            // The group that is about to start reads this; the group that just
            // finished is done with it.
            for (int p = 0; p < N_PORTS; p++)
                head_base[p*AW +: AW] <= head_base[p*AW +: AW] + AW'(head_stride);
        end
    end

    logic scan_start, store_busy, row_valid, row_ready;
    logic [ROW_BYTES*8-1:0] row_data;
    logic [31:0] store_starve;

    kv_store_ddr #(.DW(DW), .AW(AW), .IDW(IDW), .N_PORTS(N_PORTS),
                   .ROW_BYTES(ROW_BYTES), .PLANE_W(PLANE_W))
    store (.clk, .rstn, .start(scan_start), .head_base, .n_tokens,
           .busy(store_busy), .row_valid, .row_ready, .row_data,
           .starve_cycles(store_starve),
           .arid, .araddr, .arlen, .arsize, .arburst, .arlock, .arcache,
           .arprot, .arqos, .arvalid, .arready, .rid, .rdata, .rresp,
           .rlast, .rvalid, .rready);

    logic wr_valid, wr_ready;
    kv_write #(.DW(DW), .AW(AW), .IDW(IDW), .N_PORTS(N_PORTS),
               .ROW_BYTES(ROW_BYTES), .PLANE_W(PLANE_W))
    wr (.clk, .rstn, .in_valid(wr_valid), .in_ready(wr_ready), .head_base,
        .token(n_tokens - 32'd1), .row_data(row_out),
        .awid, .awaddr, .awlen, .awsize, .awburst, .awcache, .awprot,
        .awvalid, .awready, .wdata, .wstrb, .wlast, .wvalid, .wready,
        .bresp, .bvalid, .bready);

    // -------------------------------------------------------- four lanes
    logic [GROUPS-1:0] q_valid, q_ready, tab_ready, tab_commit;
    logic [GROUPS-1:0] lane_row_ready, sc_valid, sc_clip;
    logic [GROUPS-1:0] sm_ready, sm_lovf;
    logic [GROUPS-1:0] fin_valid, fin_ready, fin_out_valid, fin_range, fin_got;
    logic [GROUPS-1:0] tok_done;
    // A registered write enable per lane, not the FSM state.
    //
    // `q_vec` is 4 x 1,536 flops and decoding `st == S_QLOAD` into their clock
    // enables put the state register on 6,144 CE pins -- the same shape as the
    // egress register that cost -1.953 ns in the full system. One flop per lane
    // drives them instead, and synthesis is free to replicate it.
    (* max_fanout = 64 *)
    logic [GROUPS-1:0] q_we;
    logic [VEC_BITS-1:0] q_vec [GROUPS];

    always_ff @(posedge clk)
        for (int i = 0; i < GROUPS; i++)
            if (q_we[i]) q_vec[i] <= rvec;
    logic [D*OUT_BITS-1:0] fin_vec [GROUPS];
    logic [31:0] lane_ovf [GROUPS];
    logic [31:0] scan_count [GROUPS];

    // Every lane sees the same row.  One stalling lane stalls the row, which
    // is what keeps four softmaxes on the same token with no per-lane skid.
    assign row_ready = &lane_row_ready;

    genvar g;
    generate
        for (g = 0; g < GROUPS; g = g + 1) begin : lane
            logic signed [SCORE_WIDTH-1:0] score;
            logic [D*VAL_BITS-1:0] vcodes;
            logic signed [NORM_BITS-1:0] vnorm;

            score_lane #(.D(D), .IN_BITS(LANE_BITS), .KEY_BITS(KEY_BITS),
                         .VAL_BITS(VAL_BITS), .NORM_BITS(NORM_BITS))
            sl (.clk, .rstn,
                .q_valid(q_valid[g]), .q_ready(q_ready[g]), .q_vec(q_vec[g]),
                .tab_ready(tab_ready[g]), .tab_commit(tab_commit[g]),
                .row_valid(row_valid && row_ready), .row_ready(lane_row_ready[g]),
                .row_data(row_data),
                .out_valid(sc_valid[g]), .out_ready(sm_ready[g]),
                .out_score(score), .out_vcodes(vcodes), .out_vnorm(vnorm),
                .out_clip(sc_clip[g]));

            logic op_valid, op_ready, op_scale;
            logic [15:0] op_factor, op_p;
            logic [D*VAL_BITS-1:0] op_vcodes;
            logic signed [NORM_BITS-1:0] op_vnorm;
            logic signed [SCORE_WIDTH-1:0] out_m;
            logic [L_WIDTH-1:0] out_l;
            logic [5:0] hw_max_fill;

            softmax_online #(.D(D), .VAL_BITS(VAL_BITS), .SCORE_WIDTH(SCORE_WIDTH),
                             .NORM_BITS(NORM_BITS), .L_WIDTH(L_WIDTH))
            sm (.clk, .rstn, .start(scan_start),
                .in_valid(sc_valid[g]), .in_ready(sm_ready[g]), .in_score(score),
                .in_vcodes(vcodes), .in_vnorm(vnorm),
                .op_valid, .op_ready, .op_scale, .op_factor, .op_p,
                .op_vcodes, .op_vnorm, .out_m, .out_l, .hw_max_fill,
                .l_overflow(sm_lovf[g]));

            logic scale_done;
            logic [D*ACC_WIDTH-1:0] out_acc;
            logic [31:0] out_scale_cycles;

            accum #(.D(D), .VAL_BITS(VAL_BITS), .NORM_BITS(NORM_BITS),
                    .ACC_WIDTH(ACC_WIDTH))
            ac (.clk, .rstn, .start(scan_start), .op_valid, .op_ready, .op_scale,
                .op_factor, .op_p, .op_vcodes, .op_vnorm,
                .tok_done(tok_done[g]), .scale_done, .out_acc,
                .out_overflows(lane_ovf[g]), .out_scale_cycles(out_scale_cycles));

            // The denominator is sampled with the accumulator, at the retire
            // edge of the last add -- `out_l` is registered when the op is
            // Issued, so reading it any earlier reads a scan that is still one
            // probability short.
            logic [L_WIDTH-1:0] held_l;
            logic [D*ACC_WIDTH-1:0] held_acc;
            always_ff @(posedge clk) if (tok_done[g]) begin
                held_l   <= out_l;
                held_acc <= out_acc;
            end

            finalize #(.D(D), .ACC_WIDTH(ACC_WIDTH), .L_WIDTH(L_WIDTH),
                       .OUT_BITS(OUT_BITS), .SEAM_BITS(LANE_BITS))
            fn (.clk, .rstn, .in_valid(fin_valid[g]), .in_ready(fin_ready[g]),
                .in_acc(held_acc), .in_l(held_l),
                .out_valid(fin_out_valid[g]), .out_ready(1'b1),
                .out_vec(lane_vec), .out_recip(), .range_error(fin_range[g]));

            // The divide is data-dependent -- 50 cycles, or 19 when `l` is
            // zero and `reciprocal` answers without dividing -- so the four
            // lanes do not retire together and a bare and of `out_valid` would
            // wait for a pulse that has already gone. Each lane holds its own
            // result and raises its own flag.
            logic [D*OUT_BITS-1:0] lane_vec;
            always_ff @(posedge clk)
                if (!rstn || st == S_COMMIT) fin_got[g] <= 1'b0;
                else if (fin_out_valid[g]) begin
                    fin_got[g]   <= 1'b1;
                    fin_vec[g]   <= lane_vec;
                end

            // Tokens this lane has retired into its accumulator.  The four are
            // counted separately because a rescale stalls one lane and not the
            // others; they are only in step again at the end of the scan.
            //
            // Cleared in S_COMMIT and not on `scan_start`: `scan_start` is
            // registered, so it is high during the first cycle of S_SCAN and
            // the counters would still hold the previous group's 65 when
            // `scan_complete` is first evaluated -- which ends the scan before
            // it begins, for every group after the first.
            always_ff @(posedge clk)
                if (!rstn || st == S_COMMIT) scan_count[g] <= '0;
                else if (tok_done[g]) scan_count[g] <= scan_count[g] + 32'd1;
        end
    endgenerate

    wire scan_complete = (scan_count[0] >= n_tokens) && (scan_count[1] >= n_tokens)
                      && (scan_count[2] >= n_tokens) && (scan_count[3] >= n_tokens);

    // -------------------------------------------------------- egress
    //
    // The four finalized vectors are the result; there is no second copy of
    // them. An earlier version loaded a 6,144-bit register and shifted it 512
    // bits per beat, which cost 6,144 flops to hold what `fin_vec` already
    // held and -- the reason it is gone -- put the FSM state on the enable of
    // every one of them. In the full system that path routed to -1.953 ns with
    // 49,557 failing endpoints, almost entirely wire delay: one LUT and 5.6 ns
    // of getting there.
    //
    // Indexing instead of shifting turns that into one 12:1 mux of 512 bits,
    // driven by a 4-bit counter, feeding a single registered output beat.
    logic [$clog2(NBEAT_EGR+1)-1:0] egr_beat;
    logic [EGR_BITS-1:0] egr_flat;

    // The 24-bit seam takes the low bits of each 48-bit result; `range_error`
    // says when that lost something, which is the convention `rot_fwht` and
    // `finalize` already use.
    always_comb
        for (int i = 0; i < GROUPS; i++)
            for (int c = 0; c < D; c++)
                egr_flat[(i*D + c)*LANE_BITS +: LANE_BITS] =
                    fin_vec[i][c*OUT_BITS +: LANE_BITS];

    assign m_last = (egr_beat == NBEAT_EGR-1) && (grp == KV_HEADS-1);

    // Wraps rather than running one past the end: a part-select off the end of
    // `egr_flat` is X, and that X would be the first beat of the next group
    // until S_FIN overwrote it -- true today and a trap for tomorrow.
    wire [$clog2(NBEAT_EGR+1)-1:0] egr_next =
        (egr_beat == NBEAT_EGR-1) ? '0 : egr_beat + 1'b1;

    // -------------------------------------------------------- counters
    //
    // `scan_cycles` and `busy_cycles` are what a device-side time is made of,
    // and they are counted here rather than by the host for a reason step 13
    // depends on: the PS reads a millisecond-scale wall clock across a bus it
    // shares, so it can time a decode step but it cannot time the scan inside
    // one. The DDR bandwidth claim is `8 x n_tokens x row_bytes` over
    // `scan_cycles`, and every term in it has to come from the same clock the
    // reads were issued on.
    //
    // Both are free-running while the step is in flight and both are cleared
    // by `start`, so a host that reads them after `done` reads the step it
    // just ran and not a sum over the session.
    always_ff @(posedge clk) begin
        if (!rstn) begin
            clip_count <= '0;
            scan_cycles <= '0; busy_cycles <= '0;
            range_error <= 1'b0; norm_saturated <= 1'b0;
        end else begin
            if (start) begin
                clip_count <= '0;
                scan_cycles <= '0; busy_cycles <= '0;
                range_error <= 1'b0; norm_saturated <= 1'b0;
            end else begin
                clip_count <= clip_count + 32'(clips_now);
                if (st == S_SCAN) scan_cycles <= scan_cycles + 1'b1;
                if (busy)         busy_cycles <= busy_cycles + 1'b1;
                if (rot_out_valid && rot_out_ready && rot_range) range_error <= 1'b1;
                if (nrm_out_valid && nrm_out_ready && nrm_sat) norm_saturated <= 1'b1;
                for (int i = 0; i < GROUPS; i++)
                    if (fin_out_valid[i] && fin_range[i]) range_error <= 1'b1;
            end
        end
    end

    // The cache's own counter is cumulative across the eight scans, so it is
    // taken as it stands rather than re-counted here.
    assign starve_cycles  = store_starve;
    assign overflow_count = lane_ovf[0] + lane_ovf[1] + lane_ovf[2] + lane_ovf[3];

    // A score that saturated is a numeric event, so it is counted at the edge
    // it retires on -- all four lanes, which is why this is a sum and not a
    // single increment.
    logic [2:0] clips_now;
    always_comb begin
        clips_now = '0;
        for (int i = 0; i < GROUPS; i++)
            if (sc_valid[i] && sm_ready[i] && sc_clip[i])
                clips_now = clips_now + 3'd1;
    end

    // -------------------------------------------------------- the sequence
    assign busy = (st != S_IDLE);

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= S_IDLE;
            ing_ready <= 1'b0; rot_in_valid <= 1'b0; rot_out_ready <= 1'b0;
            nrm_in_valid <= 1'b0; nrm_out_ready <= 1'b0;
            enck_valid <= 1'b0; encv_valid <= 1'b0;
            wr_valid <= 1'b0; scan_start <= 1'b0;
            q_valid <= '0; tab_commit <= '0; fin_valid <= '0; q_we <= '0;
            m_valid <= 1'b0; done <= 1'b0; grp <= '0; egr_beat <= '0;
            m_data <= '0;
        end else begin
            ing_ready <= 1'b0;
            scan_start <= 1'b0;
            tab_commit <= '0;
            done <= 1'b0;
            q_we <= '0;

            case (st)
                S_IDLE: if (start) begin
                    grp <= '0;
                    st <= S_RECV;
                end

                // One vector at a time, in the order the group defines.
                S_RECV: if (ing_valid) begin
                    cur  <= ing_vec;
                    vidx <= ing_idx;
                    ing_ready <= 1'b1;
                    rot_in_valid <= 1'b1;
                    rot_out_ready <= 1'b1;
                    st <= S_ROT;
                end

                S_ROT: begin
                    if (rot_in_valid && rot_in_ready) rot_in_valid <= 1'b0;
                    if (rot_out_valid && rot_out_ready) begin
                        rvec <= rot_out;
                        rot_out_ready <= 1'b0;
                        if (vidx < 3'd2) begin
                            nrm_in_valid <= 1'b1;
                            nrm_out_ready <= 1'b1;
                            st <= S_NORM;
                        end else begin
                            // The vector is latched by `q_we` on the way into
                            // S_QLOAD, one cycle before `q_valid` is raised.
                            q_we[vidx - 3'd2] <= 1'b1;
                            st <= S_QLOAD;
                        end
                    end
                end

                S_NORM: begin
                    if (nrm_in_valid && nrm_in_ready) nrm_in_valid <= 1'b0;
                    if (nrm_out_valid && nrm_out_ready) begin
                        norm_r <= nrm_out;
                        if (vidx == 3'd0) k_norm_r <= nrm_out;
                        else              v_norm_r <= nrm_out;
                        nrm_out_ready <= 1'b0;
                        if (vidx == 3'd0) enck_valid <= 1'b1;
                        else              encv_valid <= 1'b1;
                        st <= S_ENC;
                    end
                end

                // The key plane is KEY_BITS wide and the value plane VAL_BITS;
                // they are two encoders and not one parameterised at runtime,
                // because the boundary table is a different length.
                S_ENC: begin
                    if (vidx == 3'd0) begin
                        if (enck_valid && enck_ready) enck_valid <= 1'b0;
                        if (enck_out) begin
                            k_codes_r <= enck_codes;
                            st <= S_RECV;
                        end
                    end else begin
                        if (encv_valid && encv_ready) encv_valid <= 1'b0;
                        if (encv_out) begin
                            v_codes_r <= encv_codes;
                            st <= S_WRITE;
                        end
                    end
                end

                // The row goes to DDR before the scan reads it back: this
                // token attends to itself.
                S_WRITE: begin
                    wr_valid <= 1'b1;
                    if (wr_valid && wr_ready) begin
                        wr_valid <= 1'b0;
                        st <= S_WRWAIT;
                    end
                end

                S_WRWAIT: if (wr_ready) st <= S_RECV;   // all four planes retired

                S_QLOAD: begin
                    q_valid[vidx - 3'd2] <= 1'b1;
                    if (q_valid[vidx - 3'd2] && q_ready[vidx - 3'd2]) begin
                        q_valid[vidx - 3'd2] <= 1'b0;
                        st <= (vidx == 3'd5) ? S_COMMIT : S_RECV;
                    end
                end

                // The previous scan has drained -- groups are sequential -- so
                // the swap is unconditional once the four tables are built.
                S_COMMIT: if (&tab_ready) begin
                    tab_commit <= '1;
                    scan_start <= 1'b1;
                    st <= S_SCAN;
                end

                S_SCAN: if (scan_complete) begin
                    fin_valid <= '1;
                    st <= S_FIN;
                end

                S_FIN: begin
                    for (int i = 0; i < GROUPS; i++)
                        if (fin_valid[i] && fin_ready[i]) fin_valid[i] <= 1'b0;
                    if (&fin_got) begin
                        m_data   <= egr_flat[0 +: BEAT_BITS];
                        st       <= S_EGRESS;
                        egr_beat <= '0;
                        m_valid  <= 1'b1;
                    end
                end

                S_EGRESS: if (m_valid && m_ready) begin
                    m_data <= egr_flat[egr_next*BEAT_BITS +: BEAT_BITS];
                    if (egr_beat == NBEAT_EGR-1) begin
                        m_valid <= 1'b0;
                        egr_beat <= '0;
                        if (grp == KV_HEADS-1) begin
                            st <= S_DONE;
                        end else begin
                            grp <= grp + 1'b1;
                            st <= S_RECV;
                        end
                    end else
                        egr_beat <= egr_beat + 1'b1;
                end

                S_DONE: begin
                    done <= 1'b1;
                    st <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end

endmodule
