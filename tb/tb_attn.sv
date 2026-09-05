// Self-checking testbench for attn_top.sv: the whole block, end to end.
//
// THE GOLDEN IS `out.online.hex`, AND IT IS THE SEAM
// --------------------------------------------------
// `forward` runs the two-pass softmax and the two are NOT bit-identical -- the
// online form truncates the accumulator once per rescale. The hardware is
// online, so `out.online.hex` is the file this bench checks, exactly as
// `plan.MD` says. A bench written against `out.hex` can never pass, and both
// files are emitted so that mistake is at least an explicit one.
//
// The stimulus is the wire stream: `ingress.hex`, 8 groups of 6 vectors of 3
// beats. Nothing between it and the answer is handed to the DUT -- no rotated
// vectors, no norms, no codes, no cache row. The rotation, the quantizer, the
// cache write, the score, the softmax and the divide are all under test at
// once, which is the only thing this level can check that the block benches
// cannot.
//
// THE NEW TOKEN IS BLANKED IN DDR, ON PURPOSE
// -------------------------------------------
// `ddr_image.hex` holds all 65 tokens, token 64 included, because the kernel
// had already appended it when `collect` copied the cache. If the bench left
// it there, a DUT whose write path did nothing at all would still produce the
// right answer -- the current token attends to itself and the row it needs
// would already be in memory. So every plane's slot for token 64 is zeroed
// before the run, and the only way it comes back is through `kv_write`'s four
// AXI writes. The blanking is asserted to actually break the answer first.
//
// BACK-PRESSURE ON BOTH SIDES
// ---------------------------
// The second run stalls the ingress stream, stalls the egress stream, and
// stalls R at random inside the DDR. The result must be BIT-IDENTICAL to the
// first run, not merely still plausible: a block that only works when nothing
// ever waits is a block that fails the first time the softmax rescales.
`timescale 1ns/1ps
module tb_attn;
    localparam int D    = 64;
    localparam int W    = 24;
    localparam int BW   = 512;
    localparam int KVH  = 8;
    localparam int GRP  = 4;
    localparam int HEADS= KVH * GRP;
    localparam int T    = 65;                  // ctx 64 + this token
    localparam int RB   = 52;
    localparam int DW   = 128;
    localparam int AW   = 49;
    localparam int IDW  = 6;
    localparam int NP   = 4;
    localparam int HEAD_STRIDE = 16384;
    localparam int PLANE_SPAN  = 4096;
    localparam int IMG_BEATS   = 8192;
    localparam int IN_BEATS    = KVH * 6 * 3;         // 144
    localparam int OUT_BEATS   = HEADS * D * W / BW;  // 96
    localparam int LATENCY     = 40;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic [BW-1:0]  stim [0:IN_BEATS-1];
    logic [DW-1:0]  img  [0:IMG_BEATS-1];
    logic [DW-1:0]  mem  [0:IMG_BEATS-1];
    logic [HEADS*D*W-1:0] gold [0:0];

    int errors = 0;

    // -- DUT ---------------------------------------------------------------
    logic start, busy, done;
    logic [AW-1:0] cache_base;
    logic [31:0] head_stride, plane_span, n_tokens;
    logic s_valid, s_ready, m_valid, m_ready, m_last;
    logic [BW-1:0] s_data, m_data;

    logic [NP*IDW-1:0] arid, rid;
    logic [NP*AW-1:0]  araddr;
    logic [NP*8-1:0]   arlen;
    logic [NP*3-1:0]   arsize, arprot;
    logic [NP*2-1:0]   arburst, rresp;
    logic [NP*4-1:0]   arcache, arqos;
    logic [NP-1:0]     arlock, arvalid, arready, rlast, rvalid, rready;
    logic [NP*DW-1:0]  rdata;

    logic [IDW-1:0] awid;
    logic [AW-1:0]  awaddr;
    logic [7:0]     awlen;
    logic [2:0]     awsize, awprot;
    logic [1:0]     awburst, bresp;
    logic [3:0]     awcache;
    logic           awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [DW-1:0]  wdata;
    logic [DW/8-1:0] wstrb;

    logic [31:0] starve_cycles, clip_count, overflow_count;
    logic range_error, norm_saturated;

    attn_top dut (
        .clk, .rstn, .start, .cache_base, .head_stride, .plane_span, .n_tokens,
        .busy, .done,
        .s_valid, .s_ready, .s_data,
        .m_valid, .m_ready, .m_data, .m_last,
        .arid, .araddr, .arlen, .arsize, .arburst, .arlock, .arcache, .arprot,
        .arqos, .arvalid, .arready, .rid, .rdata, .rresp, .rlast, .rvalid, .rready,
        .awid, .awaddr, .awlen, .awsize, .awburst, .awcache, .awprot, .awvalid,
        .awready, .wdata, .wstrb, .wlast, .wvalid, .wready, .bresp, .bvalid, .bready,
        .starve_cycles, .clip_count, .overflow_count, .range_error, .norm_saturated);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    task automatic tick; @(negedge clk); endtask

    // -- the behavioural DDR, one independent slave per port ----------------
    //
    // The same model `tb_kv_store` established: a latency every accepted AR
    // pays in parallel, two outstanding, and `rvalid` dropped the cycle a beat
    // is taken. An always-ready memory would let a prefetcher that hides no
    // latency at all pass this bench.
    logic stall_r = 0;
    int seed = 32'hA77;
    localparam int QD = 2;

    generate
        genvar q;
        for (q = 0; q < NP; q = q + 1) begin : port
            logic [AW-1:0] qaddr [0:QD-1];
            logic [8:0]    qlen  [0:QD-1];
            logic [15:0]   qwait [0:QD-1];
            logic [1:0]    qcnt;
            logic [0:0]    qh, qt;

            assign arready[q] = qcnt < 2'(QD);
            assign rid[q*IDW +: IDW] = '0;
            assign rresp[q*2 +: 2] = 2'b00;

            logic stall_now;
            always_ff @(posedge clk) stall_now <= stall_r && (($random(seed) & 7) == 0);

            wire head_ready = (qcnt != 0) && (qwait[qh] == 16'd0);
            wire beat_go = head_ready && (!rvalid[q] || rready[q]) && !stall_now;
            wire last_beat = beat_go && (qlen[qh] == 9'd1);

            always_ff @(posedge clk) begin
                if (!rstn) begin
                    qcnt <= '0; qh <= '0; qt <= '0;
                    rvalid[q] <= 1'b0; rlast[q] <= 1'b0;
                end else begin
                    if (rvalid[q] && rready[q]) begin
                        rvalid[q] <= 1'b0;
                        rlast[q]  <= 1'b0;
                    end
                    for (int i = 0; i < QD; i++)
                        if (qwait[i] != 16'd0) qwait[i] <= qwait[i] - 16'd1;

                    if (arvalid[q] && arready[q]) begin
                        qaddr[qt] <= araddr[q*AW +: AW];
                        qlen[qt]  <= 9'(arlen[q*8 +: 8]) + 9'd1;
                        qwait[qt] <= 16'(LATENCY);
                        qt <= qt + 1'b1;
                        if (!last_beat) qcnt <= qcnt + 2'd1;
                    end else if (last_beat) begin
                        qcnt <= qcnt - 2'd1;
                    end

                    if (beat_go) begin
                        rvalid[q] <= 1'b1;
                        rdata[q*DW +: DW] <= mem[qaddr[qh][AW-1:4]];
                        rlast[q] <= (qlen[qh] == 9'd1);
                        qaddr[qh] <= qaddr[qh] + 16;
                        qlen[qh]  <= qlen[qh] - 9'd1;
                        if (qlen[qh] == 9'd1) qh <= qh + 1'b1;
                    end
                end
            end
        end
    endgenerate

    // The write slave, honouring WSTRB byte by byte: the norms are a 4-byte
    // narrow transfer into a 16-byte beat, so a strobe the slave ignored would
    // let a broken byte lane pass.
    logic [AW-1:0] w_addr_l;
    assign awready = 1'b1;
    assign wready  = 1'b1;
    assign bresp   = 2'b00;
    int writes = 0;
    always_ff @(posedge clk) begin
        if (!rstn) bvalid <= 1'b0;
        else begin
            if (awvalid && awready) w_addr_l <= awaddr;
            bvalid <= wvalid && wready;
            if (wvalid && wready) begin
                writes <= writes + 1;
                for (int b = 0; b < DW/8; b++)
                    if (wstrb[b])
                        mem[w_addr_l[AW-1:4]][b*8 +: 8] <= wdata[b*8 +: 8];
            end
        end
    end

    // -- the streams --------------------------------------------------------
    int sent = 0, got = 0;
    logic [HEADS*D*W-1:0] result;
    logic stall_in = 0, stall_out = 0;
    // The stream is fed only AFTER `start`: `attn_ingress` clears its beat
    // counter on that pulse, so a beat pushed in beside it is counted by the
    // driver and dropped by the DUT, and the step then waits forever for a
    // vector that is one beat short.
    logic feeding = 0;
    // A CYCLE counter, not `$time`: the clock period is 4 time units and any
    // modulus that divides it turns a random-looking stall into a permanent
    // one -- `($time/4) % 5` is zero on every single cycle, which held the
    // egress stream off forever and looked exactly like a deadlocked DUT.
    int cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;
    int last_seen = 0;

    always @(posedge clk) if (rstn && m_valid && m_ready) begin
        result[got*BW +: BW] = m_data;
        if (m_last) last_seen = last_seen + 1;
        got = got + 1;
    end

    always_comb begin
        s_valid = feeding && (sent < IN_BEATS) && !(stall_in && (cyc % 3 == 0));
        s_data  = stim[sent < IN_BEATS ? sent : IN_BEATS-1];
        m_ready = !(stall_out && (cyc % 5 == 0));
    end
    always @(posedge clk) if (rstn && s_valid && s_ready) sent <= sent + 1;

    // -- one decode step ----------------------------------------------------
    int t0, cycles, clean_cycles;
    task automatic run_step;
        sent = 0; got = 0; last_seen = 0; feeding = 0;
        tick; start = 1; tick; start = 0; feeding = 1;
        t0 = $time;
        while (got < OUT_BEATS) tick;
        while (!done) tick;
        cycles = ($time - t0) / 4;
    endtask

    task automatic compare(input string label);
        for (int h = 0; h < HEADS; h++)
            for (int c = 0; c < D; c++)
                check(result[(h*D + c)*W +: W] === gold[0][(h*D + c)*W +: W],
                      $sformatf("%s: head %0d channel %0d: %06x != %06x", label,
                                h, c, result[(h*D + c)*W +: W],
                                gold[0][(h*D + c)*W +: W]));
    endtask

    task automatic reload_image(input logic blank_new_token);
        for (int i = 0; i < IMG_BEATS; i++) mem[i] = img[i];
        // Token 64's slot in every plane of every head. The norms plane is
        // 4 B/token, so token 64 shares its beat with tokens 61..63 and only
        // its own four bytes may be cleared -- blanking the whole beat would
        // destroy three cached tokens the golden depends on.
        if (blank_new_token)
            for (int h = 0; h < KVH; h++) begin
                mem[(h*HEAD_STRIDE + 0*PLANE_SPAN)/16 + 64] = '0;
                mem[(h*HEAD_STRIDE + 1*PLANE_SPAN)/16 + 64] = '0;
                mem[(h*HEAD_STRIDE + 2*PLANE_SPAN)/16 + 64] = '0;
                mem[(h*HEAD_STRIDE + 3*PLANE_SPAN)/16 + 16][31:0] = '0;
            end
    endtask

    int mismatches;
    logic [HEADS*D*W-1:0] first_run;

    initial begin
        $readmemh("tb/vectors/ingress.hex", stim);
        $readmemh("tb/vectors/ddr_image.hex", img);
        $readmemh("tb/vectors/out.online.hex", gold);

        start = 0; cache_base = '0; head_stride = HEAD_STRIDE;
        plane_span = PLANE_SPAN; n_tokens = T;
        reload_image(1);
        repeat (4) @(posedge clk);
        rstn = 1;

        // -- the run ---------------------------------------------------------
        run_step();
        check(got == OUT_BEATS, "every head came back as a whole beat");
        check(last_seen == 1, "TLAST is asserted exactly once, on the last beat");
        compare("clean");
        first_run = result;
        clean_cycles = cycles;
        $display("  %0d heads x %0d channels in %0d cycles (%0d rows scanned)",
                 HEADS, D, cycles, KVH*T);
        $display("  starve %0d, clips %0d, overflows %0d, range_error %0d, norm_sat %0d",
                 starve_cycles, clip_count, overflow_count, range_error, norm_saturated);
        check(!range_error, "no channel escaped the 24-bit seam");
        check(clip_count == 0, "no score saturated at these widths");
        check(writes == KVH*NP, "one AXI write per plane per head, and no more");

        // -- the blanking must actually matter --------------------------------
        //
        // Without this the run above proves nothing about the write path: the
        // image already contained token 64, so a DUT that wrote nothing would
        // have read the right row anyway. Here the row is blanked AND the write
        // path is starved of its own data by holding the memory unwritten, so
        // the answer must change.
        reload_image(1);
        force wstrb = '0;
        run_step();
        release wstrb;
        mismatches = 0;
        for (int i = 0; i < HEADS*D; i++)
            if (result[i*W +: W] !== gold[0][i*W +: W]) mismatches = mismatches + 1;
        check(mismatches > 0,
              "with every write byte lane masked, the blanked token must change the answer");
        $display("  write bytes masked: %0d of %0d channels differ",
                 mismatches, HEADS*D);

        // -- back-pressure, both sides, and a stalling DDR --------------------
        reload_image(1);
        stall_in = 1; stall_out = 1; stall_r = 1;
        run_step();
        stall_in = 0; stall_out = 0; stall_r = 0;
        check(result === first_run,
              "the stalled run must be bit-identical to the clean one");
        compare("stalled");
        $display("  stalled run: %0d cycles against %0d clean, starve %0d",
                 cycles, clean_cycles, starve_cycles);

        // -- a second step on the same instance -------------------------------
        //
        // The block is not a one-shot: every counter, every FIFO and the
        // ingress group counter must come back to where they started, or the
        // second token of a real decode is wrong in a way no single-step bench
        // can see.
        reload_image(1);
        run_step();
        check(result === first_run, "a second step on the same instance repeats it");

        if (errors == 0) $display("=== tb_attn: ALL TESTS PASSED ===");
        else $display("=== tb_attn: %0d TESTS FAILED ===", errors);
        $finish;
    end

    initial begin
        #200000000;
        $display("  FAIL: watchdog -- the step never finished");
        $display("=== tb_attn: 1 TESTS FAILED ===");
        $finish;
    end
endmodule
