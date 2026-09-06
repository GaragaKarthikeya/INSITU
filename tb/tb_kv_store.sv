// Self-checking testbench for kv_store_ddr.sv + kv_plane_rd.sv.
//
// A behavioural DDR, not an ideal one
// -----------------------------------
// The slave holds the image `hw/ddr_layout.py` produced, answers AR after a
// programmable latency, and can stall R at random. An always-ready memory would
// let a prefetcher that hides no latency at all pass, which is the one property
// this block exists to have.
//
// The golden is the cache itself
// ------------------------------
// `cache_rows_hN.hex` is what `KVQuantizer.pack` wrote. The image is the same
// bytes transposed into planes, and this checks that what comes back out of the
// four ports is the row again, byte for byte, in causal order. `hw/vectors.py`
// already refuses to emit an image whose `row_at` disagrees with the cache, so
// a failure here is the RTL's reassembly and not the layout's.
//
// Starvation is the other half of the exit criterion
// --------------------------------------------------
// Correct rows arriving too slowly is a failure of this block specifically. The
// bench runs the consumer at full rate and counts every cycle it asked for a row
// and got nothing, then asserts the steady-state rate against what step 1
// measured on the board.
`timescale 1ns/1ps
module tb_kv_store;
    localparam int DW  = 128;
    localparam int AW  = 49;
    localparam int IDW = 6;
    localparam int NP  = 4;
    localparam int RB  = 52;
    localparam int T   = 65;
    localparam int KVH = 8;
    localparam int HEAD_STRIDE = 16384;
    localparam int PLANE_SPAN  = 4096;
    // What `ddr_image.hex` actually holds: one head stride per KV head.
    localparam int IMG_WORDS   = KVH * HEAD_STRIDE / (DW/8);   // 8,192
    // The array is twice that, so the non-uniform layout case further down has
    // somewhere to write that the image does not already occupy.
    localparam int IMG_BEATS   = 2 * IMG_WORDS;
    // Read latency, in cycles, from AR accepted to the first R beat. The board
    // measured ~120 ns at 250 MHz under load; 40 is deliberately worse.
    localparam int LATENCY = 40;
    // Long enough to leave startup behind: 16 bursts on a code plane.
    localparam int LONG = 1024;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic [DW-1:0] mem [0:IMG_BEATS-1];
    logic [RB*8-1:0] gold [0:KVH-1][0:T-1];
    logic [RB*8-1:0] onehead [0:T-1];

    int errors = 0;

    logic start, row_ready;
    logic busy, row_valid;
    logic [NP*AW-1:0] head_base;
    logic [31:0] n_tokens, starve_cycles;
    logic [RB*8-1:0] row_data;

    logic [NP*IDW-1:0] arid;
    logic [NP*AW-1:0]  araddr;
    logic [NP*8-1:0]   arlen;
    logic [NP*3-1:0]   arsize, arprot;
    logic [NP*2-1:0]   arburst, rresp;
    logic [NP*4-1:0]   arcache, arqos;
    logic [NP-1:0]     arlock, arvalid, arready, rlast, rvalid, rready;
    logic [NP*IDW-1:0] rid;
    logic [NP*DW-1:0]  rdata;

    kv_store_ddr #(.DW(DW), .AW(AW), .IDW(IDW), .N_PORTS(NP), .ROW_BYTES(RB))
    dut (
        .clk, .rstn, .start, .head_base, .n_tokens, .busy,
        .row_valid, .row_ready, .row_data, .starve_cycles,
        .arid, .araddr, .arlen, .arsize, .arburst, .arlock, .arcache,
        .arprot, .arqos, .arvalid, .arready, .rid, .rdata, .rresp,
        .rlast, .rvalid, .rready);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    // -- the behavioural DDR, one instance per port -------------------------
    //
    // Each port is independent, as four dedicated AXI-HP masters are: they do
    // not share a crossbar here, because funnelling them through one would
    // measure the interconnect rather than the memory, and `plan.MD` requires
    // one port per master for exactly that reason.
    logic stall_r = 0;
    int seed = 32'hD08;

    // A slave that holds two outstanding ARs, because the design is sized for
    // two and a single-outstanding model would never exercise the depth the
    // board measurement chose. Beats are returned in order, after a latency,
    // and `rvalid` is deasserted the cycle a beat is accepted -- an earlier
    // version of this model left `rvalid` and `rlast` asserted after the last
    // beat of a burst, which made the master count an RLAST every cycle and
    // underflow its in-flight counter. That is a bench bug that looks exactly
    // like a prefetcher bug, so the master's counter is asserted below.
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

            // The stall decision is registered. A `$random` in a continuous
            // assignment is re-evaluated on every event that touches the wire,
            // which is not a coin flip per cycle and, in iverilog, is not even
            // legal in this context.
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
                    // Every pending burst's latency runs from when it was
                    // accepted, not from when it reaches the head of the
                    // queue. Counting only the head serialises the round
                    // trips and makes two outstanding behave exactly like one
                    // -- which is the 8,458 MB/s row of the step-1 sweep, not
                    // the 14,708 one, and it made a working prefetcher look
                    // like it hid no latency at all.
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

            // The bug that cost an afternoon: an in-flight counter that goes
            // below zero stops the port issuing reads forever, and every
            // symptom of it looks like stale data rather than like a count.
            always @(posedge clk) if (rstn)
                check(dut.plane[q].rd.inflight <= 8'd4,
                      $sformatf("port %0d in-flight counter underflowed (%0d)",
                                q, dut.plane[q].rd.inflight));
        end
    endgenerate

    // -- the write path -----------------------------------------------------
    logic w_valid, w_ready;
    logic [31:0] w_token;
    logic [RB*8-1:0] w_row;
    logic [IDW-1:0] awid;
    logic [AW-1:0]  awaddr;
    logic [7:0]     awlen;
    logic [2:0]     awsize, awprot;
    logic [1:0]     awburst, bresp;
    logic [3:0]     awcache;
    logic           awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [DW-1:0]  wdata;
    logic [DW/8-1:0] wstrb;

    kv_write #(.DW(DW), .AW(AW), .IDW(IDW), .N_PORTS(NP), .ROW_BYTES(RB))
    wr (
        .clk, .rstn, .in_valid(w_valid), .in_ready(w_ready),
        .head_base, .token(w_token), .row_data(w_row),
        .awid, .awaddr, .awlen, .awsize, .awburst, .awcache, .awprot,
        .awvalid, .awready, .wdata, .wstrb, .wlast, .wvalid, .wready,
        .bresp, .bvalid, .bready);

    // A write slave over the same memory, honouring WSTRB byte by byte -- the
    // norms are a 4-byte narrow transfer into a 16-byte beat, so a strobe the
    // slave ignored would let a broken byte lane pass.
    logic [AW-1:0] w_addr_l;
    assign awready = 1'b1;
    assign wready  = 1'b1;
    assign bresp   = 2'b00;
    always_ff @(posedge clk) begin
        if (!rstn) bvalid <= 1'b0;
        else begin
            if (awvalid && awready) w_addr_l <= awaddr;
            bvalid <= wvalid && wready;
            if (wvalid && wready)
                for (int b = 0; b < DW/8; b++)
                    if (wstrb[b])
                        mem[w_addr_l[AW-1:4]][b*8 +: 8] <= wdata[b*8 +: 8];
        end
    end

    // -- a second, non-UNIFORM layout ---------------------------------------
    //
    // At 4b/2b the planes are 16, 16, 16 and 4 bytes, so plane `p` starts at
    // byte `16*p` and an assembler that assumed a uniform 16-byte plane is
    // indistinguishable from one that sums the widths. `key_bits = 3` gives
    // 16, 8, 16, 4 -- offsets 0, 16, 24, 40 -- and there the two disagree.
    // Step 15 sweeps `key_bits`, so this is a configuration the design has to
    // hold, not a hypothetical.
    //
    // The stimulus is synthetic and the golden is arithmetic, deliberately:
    // the property under test is where a plane's bytes land in the row, and
    // `tests/test_ddr_layout.py` already checks the layout against the kernel.
    localparam int NU_RB   = 44;
    localparam int NU_BASE = 8192 * 16;             // beats 8192.. in `mem`
    localparam int NU_SPAN = 4096;
    localparam int NU_T    = 64;

    logic nu_start, nu_row_ready, nu_busy, nu_row_valid;
    logic [NP*AW-1:0] nu_base;
    logic [31:0] nu_ntok, nu_starve;
    logic [NU_RB*8-1:0] nu_row;
    logic [NP*IDW-1:0] nu_arid, nu_rid;
    logic [NP*AW-1:0]  nu_araddr;
    logic [NP*8-1:0]   nu_arlen;
    logic [NP*3-1:0]   nu_arsize, nu_arprot;
    logic [NP*2-1:0]   nu_arburst, nu_rresp;
    logic [NP*4-1:0]   nu_arcache, nu_arqos;
    logic [NP-1:0]     nu_arlock, nu_arvalid, nu_arready, nu_rlast,
                       nu_rvalid, nu_rready;
    logic [NP*DW-1:0]  nu_rdata;

    kv_store_ddr #(.DW(DW), .AW(AW), .IDW(IDW), .N_PORTS(NP),
                   .ROW_BYTES(NU_RB), .PLANE_W({8'd4, 8'd16, 8'd8, 8'd16}))
    nu (
        .clk, .rstn, .start(nu_start), .head_base(nu_base), .n_tokens(nu_ntok),
        .busy(nu_busy), .row_valid(nu_row_valid), .row_ready(nu_row_ready),
        .row_data(nu_row), .starve_cycles(nu_starve),
        .arid(nu_arid), .araddr(nu_araddr), .arlen(nu_arlen),
        .arsize(nu_arsize), .arburst(nu_arburst), .arlock(nu_arlock),
        .arcache(nu_arcache), .arprot(nu_arprot), .arqos(nu_arqos),
        .arvalid(nu_arvalid), .arready(nu_arready), .rid(nu_rid),
        .rdata(nu_rdata), .rresp(nu_rresp), .rlast(nu_rlast),
        .rvalid(nu_rvalid), .rready(nu_rready));

    // A zero-latency slave: this bench measures layout, not bandwidth.
    generate
        genvar z;
        for (z = 0; z < NP; z = z + 1) begin : nuport
            logic [AW-1:0] za;
            logic [8:0] zl;
            logic zb;
            assign nu_arready[z] = !zb;
            assign nu_rid[z*IDW +: IDW] = '0;
            assign nu_rresp[z*2 +: 2] = 2'b00;
            always_ff @(posedge clk) begin
                if (!rstn) begin zb <= 1'b0; nu_rvalid[z] <= 1'b0; end
                else begin
                    if (nu_rvalid[z] && nu_rready[z]) nu_rvalid[z] <= 1'b0;
                    if (nu_arvalid[z] && nu_arready[z]) begin
                        za <= nu_araddr[z*AW +: AW];
                        zl <= 9'(nu_arlen[z*8 +: 8]) + 9'd1;
                        zb <= 1'b1;
                    end else if (zb && (!nu_rvalid[z] || nu_rready[z])) begin
                        nu_rvalid[z] <= 1'b1;
                        nu_rdata[z*DW +: DW] <= mem[za[AW-1:4]];
                        nu_rlast[z] <= (zl == 9'd1);
                        za <= za + 16;
                        zl <= zl - 9'd1;
                        if (zl == 9'd1) zb <= 1'b0;
                    end
                end
            end
        end
    endgenerate

    // -- collector ----------------------------------------------------------
    int head = 0, tok = 0, got = 0;
    // Steady state is measured from the first row to the last, so the DDR
    // round trip -- paid once per scan and hidden by the next scan in
    // `attn_top` -- is not charged against the streaming rate.
    int t_first = 0, t_last = 0;
    always @(posedge clk) if (rstn && row_valid && row_ready) begin
        if (check_data)
            check(row_data === gold[head][tok],
                  $sformatf("head %0d token %0d: row mismatch", head, tok));
        if (tok == 0) t_first = $time;
        t_last = $time;
        tok = tok + 1;
        got = got + 1;
    end

    // Offsets 0, 16, 24, 40 -- the sums of the widths before each plane, which
    // is exactly what `p*16` gets wrong.
    always @(posedge clk) if (rstn && nu_row_valid && nu_row_ready) begin
        for (int p = 0; p < NP; p++) begin
            nu_cw  = (p == 1) ? 8 : ((p == 3) ? 4 : 16);
            nu_off = (p == 0) ? 0 : ((p == 1) ? 16 : ((p == 2) ? 24 : 40));
            for (int b = 0; b < nu_cw; b++)
                check(nu_row[(nu_off + b)*8 +: 8] === 8'((p*64 + nu_tok) ^ (b*17)),
                      $sformatf("non-uniform: plane %0d token %0d byte %0d",
                                p, nu_tok, b));
        end
        nu_tok = nu_tok + 1;
    end

    // -- the scan -----------------------------------------------------------
    int t0, cycles, total_starve = 0, steady = 0;
    logic check_data = 1, blanked = 0;
    int nu_tok = 0, nu_w = 0, nu_cw = 0, nu_off = 0;
    task automatic tick; @(negedge clk); endtask

    task automatic scan(input int h);
        tok = 0;
        for (int p = 0; p < NP; p++)
            head_base[p*AW +: AW] = h * HEAD_STRIDE + p * PLANE_SPAN;
        n_tokens = T;
        tick; start = 1; tick; start = 0;
        t0 = $time;
        while (tok < T) tick;
        cycles = ($time - t0) / 4;
        total_starve += starve_cycles;
    endtask

    initial begin
        // The range is explicit. Without it Icarus warns on every run that the
        // file is shorter than `mem`, which it is meant to be, and a real
        // short-file warning would then be lost in the noise.
        $readmemh("tb/vectors/ddr_image.hex", mem, 0, IMG_WORDS-1);
        for (int h = 0; h < KVH; h++) begin
            case (h)
                0: $readmemh("tb/vectors/cache_rows_h0.hex", onehead);
                1: $readmemh("tb/vectors/cache_rows_h1.hex", onehead);
                2: $readmemh("tb/vectors/cache_rows_h2.hex", onehead);
                3: $readmemh("tb/vectors/cache_rows_h3.hex", onehead);
                4: $readmemh("tb/vectors/cache_rows_h4.hex", onehead);
                5: $readmemh("tb/vectors/cache_rows_h5.hex", onehead);
                6: $readmemh("tb/vectors/cache_rows_h6.hex", onehead);
                7: $readmemh("tb/vectors/cache_rows_h7.hex", onehead);
            endcase
            for (int t = 0; t < T; t++) gold[h][t] = onehead[t];
        end

        start = 0; row_ready = 1; n_tokens = 0; head_base = '0;
        nu_start = 0; nu_row_ready = 0; nu_ntok = 0; nu_base = '0;
        w_valid = 0; w_token = 0; w_row = '0;
        repeat (4) @(posedge clk);
        rstn = 1;

        // The measured depth is a parameter, and a build that quietly raised
        // it would be 19% slower on the board and pass every other check here.
        check(dut.plane[0].rd.OUTST == 2, "outstanding must be 2, as measured");
        check(dut.plane[0].rd.BURST == 64, "burst must be 64, as measured");

        for (head = 0; head < KVH; head++) scan(head);
        check(got == KVH * T, "every head returned ctx+1 rows");

        // Steady state: one row per cycle once the pipe is full. The startup
        // cost is the DDR round trip and is paid once per scan, not per token.
        $display("  %0d heads x %0d rows, last scan %0d cycles (%0d tokens)",
                 KVH, T, cycles, T);
        $display("  starve cycles %0d total", total_starve);

        // -- steady state, over a scan long enough to have one ---------------
        //
        // 65 tokens is two bursts; the DDR round trip is most of it and the
        // rate says nothing. This runs 1,024 tokens through the same four
        // planes -- reading past the cached region into the plane's own zeroed
        // span, which is a legal address and the right access pattern, which is
        // what the rate depends on. Data is not checked here; the eight scans
        // above did that.
        check_data = 0;
        tok = 0; head = 0;
        for (int p = 0; p < NP; p++)
            head_base[p*AW +: AW] = p * PLANE_SPAN;
        n_tokens = LONG;
        tick; start = 1; tick; start = 0;
        while (tok < LONG) tick;
        steady = (t_last - t_first) / 4 + 1;
        $display("  long scan: %0d rows in %0d cycles after the first (%0d starve)",
                 LONG, steady, starve_cycles);
        // One row per cycle is the whole design point: 52 B/cycle is 13.0 GB/s
        // at 250 MHz, which is what `PARALLEL_KV = 1` was sized against.
        check(steady <= LONG + LONG/20,
              $sformatf("steady state is %0d cycles for %0d rows", steady, LONG));
        check_data = 1;

        // -- the round trip --------------------------------------------------
        //
        // Blank every plane's slot for token 64 across all eight heads, write
        // the rows back through the AXI write path, then scan and check. This
        // is the exit criterion's "row round-trip byte-exact against
        // quantize.pack": the bytes leave through WSTRB byte lanes and come
        // back through four read masters, and nothing in between knows what a
        // row is.
        // Tokens 61..64, not just 64. The norms are 4 B/token, so token 64
        // lands at byte lane 0 of its beat and exercises nothing: a write path
        // that ignored the byte lane entirely would round-trip it perfectly.
        // 61, 62 and 63 sit at lanes 4, 8 and 12.
        for (int h = 0; h < KVH; h++) begin
            for (int t = 61; t <= 64; t++) begin
                mem[(h*HEAD_STRIDE + 0*PLANE_SPAN)/16 + t] = '0;
                mem[(h*HEAD_STRIDE + 1*PLANE_SPAN)/16 + t] = '0;
                mem[(h*HEAD_STRIDE + 2*PLANE_SPAN)/16 + t] = '0;
            end
            mem[(h*HEAD_STRIDE + 3*PLANE_SPAN)/16 + 15] = '0;   // tokens 60..63
            mem[(h*HEAD_STRIDE + 3*PLANE_SPAN)/16 + 16] = '0;   // token  64
        end
        // The blanking must actually break the scan, or the round trip proves
        // nothing. One head is enough to establish it.
        check_data = 0; scan(0); check_data = 1;
        blanked = (dut.row_data !== gold[0][64]);
        check(blanked, "blanking token 64 must change what the scan returns");

        // Token 60 was blanked as collateral (it shares the norms beat with
        // 61..63) so it is rewritten too: four whole rows plus one norm.
        for (int h = 0; h < KVH; h++) begin
            for (int p = 0; p < NP; p++)
                head_base[p*AW +: AW] = h * HEAD_STRIDE + p * PLANE_SPAN;
            for (int t = 60; t <= 64; t++) begin
                w_token = t;
                w_row   = gold[h][t];
                w_valid = 1;
                @(posedge clk);
                while (!w_ready) begin tick; @(posedge clk); end
                tick; w_valid = 0;
                while (!w_ready) tick;
            end
        end

        got = 0;
        for (head = 0; head < KVH; head++) scan(head);
        check(got == KVH * T, "every head read back after the write path");
        $display("  round trip: %0d rows written and read back through AXI",
                 KVH*5);

        // -- the non-uniform layout -------------------------------------------
        //
        // Plane widths 16, 8, 16, 4 -- offsets 0, 16, 24, 40. Each token's
        // slice in plane `p` is stamped with a value that names (p, token), so
        // a row assembled at the wrong offsets cannot look right by accident.
        for (int p = 0; p < NP; p++)
            for (int t = 0; t < NU_T; t++) begin
                nu_w = 0;
                case (p) 0: nu_w = 16; 1: nu_w = 8; 2: nu_w = 16; 3: nu_w = 4;
                endcase
                for (int b = 0; b < nu_w; b++)
                    mem[(NU_BASE + p*NU_SPAN + t*nu_w + b) / 16]
                       [((NU_BASE + p*NU_SPAN + t*nu_w + b) % 16)*8 +: 8]
                       = 8'((p*64 + t) ^ (b*17));
            end
        for (int p = 0; p < NP; p++)
            nu_base[p*AW +: AW] = NU_BASE + p*NU_SPAN;
        nu_ntok = NU_T; nu_row_ready = 1;
        nu_tok = 0;
        tick; nu_start = 1; tick; nu_start = 0;
        while (nu_tok < NU_T) tick;
        $display("  non-uniform layout: %0d rows of %0d B at widths 16/8/16/4",
                 NU_T, NU_RB);

        // Now with the memory stalling R at random.
        stall_r = 1;
        head = 0; got = 0;
        for (head = 0; head < KVH; head++) scan(head);
        check(got == KVH * T, "every head returned ctx+1 rows under R stalls");

        if (errors == 0) $display("=== tb_kv_store: ALL TESTS PASSED ===");
        else $display("=== tb_kv_store: %0d TESTS FAILED ===", errors);
        $finish;
    end

    initial begin
        #20000000;
        $display("  FAIL: watchdog -- the scan never finished");
        $display("=== tb_kv_store: 1 TESTS FAILED ===");
        $finish;
    end
endmodule
