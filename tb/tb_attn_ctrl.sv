// Self-checking testbench for attn_ctrl.sv: the block as JTAG will drive it.
//
// THE POINT IS THE PROTOCOL, NOT THE ARITHMETIC
// ---------------------------------------------
// `tb_attn` already proved the datapath against `out.online.hex`. This bench
// proves the SHIM: that a host which writes 2,304 words into the input window,
// pokes one control register, polls a status bit and reads 1,536 words back
// gets those same numbers. Every access is a 32-bit AXI-Lite transaction at a
// real address, because that is what `create_hw_axi_txn` issues -- and it is
// the one thing that cannot be checked on the board without a working
// bitstream to check it with.
//
// WHAT WOULD OTHERWISE BE FOUND AT 1 ms PER TRANSACTION
// -----------------------------------------------------
// Word order inside a 512-bit beat, the buffer base addresses, a status bit
// that never latches, a `done` pulse missed between two polls, and reads that
// wedge because `rvalid` came out of reset as X. Each of those is a morning on
// hardware and a second here.
//
// The input buffer is READ BACK before the run, because a host that cannot
// trust what it loaded cannot interpret what it gets out.
`timescale 1ns/1ps
module tb_attn_ctrl;
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


    // -- AXI-Lite master, one transaction at a time, as JTAG is --------------
    localparam int LAW = 17;
    localparam logic [LAW-1:0] IN_BASE  = 17'h0_2000;
    localparam logic [LAW-1:0] OUT_BASE = 17'h0_8000;

    logic [LAW-1:0] s_awaddr, s_araddr;
    logic [31:0]    s_wdata, s_rdata;
    logic           s_awvalid, s_awready, s_wvalid, s_wready, s_bvalid, s_bready;
    logic           s_arvalid, s_arready, s_rvalid, s_rready;
    logic [1:0]     s_bresp, s_rresp;

    attn_ctrl #(.LAW(LAW), .DW(DW), .AW(AW), .IDW(IDW), .N_PORTS(NP))
    dut (
        .clk, .rstn,
        .s_axi_awaddr(s_awaddr), .s_axi_awprot(3'b000), .s_axi_awvalid(s_awvalid),
        .s_axi_awready(s_awready), .s_axi_wdata(s_wdata), .s_axi_wstrb(4'hF),
        .s_axi_wvalid(s_wvalid), .s_axi_wready(s_wready), .s_axi_bresp(s_bresp),
        .s_axi_bvalid(s_bvalid), .s_axi_bready(s_bready),
        .s_axi_araddr(s_araddr), .s_axi_arprot(3'b000), .s_axi_arvalid(s_arvalid),
        .s_axi_arready(s_arready), .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp),
        .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready),
        .m00_axi_arid(arid[0*IDW +: IDW]), .m00_axi_araddr(araddr[0*AW +: AW]),
        .m00_axi_arlen(arlen[0*8 +: 8]), .m00_axi_arsize(arsize[0*3 +: 3]),
        .m00_axi_arburst(arburst[0*2 +: 2]), .m00_axi_arlock(arlock[0]),
        .m00_axi_arcache(arcache[0*4 +: 4]), .m00_axi_arprot(arprot[0*3 +: 3]),
        .m00_axi_arqos(arqos[0*4 +: 4]), .m00_axi_arvalid(arvalid[0]),
        .m00_axi_arready(arready[0]), .m00_axi_rid(rid[0*IDW +: IDW]),
        .m00_axi_rdata(rdata[0*DW +: DW]), .m00_axi_rresp(rresp[0*2 +: 2]),
        .m00_axi_rlast(rlast[0]), .m00_axi_rvalid(rvalid[0]),
        .m00_axi_rready(rready[0]),
        .m01_axi_arid(arid[1*IDW +: IDW]), .m01_axi_araddr(araddr[1*AW +: AW]),
        .m01_axi_arlen(arlen[1*8 +: 8]), .m01_axi_arsize(arsize[1*3 +: 3]),
        .m01_axi_arburst(arburst[1*2 +: 2]), .m01_axi_arlock(arlock[1]),
        .m01_axi_arcache(arcache[1*4 +: 4]), .m01_axi_arprot(arprot[1*3 +: 3]),
        .m01_axi_arqos(arqos[1*4 +: 4]), .m01_axi_arvalid(arvalid[1]),
        .m01_axi_arready(arready[1]), .m01_axi_rid(rid[1*IDW +: IDW]),
        .m01_axi_rdata(rdata[1*DW +: DW]), .m01_axi_rresp(rresp[1*2 +: 2]),
        .m01_axi_rlast(rlast[1]), .m01_axi_rvalid(rvalid[1]),
        .m01_axi_rready(rready[1]),
        .m02_axi_arid(arid[2*IDW +: IDW]), .m02_axi_araddr(araddr[2*AW +: AW]),
        .m02_axi_arlen(arlen[2*8 +: 8]), .m02_axi_arsize(arsize[2*3 +: 3]),
        .m02_axi_arburst(arburst[2*2 +: 2]), .m02_axi_arlock(arlock[2]),
        .m02_axi_arcache(arcache[2*4 +: 4]), .m02_axi_arprot(arprot[2*3 +: 3]),
        .m02_axi_arqos(arqos[2*4 +: 4]), .m02_axi_arvalid(arvalid[2]),
        .m02_axi_arready(arready[2]), .m02_axi_rid(rid[2*IDW +: IDW]),
        .m02_axi_rdata(rdata[2*DW +: DW]), .m02_axi_rresp(rresp[2*2 +: 2]),
        .m02_axi_rlast(rlast[2]), .m02_axi_rvalid(rvalid[2]),
        .m02_axi_rready(rready[2]),
        .m03_axi_arid(arid[3*IDW +: IDW]), .m03_axi_araddr(araddr[3*AW +: AW]),
        .m03_axi_arlen(arlen[3*8 +: 8]), .m03_axi_arsize(arsize[3*3 +: 3]),
        .m03_axi_arburst(arburst[3*2 +: 2]), .m03_axi_arlock(arlock[3]),
        .m03_axi_arcache(arcache[3*4 +: 4]), .m03_axi_arprot(arprot[3*3 +: 3]),
        .m03_axi_arqos(arqos[3*4 +: 4]), .m03_axi_arvalid(arvalid[3]),
        .m03_axi_arready(arready[3]), .m03_axi_rid(rid[3*IDW +: IDW]),
        .m03_axi_rdata(rdata[3*DW +: DW]), .m03_axi_rresp(rresp[3*2 +: 2]),
        .m03_axi_rlast(rlast[3]), .m03_axi_rvalid(rvalid[3]),
        .m03_axi_rready(rready[3]),
        .m_wr_axi_awid(awid), .m_wr_axi_awaddr(awaddr), .m_wr_axi_awlen(awlen),
        .m_wr_axi_awsize(awsize), .m_wr_axi_awburst(awburst),
        .m_wr_axi_awcache(awcache), .m_wr_axi_awprot(awprot),
        .m_wr_axi_awvalid(awvalid), .m_wr_axi_awready(awready),
        .m_wr_axi_wdata(wdata), .m_wr_axi_wstrb(wstrb), .m_wr_axi_wlast(wlast),
        .m_wr_axi_wvalid(wvalid), .m_wr_axi_wready(wready),
        .m_wr_axi_bresp(bresp), .m_wr_axi_bvalid(bvalid), .m_wr_axi_bready(bready));

    task automatic axi_write(input [LAW-1:0] a, input [31:0] d);
        @(negedge clk);
        s_awaddr = a; s_wdata = d; s_awvalid = 1; s_wvalid = 1; s_bready = 1;
        while (!(s_awvalid && s_awready)) @(negedge clk);
        @(negedge clk); s_awvalid = 0;
        while (s_wvalid && !s_wready) @(negedge clk);
        @(negedge clk); s_wvalid = 0;
        while (!s_bvalid) @(negedge clk);
        @(negedge clk); s_bready = 0;
    endtask

    task automatic axi_read(input [LAW-1:0] a, output [31:0] d);
        @(negedge clk);
        s_araddr = a; s_arvalid = 1; s_rready = 1;
        while (!(s_arvalid && s_arready)) @(negedge clk);
        @(negedge clk); s_arvalid = 0;
        while (!s_rvalid) @(negedge clk);
        d = s_rdata;
        @(negedge clk); s_rready = 0;
    endtask

    task automatic reload_image(input logic blank_new_token);
        for (int i = 0; i < IMG_BEATS; i++) mem[i] = img[i];
        // Token 64's slot in every plane, so the answer depends on the write
        // path having run. The norms plane is 4 B/token, so only token 64's own
        // four bytes may go -- blanking the beat would take tokens 61..63 with
        // it, and those are cached tokens the golden depends on.
        if (blank_new_token)
            for (int h = 0; h < KVH; h++) begin
                mem[(h*HEAD_STRIDE + 0*PLANE_SPAN)/16 + 64] = '0;
                mem[(h*HEAD_STRIDE + 1*PLANE_SPAN)/16 + 64] = '0;
                mem[(h*HEAD_STRIDE + 2*PLANE_SPAN)/16 + 64] = '0;
                mem[(h*HEAD_STRIDE + 3*PLANE_SPAN)/16 + 16][31:0] = '0;
            end
    endtask

    logic [31:0] rv, word;
    logic [HEADS*D*W-1:0] result;
    int polls;

    initial begin
        $readmemh("tb/vectors/ingress.hex", stim);
        $readmemh("tb/vectors/ddr_image.hex", img);
        $readmemh("tb/vectors/out.online.hex", gold);

        s_awvalid = 0; s_wvalid = 0; s_bready = 0; s_arvalid = 0; s_rready = 0;
        s_awaddr = '0; s_araddr = '0; s_wdata = '0;
        reload_image(1);
        repeat (4) @(posedge clk);
        rstn = 1;

        // -- the bitstream is the one the host thinks it is -------------------
        axi_read(17'h04, rv);
        check(rv === 32'hA77E_0001, $sformatf("MAGIC reads %08x", rv));
        axi_read(17'h28, rv);
        check(rv === 32'd2304, $sformatf("input window is %0d words", rv));
        axi_read(17'h2C, rv);
        check(rv === 32'd1536, $sformatf("output window is %0d words", rv));
        // An address in neither window must be a hole, not a neighbour.
        axi_read(17'h00FC, rv);
        check(rv === 32'hDEAD_BEEF, "an unmapped register reads DEADBEEF");

        // -- load the token, low word of each beat first ----------------------
        for (int b = 0; b < IN_BEATS; b++)
            for (int w = 0; w < 16; w++)
                axi_write(IN_BASE + LAW'((b*16 + w)*4), stim[b][w*32 +: 32]);

        // Read it back. A host that cannot trust what it loaded cannot
        // interpret what comes out, and on JTAG a dropped write is a real
        // possibility rather than a hypothetical one.
        for (int b = 0; b < IN_BEATS; b += 17)
            for (int w = 0; w < 16; w++) begin
                axi_read(IN_BASE + LAW'((b*16 + w)*4), word);
                check(word === stim[b][w*32 +: 32],
                      $sformatf("input beat %0d word %0d read back %08x", b, w, word));
            end

        axi_write(17'h08, T);
        axi_write(17'h0C, 32'd0);
        axi_write(17'h10, 32'd0);
        axi_write(17'h14, HEAD_STRIDE);
        axi_write(17'h18, PLANE_SPAN);
        axi_read(17'h08, rv);
        check(rv === T, "n_tokens reads back what was written");

        // -- go ---------------------------------------------------------------
        axi_write(17'h00, 32'd1);
        polls = 0; rv = '0;
        while (!rv[1] && polls <= 2000) begin       // bit 1 is `done`, latched
            axi_read(17'h00, rv);
            polls = polls + 1;
        end
        check(polls <= 2000, "the step never reported done");
        // `done` is one pulse and a host polls at millisecond intervals, so the
        // bit must LATCH. Reading it twice proves it did.
        axi_read(17'h00, rv);
        check(rv[1] === 1'b1, "the done bit is latched, not a pulse");
        check(rv[0] === 1'b0, "busy is clear once done is set");
        check(rv[2] === 1'b0, "no channel escaped the 24-bit seam");

        // -- read the answer ---------------------------------------------------
        for (int b = 0; b < OUT_BEATS; b++)
            for (int w = 0; w < 16; w++) begin
                axi_read(OUT_BASE + LAW'((b*16 + w)*4), word);
                result[(b*16 + w)*32 +: 32] = word;
            end

        for (int h = 0; h < HEADS; h++)
            for (int c = 0; c < D; c++)
                check(result[(h*D + c)*W +: W] === gold[0][(h*D + c)*W +: W],
                      $sformatf("head %0d channel %0d: %06x != %06x", h, c,
                                result[(h*D + c)*W +: W], gold[0][(h*D + c)*W +: W]));

        axi_read(17'h1C, rv);
        $display("  %0d heads read back over AXI-Lite, %0d polls, %0d starve cycles",
                 HEADS, polls, rv);

        // -- the output window is re-readable ----------------------------------
        //
        // A JTAG read that has to be repeated must return the same word. A
        // pop-on-read FIFO would pass every check above and fail this one.
        axi_read(OUT_BASE, rv);
        axi_read(OUT_BASE, word);
        check(rv === word, "reading the output window twice returns the same word");

        // -- a second step on the same instance ---------------------------------
        reload_image(1);
        axi_write(17'h00, 32'd1);
        polls = 0; rv = '0;
        while (!rv[1] && polls <= 2000) begin
            axi_read(17'h00, rv);
            polls = polls + 1;
        end
        check(polls <= 2000, "the second step never reported done");
        for (int b = 0; b < OUT_BEATS; b++)
            for (int w = 0; w < 16; w++) begin
                axi_read(OUT_BASE + LAW'((b*16 + w)*4), word);
                check(word === result[(b*16 + w)*32 +: 32],
                      $sformatf("second step beat %0d word %0d differs", b, w));
            end

        if (errors == 0) $display("=== tb_attn_ctrl: ALL TESTS PASSED ===");
        else $display("=== tb_attn_ctrl: %0d TESTS FAILED ===", errors);
        $finish;
    end

    initial begin
        #500000000;
        $display("  FAIL: watchdog -- the step never finished");
        $display("=== tb_attn_ctrl: 1 TESTS FAILED ===");
        $finish;
    end
endmodule
