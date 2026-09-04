// Self-checking testbench for the DDR bandwidth probe.
//
// The point of the probe is to measure a real memory system, so this bench
// cannot validate the ANSWER -- only the INSTRUMENT. It checks the three ways
// the instrument could lie:
//
//   1. dropped or double-counted beats  -> the byte count would be wrong
//   2. a cycle counter that stops early -> the bandwidth would be overstated
//   3. outstanding-transaction credit that does not actually gate           ->
//      latency hiding would look free when it is not
//
// (3) is the one worth a bench: if the credit logic were broken the probe
// would report the same bandwidth at outstanding=1 and outstanding=16, and the
// prefetch depth in kv_store_ddr would then be sized from a number that means
// nothing.
//
// Written for Icarus Verilog 12, which has no `break` and no queues -- hence
// the explicit while-loops and the circular buffer in the slave model.
`timescale 1ns/1ps
module tb_ddr_probe;
    localparam int DW  = 128;
    localparam int AW  = 49;
    localparam int IDW = 6;
    localparam int LAW = 12;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;              // 250 MHz

    int errors = 0;
    task check(input logic cond, input string what);
        if (!cond) begin $display("  FAIL: %s", what); errors = errors + 1; end
    endtask

    // Drive and sample on the NEGEDGE. The DUT registers everything off the
    // posedge, so mid-cycle is the only place where a signal is unambiguously
    // settled. Sampling at the posedge instead reads the value the DUT is in
    // the act of replacing, and a driver built that way deasserts VALID one
    // cycle before the handshake it thinks it saw -- which hangs the bus.
    task tick; @(negedge clk); endtask

    // ---------------- AXI-Lite driver ----------------
    logic [LAW-1:0] awaddr, araddr;
    logic           awvalid, wvalid, bready, arvalid, rready;
    logic [31:0]    wdata, rdata;
    logic           awready, wready, bvalid, arready, rvalid;
    logic [3:0]     wstrb = 4'hF;

    // The DUT asserts AWREADY and WREADY together, so one wait covers both.
    // After READY is observed at a negedge, the handshake fires at the NEXT
    // posedge -- hence the extra tick before dropping VALID.
    task automatic lite_write(input int addr, input int unsigned data);
        tick;
        awaddr  = addr[LAW-1:0]; awvalid = 1'b1;
        wdata   = data;          wvalid  = 1'b1;
        bready  = 1'b1;
        tick;
        while (!(awready && wready)) tick;
        tick;
        awvalid = 1'b0; wvalid = 1'b0;
        while (!bvalid) tick;
        tick;
        bready = 1'b0;
    endtask

    task automatic lite_read(input int addr, output int unsigned data);
        tick;
        araddr = addr[LAW-1:0]; arvalid = 1'b1; rready = 1'b1;
        tick;
        while (!arready) tick;
        tick;
        arvalid = 1'b0;
        while (!rvalid) tick;
        data = rdata;
        tick;
        rready = 1'b0;
    endtask

    // ---------------- three AXI4 read ports ----------------
    logic [AW-1:0] araddr_m [0:3];
    logic [7:0]    arlen_m  [0:3];
    logic          arvalid_m[0:3], arready_m[0:3];
    logic [DW-1:0] rdata_m  [0:3];
    logic          rlast_m  [0:3], rvalid_m [0:3], rready_m [0:3];

    ddr_probe #(.DW(DW), .AW(AW), .IDW(IDW), .LAW(LAW)) dut (
        .clk(clk), .rstn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid),
        .s_axi_awready(awready), .s_axi_wdata(wdata), .s_axi_wstrb(wstrb),
        .s_axi_wvalid(wvalid), .s_axi_wready(wready), .s_axi_bresp(),
        .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready),

        .m00_axi_arid(), .m00_axi_araddr(araddr_m[0]), .m00_axi_arlen(arlen_m[0]),
        .m00_axi_arsize(), .m00_axi_arburst(), .m00_axi_arlock(), .m00_axi_arcache(),
        .m00_axi_arprot(), .m00_axi_arqos(), .m00_axi_arvalid(arvalid_m[0]),
        .m00_axi_arready(arready_m[0]), .m00_axi_rid(6'd0), .m00_axi_rdata(rdata_m[0]),
        .m00_axi_rresp(2'b0), .m00_axi_rlast(rlast_m[0]), .m00_axi_rvalid(rvalid_m[0]),
        .m00_axi_rready(rready_m[0]),

        .m01_axi_arid(), .m01_axi_araddr(araddr_m[1]), .m01_axi_arlen(arlen_m[1]),
        .m01_axi_arsize(), .m01_axi_arburst(), .m01_axi_arlock(), .m01_axi_arcache(),
        .m01_axi_arprot(), .m01_axi_arqos(), .m01_axi_arvalid(arvalid_m[1]),
        .m01_axi_arready(arready_m[1]), .m01_axi_rid(6'd0), .m01_axi_rdata(rdata_m[1]),
        .m01_axi_rresp(2'b0), .m01_axi_rlast(rlast_m[1]), .m01_axi_rvalid(rvalid_m[1]),
        .m01_axi_rready(rready_m[1]),

        .m02_axi_arid(), .m02_axi_araddr(araddr_m[2]), .m02_axi_arlen(arlen_m[2]),
        .m02_axi_arsize(), .m02_axi_arburst(), .m02_axi_arlock(), .m02_axi_arcache(),
        .m02_axi_arprot(), .m02_axi_arqos(), .m02_axi_arvalid(arvalid_m[2]),
        .m02_axi_arready(arready_m[2]), .m02_axi_rid(6'd0), .m02_axi_rdata(rdata_m[2]),
        .m02_axi_rresp(2'b0), .m02_axi_rlast(rlast_m[2]), .m02_axi_rvalid(rvalid_m[2]),
        .m02_axi_rready(rready_m[2]),

        .m03_axi_arid(), .m03_axi_araddr(araddr_m[3]), .m03_axi_arlen(arlen_m[3]),
        .m03_axi_arsize(), .m03_axi_arburst(), .m03_axi_arlock(), .m03_axi_arcache(),
        .m03_axi_arprot(), .m03_axi_arqos(), .m03_axi_arvalid(arvalid_m[3]),
        .m03_axi_arready(arready_m[3]), .m03_axi_rid(6'd0), .m03_axi_rdata(rdata_m[3]),
        .m03_axi_rresp(2'b0), .m03_axi_rlast(rlast_m[3]), .m03_axi_rvalid(rvalid_m[3]),
        .m03_axi_rready(rready_m[3]));

    // ---------------- behavioural AXI read slave, x3 ----------------
    // A request accepted at cycle T has its data ready at T+LAT; the data
    // channel then serves bursts in order, back-to-back, never starting one
    // before its ready time.
    //
    // The LATENCY IS PIPELINED, which is the whole point. A model that re-pays
    // LAT for every burst even when the queue is full would report the same
    // bandwidth at every outstanding depth -- and the prefetch depth in
    // kv_store_ddr would then be sized from a number that means nothing.
    localparam int LAT   = 40;
    localparam int QDEEP = 64;
    integer ar_count [0:3];

    generate
        genvar p;
        for (p = 0; p < 4; p = p + 1) begin : slv
            integer len_fifo [0:QDEEP-1];
            integer rdy_fifo [0:QDEEP-1];
            integer wr_ptr, rd_ptr, beats_left, now;
            logic   serving;

            assign arready_m[p] = 1'b1;              // always take an AR

            always_ff @(posedge clk) begin
                if (!rstn) begin
                    wr_ptr <= 0; rd_ptr <= 0; beats_left <= 0; now <= 0;
                    serving <= 1'b0; rvalid_m[p] <= 1'b0; rlast_m[p] <= 1'b0;
                    ar_count[p] <= 0;
                end else begin
                    now <= now + 1;

                    if (arvalid_m[p] && arready_m[p]) begin
                        len_fifo[wr_ptr % QDEEP] <= integer'(arlen_m[p]) + 1;
                        rdy_fifo[wr_ptr % QDEEP] <= now + LAT;
                        wr_ptr      <= wr_ptr + 1;
                        ar_count[p] <= ar_count[p] + 1;
                    end

                    if (!serving) begin
                        rvalid_m[p] <= 1'b0;
                        rlast_m[p]  <= 1'b0;
                        if (rd_ptr != wr_ptr && now >= rdy_fifo[rd_ptr % QDEEP]) begin
                            serving    <= 1'b1;
                            beats_left <= len_fifo[rd_ptr % QDEEP];
                            rd_ptr     <= rd_ptr + 1;
                        end
                    end else begin
                        // A beat moves only when RVALID and RREADY are both
                        // high at an edge. Decrementing on RREADY alone --
                        // before RVALID has come up -- silently drops the
                        // first beat of every burst, and the engine then waits
                        // forever for a count it will never reach.
                        rdata_m[p] <= {DW{1'b1}};
                        if (!rvalid_m[p]) begin
                            rvalid_m[p] <= 1'b1;
                            rlast_m[p]  <= (beats_left == 1);
                        end else if (rready_m[p]) begin
                            if (beats_left == 1) begin
                                serving     <= 1'b0;
                                beats_left  <= 0;
                                rvalid_m[p] <= 1'b0;
                                rlast_m[p]  <= 1'b0;
                            end else begin
                                beats_left <= beats_left - 1;
                                rlast_m[p] <= (beats_left == 2);
                            end
                        end
                    end
                end
            end
        end
    endgenerate

    // ---------------- run one measurement ----------------
    task automatic run(input int unsigned mask, input int unsigned beats,
                       input int unsigned burst, input int unsigned outst,
                       output int unsigned cycles, output int unsigned total);
        int unsigned st, g0, g1, g2, g3;
        lite_write('h04, mask);
        lite_write('h08, 32'h1000_0000);
        lite_write('h10, 32'h0400_0000);
        lite_write('h14, beats);
        lite_write('h18, burst);
        lite_write('h1C, outst);
        lite_write('h00, 1);
        st = 32'h2;
        while (st[1]) begin
            lite_read('h00, st);
            if (st[1]) repeat (20) tick;
        end
        lite_read('h20, cycles);
        lite_read('h24, g0); lite_read('h28, g1);
        lite_read('h2C, g2); lite_read('h34, g3);
        total = g0 + g1 + g2 + g3;
    endtask

    int unsigned v, c1, c16, t1, t16, c, t;

    initial begin
        awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
        awaddr = 0; araddr = 0; wdata = 0;
        repeat (20) @(posedge clk);
        rstn = 1;
        repeat (10) @(posedge clk);

        lite_read('h30, v);
        check(v == 32'h44445250, "MAGIC readback");

        // ---- the credit knob must actually gate ----
        // 64 bursts of 64 beats = 4096 beats on one port.
        run(4'b0001, 4096, 64, 1,  c1,  t1);
        check(t1 == 4096, $sformatf("outstanding=1 delivered %0d/4096 beats", t1));

        run(4'b0001, 4096, 64, 16, c16, t16);
        check(t16 == 4096, $sformatf("outstanding=16 delivered %0d/4096 beats", t16));

        $display("  outstanding=1  : %0d cyc  (%0.3f beats/cyc)", c1,  $itor(t1)/$itor(c1));
        $display("  outstanding=16 : %0d cyc  (%0.3f beats/cyc)", c16, $itor(t16)/$itor(c16));

        // With LAT=40 and 64-beat bursts, depth 1 pays the latency on every
        // burst and depth 16 hides all of it. If the credit logic does not
        // gate, these come out equal.
        check(c1 > (c16 * 3) / 2, "outstanding depth must change the cycle count");
        check($itor(t16)/$itor(c16) > 0.9, "deep queue should approach 1 beat/cyc");

        // ---- all four ports issue and are counted ----
        run(4'b1111, 4096, 64, 16, c, t);
        check(t == 4*4096, $sformatf("4 ports delivered %0d/16384 beats", t));
        check(ar_count[1] > 0 && ar_count[2] > 0 && ar_count[3] > 0,
              "all four ports must issue ARs");
        $display("  4 ports        : %0d cyc  (%0.3f beats/cyc aggregate)",
                 c, $itor(t)/$itor(c));

        // ---- a disabled port must not report a stale count ----
        run(4'b0001, 2048, 64, 16, c, t);
        check(t == 2048, $sformatf("after a 4-port run, 1-port run reported %0d/2048", t));

        // ---- burst length extremes ----
        run(4'b0001, 4096, 256, 16, c, t);
        check(t == 4096, "burst=256 delivered all beats");
        run(4'b0001, 512, 1, 16, c, t);
        check(t == 512, "burst=1 delivered all beats");

        if (errors == 0) $display("=== tb_ddr_probe: ALL TESTS PASSED ===");
        else             $display("=== tb_ddr_probe: %0d FAILURES ===", errors);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("=== tb_ddr_probe: TIMEOUT ===");
        $finish;
    end
endmodule
