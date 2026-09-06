// DDR bandwidth probe -- Step 1 of the implementation plan, and the gate on
// everything after it.
//
// Four AXI-HP read masters, an AXI-Lite control block, and a cycle counter.
// No attention logic. The single question it answers:
//
//     does PS DDR4-2400 sustain the 13.0 GB/s that PARALLEL_KV=1 needs?
//
// 52 B/cycle at 250 MHz is the appetite of one score-lane group. The plan
// assumes ~13.4 GB/s sustained (about 70% of the 19.2 GB/s peak). If the real
// number is materially lower, PARALLEL_KV=1 is starved and every throughput
// figure in the plan is optimistic -- which is exactly why this is measured
// before any attention RTL exists.
//
// Why four ports and not three
// ----------------------------
// A 128-bit HP port moves 16 B per PL clock. At one clock domain of 250 MHz
// that is 4.0 GB/s, so three ports reach only 12.0 GB/s -- Short of the 13.0
// the engine needs. Three ports would suffice at 333 MHz, but only by running
// the HP interfaces in a second clock domain and paying for the CDC.
//
//     4 ports x 16 B x 250 MHz = 16.0 GB/s, single clock domain, 23% headroom
//
// The headroom is the point: 16.0 GB/s of interface against 13.0 GB/s of
// appetite leaves room for the DRAM controller to fall short of its own peak,
// which it certainly will.
//
// One bitstream, swept at runtime
// -------------------------------
// Port count, burst length and outstanding depth are registers, not
// parameters, so the whole sweep runs from software without rebuilding. A
// Vivado implementation run is ~20 minutes and the interesting space is
// 4 x 4 x 5 points.
//
// The ports read disjoint regions (base + port*stride). Pointing them at the
// same region would let the DRAM controller's row buffer serve two of them
// from one activate and report a bandwidth the real design cannot reproduce --
// the KV cache scan is one sequential stream per active head, not four aliased
// ones.
`timescale 1ns/1ps
module ddr_probe #(
    parameter int DW  = 128,
    parameter int AW  = 49,
    parameter int IDW = 6,
    parameter int LAW = 12          // AXI-Lite address width
)(
    input  logic            clk,
    input  logic            rstn,

    // ---------------- AXI-Lite control ----------------
    input  logic [LAW-1:0]  s_axi_awaddr,
    input  logic [2:0]      s_axi_awprot,
    input  logic            s_axi_awvalid,
    output logic            s_axi_awready,
    input  logic [31:0]     s_axi_wdata,
    input  logic [3:0]      s_axi_wstrb,
    input  logic            s_axi_wvalid,
    output logic            s_axi_wready,
    output logic [1:0]      s_axi_bresp,
    output logic            s_axi_bvalid,
    input  logic            s_axi_bready,
    input  logic [LAW-1:0]  s_axi_araddr,
    input  logic [2:0]      s_axi_arprot,
    input  logic            s_axi_arvalid,
    output logic            s_axi_arready,
    output logic [31:0]     s_axi_rdata,
    output logic [1:0]      s_axi_rresp,
    output logic            s_axi_rvalid,
    input  logic            s_axi_rready,

    // ---------------- four AXI4 read masters ----------------
    // Flat, individually named, because that is how Vivado infers an interface
    // on a module reference. The logic below works on arrays and these are
    // wired to them at the bottom.
    output logic [IDW-1:0] m00_axi_arid,   output logic [AW-1:0] m00_axi_araddr,
    output logic [7:0]     m00_axi_arlen,  output logic [2:0]    m00_axi_arsize,
    output logic [1:0]     m00_axi_arburst,output logic          m00_axi_arlock,
    output logic [3:0]     m00_axi_arcache,output logic [2:0]    m00_axi_arprot,
    output logic [3:0]     m00_axi_arqos,  output logic          m00_axi_arvalid,
    input  logic           m00_axi_arready,
    input  logic [IDW-1:0] m00_axi_rid,    input  logic [DW-1:0] m00_axi_rdata,
    input  logic [1:0]     m00_axi_rresp,  input  logic          m00_axi_rlast,
    input  logic           m00_axi_rvalid, output logic          m00_axi_rready,

    output logic [IDW-1:0] m01_axi_arid,   output logic [AW-1:0] m01_axi_araddr,
    output logic [7:0]     m01_axi_arlen,  output logic [2:0]    m01_axi_arsize,
    output logic [1:0]     m01_axi_arburst,output logic          m01_axi_arlock,
    output logic [3:0]     m01_axi_arcache,output logic [2:0]    m01_axi_arprot,
    output logic [3:0]     m01_axi_arqos,  output logic          m01_axi_arvalid,
    input  logic           m01_axi_arready,
    input  logic [IDW-1:0] m01_axi_rid,    input  logic [DW-1:0] m01_axi_rdata,
    input  logic [1:0]     m01_axi_rresp,  input  logic          m01_axi_rlast,
    input  logic           m01_axi_rvalid, output logic          m01_axi_rready,

    output logic [IDW-1:0] m02_axi_arid,   output logic [AW-1:0] m02_axi_araddr,
    output logic [7:0]     m02_axi_arlen,  output logic [2:0]    m02_axi_arsize,
    output logic [1:0]     m02_axi_arburst,output logic          m02_axi_arlock,
    output logic [3:0]     m02_axi_arcache,output logic [2:0]    m02_axi_arprot,
    output logic [3:0]     m02_axi_arqos,  output logic          m02_axi_arvalid,
    input  logic           m02_axi_arready,
    input  logic [IDW-1:0] m02_axi_rid,    input  logic [DW-1:0] m02_axi_rdata,
    input  logic [1:0]     m02_axi_rresp,  input  logic          m02_axi_rlast,
    input  logic           m02_axi_rvalid, output logic          m02_axi_rready,

    output logic [IDW-1:0] m03_axi_arid,   output logic [AW-1:0] m03_axi_araddr,
    output logic [7:0]     m03_axi_arlen,  output logic [2:0]    m03_axi_arsize,
    output logic [1:0]     m03_axi_arburst,output logic          m03_axi_arlock,
    output logic [3:0]     m03_axi_arcache,output logic [2:0]    m03_axi_arprot,
    output logic [3:0]     m03_axi_arqos,  output logic          m03_axi_arvalid,
    input  logic           m03_axi_arready,
    input  logic [IDW-1:0] m03_axi_rid,    input  logic [DW-1:0] m03_axi_rdata,
    input  logic [1:0]     m03_axi_rresp,  input  logic          m03_axi_rlast,
    input  logic           m03_axi_rvalid, output logic          m03_axi_rready
);
    localparam int NP = 4;
    localparam logic [31:0] MAGIC = 32'h44445250;   // "DDRP"

    // ---------------- registers ----------------
    logic [NP-1:0] r_mask;
    logic [AW-1:0] r_base;
    logic [31:0]   r_stride;    // bytes between port regions
    logic [31:0]   r_beats;     // beats per port
    logic [8:0]    r_burst;     // 1..256
    logic [7:0]    r_outst;     // 1..255
    logic [31:0]   r_cycles;
    logic          start_pulse;

    logic [NP-1:0] busy;
    logic [31:0]   got [NP];
    wire           any_busy = |(busy & r_mask);
    logic          run;         // high from start until all enabled engines idle

    // ---------------- engines ----------------
    logic [IDW-1:0] arid  [NP];
    logic [AW-1:0]  araddr[NP];
    logic [7:0]     arlen [NP];
    logic [2:0]     arsize[NP];
    logic [1:0]     arburst[NP];
    logic           arlock[NP];
    logic [3:0]     arcache[NP];
    logic [2:0]     arprot[NP];
    logic [3:0]     arqos [NP];
    logic           arvalid[NP], arready[NP];
    logic [IDW-1:0] rid   [NP];
    logic [DW-1:0]  rdata_m[NP];
    logic [1:0]     rresp [NP];
    logic           rlast [NP], rvalid_m[NP], rready[NP];

    generate
        genvar i;
        for (i = 0; i < NP; i = i + 1) begin : eng
            // Disjoint regions: port i reads [base + i*stride, +beats*DW/8).
            wire [AW-1:0] base_i = r_base + AW'(i) * AW'(r_stride);
            axi_rd_engine #(.DW(DW), .AW(AW), .IDW(IDW)) u (
                .clk(clk), .rstn(rstn),
                .start(start_pulse), .enable(r_mask[i]), .base(base_i),
                .beats_total(r_beats), .burst_len(r_burst), .outstanding(r_outst),
                .busy(busy[i]), .beats_got(got[i]),
                .arid(arid[i]), .araddr(araddr[i]), .arlen(arlen[i]),
                .arsize(arsize[i]), .arburst(arburst[i]), .arlock(arlock[i]),
                .arcache(arcache[i]), .arprot(arprot[i]), .arqos(arqos[i]),
                .arvalid(arvalid[i]), .arready(arready[i]),
                .rid(rid[i]), .rdata(rdata_m[i]), .rresp(rresp[i]),
                .rlast(rlast[i]), .rvalid(rvalid_m[i]), .rready(rready[i]));
        end
    endgenerate

    // ---------------- array <-> flat port wiring ----------------
    `define WIRE_PORT(N, IDX)                                                  \
        assign m``N``_axi_arid   = arid[IDX];                                  \
        assign m``N``_axi_araddr = araddr[IDX];                                \
        assign m``N``_axi_arlen  = arlen[IDX];                                 \
        assign m``N``_axi_arsize = arsize[IDX];                                \
        assign m``N``_axi_arburst= arburst[IDX];                               \
        assign m``N``_axi_arlock = arlock[IDX];                                \
        assign m``N``_axi_arcache= arcache[IDX];                               \
        assign m``N``_axi_arprot = arprot[IDX];                                \
        assign m``N``_axi_arqos  = arqos[IDX];                                 \
        assign m``N``_axi_arvalid= arvalid[IDX];                               \
        assign arready[IDX]      = m``N``_axi_arready;                         \
        assign rid[IDX]          = m``N``_axi_rid;                             \
        assign rdata_m[IDX]      = m``N``_axi_rdata;                           \
        assign rresp[IDX]        = m``N``_axi_rresp;                           \
        assign rlast[IDX]        = m``N``_axi_rlast;                           \
        assign rvalid_m[IDX]     = m``N``_axi_rvalid;                          \
        assign m``N``_axi_rready = rready[IDX];

    `WIRE_PORT(00, 0)
    `WIRE_PORT(01, 1)
    `WIRE_PORT(02, 2)
    `WIRE_PORT(03, 3)
    `undef WIRE_PORT

    // ---------------- cycle counter ----------------
    // Counts from the start pulse until the last enabled engine drops busy, so
    // it measures the whole parallel transfer, not one port's share.
    always_ff @(posedge clk) begin
        if (!rstn) begin
            run      <= 1'b0;
            r_cycles <= '0;
        end else if (start_pulse) begin
            run      <= 1'b1;
            r_cycles <= '0;
        end else if (run) begin
            r_cycles <= r_cycles + 32'd1;
            // A couple of cycles of settle before believing `busy`: the
            // engines register it, so it is not asserted on the start cycle.
            if (r_cycles > 32'd2 && !any_busy) run <= 1'b0;
        end
    end

    // ---------------- AXI-Lite ----------------
    logic aw_hit, w_hit;
    logic [LAW-1:0] awaddr_q;
    logic [31:0]    wdata_q;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            aw_hit <= 1'b0; w_hit <= 1'b0;
            awaddr_q <= '0; wdata_q <= '0;
            start_pulse <= 1'b0;
            r_mask   <= 4'b0001;
            r_base   <= '0;
            r_stride <= 32'h0400_0000;   // 64 MB apart
            r_beats  <= 32'd65536;
            r_burst  <= 9'd256;
            r_outst  <= 8'd16;
        end else begin
            start_pulse   <= 1'b0;
            s_axi_awready <= !aw_hit && s_axi_awvalid && !s_axi_bvalid;
            s_axi_wready  <= !w_hit  && s_axi_wvalid  && !s_axi_bvalid;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_hit <= 1'b1; awaddr_q <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_hit <= 1'b1; wdata_q <= s_axi_wdata;
            end

            if (aw_hit && w_hit && !s_axi_bvalid) begin
                aw_hit <= 1'b0; w_hit <= 1'b0;
                s_axi_bvalid <= 1'b1;
                // Writes are ignored while a run is in flight, so a stray poke
                // cannot change burst length halfway through a measurement.
                if (!run) case (awaddr_q[7:0])
                    8'h00: start_pulse <= wdata_q[0];
                    8'h04: r_mask      <= wdata_q[NP-1:0];
                    8'h08: r_base[31:0]        <= wdata_q;
                    8'h0C: r_base[AW-1:32]     <= wdata_q[AW-33:0];
                    8'h10: r_stride    <= wdata_q;
                    8'h14: r_beats     <= wdata_q;
                    8'h18: r_burst     <= (wdata_q[8:0] == 9'd0) ? 9'd1 : wdata_q[8:0];
                    8'h1C: r_outst     <= (wdata_q[7:0] == 8'd0) ? 8'd1 : wdata_q[7:0];
                    default: ;
                endcase
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end
    assign s_axi_bresp = 2'b00;

    // Read channel. s_axi_rvalid must be reset -- unreset, `arready = !rvalid`
    // is X and every read wedges, which on hardware looks like "the FPGA
    // returns garbage". (systolic_zcu104 README, bug 5.)
    always_ff @(posedge clk) begin
        if (!rstn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= '0;
        end else begin
            s_axi_arready <= s_axi_arvalid && !s_axi_rvalid;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= {30'd0, run, any_busy};
                    8'h04: s_axi_rdata <= {28'd0, r_mask};
                    8'h08: s_axi_rdata <= r_base[31:0];
                    8'h0C: s_axi_rdata <= {{(64-AW){1'b0}}, r_base[AW-1:32]};
                    8'h10: s_axi_rdata <= r_stride;
                    8'h14: s_axi_rdata <= r_beats;
                    8'h18: s_axi_rdata <= {23'd0, r_burst};
                    8'h1C: s_axi_rdata <= {24'd0, r_outst};
                    8'h20: s_axi_rdata <= r_cycles;
                    8'h24: s_axi_rdata <= got[0];
                    8'h28: s_axi_rdata <= got[1];
                    8'h2C: s_axi_rdata <= got[2];
                    8'h30: s_axi_rdata <= MAGIC;
                    8'h34: s_axi_rdata <= got[3];
                    8'h38: s_axi_rdata <= 32'(NP);
                    default: s_axi_rdata <= 32'hDEAD_BEEF;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
    assign s_axi_rresp = 2'b00;
endmodule
