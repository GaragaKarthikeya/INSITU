// AXI-Lite shim around `attn_top`: one decode step, driven over JTAG.
//
// Why buffers and not a FIFO window
// ---------------------------------
// The obvious shim is two magic addresses -- write here to push a beat, read
// there to pop one. It is smaller and it is the wrong choice for a JTAG bring-
// up. A `create_hw_axi_txn` that is retried, reordered or simply typed twice
// then silently shifts the whole stream by one word, and the failure looks
// like a numeric bug in the datapath rather than like a lost transaction. So
// ingress and egress are addressed memories: every write lands where its
// address says, every read is repeatable, and the host can read back what it
// wrote before starting anything.
//
// 9,216 B in (8 groups x 6 vectors x 3 beats of 512 b) and 6,144 B out (32
// heads x 64 channels x 24 b). `attn_top` reports zero BRAM, so the ~5 block
// RAMs these cost are free.
//
// The stream is played, not pushed
// --------------------------------
// `start` plays the whole input buffer into `attn_top`'s 512-bit stream and
// collects its egress back into the output buffer, 32 bits per cycle each way.
// 2,304 cycles to feed and 1,536 to collect, against a scan that is 520 rows
// at ctx 64 and 8 x T at any real context -- so the shim is never what the
// measurement is measuring, and at ctx 8k it is 0.7% of the step.
//
// ~1 ms per JTAG round trip, so this block makes NO throughput claim. It
// exists to answer one question: does the placed and routed design produce the
// same 2,048 numbers the numpy kernel does. Step 13's DMA path is where a
// device-side time means anything.
`timescale 1ns/1ps
module attn_ctrl #(
    parameter int BEAT_BITS = 512,
    parameter int IN_BEATS  = 144,      // 8 groups x 6 vectors x 3 beats
    parameter int OUT_BEATS = 96,       // 32 heads x 64 channels x 24 b
    parameter int LAW       = 17,       // AXI-Lite address width
    parameter int DW        = 128,
    parameter int AW        = 49,
    parameter int IDW       = 6,
    parameter int N_PORTS   = 4,
    // A build stamp the host reads back first.  A bitstream that is not the
    // one the host thinks it is looks exactly like a datapath bug.
    parameter logic [31:0] MAGIC = 32'hA77E_0001
) (
    input  logic clk,
    input  logic rstn,

    // ---------------- the DMA stream (step 13's measured bill) ----------------
    //
    // Step 13 measured 366 us of every 433 us decode step as the A53 walking
    // 15 KB through the 32-bit AXI-Lite window one store at a time -- 85% of
    // the step, against 67 us of block. That is what these carry instead.
    //
    // The buffers are not removed. They are how step 12 and step 13 ran, they
    // are readable back, and a bring-up that cannot re-read what it loaded
    // cannot interpret what it gets out. `MODE` picks which one feeds the
    // core, so the slow path stays available as the thing a failing fast path
    // is diffed against.
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,
    input  logic [BEAT_BITS-1:0]  s_axis_tdata,
    input  logic                  s_axis_tlast,
    output logic                  m_axis_tvalid,
    input  logic                  m_axis_tready,
    output logic [BEAT_BITS-1:0]  m_axis_tdata,
    output logic                  m_axis_tlast,

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
    //
    // Flat and individually named, because that is how Vivado infers an
    // interface on a module reference -- a packed `[N_PORTS*AW-1:0]` port is
    // just a wide bus to the block designer, and the four masters would never
    // appear in the address map. The arrays live inside.
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
    input  logic           m03_axi_rvalid, output logic          m03_axi_rready,

    // ---------------- the write master ----------------
    output logic [IDW-1:0]  m_wr_axi_awid,
    output logic [AW-1:0]   m_wr_axi_awaddr,
    output logic [7:0]      m_wr_axi_awlen,
    output logic [2:0]      m_wr_axi_awsize,
    output logic [1:0]      m_wr_axi_awburst,
    output logic [3:0]      m_wr_axi_awcache,
    output logic [2:0]      m_wr_axi_awprot,
    output logic            m_wr_axi_awvalid,
    input  logic            m_wr_axi_awready,
    output logic [DW-1:0]   m_wr_axi_wdata,
    output logic [DW/8-1:0] m_wr_axi_wstrb,
    output logic            m_wr_axi_wlast,
    output logic            m_wr_axi_wvalid,
    input  logic            m_wr_axi_wready,
    input  logic [1:0]      m_wr_axi_bresp,
    input  logic            m_wr_axi_bvalid,
    output logic            m_wr_axi_bready
);
    // ---------------- array <-> flat port wiring ----------------
    logic [N_PORTS*IDW-1:0] arid, rid;
    logic [N_PORTS*AW-1:0]  araddr;
    logic [N_PORTS*8-1:0]   arlen;
    logic [N_PORTS*3-1:0]   arsize, arprot;
    logic [N_PORTS*2-1:0]   arburst, rresp;
    logic [N_PORTS*4-1:0]   arcache, arqos;
    logic [N_PORTS-1:0]     arlock, arvalid, arready, rlast, rvalid, rready;
    logic [N_PORTS*DW-1:0]  rdata;

    logic [IDW-1:0]  awid;
    logic [AW-1:0]   awaddr;
    logic [7:0]      awlen;
    logic [2:0]      awsize, awprot;
    logic [1:0]      awburst, bresp;
    logic [3:0]      awcache;
    logic            awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [DW-1:0]   wdata;
    logic [DW/8-1:0] wstrb;

    `define WIRE_PORT(N, IDX)                                                  \
        assign m``N``_axi_arid   = arid[IDX*IDW +: IDW];                       \
        assign m``N``_axi_araddr = araddr[IDX*AW +: AW];                       \
        assign m``N``_axi_arlen  = arlen[IDX*8 +: 8];                          \
        assign m``N``_axi_arsize = arsize[IDX*3 +: 3];                         \
        assign m``N``_axi_arburst= arburst[IDX*2 +: 2];                        \
        assign m``N``_axi_arlock = arlock[IDX];                                \
        assign m``N``_axi_arcache= arcache[IDX*4 +: 4];                        \
        assign m``N``_axi_arprot = arprot[IDX*3 +: 3];                         \
        assign m``N``_axi_arqos  = arqos[IDX*4 +: 4];                          \
        assign m``N``_axi_arvalid= arvalid[IDX];                               \
        assign arready[IDX]      = m``N``_axi_arready;                         \
        assign rid[IDX*IDW +: IDW]   = m``N``_axi_rid;                         \
        assign rdata[IDX*DW +: DW]   = m``N``_axi_rdata;                       \
        assign rresp[IDX*2 +: 2]     = m``N``_axi_rresp;                       \
        assign rlast[IDX]            = m``N``_axi_rlast;                       \
        assign rvalid[IDX]           = m``N``_axi_rvalid;                      \
        assign m``N``_axi_rready = rready[IDX];

    `WIRE_PORT(00, 0)
    `WIRE_PORT(01, 1)
    `WIRE_PORT(02, 2)
    `WIRE_PORT(03, 3)
    `undef WIRE_PORT

    assign m_wr_axi_awid    = awid;
    assign m_wr_axi_awaddr  = awaddr;
    assign m_wr_axi_awlen   = awlen;
    assign m_wr_axi_awsize  = awsize;
    assign m_wr_axi_awburst = awburst;
    assign m_wr_axi_awcache = awcache;
    assign m_wr_axi_awprot  = awprot;
    assign m_wr_axi_awvalid = awvalid;
    assign awready          = m_wr_axi_awready;
    assign m_wr_axi_wdata   = wdata;
    assign m_wr_axi_wstrb   = wstrb;
    assign m_wr_axi_wlast   = wlast;
    assign m_wr_axi_wvalid  = wvalid;
    assign wready           = m_wr_axi_wready;
    assign bresp            = m_wr_axi_bresp;
    assign bvalid           = m_wr_axi_bvalid;
    assign m_wr_axi_bready  = bready;

    localparam int WPB     = BEAT_BITS / 32;          // 16 words per beat
    localparam int IN_W    = IN_BEATS  * WPB;         // 2,304 words
    localparam int OUT_W   = OUT_BEATS * WPB;         // 1,536 words
    localparam int IN_AW   = $clog2(IN_W);
    localparam int OUT_AW  = $clog2(OUT_W);

    // Register file at 0x0000, input buffer at 0x2000, output at 0x8000.  The
    // windows are far apart so a mistyped address lands in a hole and reads
    // 0xDEADBEEF rather than in the neighbouring buffer.
    localparam logic [LAW-1:0] IN_BASE  = 17'h0_2000;
    localparam logic [LAW-1:0] OUT_BASE = 17'h0_8000;

    logic [31:0] inbuf  [0:IN_W-1];
    logic [31:0] outbuf [0:OUT_W-1];

    // ---------------- the block ----------------
    logic start, busy, done;
    logic [AW-1:0] cache_base;
    logic [31:0] head_stride, plane_span, n_tokens;
    logic s_valid, s_ready, m_valid, m_ready, m_last;
    logic feed_valid, coll_ready;
    logic stream_mode;
    logic [BEAT_BITS-1:0] s_data, m_data;
    logic [31:0] starve_cycles, clip_count, overflow_count;
    logic [31:0] scan_cycles, busy_cycles;
    logic range_error, norm_saturated;

    attn_top #(.BEAT_BITS(BEAT_BITS), .DW(DW), .AW(AW), .IDW(IDW),
               .N_PORTS(N_PORTS))
    core (.clk, .rstn, .start, .cache_base, .head_stride, .plane_span, .n_tokens,
          .busy, .done, .s_valid, .s_ready, .s_data,
          .m_valid, .m_ready, .m_data, .m_last,
          .arid, .araddr, .arlen, .arsize, .arburst, .arlock, .arcache, .arprot,
          .arqos, .arvalid, .arready, .rid, .rdata, .rresp, .rlast, .rvalid, .rready,
          .awid, .awaddr, .awlen, .awsize, .awburst, .awcache, .awprot, .awvalid,
          .awready, .wdata, .wstrb, .wlast, .wvalid, .wready, .bresp, .bvalid, .bready,
          .scan_cycles, .busy_cycles,
          .starve_cycles, .clip_count, .overflow_count, .range_error, .norm_saturated);

    // ---------------- the feeder ----------------
    //
    // One word per cycle out of `inbuf` into a 512-bit shift register, low word
    // first -- the order `hw/vectors.py::beats` writes and the host loads.
    logic [IN_AW:0] feed_w;
    logic [$clog2(WPB+1)-1:0] feed_sub;
    logic feeding;
    logic [BEAT_BITS-1:0] beat_sh;

    // Mode 0 plays the input buffer into the core, as steps 12 and 13 did.
    // Mode 1 hands the core straight to the DMA. One mux, and everything below
    // it -- the core, the four read masters, the write master, the register
    // map -- is untouched, which is the same promise the shim made when the
    // JTAG master became the PS.
    assign s_valid = stream_mode ? s_axis_tvalid : feed_valid;
    assign s_data  = stream_mode ? s_axis_tdata  : beat_sh;
    // Gated ON `busy`, and that IS de-facto A protocol rule made into logic.
    //
    // `attn_ingress` clears its beat counter on `start`. A beat accepted
    // Before the step begins is therefore swallowed and then thrown away with
    // that reset, and the stream ends one vector short -- which surfaces as
    // the last group hanging in S_RECV forever, with every beat apparently
    // delivered. The buffered path cannot hit it because its feeder is armed
    // by `start` itself.
    //
    // Requiring software to start the DMA after the block would work and would
    // be a rule nothing enforces. Holding `tready` low until the core is
    // running makes the ordering unnecessary: a DMA armed early simply waits.
    assign s_axis_tready = stream_mode && s_ready && busy;

    assign m_axis_tvalid = stream_mode && m_valid;
    assign m_axis_tdata  = m_data;
    assign m_axis_tlast  = m_last;
    assign m_ready = stream_mode ? m_axis_tready : coll_ready;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            feeding <= 1'b0; feed_w <= '0; feed_sub <= '0; feed_valid <= 1'b0;
        end else begin
            if (start && !stream_mode) begin
                feeding <= 1'b1; feed_w <= '0; feed_sub <= '0; feed_valid <= 1'b0;
            end else if (feeding) begin
                if (feed_valid) begin
                    if (s_ready) begin
                        feed_valid <= 1'b0;
                        // Every word has been shifted in and the last beat has
                        // been taken: the stream is complete.
                        if (feed_w == (IN_AW+1)'(IN_W)) feeding <= 1'b0;
                    end
                end else begin
                    beat_sh <= {inbuf[feed_w[IN_AW-1:0]], beat_sh[BEAT_BITS-1:32]};
                    feed_w  <= feed_w + 1'b1;
                    if (feed_sub == WPB-1) begin
                        feed_sub <= '0;
                        feed_valid <= 1'b1;
                    end else
                        feed_sub <= feed_sub + 1'b1;
                end
            end
        end
    end

    // ---------------- the collector ----------------
    logic [OUT_AW:0] coll_w;
    logic [$clog2(WPB+1)-1:0] coll_sub;
    logic [BEAT_BITS-1:0] coll_sh;
    logic collecting;

    assign coll_ready = !collecting;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            collecting <= 1'b0; coll_w <= '0; coll_sub <= '0;
        end else begin
            if (start) begin
                collecting <= 1'b0; coll_w <= '0; coll_sub <= '0;
            end else if (collecting) begin
                outbuf[coll_w[OUT_AW-1:0]] <= coll_sh[31:0];
                coll_sh <= {32'd0, coll_sh[BEAT_BITS-1:32]};
                coll_w  <= coll_w + 1'b1;
                if (coll_sub == WPB-1) begin
                    coll_sub   <= '0;
                    collecting <= 1'b0;
                end else
                    coll_sub <= coll_sub + 1'b1;
            end else if (m_valid && !stream_mode) begin
                coll_sh    <= m_data;
                coll_sub   <= '0;
                collecting <= 1'b1;
            end
        end
    end

    // ---------------- run state ----------------
    //
    // `done` is a pulse and JTAG polls at millisecond intervals, so it is
    // Latched. A status bit the host can miss is a status bit that reports a
    // hung run.
    logic run, done_l;
    always_ff @(posedge clk) begin
        if (!rstn) begin
            run <= 1'b0; done_l <= 1'b0;
        end else begin
            if (start) begin run <= 1'b1; done_l <= 1'b0; end
            else if (done) begin run <= 1'b0; done_l <= 1'b1; end
        end
    end

    // ---------------- AXI-Lite ----------------
    logic aw_hit, w_hit;
    logic [LAW-1:0] awaddr_q;
    logic [31:0]    wdata_q;

    wire aw_in  = (awaddr_q >= IN_BASE)  && (awaddr_q < IN_BASE  + LAW'(IN_W*4));
    wire [IN_AW-1:0] aw_in_idx = (awaddr_q - IN_BASE) >> 2;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            s_axi_awready <= 1'b0; s_axi_wready <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            aw_hit <= 1'b0; w_hit <= 1'b0;
            awaddr_q <= '0; wdata_q <= '0;
            start <= 1'b0;
            cache_base  <= '0;
            head_stride <= 32'd16384;
            plane_span  <= 32'd4096;
            n_tokens    <= 32'd1;
            stream_mode <= 1'b0;      // the buffers, until a host asks for the DMA
        end else begin
            start <= 1'b0;
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
                // Writes are ignored while a step is in flight: a stray poke
                // must not move the cache base under a running scan.
                if (!run) begin
                    if (aw_in)
                        inbuf[aw_in_idx] <= wdata_q;
                    else case (awaddr_q[7:0])
                        8'h00: start       <= wdata_q[0];
                        8'h08: n_tokens    <= wdata_q;
                        8'h0C: cache_base[31:0]    <= wdata_q;
                        8'h10: cache_base[AW-1:32] <= wdata_q[AW-33:0];
                        8'h14: head_stride <= wdata_q;
                        8'h18: plane_span  <= wdata_q;
                        8'h38: stream_mode <= wdata_q[0];
                        default: ;
                    endcase
                end
            end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
        end
    end
    assign s_axi_bresp = 2'b00;

    // Read channel.  `s_axi_rvalid` must be reset: unreset, `arready = !rvalid`
    // is X and every read wedges, which on hardware looks like "the FPGA
    // returns garbage". (`systolic_zcu104` README, bug 5.)
    wire ar_in  = (s_axi_araddr >= IN_BASE)  && (s_axi_araddr < IN_BASE  + LAW'(IN_W*4));
    wire ar_out = (s_axi_araddr >= OUT_BASE) && (s_axi_araddr < OUT_BASE + LAW'(OUT_W*4));
    wire [IN_AW-1:0]  ar_in_idx  = (s_axi_araddr - IN_BASE)  >> 2;
    wire [OUT_AW-1:0] ar_out_idx = (s_axi_araddr - OUT_BASE) >> 2;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= '0;
        end else begin
            s_axi_arready <= s_axi_arvalid && !s_axi_rvalid;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                if (ar_out)      s_axi_rdata <= outbuf[ar_out_idx];
                else if (ar_in)  s_axi_rdata <= inbuf[ar_in_idx];
                else case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= {28'd0, norm_saturated, range_error,
                                           done_l, busy};
                    8'h04: s_axi_rdata <= MAGIC;
                    8'h08: s_axi_rdata <= n_tokens;
                    8'h0C: s_axi_rdata <= cache_base[31:0];
                    8'h10: s_axi_rdata <= {{(64-AW){1'b0}}, cache_base[AW-1:32]};
                    8'h14: s_axi_rdata <= head_stride;
                    8'h18: s_axi_rdata <= plane_span;
                    8'h1C: s_axi_rdata <= starve_cycles;
                    8'h20: s_axi_rdata <= clip_count;
                    8'h24: s_axi_rdata <= overflow_count;
                    8'h28: s_axi_rdata <= 32'(IN_W);
                    8'h2C: s_axi_rdata <= 32'(OUT_W);
                    // The step's own clock. `busy_cycles` is start-to-done and
                    // `scan_cycles` is the part of it that reads DDR, which is
                    // the only one a bandwidth number may be divided by.
                    8'h30: s_axi_rdata <= scan_cycles;
                    8'h34: s_axi_rdata <= busy_cycles;
                    8'h38: s_axi_rdata <= {31'd0, stream_mode};
                    default: s_axi_rdata <= 32'hDEAD_BEEF;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
    assign s_axi_rresp = 2'b00;
endmodule
