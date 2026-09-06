// One read-only AXI4 master that streams a contiguous region as fast as the
// slave will allow, and counts what it got.
//
// This exists to answer one question: what does PS DDR4 actually sustain
// through an AXI-HP port on this board? The whole attention design is sized
// from that number (52 B/cycle = 13.0 GB/s at 250 MHz), and it is currently an
// assumption -- ~70% of DDR4-2400 peak -- not a measurement.
//
// Why it only reads
// -----------------
// The access pattern being modelled is a full KV-cache scan: every layer reads
// its entire cache every token, perfectly sequentially, with the address known
// a full scan ahead. Writes are 416 B/token/layer against megabytes of reads.
// Measuring reads alone measures the thing that binds.
//
// Two knobs, separate ON purpose, mirroring kernel/hw/memory.py:
//   burst_len     beats per AR. Longer bursts amortise address overhead.
//   outstanding   how many ARs may be in flight. This is what hides latency.
// A design with one knob cannot tell you which of the two you are short of.
`timescale 1ns/1ps
module axi_rd_engine #(
    parameter int DW = 128,          // AXI data width, bits
    parameter int AW = 49,           // AXI address width (Zynq MPSoC FPD)
    parameter int IDW = 6
)(
    input  logic            clk,
    input  logic            rstn,

    // control (held stable while busy)
    //
    // `start` is the global pulse and `enable` selects whether this port takes
    // part. They are separate so that a disabled port still clears its beat
    // counter: leaving a stale count in a port that sat out the run makes the
    // software read back the previous measurement's bytes and compute a
    // bandwidth that never happened.
    input  logic            start,          // one-cycle pulse, all ports
    input  logic            enable,         // this port participates
    input  logic [AW-1:0]   base,           // 4KB-aligned
    input  logic [31:0]     beats_total,    // beats to read, multiple of burst_len
    input  logic [8:0]      burst_len,      // 1..256 beats per AR
    input  logic [7:0]      outstanding,    // max ARs in flight, >=1
    output logic            busy,

    // results
    output logic [31:0]     beats_got,

    // AXI4 read address
    output logic [IDW-1:0]  arid,
    output logic [AW-1:0]   araddr,
    output logic [7:0]      arlen,
    output logic [2:0]      arsize,
    output logic [1:0]      arburst,
    output logic            arlock,
    output logic [3:0]      arcache,
    output logic [2:0]      arprot,
    output logic [3:0]      arqos,
    output logic            arvalid,
    input  logic            arready,

    // AXI4 read data
    input  logic [IDW-1:0]  rid,
    input  logic [DW-1:0]   rdata,
    input  logic [1:0]      rresp,
    input  logic            rlast,
    input  logic            rvalid,
    output logic            rready
);
    localparam int BYTES = DW/8;
    localparam int SIZE  = $clog2(BYTES);

    // Constant AXI attributes. ARCACHE 4'b1111 = write-back read-allocate:
    // required for the transaction to be cacheable/bufferable at the HP port,
    // which is worth real bandwidth. ARPROT 3'b000 = unprivileged, secure, data.
    assign arsize  = SIZE[2:0];
    assign arburst = 2'b01;          // INCR
    assign arlock  = 1'b0;
    assign arcache = 4'b1111;
    assign arprot  = 3'b000;
    assign arqos   = 4'b0000;
    assign arid    = '0;
    assign rready  = 1'b1;           // never back-pressure; we are measuring the slave

    logic [AW-1:0]  next_addr;
    logic [31:0]    beats_asked;
    logic [7:0]     inflight;
    logic [31:0]    beats_target;
    logic [8:0]     blen;

    wire ar_fire = arvalid & arready;
    wire r_fire  = rvalid  & rready;

    wire more_to_ask = beats_asked < beats_target;
    wire has_credit  = inflight < outstanding;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            busy         <= 1'b0;
            arvalid      <= 1'b0;
            araddr       <= '0;
            arlen        <= '0;
            next_addr    <= '0;
            beats_asked  <= '0;
            beats_got    <= '0;
            inflight     <= '0;
            beats_target <= '0;
            blen         <= 9'd1;
        end else if (start) begin
            busy         <= enable;
            next_addr    <= base;
            beats_asked  <= '0;
            beats_got    <= '0;
            inflight     <= '0;
            beats_target <= beats_total;
            blen         <= burst_len;
            arvalid      <= 1'b0;
        end else begin
            // -------- address channel --------
            if (ar_fire) begin
                arvalid     <= 1'b0;
                next_addr   <= next_addr + AW'(blen) * AW'(BYTES);
                beats_asked <= beats_asked + 32'(blen);
            end
            if (busy && !arvalid && more_to_ask && has_credit) begin
                arvalid <= 1'b1;
                araddr  <= next_addr;
                arlen   <= 8'(blen - 9'd1);
            end

            // -------- in-flight accounting --------
            // A burst leaves flight on its RLAST, not on each beat.
            case ({ar_fire, rvalid & rready & rlast})
                2'b10:   inflight <= inflight + 8'd1;
                2'b01:   inflight <= inflight - 8'd1;
                default: ;                       // 2'b11 nets to zero, 2'b00 idle
            endcase

            // -------- data channel --------
            if (r_fire) beats_got <= beats_got + 32'd1;

            if (busy && !more_to_ask && (beats_got + 32'(r_fire)) >= beats_target)
                busy <= 1'b0;
        end
    end
endmodule
