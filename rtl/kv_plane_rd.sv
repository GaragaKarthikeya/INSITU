// One AXI-HP read master streaming one plane of the KV cache.
//
// A plane is a byte range of the cache row -- 16, 16, 16 and 4 bytes at
// KEY_BITS=4 -- held contiguously for every token, so this reads a flat
// sequential region and hands out `WIDTH` bytes per token in order.  Four of
// these, one per port, reconstruct a 52-byte row per cycle without any of them
// ever reading out of order.  See `hw/ddr_layout.py` for why the cache is
// transposed into planes at all.
//
// BURST 64, OUTSTANDING 2, AND NOT DEEPER
// ---------------------------------------
// Measured on the board in step 1: at four ports and burst 64, two outstanding
// reads give 14,708 MB/s and four give 11,987 -- a deeper queue costs 19% of
// the bandwidth and triples the variance.  The mechanism is row-buffer
// locality; a deep queue interleaves requests from four streams across four
// DRAM rows and thrashes activate/precharge.  The instinct to build a deep
// prefetch queue is actively wrong here, so the depth is a parameter with a
// measured default and the testbench asserts it is what shipped.
//
// THE BUFFER IS SIZED BY LATENCY, NOT BY CONTEXT
// ----------------------------------------------
// This is the whole point of the transpose.  The consumer eats WIDTH bytes per
// token and this port delivers 16 B per beat, so the FIFO only has to cover the
// round trip of two outstanding bursts -- hundreds of bytes -- rather than a
// quarter of the cache.
`timescale 1ns/1ps
module kv_plane_rd #(
    parameter int DW    = 128,       // AXI data width
    parameter int AW    = 49,        // Zynq MPSoC FPD address width
    parameter int IDW   = 6,
    parameter int WIDTH = 16,        // plane bytes per token
    parameter int BURST = 64,        // beats per AR
    parameter int OUTST = 2,         // ARs in flight.  MEASURED.  Do not raise.
    parameter int DEPTH = 256        // beat FIFO, >= BURST*OUTST
) (
    input  logic clk,
    input  logic rstn,

    // -- scan control ---------------------------------------------------
    input  logic start,                    // begin a scan
    input  logic [AW-1:0] base,            // this plane's base, 4 KB aligned
    input  logic [31:0] n_tokens,          // tokens in the scan
    output logic busy,

    // -- one token's slice of the row -----------------------------------
    output logic tok_valid,
    input  logic tok_ready,
    output logic [WIDTH*8-1:0] tok_data,
    output logic starved,                  // consumer asked and we had nothing

    // -- AXI4 read ------------------------------------------------------
    output logic [IDW-1:0] arid,
    output logic [AW-1:0]  araddr,
    output logic [7:0]     arlen,
    output logic [2:0]     arsize,
    output logic [1:0]     arburst,
    output logic           arlock,
    output logic [3:0]     arcache,
    output logic [2:0]     arprot,
    output logic [3:0]     arqos,
    output logic           arvalid,
    input  logic           arready,
    input  logic [IDW-1:0] rid,
    input  logic [DW-1:0]  rdata,
    input  logic [1:0]     rresp,
    input  logic           rlast,
    input  logic           rvalid,
    output logic           rready
);
    localparam int BEAT  = DW/8;                       // 16 B
    localparam int SIZE  = $clog2(BEAT);
    localparam int AD    = $clog2(DEPTH);

    generate
        if (DEPTH < BURST * OUTST)
            initial $fatal(1, "FIFO must hold every beat that can be in flight");
        // `hw/ddr_layout.py` splits the row at its field boundaries before it
        // splits at beats, so every plane is 16, 8 or 4 bytes and all three
        // divide the beat. A plane wider than a beat, or one that straddled
        // beats, would need a byte shift register here; the layout is what
        // makes that unreachable, and this is where the assumption is stated.
        if (WIDTH > BEAT || BEAT % WIDTH != 0)
            initial $fatal(1, "a plane must divide the beat; split at fields");
    endgenerate

    assign arsize  = SIZE[2:0];
    assign arburst = 2'b01;                            // INCR
    assign arlock  = 1'b0;
    assign arcache = 4'b1111;                          // cacheable, bufferable
    assign arprot  = 3'b000;
    assign arqos   = 4'b0000;
    assign arid    = '0;

    // -- beat FIFO ---------------------------------------------------------
    logic [DW-1:0] fifo [0:DEPTH-1];
    logic [AD:0] wptr, rptr;
    wire [AD:0] fill = wptr - rptr;
    wire f_full  = fill >= DEPTH[AD:0];
    wire f_empty = fill == 0;

    assign rready = !f_full;

    // -- address generation ------------------------------------------------
    // The scan is `n_tokens * WIDTH` bytes.  The last burst is short when that
    // is not a multiple of BURST beats, which it usually is not -- rounding it
    // up would read past the plane into the next one.
    logic [31:0] beats_total, beats_asked;
    logic [AW-1:0] next_addr;
    logic [7:0] inflight;
    wire ar_fire = arvalid && arready;
    wire more_to_ask = beats_asked < beats_total;
    wire [31:0] beats_left = beats_total - beats_asked;
    wire [8:0] this_len = (beats_left < BURST) ? beats_left[8:0] : 9'(BURST);

    // -- token extraction --------------------------------------------------
    // One beat serves BEAT/WIDTH tokens: 1 for the code planes, 4 for the
    // norms. `sub` selects the slice, so nothing here shifts bytes.
    localparam int PER_BEAT = BEAT/WIDTH;

    logic [$clog2(PER_BEAT+1)-1:0] sub;
    logic [31:0] toks_out;

    wire have_beat = !f_empty;
    wire [DW-1:0] head = fifo[rptr[AD-1:0]];

    assign tok_data  = head[sub*WIDTH*8 +: WIDTH*8];
    assign tok_valid = busy && have_beat;

    wire tok_fire = tok_valid && tok_ready;
    assign starved = busy && tok_ready && !tok_valid;

    always_ff @(posedge clk) begin
        if (!rstn || start) begin
            wptr <= '0; rptr <= '0; sub <= '0; toks_out <= '0;
            arvalid <= 1'b0; inflight <= '0; beats_asked <= '0;
            next_addr <= start ? base : '0;
            beats_total <= start
                ? ((n_tokens * WIDTH + BEAT - 1) / BEAT) : '0;
            busy <= start && (n_tokens != 0);
        end else begin
            // -- address channel ------------------------------------------
            if (ar_fire) begin
                arvalid     <= 1'b0;
                next_addr   <= next_addr + AW'(this_len) * AW'(BEAT);
                beats_asked <= beats_asked + 32'(this_len);
            end
            // Only ask for what the FIFO can still hold: an AXI slave may not
            // be back-pressured indefinitely on R, and `rready` low with two
            // bursts outstanding is exactly the stall the depth-2 measurement
            // was taken under.
            if (busy && !arvalid && more_to_ask && inflight < OUTST[7:0]
                && (fill + (AD+1)'(inflight) * (AD+1)'(BURST)
                    + (AD+1)'(BURST) <= DEPTH[AD:0]))
                begin
                    arvalid <= 1'b1;
                    araddr  <= next_addr;
                    arlen   <= 8'(this_len - 9'd1);
                end

            case ({ar_fire, rvalid && rready && rlast})
                2'b10:   inflight <= inflight + 8'd1;
                2'b01:   inflight <= inflight - 8'd1;
                default: ;
            endcase

            // -- data in ----------------------------------------------------
            if (rvalid && rready) begin
                fifo[wptr[AD-1:0]] <= rdata;
                wptr <= wptr + 1'b1;
            end

            // -- tokens out -------------------------------------------------
            if (tok_fire) begin
                toks_out <= toks_out + 1'b1;
                if (sub == PER_BEAT-1) begin
                    sub <= '0;
                    rptr <= rptr + 1'b1;
                end else
                    sub <= sub + 1'b1;
                if (toks_out + 1 == n_tokens) busy <= 1'b0;
            end
        end
    end
endmodule
