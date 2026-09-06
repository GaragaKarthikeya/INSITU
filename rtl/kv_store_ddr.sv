// The KV cache, and the only cache in this design.
//
// Four AXI-HP read masters, one per plane of the transposed cache, streaming
// one 52-byte row per cycle into `score_lane`.  Plus a write path for the eight
// rows the arriving token adds, which must never stall the read stream.
//
// The cache is transposed, and that is a correction to `plan.MD`
// --------------------------------------------------------------
// The plan specified packed 52-byte rows at a flat stride.  That cannot feed
// four ports.  A row is 52 B and one 128-bit HP port carries 16 B per cycle, so
// four ports must run at once and each needs its own sequential stream -- and a
// flat row layout has exactly one stream in it.  Splitting the token range into
// quarters does not rescue it: the scan is causal, so every token of the first
// quarter comes from port 0 alone, at 16 B/cycle against the 52 B/cycle the
// score lane eats, and the only way port 0 keeps up is if its whole quarter is
// already on chip.  Four of those is the entire cache resident in BRAM, which
// is the thing this project exists not to do.
//
// So the cache is stored as planes: row bytes 0..16 for every token, then
// 16..32, then 32..48, then 48..52.  Each port reads one plane sequentially and
// the four together deliver one row per cycle -- three at a beat per token and
// the fourth at a beat per four tokens.  `hw/ddr_layout.py` owns the address
// map and `tests/test_ddr_layout.py` asserts the round trip is byte-exact
// against `KVQuantizer.pack`.  The row format is untouched; only its
// arrangement in memory changed.
//
// BURST 64, outstanding 2
// -----------------------
// Measured, in step 1, on the board.  Deeper is worse: 14,708 MB/s at two
// outstanding against 11,987 at four, with three times the variance.  See
// `kv_plane_rd.sv`.
//
// The write path is separate and non-blocking
// -------------------------------------------
// 416 B per token per layer against megabytes of reads, so it is negligible
// traffic -- but it shares the DRAM, and a write that blocked the read stream
// would stall the score pipe for the sake of 0.02% of the bytes.  It runs on
// its own master with its own buffer and its own address generator, and the
// read masters never see it.
`timescale 1ns/1ps
module kv_store_ddr #(
    parameter int DW        = 128,
    parameter int AW        = 49,
    parameter int IDW       = 6,
    parameter int N_PORTS   = 4,
    parameter int ROW_BYTES = 52,
    parameter int BURST     = 64,
    parameter int OUTST     = 2,
    parameter int DEPTH     = 256,
    // Plane widths, LSB-first: plane 0 in the low byte.  A format artifact of
    // `hw/ddr_layout.py`, like every other table in this datapath.
    parameter logic [N_PORTS*8-1:0] PLANE_W = {8'd4, 8'd16, 8'd16, 8'd16}
) (
    input  logic clk,
    input  logic rstn,

    // -- scan ------------------------------------------------------------
    input  logic start,
    // Packed, plane 0 in the low bits -- unpacked array ports are SV-2012 and
    // not portable across the simulators this repo runs on.
    input  logic [N_PORTS*AW-1:0] head_base,         // this head's plane bases
    input  logic [31:0] n_tokens,
    output logic busy,

    output logic row_valid,
    input  logic row_ready,
    output logic [ROW_BYTES*8-1:0] row_data,
    // Cycles the consumer asked for a row and no plane had one.  The number
    // step 1 exists to make zero; reported, never assumed.
    output logic [31:0] starve_cycles,

    // -- AXI4 read, one master per plane ---------------------------------
    output logic [N_PORTS*IDW-1:0] arid,
    output logic [N_PORTS*AW-1:0] araddr,
    output logic [N_PORTS*8-1:0] arlen,
    output logic [N_PORTS*3-1:0] arsize,
    output logic [N_PORTS*2-1:0] arburst,
    output logic [N_PORTS-1:0] arlock,
    output logic [N_PORTS*4-1:0] arcache,
    output logic [N_PORTS*3-1:0] arprot,
    output logic [N_PORTS*4-1:0] arqos,
    output logic [N_PORTS-1:0] arvalid,
    input  logic [N_PORTS-1:0] arready,
    input  logic [N_PORTS*IDW-1:0] rid,
    input  logic [N_PORTS*DW-1:0] rdata,
    input  logic [N_PORTS*2-1:0] rresp,
    input  logic [N_PORTS-1:0] rlast,
    input  logic [N_PORTS-1:0] rvalid,
    output logic [N_PORTS-1:0] rready
);
    // Where plane `p`'s slice sits in the row: the sum of the widths before
    // it. This is `pack`'s own byte order, which is what makes the assembled
    // row byte-exact against `KVQuantizer.pack` rather than merely the right
    // length.
    function automatic int plane_off(input int upto);
        plane_off = 0;
        for (int i = 0; i < upto; i++) plane_off += PLANE_W[i*8 +: 8];
    endfunction

    logic [N_PORTS-1:0] p_valid, p_ready, p_busy, p_starved;

    // A row exists only when every plane has its slice: the planes run at
    // different rates -- a beat per token for the codes, a beat per four for
    // the norms -- so they are joined here, not assumed to be in step.
    assign row_valid = &p_valid;
    wire row_fire = row_valid && row_ready;
    assign busy = |p_busy;

    generate
        genvar p;
        for (p = 0; p < N_PORTS; p = p + 1) begin : plane
            localparam int W = PLANE_W[p*8 +: 8];
            wire [W*8-1:0] slice;
            // Plane p carries row bytes [sum of earlier widths ...], so the
            // concatenation below is the row in `pack`'s own byte order.
            localparam int OFF = plane_off(p);
            assign row_data[OFF*8 +: W*8] = slice;
            assign p_ready[p] = row_fire;

            kv_plane_rd #(.DW(DW), .AW(AW), .IDW(IDW), .WIDTH(W),
                          .BURST(BURST), .OUTST(OUTST), .DEPTH(DEPTH))
            rd (
                .clk, .rstn, .start, .base(head_base[p*AW +: AW]), .n_tokens,
                .busy(p_busy[p]),
                .tok_valid(p_valid[p]), .tok_ready(p_ready[p]),
                .tok_data(slice), .starved(p_starved[p]),
                .arid(arid[p*IDW +: IDW]), .araddr(araddr[p*AW +: AW]),
                .arlen(arlen[p*8 +: 8]), .arsize(arsize[p*3 +: 3]),
                .arburst(arburst[p*2 +: 2]), .arlock(arlock[p]),
                .arcache(arcache[p*4 +: 4]), .arprot(arprot[p*3 +: 3]),
                .arqos(arqos[p*4 +: 4]),
                .arvalid(arvalid[p]), .arready(arready[p]),
                .rid(rid[p*IDW +: IDW]), .rdata(rdata[p*DW +: DW]),
                .rresp(rresp[p*2 +: 2]),
                .rlast(rlast[p]), .rvalid(rvalid[p]), .rready(rready[p]));
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rstn || start) starve_cycles <= '0;
        else if (busy && row_ready && !row_valid)
            starve_cycles <= starve_cycles + 1'b1;
    end
endmodule
