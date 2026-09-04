// The write path: one arriving token's row, scattered into its planes.
//
// SEPARATE MASTER, ON PURPOSE
// ---------------------------
// 416 B per token per layer against megabytes of reads -- 0.02% of the traffic.
// It is still on its own AXI master with its own address generator, because
// sharing one with the read stream would let a write stall the score pipe for
// the sake of a rounding error's worth of bytes.  `plan.MD` requires it and the
// bandwidth is not the reason.
//
// ONE ROW IS N_PORTS WRITES
// -------------------------
// The cache is transposed, so a 52-byte row does not live anywhere contiguous:
// it is 16 B in plane 0 at `token*16`, 16 B in plane 1, 16 B in plane 2 and 4 B
// in plane 3 at `token*4`.  So the row goes out as one single-beat write per
// plane, each naturally aligned to its own width -- a narrow AXI transfer for
// the norms, at AWSIZE=2, rather than a read-modify-write of a 16-byte beat.
//
// WSTRB AND THE BYTE LANE
// -----------------------
// A narrow write still drives the full data bus, and the bytes must sit in the
// lanes the address selects: byte `addr % 16` upward.  Getting that wrong puts
// the norms four bytes away from where the reader looks for them, which decodes
// as a plausible norm belonging to a different token.
`timescale 1ns/1ps
module kv_write #(
    parameter int DW        = 128,
    parameter int AW        = 49,
    parameter int IDW       = 6,
    parameter int N_PORTS   = 4,
    parameter int ROW_BYTES = 52,
    parameter logic [N_PORTS*8-1:0] PLANE_W = {8'd4, 8'd16, 8'd16, 8'd16}
) (
    input  logic clk,
    input  logic rstn,

    input  logic in_valid,
    output logic in_ready,
    input  logic [N_PORTS*AW-1:0] head_base,       // the target head's planes
    input  logic [31:0] token,
    input  logic [ROW_BYTES*8-1:0] row_data,

    output logic [IDW-1:0] awid,
    output logic [AW-1:0]  awaddr,   // combinational: see below
    output logic [7:0]     awlen,
    output logic [2:0]     awsize,
    output logic [1:0]     awburst,
    output logic [3:0]     awcache,
    output logic [2:0]     awprot,
    output logic           awvalid,
    input  logic           awready,
    output logic [DW-1:0]  wdata,
    output logic [DW/8-1:0] wstrb,
    output logic           wlast,
    output logic           wvalid,
    input  logic           wready,
    input  logic [1:0]     bresp,
    input  logic           bvalid,
    output logic           bready
);
    localparam int BEAT = DW/8;

    function automatic int plane_off(input int upto);
        plane_off = 0;
        for (int i = 0; i < upto; i++) plane_off += PLANE_W[i*8 +: 8];
    endfunction

    typedef enum logic [1:0] {IDLE, ADDR, DATA, RESP} state_t;
    state_t st;
    logic [$clog2(N_PORTS+1)-1:0] pl;
    logic [ROW_BYTES*8-1:0] held;
    logic [N_PORTS*AW-1:0] bases;
    logic [31:0] tok;

    assign in_ready = (st == IDLE);
    // Driven combinationally from `pl`, which is registered. Registering it
    // inside ADDR would present the FIRST plane's AW with the previous
    // row's address, since `awvalid` is raised on the way in.
    assign awaddr   = w_addr;
    assign awid     = '0;
    assign awlen    = 8'd0;                 // one beat per plane
    assign awburst  = 2'b01;
    assign awcache  = 4'b1111;
    assign awprot   = 3'b000;
    assign wlast    = 1'b1;
    assign bready   = 1'b1;

    // Current plane's width, byte offset in the row, and address.
    logic [7:0] w_bytes;
    logic [31:0] w_off;
    logic [AW-1:0] w_addr;
    always_comb begin
        w_bytes = '0; w_off = '0;
        for (int i = 0; i < N_PORTS; i++) if (i == pl) begin
            w_bytes = PLANE_W[i*8 +: 8];
            w_off   = plane_off(i);
        end
        w_addr = '0;
        for (int i = 0; i < N_PORTS; i++) if (i == pl)
            w_addr = bases[i*AW +: AW] + AW'(tok) * AW'(w_bytes);
    end

    // AWSIZE is log2 of the plane width: 4 for a 16-byte plane, 2 for the
    // 4-byte norms. A narrow transfer, naturally aligned, never a read-modify-
    // write of a beat that holds three other tokens' norms.
    always_comb begin
        awsize = 3'd4;
        case (w_bytes)
            8'd1: awsize = 3'd0;
            8'd2: awsize = 3'd1;
            8'd4: awsize = 3'd2;
            8'd8: awsize = 3'd3;
            default: awsize = 3'd4;
        endcase
    end

    wire [3:0] lane = w_addr[3:0];          // byte lane within the 128-bit bus

    always_ff @(posedge clk) begin
        if (!rstn) begin
            st <= IDLE; pl <= '0;
            awvalid <= 1'b0; wvalid <= 1'b0;
        end else case (st)
            IDLE: if (in_valid) begin
                held <= row_data; bases <= head_base; tok <= token;
                pl <= '0; st <= ADDR; awvalid <= 1'b1;
            end
            ADDR: begin
                if (awready) begin
                    awvalid <= 1'b0;
                    wvalid  <= 1'b1;
                    // The bytes go to the lanes the address selects.
                    wdata <= '0; wstrb <= '0;
                    for (int b = 0; b < BEAT; b++)
                        if (b >= int'(lane) && b < int'(lane) + int'(w_bytes)) begin
                            wdata[b*8 +: 8] <=
                                held[(w_off + (b - int'(lane)))*8 +: 8];
                            wstrb[b] <= 1'b1;
                        end
                    st <= DATA;
                end else
                    awvalid <= 1'b1;
            end
            DATA: if (wready) begin
                wvalid <= 1'b0;
                st <= RESP;
            end
            RESP: if (bvalid) begin
                if (pl == N_PORTS-1) begin
                    st <= IDLE;
                end else begin
                    pl <= pl + 1'b1;
                    awvalid <= 1'b1;
                    st <= ADDR;
                end
            end
        endcase
    end
endmodule
