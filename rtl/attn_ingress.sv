// 512-bit AXI4-Stream in, one 1,536-bit rotated-lane vector out.
//
// WHY THIS IS A SHIFT REGISTER AND A COUNTER, AND NOTHING MORE
// -----------------------------------------------------------
// `d = 64` lanes of Q8.16 in 24 bits is 1,536 b, which is exactly three
// 512-bit beats, so there is no TKEEP to decode and no partial beat to hold.
// A LANE still straddles a beat boundary -- 24 does not divide 512 -- but the
// straddle disappears the moment three beats are concatenated, which is what
// this block does. `hw/vectors.py::beats` is the other end of the convention:
// byte 0 of the stream is bits [7:0] of beat 0, so beat 0 is the LOW beat of
// the vector.
//
// THE GROUP STRUCTURE IS COUNTED HERE, NOT IN THE FSM
// ---------------------------------------------------
// The stream is 8 groups of 6 vectors, `[k][v][q0..q3]`. `out_idx` says which
// of the six a vector is and `out_group` which group, so `attn_top`'s control
// never counts beats -- getting the k/v/q order wrong is then a single
// comparison to read rather than a beat arithmetic bug.
`timescale 1ns/1ps
module attn_ingress #(
    parameter int D          = 64,
    parameter int LANE_BITS  = 24,
    parameter int BEAT_BITS  = 512,
    parameter int VEC_BITS   = D * LANE_BITS,
    parameter int PER_GROUP  = 6,
    parameter int GROUPS     = 8
) (
    input  logic clk,
    input  logic rstn,
    input  logic start,                       // reset the group/vector counters

    input  logic s_valid,
    output logic s_ready,
    input  logic [BEAT_BITS-1:0] s_data,

    output logic out_valid,
    input  logic out_ready,
    output logic [VEC_BITS-1:0] out_vec,
    output logic [$clog2(PER_GROUP)-1:0] out_idx,
    output logic [$clog2(GROUPS)-1:0] out_group
);
    localparam int NBEAT = VEC_BITS / BEAT_BITS;

    initial if (VEC_BITS % BEAT_BITS != 0)
        $fatal(1, "a vector must be a whole number of beats; TKEEP is not modelled");

    logic [VEC_BITS-1:0] sh;
    logic [$clog2(NBEAT+1)-1:0] cnt;
    logic [$clog2(PER_GROUP)-1:0] vidx;
    logic [$clog2(GROUPS)-1:0] grp;

    assign s_ready  = !out_valid;
    assign out_vec  = sh;
    assign out_idx  = vidx;
    assign out_group = grp;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            cnt <= '0; vidx <= '0; grp <= '0;
            out_valid <= 1'b0;
        end else if (start) begin
            cnt <= '0; vidx <= '0; grp <= '0;
            out_valid <= 1'b0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
                if (vidx == PER_GROUP-1) begin
                    vidx <= '0;
                    grp <= grp + 1'b1;
                end else
                    vidx <= vidx + 1'b1;
            end
            if (s_valid && s_ready) begin
                sh <= {s_data, sh[VEC_BITS-1:BEAT_BITS]};
                if (cnt == NBEAT-1) begin
                    cnt <= '0;
                    out_valid <= 1'b1;
                end else
                    cnt <= cnt + 1'b1;
            end
        end
    end
endmodule
