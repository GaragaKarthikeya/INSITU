// Forward randomized Walsh-Hadamard rotation for the d=64 attention seam.
//
// This is deliberately NOT a generic transform.  The accelerator format is
// one round of R = H D / sqrt(64): 64 signed Q8.16 lanes, a seed-selected
// sign diagonal, six add/subtract levels, and an exact arithmetic >> 3.
// Other dimensions need the reciprocal-multiply path in ops/rotate.py and
// must not silently acquire a rounded shift here.
`timescale 1ns/1ps
module rot_fwht #(
    // Bit i applies to lane i.  A one means negate.  The default is the
    // deterministic QuantConfig(seed=0) diagonal; a build for another seed
    // supplies the generated sign word as this parameter.
    parameter logic [63:0] SIGN_NEG =
        64'h81191ff5899001f8
) (
    input  logic              clk,
    input  logic              rstn,
    input  logic              in_valid,
    output logic              in_ready,
    input  logic signed [1535:0] in_vec,   // lane i is [24*i +: 24]
    output logic              out_valid,
    input  logic              out_ready,
    output logic signed [1535:0] out_vec,
    // Narrowing the normalized result to 24 bits is legal only when this is
    // low.  Production integration treats a high value as a format error;
    // it never wraps or saturates behind the numeric model's back.
    output logic              range_error
);
    localparam int D       = 64;
    localparam int IN_BITS = 24;
    // Six butterfly additions plus the one-bit negation endpoint need 31 b.
    localparam int WIDE    = 31;
    localparam int SHIFT   = 3;
    localparam logic signed [WIDE-1:0] OUT_MIN = -(31'sd1 <<< 23);
    localparam logic signed [WIDE-1:0] OUT_MAX =  (31'sd1 <<< 23) - 1;

    logic signed [WIDE-1:0] signed_in [0:D-1];
    logic signed [WIDE-1:0] b1 [0:D-1], b2 [0:D-1], b3 [0:D-1];
    logic signed [WIDE-1:0] b4 [0:D-1], b5 [0:D-1], b6 [0:D-1];
    logic signed [WIDE-1:0] s1 [0:D-1], s2 [0:D-1], s3 [0:D-1];
    logic signed [WIDE-1:0] s4 [0:D-1], s5 [0:D-1], s6 [0:D-1];
    logic [5:0] valid;
    logic [D-1:0] range_bad;

    // A stalled consumer freezes the complete pipe.  This is intentionally
    // simple for this block; once released, all six levels advance together
    // and the input is accepted again, preserving vector order.
    wire advance = !valid[5] || out_ready;
    assign in_ready = advance;
    assign out_valid = valid[5];

    generate
        genvar g;
        for (g = 0; g < D; g = g + 1) begin : lanes
            wire signed [IN_BITS-1:0] lane = in_vec[g*IN_BITS +: IN_BITS];
            wire signed [WIDE-1:0] lane_extended =
                {{(WIDE-IN_BITS){lane[IN_BITS-1]}}, lane};
            // Extend BEFORE negating: -24'sh800000 is +2^23, which does not
            // fit in 24 signed bits but does fit in this internal width.
            assign signed_in[g] = SIGN_NEG[g] ? -lane_extended : lane_extended;

            wire signed [WIDE-1:0] scaled = s6[g] >>> SHIFT;
            assign out_vec[g*IN_BITS +: IN_BITS] = scaled[IN_BITS-1:0];
            assign range_bad[g] = (scaled < OUT_MIN) || (scaled > OUT_MAX);
        end
    endgenerate
    assign range_error = |range_bad;

    // bN is the combinational result of FWHT stage N.  A stage with h pairs
    // the low and high halves of each 2h-sized group: [lo+hi, lo-hi].
    integer i, group;
    always_comb begin
        for (i = 0; i < D; i = i + 1) begin
            b1[i] = '0; b2[i] = '0; b3[i] = '0;
            b4[i] = '0; b5[i] = '0; b6[i] = '0;
        end
        for (group = 0; group < D; group = group + 2)
            begin b1[group] = signed_in[group] + signed_in[group+1];
                  b1[group+1] = signed_in[group] - signed_in[group+1]; end
        for (group = 0; group < D; group = group + 4)
            for (i = 0; i < 2; i = i + 1)
                begin b2[group+i] = s1[group+i] + s1[group+2+i];
                      b2[group+2+i] = s1[group+i] - s1[group+2+i]; end
        for (group = 0; group < D; group = group + 8)
            for (i = 0; i < 4; i = i + 1)
                begin b3[group+i] = s2[group+i] + s2[group+4+i];
                      b3[group+4+i] = s2[group+i] - s2[group+4+i]; end
        for (group = 0; group < D; group = group + 16)
            for (i = 0; i < 8; i = i + 1)
                begin b4[group+i] = s3[group+i] + s3[group+8+i];
                      b4[group+8+i] = s3[group+i] - s3[group+8+i]; end
        for (group = 0; group < D; group = group + 32)
            for (i = 0; i < 16; i = i + 1)
                begin b5[group+i] = s4[group+i] + s4[group+16+i];
                      b5[group+16+i] = s4[group+i] - s4[group+16+i]; end
        for (i = 0; i < 32; i = i + 1)
            begin b6[i] = s5[i] + s5[32+i];
                  b6[32+i] = s5[i] - s5[32+i]; end
    end

    always_ff @(posedge clk) begin
        if (!rstn) begin
            valid <= '0;
        end else if (advance) begin
            valid <= {valid[4:0], in_valid};
            if (in_valid)
                for (i = 0; i < D; i = i + 1) s1[i] <= b1[i];
            if (valid[0])
                for (i = 0; i < D; i = i + 1) s2[i] <= b2[i];
            if (valid[1])
                for (i = 0; i < D; i = i + 1) s3[i] <= b3[i];
            if (valid[2])
                for (i = 0; i < D; i = i + 1) s4[i] <= b4[i];
            if (valid[3])
                for (i = 0; i < D; i = i + 1) s5[i] <= b5[i];
            if (valid[4])
                for (i = 0; i < D; i = i + 1) s6[i] <= b6[i];
        end
    end
endmodule
