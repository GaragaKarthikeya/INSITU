// Self-checking testbench for rot_encode.sv, at both wire widths.
//
// The key plane (4 bits, 15 boundaries) and the value plane (2 bits, 3) run on
// the same stimulus and the same norms, because the interesting bug is a
// boundary index that is right at one width and off by one at the other.
//
// The last 32 vectors sit exactly on a decision boundary.  `_encode_plane`
// compares with `>`, so those channels belong to the lower bin; a `>=` in the
// RTL passes every other vector in this file and fails those.  They are
// constructed by `hw/vectors.py` rather than sampled -- an exact tie needs
// `x << 8 == norm * boundary`, which random stimulus hits about once in 2**32.
`timescale 1ns/1ps
module tb_rot_encode;
    localparam int D = 64;
    localparam int W = 24;
    localparam int VEC = D * W;
    localparam int N = 560;
    localparam int TIES = 32;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic signed [VEC-1:0] stim [0:N-1];
    logic [15:0] norms [0:N-1];
    logic [D*8-1:0] gold_k [0:N-1];      // one golden code per byte
    logic [D*8-1:0] gold_v [0:N-1];
    logic [19:0] bounds_key [0:14];
    logic [19:0] bounds_val [0:2];

    int errors = 0;

    logic in_valid, out_ready;
    logic signed [VEC-1:0] in_vec;
    logic [15:0] in_norm;
    logic k_ready, k_valid, v_ready, v_valid;
    logic [D*4-1:0] k_codes;
    logic [D*2-1:0] v_codes;

    rot_encode #(.BITS(4)) dut_k (
        .clk, .rstn, .in_valid(in_valid && k_ready), .in_ready(k_ready),
        .in_vec, .in_norm, .out_valid(k_valid), .out_ready(out_ready),
        .out_codes(k_codes));

    rot_encode #(.BITS(2), .BOUNDS({20'h07da2, 20'h00000, 20'hf825d})) dut_v (
        .clk, .rstn, .in_valid(in_valid && v_ready), .in_ready(v_ready),
        .in_vec, .in_norm, .out_valid(v_valid), .out_ready(out_ready),
        .out_codes(v_codes));

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    int got_k = 0, got_v = 0;
    always @(posedge clk) if (rstn && out_ready) begin
        if (k_valid) begin
            for (int l = 0; l < D; l++)
                check(k_codes[l*4 +: 4] === gold_k[got_k][l*8 +: 8],
                      $sformatf("vector %0d lane %0d: key code %0d != %0d",
                                got_k, l, k_codes[l*4 +: 4], gold_k[got_k][l*8 +: 8]));
            got_k = got_k + 1;
        end
        if (v_valid) begin
            for (int l = 0; l < D; l++)
                check(v_codes[l*2 +: 2] === gold_v[got_v][l*8 +: 8],
                      $sformatf("vector %0d lane %0d: value code %0d != %0d",
                                got_v, l, v_codes[l*2 +: 2], gold_v[got_v][l*8 +: 8]));
            got_v = got_v + 1;
        end
    end

    task automatic tick; @(negedge clk); endtask

    int sent;
    initial begin
        $readmemh("tb/vectors/enc_in.hex", stim);
        $readmemh("tb/vectors/enc_norm.hex", norms);
        $readmemh("tb/vectors/enc_idx_key.hex", gold_k);
        $readmemh("tb/vectors/enc_idx_value.hex", gold_v);
        $readmemh("tb/vectors/enc_bounds_key.hex", bounds_key);
        $readmemh("tb/vectors/enc_bounds_value.hex", bounds_val);

        // The boundaries are a format artifact, exactly as the sign diagonal
        // is.  Checking the parameter against the emitted codebook makes a
        // stale build fail here and not as a scattering of one-off codes.
        for (int i = 0; i < 15; i++)
            check(dut_k.BOUNDS[i*20 +: 20] === bounds_key[i],
                  $sformatf("key boundary %0d must match the Python codebook", i));
        for (int i = 0; i < 3; i++)
            check(dut_v.BOUNDS[i*20 +: 20] === bounds_val[i],
                  $sformatf("value boundary %0d must match the Python codebook", i));

        in_valid = 0; in_vec = '0; in_norm = '0; out_ready = 1;
        repeat (4) @(posedge clk);
        rstn = 1;

        sent = 0;
        while (sent < N) begin
            tick;
            if (k_ready && v_ready) begin
                in_valid = 1;
                in_vec = stim[sent];
                in_norm = norms[sent];
                sent = sent + 1;
            end else begin
                in_valid = 0;
            end
        end
        tick; in_valid = 0;
        while (got_k < N || got_v < N) tick;
        check(got_k == N && got_v == N, "every vector encoded exactly once at both widths");
        $display("  %0d vectors, of which the last %0d sit on a boundary", N, TIES);

        if (errors == 0) $display("=== tb_rot_encode: ALL TESTS PASSED ===");
        else $display("=== tb_rot_encode: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
