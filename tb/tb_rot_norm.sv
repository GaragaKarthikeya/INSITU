// Self-checking testbench for rot_norm.sv.
//
// 560 vectors: the 16 real rotated k/v tensors off the unmodified kernel, 512
// random ones, and 32 constructed to sit exactly on a codebook boundary.  The
// golden is `KVQuantizer._norm_wire` itself, so a norm that is one LSB out --
// the half-to-even rounding being a half-up, say -- fails here rather than
// several thousand cache rows later.
`timescale 1ns/1ps
module tb_rot_norm;
    localparam int D = 64;
    localparam int W = 24;
    localparam int VEC = D * W;
    localparam int N = 560;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic in_valid, in_ready, out_valid, out_ready, out_saturated;
    logic signed [VEC-1:0] in_vec;
    logic [15:0] out_norm;

    logic signed [VEC-1:0] stim [0:N-1];
    logic [15:0] golden [0:N-1];
    int errors = 0, received = 0, saturated = 0;
    logic checking = 1'b1;

    rot_norm dut (.clk, .rstn, .in_valid, .in_ready, .in_vec,
                  .out_valid, .out_ready, .out_norm, .out_saturated);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    always @(posedge clk) begin
        if (rstn && checking && out_valid && out_ready) begin
            check(out_norm === golden[received],
                  $sformatf("vector %0d: norm %0d must equal _norm_wire's %0d",
                            received, out_norm, golden[received]));
            if (out_saturated) saturated = saturated + 1;
            received = received + 1;
        end
    end

    task automatic tick; @(negedge clk); endtask

    int sent;
    logic [15:0] held;
    initial begin
        $readmemh("tb/vectors/enc_in.hex", stim);
        $readmemh("tb/vectors/enc_norm.hex", golden);

        in_valid = 0; in_vec = '0; out_ready = 1;
        repeat (4) @(posedge clk);
        rstn = 1;

        sent = 0;
        while (sent < N) begin
            tick;
            if (in_ready) begin
                in_valid = 1;
                in_vec = stim[sent];
                sent = sent + 1;
            end else begin
                in_valid = 0;
            end
        end
        tick; in_valid = 0;
        while (received < N) tick;
        check(received == N, "every vector retired exactly once, in order");

        // The stimulus is bounded to the 24-bit seam, so nothing in it should
        // reach the Q7.9 clamp.  If this ever fires, the wire format -- not the
        // block -- is what needs revisiting.
        check(saturated == 0, $sformatf("%0d norms saturated the wire format", saturated));

        // A blocked consumer must freeze the square-root pipeline as a unit.
        // It has no internal elasticity, so a stage that kept advancing here
        // would overwrite a result that was never read.
        checking = 0;
        out_ready = 0;
        in_valid = 1; in_vec = stim[0];
        while (!out_valid) tick;
        in_valid = 0;
        held = out_norm;
        repeat (5) begin
            @(posedge clk);
            check(out_valid, "out_valid holds while the consumer is blocked");
            check(out_norm === held, "the norm is stable while the consumer is blocked");
        end
        out_ready = 1;
        @(posedge clk);
        tick;
        check(!out_valid, "the held result retires once out_ready returns");

        if (errors == 0) $display("=== tb_rot_norm: ALL TESTS PASSED ===");
        else $display("=== tb_rot_norm: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
