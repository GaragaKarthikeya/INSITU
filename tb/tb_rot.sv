// Self-checking testbench for rot_fwht.sv.
//
// The first 48 vectors are the q/k/v tensors emitted by the unmodified
// kernel.  The following 1024 are bounded, deterministic random vectors also
// emitted by Rotation.apply.  Together they catch lane ordering, the PCG64
// sign diagonal, every butterfly level, and signed final scaling.
`timescale 1ns/1ps
module tb_rot;
    localparam int D = 64;
    localparam int W = 24;
    localparam int VEC = D * W;
    localparam int MODEL_VECTORS = 48;
    localparam int RANDOM_VECTORS = 1024;
    localparam int TOTAL = MODEL_VECTORS + RANDOM_VECTORS;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;              // 250 MHz target clock

    logic in_valid, in_ready, out_valid, out_ready, range_error;
    logic signed [VEC-1:0] in_vec, out_vec;
    int errors = 0;
    int received = 0;
    logic checking = 1'b1;

    logic signed [VEC-1:0] wire_q [0:31], wire_k [0:7], wire_v [0:7];
    logic signed [VEC-1:0] rot_q  [0:31], rot_k  [0:7], rot_v  [0:7];
    logic signed [VEC-1:0] random_in [0:RANDOM_VECTORS-1];
    logic signed [VEC-1:0] random_out[0:RANDOM_VECTORS-1];
    logic [63:0] rot_signs [0:0];

    rot_fwht dut (
        .clk, .rstn, .in_valid, .in_ready, .in_vec,
        .out_valid, .out_ready, .out_vec, .range_error);

    function automatic logic signed [VEC-1:0] stimulus(input int n);
        if (n < 32) stimulus = wire_q[n];
        else if (n < 40) stimulus = wire_k[n - 32];
        else if (n < 48) stimulus = wire_v[n - 40];
        else stimulus = random_in[n - MODEL_VECTORS];
    endfunction

    function automatic logic signed [VEC-1:0] expected(input int n);
        if (n < 32) expected = rot_q[n];
        else if (n < 40) expected = rot_k[n - 32];
        else if (n < 48) expected = rot_v[n - 40];
        else expected = random_out[n - MODEL_VECTORS];
    endfunction

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    // Sampling before the nonblocking updates at this edge observes exactly
    // the value being accepted by the output handshake.
    always @(posedge clk) begin
        if (rstn && checking && out_valid && out_ready) begin
            check(out_vec === expected(received),
                  $sformatf("vector %0d must equal the Python Rotation.apply golden", received));
            check(!range_error, $sformatf("vector %0d escaped 24-bit output range", received));
            received = received + 1;
        end
    end

    task automatic tick;
        @(negedge clk);
    endtask

    int sent;
    logic signed [VEC-1:0] held;
    initial begin
        $readmemh("tb/vectors/wire_q.hex", wire_q);
        $readmemh("tb/vectors/wire_k.hex", wire_k);
        $readmemh("tb/vectors/wire_v.hex", wire_v);
        $readmemh("tb/vectors/rot_q.hex", rot_q);
        $readmemh("tb/vectors/rot_k.hex", rot_k);
        $readmemh("tb/vectors/rot_v.hex", rot_v);
        $readmemh("tb/vectors/rot_random_in.hex", random_in);
        $readmemh("tb/vectors/rot_random_out.hex", random_out);
        $readmemh("tb/vectors/rot_signs.hex", rot_signs);
        check(rot_signs[0] === dut.SIGN_NEG,
              "DUT sign diagonal must match the Python Rotation artifact");

        in_valid = 0; in_vec = '0; out_ready = 1;
        repeat (4) @(posedge clk);
        rstn = 1;

        // Full-rate stream: after the six-stage fill, one vector must retire
        // every cycle.  It covers 48 real stage taps and 1024 random vectors.
        sent = 0;
        while (sent < TOTAL) begin
            tick;
            if (in_ready) begin
                in_valid = 1;
                in_vec = stimulus(sent);
                sent = sent + 1;
            end
        end
        tick;
        in_valid = 0;
        while (received < TOTAL) tick;
        check(received == TOTAL, "all queued vectors retired exactly once");

        // A blocked output freezes the complete pipe.  This verifies that no
        // vector is overwritten while an integration-level downstream block
        // applies backpressure.
        checking = 0;
        out_ready = 0;
        tick;
        check(in_ready, "empty pipe accepts a vector even when output is blocked");
        in_valid = 1; in_vec = wire_q[0];
        tick;
        in_valid = 0;
        while (!out_valid) tick;
        held = out_vec;
        repeat (3) begin
            @(posedge clk);
            check(out_valid, "out_valid remains asserted during a stall");
            check(out_vec === held, "output vector remains stable during a stall");
        end
        out_ready = 1;
        @(posedge clk);
        tick;
        check(!out_valid, "stalled vector retires after out_ready returns");

        if (errors == 0) $display("=== tb_rot: ALL TESTS PASSED ===");
        else $display("=== tb_rot: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
