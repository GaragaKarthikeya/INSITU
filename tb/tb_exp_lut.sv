// Self-checking testbench for exp_lut.sv.
//
// EVERY REACHABLE OUTPUT, AND BOTH SIDES OF EVERY TRANSITION
// -----------------------------------------------------------
// `exp_in.hex` is not random stimulus. For each of the 4,353 flat indices the
// generator computes the SMALLEST delta that reaches it, and emits that delta
// with the one below and the one above. So every entry of the table is read at
// every shift that can reach it, and every step between two outputs is
// straddled -- which is where an off-by-one in the fraction/integer split
// lives, and exactly what a uniform random sweep plateaus over.
//
// The saturation point is in the stimulus three times (DELTA_MAX-1, DELTA_MAX,
// DELTA_MAX+1) plus a delta of 2**40, because clamping there is a CORRECTNESS
// claim -- the model asserts it changes no result -- and not a safety net.
`timescale 1ns/1ps
module tb_exp_lut;
    localparam int N = 13565;
    localparam int DB = 48;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic [DB-1:0] stim [0:N-1];
    logic [15:0]   gold [0:N-1];
    logic [15:0]   table_gold [0:255];

    int errors = 0, nonzero = 0;

    logic en;
    logic [DB-1:0] in_delta;
    logic [15:0] out_p;

    exp_lut dut (.clk, .rstn, .en, .in_delta, .out_p);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    task automatic tick; @(negedge clk); endtask

    initial begin
        $readmemh("tb/vectors/exp_in.hex", stim);
        $readmemh("tb/vectors/exp_out.hex", gold);
        $readmemh("tb/vectors/exp_table.hex", table_gold);

        // The table is a format artifact of `ExpLut.__init__`, like the sign
        // diagonal and the codebook. A stale build fails HERE.
        for (int i = 0; i < 256; i++)
            check(dut.TABLE[i*16 +: 16] === table_gold[i],
                  $sformatf("table entry %0d must match ExpLut.table", i));

        en = 0; in_delta = '0;
        repeat (4) @(posedge clk);
        rstn = 1;
        en = 1;

        // Fully pipelined: one delta in per cycle, results three behind.
        for (int i = 0; i < N + 3; i++) begin
            tick;
            in_delta = (i < N) ? stim[i] : '0;
            @(posedge clk);
            if (i >= 3) begin
                check(out_p === gold[i-3],
                      $sformatf("delta %0d: %0d != %0d", stim[i-3], out_p, gold[i-3]));
                if (gold[i-3] != 0) nonzero = nonzero + 1;
            end
        end

        check(nonzero > 4000, "the sweep must reach far more than the zero tail");
        $display("  %0d deltas, %0d of them non-zero, 3-cycle pipe at II=1",
                 N, nonzero);

        if (errors == 0) $display("=== tb_exp_lut: ALL TESTS PASSED ===");
        else $display("=== tb_exp_lut: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
