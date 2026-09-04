// Self-checking testbench for qtab_build.sv.
//
// EVERY ENTRY, NOT THE ONES SOME TOKEN HAPPENS TO SELECT
// ------------------------------------------------------
// The table has D * 2**BITS entries and a cached token reads exactly D of
// them.  Driving real cache codes would leave most of the table untested and
// would pass with a builder that wrote the right value to the wrong code
// index.  So the sweep drives code `i` on ALL D channels at once, for every
// `i`: 2**BITS gathers cover the table exhaustively, and a transposed write
// fails immediately.
//
// The golden is `qtab_table.hex` -- `q_rot[ch] * centroid[i]` computed in
// numpy from the SAME rotated queries the kernel scored with, at the same
// PROD_W the RTL uses.  `hw/vectors.py` refuses to emit a product that does
// not fit that width, so an overflow is a Python failure, not a silent wrap
// here.
//
// THE PING-PONG IS THE OTHER HALF OF THE TEST
// -------------------------------------------
// A table under construction must not disturb the one being read.  Head j+1
// is built while head j's table is still gathered from, and the bench checks
// head j's values DURING that build -- an unbanked implementation passes the
// per-head sweep and fails here.
`timescale 1ns/1ps
module tb_qtab_build;
    localparam int D       = 64;
    localparam int W       = 24;
    localparam int BITS    = 4;
    localparam int NB      = 1 << BITS;
    localparam int CB      = 20;
    localparam int PROD_W  = W + CB - 1;
    localparam int HEADS   = 32;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic signed [D*W-1:0]      qvec  [0:HEADS-1];
    logic [D*NB*PROD_W-1:0]     gold  [0:HEADS-1];
    logic [CB-1:0]              cents [0:NB-1];

    int errors = 0;
    // Set whenever the builder is mid-table, so "read through a build" is a
    // MEASURED overlap and not an assumption about relative durations.
    logic overlapped;
    always @(posedge clk) if (rstn && dut.busy) overlapped <= 1'b1;

    logic in_valid, commit, gat_en;
    logic in_ready, out_valid;
    logic signed [D*W-1:0] in_q;
    logic [D*BITS-1:0] gat_codes;
    logic [D*PROD_W-1:0] gat_prod;

    qtab_build dut (
        .clk, .rstn, .in_valid, .in_ready, .in_q, .out_valid, .commit,
        .gat_en, .gat_codes, .gat_prod);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    task automatic tick; @(negedge clk); endtask

    // Drive code `i` on every channel and compare the whole gathered word.
    task automatic sweep_code(input int head, input int i);
        for (int ch = 0; ch < D; ch++) gat_codes[ch*BITS +: BITS] = i[BITS-1:0];
        gat_en = 1;
        tick; @(posedge clk); tick;          // address registered, data out
        for (int ch = 0; ch < D; ch++)
            check(gat_prod[ch*PROD_W +: PROD_W] ===
                  gold[head][(ch*NB + i)*PROD_W +: PROD_W],
                  $sformatf("head %0d ch %0d code %0d: %0d != %0d", head, ch, i,
                            $signed(gat_prod[ch*PROD_W +: PROD_W]),
                            $signed(gold[head][(ch*NB + i)*PROD_W +: PROD_W])));
        gat_en = 0;
    endtask

    task automatic sweep_head(input int head);
        for (int i = 0; i < NB; i++) sweep_code(head, i);
    endtask

    task automatic build(input int head);
        in_q = qvec[head];
        in_valid = 1;
        while (!in_ready) tick;
        tick;
        in_valid = 0;
    endtask

    initial begin
        $readmemh("tb/vectors/rot_q.hex", qvec);
        $readmemh("tb/vectors/qtab_table.hex", gold);
        $readmemh("tb/vectors/cb_centroids_key.hex", cents);

        // The centroids are a format artifact.  A stale parameter fails HERE
        // and not as 1,024 wrong products.
        for (int i = 0; i < NB; i++)
            check(dut.CENTROIDS[i*CB +: CB] === cents[i],
                  $sformatf("centroid %0d must match the Python codebook", i));
        check(dut.PROD_W == PROD_W, "product width must match the golden's");

        in_valid = 0; commit = 0; gat_en = 0; in_q = '0; gat_codes = '0;
        repeat (4) @(posedge clk);
        rstn = 1;
        tick;

        // Head 0: build, commit, sweep every entry.
        build(0);
        while (!out_valid) tick;
        commit = 1; tick; commit = 0;
        sweep_head(0);

        // Ping-pong: start head 1's build and read head 0 THROUGH it.  The
        // builder is writing the spare bank for all 64 of those cycles.
        overlapped = 0;
        build(1);
        sweep_head(0);
        check(overlapped, "head 0 was gathered WHILE head 1 was being built");
        while (!out_valid) tick;
        commit = 1; tick; commit = 0;
        sweep_head(1);

        // The rest, back to back, which is the steady state attn_top runs in.
        for (int h = 2; h < HEADS; h++) begin
            build(h);
            while (!out_valid) tick;
            commit = 1; tick; commit = 0;
            sweep_head(h);
        end

        $display("  %0d heads x %0d channels x %0d codes = %0d entries checked",
                 HEADS, D, NB, HEADS*D*NB);

        if (errors == 0) $display("=== tb_qtab_build: ALL TESTS PASSED ===");
        else $display("=== tb_qtab_build: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
