// Self-checking testbench for softmax_online.sv + accum.sv + exp_lut.sv.
//
// The golden IS `attend_online`, not `attend_two_pass`
// ----------------------------------------------------
// `forward` runs the two-pass softmax and the two are not bit-identical: the
// online form truncates the accumulator once per rescale. Hardware is online,
// so the goldens here come from an observer attached to `attend_online` itself
// -- `CompressedAttention.on_step`, the same `is not None` contract the
// kernel's `on_stage` uses. A bench written against the two-pass result could
// never pass, and `plan.MD` says so.
//
// Every token, not the last one
// -----------------------------
// `m`, `l` and all 64 accumulator channels are compared after every token, not
// at the end of the scan. An accumulator that is only right at the last token
// is one whose rescales happened to cancel, and the rescale is the thing this
// step exists to build.
//
// The flush path is the point of the short context
// ------------------------------------------------
// New maxima are logarithmically rare -- about ln T + 0.577 -- so a long
// context would exercise the common path almost exclusively. At ctx 64 the
// four lanes rescale 3 to 6 times each, and the bench asserts the count it
// observed matches the count the model recorded. A DUT that never rescaled
// would pass a bench that only checked the final answer of a lane whose
// maximum arrived first.
`timescale 1ns/1ps
module tb_softmax_accum;
    localparam int D    = 64;
    localparam int VB   = 2;
    localparam int NB   = 16;
    localparam int KB   = 4;
    localparam int SW   = 48;
    localparam int AW   = 32;
    localparam int LW   = 48;
    localparam int T    = 65;
    localparam int LANE = 4;                       // query heads sharing KV head 0
    localparam int ROW_BITS = D*KB + D*VB + 2*NB;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic [LANE*SW-1:0] g_s     [0:T-1];
    logic [LANE*SW-1:0] g_m     [0:T-1];
    logic [LANE*16-1:0] g_p     [0:T-1];
    logic [LANE*LW-1:0] g_l     [0:T-1];
    logic [LANE*16-1:0] g_fac   [0:T-1];
    logic [3:0]         g_grew  [0:T-1];
    logic [D*AW-1:0]    g_acc   [0:T*LANE-1];      // token-major, then lane
    logic [ROW_BITS-1:0] rows   [0:T-1];

    int errors = 0;

    logic start, in_valid, out_ready_unused;
    logic in_ready;
    logic signed [SW-1:0] in_score;
    logic [D*VB-1:0] in_vcodes;
    logic signed [NB-1:0] in_vnorm;

    logic op_valid, op_ready, op_scale;
    logic [15:0] op_factor, op_p;
    logic [D*VB-1:0] op_vcodes;
    logic signed [NB-1:0] op_vnorm;
    logic signed [SW-1:0] out_m;
    logic [LW-1:0] out_l;
    logic [5:0] hw_max_fill;
    logic l_overflow;

    logic tok_done, scale_done;
    logic [D*AW-1:0] out_acc;
    logic [31:0] out_overflows, out_scale_cycles;

    softmax_online sm (
        .clk, .rstn, .start, .in_valid, .in_ready, .in_score, .in_vcodes, .in_vnorm,
        .op_valid, .op_ready, .op_scale, .op_factor, .op_p, .op_vcodes, .op_vnorm,
        .out_m, .out_l, .hw_max_fill, .l_overflow);

    // A SECOND accumulator, driven directly at full rate.
    //
    // `softmax_online` drains its own three-stage pipe and spends three more
    // cycles on the factor before it issues a SCALE, so `accum` never sees one
    // with its add pipe still busy -- the guard that waits for that drain is
    // unreachable through this producer, and a mutation removing it passes.
    // Replaying the same op stream back to back, with no gaps, is what puts a
    // SCALE directly behind an add. The goldens are unchanged: the arithmetic
    // does not depend on the timing, which is the claim.
    logic r_valid, r_scale;
    logic r_ready;
    logic [15:0] r_factor, r_p;
    logic [D*VB-1:0] r_vcodes;
    logic signed [NB-1:0] r_vnorm;
    logic r_tok_done, r_scale_done;
    logic [D*AW-1:0] r_acc;
    logic [31:0] r_ovf, r_cyc;

    accum replay (
        .clk, .rstn, .start, .op_valid(r_valid), .op_ready(r_ready),
        .op_scale(r_scale), .op_factor(r_factor), .op_p(r_p),
        .op_vcodes(r_vcodes), .op_vnorm(r_vnorm),
        .tok_done(r_tok_done), .scale_done(r_scale_done),
        .out_acc(r_acc), .out_overflows(r_ovf), .out_scale_cycles(r_cyc));

    accum ac (
        .clk, .rstn, .start, .op_valid, .op_ready, .op_scale, .op_factor,
        .op_p, .op_vcodes, .op_vnorm, .tok_done, .scale_done,
        .out_acc, .out_overflows, .out_scale_cycles);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    task automatic tick; @(negedge clk); endtask

    // -- the collectors ----------------------------------------------------
    //
    // Two counters, because `l` and `acc` are produced at different points.
    // `softmax_online` folds a probability into `l` when it issues the op;
    // `accum` reaches the matching accumulator two stages later. Comparing
    // both at the retire edge reads an `l` that already contains the next
    // token's probability -- which is what the first version of this bench
    // did, and it failed a correct DUT by exactly one unity.
    int lane = 0, tok = 0, itok = 0, scales_seen = 0, adds_seen = 0;
    int growing_cycles = 0, stall_cycles = 0;

    always @(posedge clk) if (rstn && tok_done) begin
        check(out_acc === g_acc[tok*LANE + lane],
              $sformatf("lane %0d token %0d: accumulator mismatch", lane, tok));
        adds_seen = adds_seen + 1;
        tok = tok + 1;
    end

    // `m` and `l` are the state after a token, registered on the issue edge,
    // so they are compared one cycle later. Blocking inside an `always` on a
    // `@(posedge clk)` to wait for them makes the block miss every event that
    // lands while it waits -- which is what the first version of this bench
    // did, and it read a finished scan's `l` against token 34.
    logic prev_add = 0;
    int prev_idx = 0;

    // lane 0's op stream, captured as it is issued
    localparam int MAXOPS = 128;
    logic [1 + 16 + 16 + D*VB + NB - 1:0] ops [0:MAXOPS-1];
    int n_ops = 0, capture = 0;
    always @(posedge clk) if (rstn && capture && op_valid && op_ready) begin
        ops[n_ops] = {op_scale, op_factor, op_p, op_vcodes, op_vnorm};
        n_ops = n_ops + 1;
    end

    always @(posedge clk) if (rstn) begin
        if (prev_add) begin
            check(out_m === $signed(g_m[prev_idx][lane*SW +: SW]),
                  $sformatf("lane %0d token %0d: m %0d != %0d", lane, prev_idx,
                            out_m, $signed(g_m[prev_idx][lane*SW +: SW])));
            check(out_l === g_l[prev_idx][lane*LW +: LW],
                  $sformatf("lane %0d token %0d: l %0d != %0d", lane, prev_idx,
                            out_l, g_l[prev_idx][lane*LW +: LW]));
        end
        prev_add <= 1'b0;
        if (op_valid && op_ready) begin
            if (op_scale) begin
                check(op_factor === g_fac[itok][lane*16 +: 16],
                      $sformatf("lane %0d token %0d: factor %0d != %0d", lane,
                                itok, op_factor, g_fac[itok][lane*16 +: 16]));
                check(g_grew[itok][lane] === 1'b1,
                      $sformatf("lane %0d token %0d: rescaled where the model did not",
                                lane, itok));
                scales_seen = scales_seen + 1;
                if (seen_factor[op_factor] !== 16'h1) begin
                    seen_factor[op_factor] = 16'h1;
                    distinct_factors = distinct_factors + 1;
                end
            end else begin
                check(op_p === g_p[itok][lane*16 +: 16],
                      $sformatf("lane %0d token %0d: p %0d != %0d", lane, itok,
                                op_p, g_p[itok][lane*16 +: 16]));
                prev_add <= 1'b1;
                prev_idx <= itok;
                itok = itok + 1;
            end
        end
    end

    // How long a new maximum actually costs, end to end: drain the exp pipe,
    // three cycles for the factor, then accum folding 64 channels through 8
    // multipliers. Measured, because `plan.MD` sized the input FIFO from an
    // estimate of it.
    always @(posedge clk) if (rstn) begin
        if (!sm.empty && !sm.pop) bubble = bubble + 1;
        if (sm.growing) growing_cycles = growing_cycles + 1;
        if (in_valid && !in_ready) stall_cycles = stall_cycles + 1;
    end

    int expect_scales, total_scales = 0;
    int stress = 0, distinct_factors = 0;
    logic [15:0] seen_factor [0:65535];

    // A deadlock is a failure, not a hang. A `>=` in the new-maximum test makes
    // the DUT rescale forever on a token equal to the running maximum, and
    // without this the bench is an iverilog process that never exits -- which
    // reads as an infrastructure problem rather than as the DUT being wrong.
    initial begin
        #4000000;
        $display("  FAIL: watchdog -- the scan never finished");
        $display("=== tb_softmax_accum: 1 TESTS FAILED ===");
        $finish;
    end
    int rtok = 0, back_to_back = 0;
    // What a new maximum actually costs the scan: cycles in which a score is
    // waiting in the FIFO and the datapath cannot take it. That is the number
    // `AttentionConfig.rescale_cycles` stands in for, and it carries 1.
    int bubble = 0;
    // Snapshotted before the replay pass, which pulses `start` and clears them.
    int fill_hw = 0, scale_cyc = 0;

    always @(posedge clk) if (rstn && r_tok_done) begin
        check(r_acc === g_acc[rtok*LANE + 0],
              $sformatf("replay token %0d: accumulator mismatch", rtok));
        rtok = rtok + 1;
    end
    // Did a SCALE actually arrive with the add pipe still in flight?
    always @(posedge clk)
        if (rstn && r_valid && r_scale && replay.pipe_busy)
            back_to_back = back_to_back + 1;
    initial begin
        $readmemh("tb/vectors/cache_rows_h0.hex", rows);

        start = 0; in_valid = 0; in_score = '0; in_vcodes = '0; in_vnorm = '0;
        r_valid = 0; r_scale = 0; r_factor = '0; r_p = '0;
        r_vcodes = '0; r_vnorm = '0;
        repeat (4) @(posedge clk);
        rstn = 1;

        expect_scales = 0;
        for (stress = 0; stress < 2; stress++) begin
        if (stress == 0) begin
            $readmemh("tb/vectors/sm_s.hex", g_s);
            $readmemh("tb/vectors/sm_m.hex", g_m);
            $readmemh("tb/vectors/sm_p.hex", g_p);
            $readmemh("tb/vectors/sm_l.hex", g_l);
            $readmemh("tb/vectors/sm_factor.hex", g_fac);
            $readmemh("tb/vectors/sm_grew.hex", g_grew);
            $readmemh("tb/vectors/sm_acc.hex", g_acc);
        end else begin
            // The same attend_online on the same cached rows, with a query
            // scaled so the running maximum actually moves by something the
            // exp LUT can distinguish. Without this set the rescale multiply
            // is a no-op in 16 of 19 events and its rounding is untested.
            $readmemh("tb/vectors/sm2_s.hex", g_s);
            $readmemh("tb/vectors/sm2_m.hex", g_m);
            $readmemh("tb/vectors/sm2_p.hex", g_p);
            $readmemh("tb/vectors/sm2_l.hex", g_l);
            $readmemh("tb/vectors/sm2_factor.hex", g_fac);
            $readmemh("tb/vectors/sm2_grew.hex", g_grew);
            $readmemh("tb/vectors/sm2_acc.hex", g_acc);
        end
        for (lane = 0; lane < LANE; lane++) begin
            tick; start = 1; tick; start = 0;
            // Capture the STRESS set's lane 0: its ops carry the rescale
            // factors that are not unity, so the replay exercises the fold
            // as well as the flow control.
            capture = (lane == 0 && stress == 1);
            if (capture) n_ops = 0;
            tok = 0; itok = 0; adds_seen = 0; scales_seen = 0;
            prev_add = 0;

            // Full rate in: the FIFO, not back-pressure, is what absorbs a
            // rescale. `hw_max_fill` reports whether 16 entries would do.
            for (int t = 0; t < T; t++) begin
                in_valid = 1;
                in_score = $signed(g_s[t][lane*SW +: SW]);
                in_vcodes = rows[t][D*KB +: D*VB];
                in_vnorm  = $signed(rows[t][D*KB + D*VB + NB +: NB]);
                @(posedge clk);
                while (!in_ready) begin tick; @(posedge clk); end
                tick;
            end
            in_valid = 0;
            while (adds_seen < T) tick;

            for (int t = 0; t < T; t++) expect_scales += g_grew[t][lane];
            check(scales_seen == expect_scales,
                  $sformatf("lane %0d: %0d rescales, model recorded %0d",
                            lane, scales_seen, expect_scales));
            total_scales += scales_seen;
            expect_scales = 0;
        end
        end

        fill_hw = hw_max_fill;
        scale_cyc = out_scale_cycles;

        // -- replay lane 0's ops into a bare accumulator, back to back ------
        rtok = 0;
        tick; start = 1; tick; start = 0;
        for (int i = 0; i < n_ops; i++) begin
            r_valid  = 1;
            r_scale  = ops[i][1 + 16 + 16 + D*VB + NB - 1];
            r_factor = ops[i][16 + 16 + D*VB + NB - 1 -: 16];
            r_p      = ops[i][16 + D*VB + NB - 1 -: 16];
            r_vcodes = ops[i][D*VB + NB - 1 -: D*VB];
            r_vnorm  = $signed(ops[i][NB-1:0]);
            @(posedge clk);
            while (!r_ready) begin tick; @(posedge clk); end
            tick;
        end
        r_valid = 0;
        while (rtok < T) tick;
        check(back_to_back > 0,
              "a SCALE must land while the add pipe is busy");
        $display("  replayed %0d ops at full rate, %0d cycles of SCALE-behind-add",
                 n_ops, back_to_back);

        // A rescale whose factor is always unity is not a rescale.
        check(distinct_factors >= 17,
              $sformatf("only %0d distinct rescale factors were applied",
                        distinct_factors));
        check(l_overflow === 1'b0, "the running sum never overflowed L_WIDTH");
        check(out_overflows == 0, "no accumulator channel saturated at ctx 64");
        $display("  2 stimulus sets x %0d lanes x %0d tokens, FIFO high-water %0d of %0d",
                 LANE, T, fill_hw, sm.FIFO_DEPTH);
        $display("  %0d distinct rescale factors applied", distinct_factors);
        $display("  %0d rescale events: %0d cycles growing, %0d folding in accum",
                 total_scales, growing_cycles, scale_cyc);
        $display("  scan bubbles %0d cycles, %0d per rescale (model charges 1)",
                 bubble, bubble / total_scales);

        if (errors == 0) $display("=== tb_softmax_accum: ALL TESTS PASSED ===");
        else $display("=== tb_softmax_accum: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
