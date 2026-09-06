// Self-checking testbench for score_lane.sv + qtab_build.sv.
//
// The golden IS `CompressedAttention.scores`, not A REIMPLEMENTATION
// ------------------------------------------------------------------
// `scores.hex` is what the kernel's own attention object returned for the
// step in `hw/vectors.py::collect` -- the same call `forward` makes, on the
// same rotated queries and the same cache.  So a bench that passes has agreed
// with the code that scores `cosine 1.000000` against `LlamaAttention`, and
// not with a second model written from the same paragraph.
//
// The stimulus is the cache those scores were computed over: `cache_rows_hN.hex`
// is `ctx + 1` rows of the 52 bytes `pack` wrote, in causal order.  The lane
// is handed the raw row and splits it itself, so the bit layout is under test
// too -- a `_pack_codes` disagreement shows up here rather than as an accuracy
// loss nobody can localise.
//
// ALL 32 QUERY HEADS, which IS the point OF the PING-PONG
// -------------------------------------------------------
// Query heads 4h..4h+3 share KV head h's rows.  The bench runs every head
// through one lane, building each head's table while the previous head's scan
// is still draining -- the steady state `attn_top` runs in, and the only way
// the table swap gets exercised against live rows.
//
// Back-pressure is part of the contract
// -------------------------------------
// The last four heads are scanned with `out_ready` pulsed low at random,
// because a
// score pipe that only works when nothing downstream ever stalls is a pipe
// that will fail the first time the softmax rescales.  The values must be
// bit-identical to the unstalled run, not merely still plausible.
`timescale 1ns/1ps
module tb_score_lane;
    localparam int D        = 64;
    localparam int W        = 24;
    localparam int KB       = 4;
    localparam int VB       = 2;
    localparam int NB       = 16;
    localparam int ROW_BITS = D*KB + D*VB + 2*NB;   // 416 = 52 B
    localparam int SW       = 48;
    localparam int HEADS    = 32;
    localparam int KVH      = 8;
    localparam int GROUPS   = HEADS / KVH;
    localparam int T        = 65;                   // ctx 64 + this token
    localparam int EDGE     = 5;                    // constructed norm rows

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic signed [D*W-1:0]  qvec  [0:HEADS-1];
    logic [ROW_BITS-1:0]    rows  [0:KVH-1][0:T-1];
    logic [T*SW-1:0]        gold  [0:HEADS-1];
    logic [ROW_BITS-1:0]    onehead [0:T-1];
    logic [ROW_BITS-1:0]    edge_rows [0:EDGE-1];
    logic [SW-1:0]          edge_gold [0:EDGE-1];

    int errors = 0;

    logic q_valid, tab_commit, row_valid, out_ready;
    logic q_ready, tab_ready, row_ready, out_valid, out_clip;
    logic signed [D*W-1:0] q_vec;
    logic [ROW_BITS-1:0] row_data;
    logic signed [SW-1:0] out_score;
    logic [D*VB-1:0] out_vcodes;
    logic signed [NB-1:0] out_vnorm;

    score_lane dut (
        .clk, .rstn, .q_valid, .q_ready, .q_vec, .tab_ready, .tab_commit,
        .row_valid, .row_ready, .row_data,
        .out_valid, .out_ready, .out_score, .out_vcodes, .out_vnorm, .out_clip);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    task automatic tick; @(negedge clk); endtask

    // -- the collector -----------------------------------------------------
    int head_out = 0, tok_out = 0, clips = 0, stalls = 0;
    // A back-pressure claim is only worth making if the pipe was actually
    // held; a random pattern that happened never to fire proves nothing.
    always @(posedge clk) if (rstn && out_valid && !out_ready)
        stalls = stalls + 1;
    logic edge_phase = 0;
    int edge_out = 0;
    always @(posedge clk) if (rstn && out_valid && out_ready && edge_phase) begin
        check(out_score === $signed(edge_gold[edge_out]),
              $sformatf("edge row %0d: %0d != %0d", edge_out,
                        out_score, $signed(edge_gold[edge_out])));
        edge_out = edge_out + 1;
    end
    always @(posedge clk) if (rstn && out_valid && out_ready && !edge_phase) begin
        check(out_score === $signed(gold[head_out][tok_out*SW +: SW]),
              $sformatf("head %0d token %0d: %0d != %0d", head_out, tok_out,
                        out_score, $signed(gold[head_out][tok_out*SW +: SW])));
        // The value plane must arrive with the score of the same token; a
        // shift register off by one here is invisible until `accum.sv` exists.
        check(out_vcodes === rows[head_out/GROUPS][tok_out][D*KB +: D*VB],
              $sformatf("head %0d token %0d: v codes not aligned with the score",
                        head_out, tok_out));
        check(out_vnorm === $signed(rows[head_out/GROUPS][tok_out][D*KB+D*VB+NB +: NB]),
              $sformatf("head %0d token %0d: v norm not aligned with the score",
                        head_out, tok_out));
        if (out_clip) clips = clips + 1;
        tok_out = tok_out + 1;
        if (tok_out == T) begin tok_out = 0; head_out = head_out + 1; end
    end

    // -- the driver --------------------------------------------------------
    int seed = 32'h5C0E;
    // `row_ready` is combinational in `out_ready`, so it must be SAMPLED at the
    // clock edge the DUT samples it -- reading it in the same delta the driver
    // assigns `out_ready` races the continuous assignment and makes the bench
    // and the DUT disagree about which rows were accepted.
    task automatic scan(input int head, input logic stall);
        int t;
        t = 0;
        while (t < T) begin
            row_valid = 1;
            row_data  = rows[head/GROUPS][t];
            @(posedge clk);
            if (row_ready) t = t + 1;
            tick;
            out_ready = stall ? ($random(seed) % 2 != 0) : 1'b1;
        end
        row_valid = 0;
        out_ready = 1;
        tick;
    endtask

    initial begin
        $readmemh("tb/vectors/rot_q.hex", qvec);
        $readmemh("tb/vectors/scores.hex", gold);
        $readmemh("tb/vectors/score_edge_rows.hex", edge_rows);
        $readmemh("tb/vectors/score_edge_gold.hex", edge_gold);
        for (int h = 0; h < KVH; h++) begin
            case (h)
                0: $readmemh("tb/vectors/cache_rows_h0.hex", onehead);
                1: $readmemh("tb/vectors/cache_rows_h1.hex", onehead);
                2: $readmemh("tb/vectors/cache_rows_h2.hex", onehead);
                3: $readmemh("tb/vectors/cache_rows_h3.hex", onehead);
                4: $readmemh("tb/vectors/cache_rows_h4.hex", onehead);
                5: $readmemh("tb/vectors/cache_rows_h5.hex", onehead);
                6: $readmemh("tb/vectors/cache_rows_h6.hex", onehead);
                7: $readmemh("tb/vectors/cache_rows_h7.hex", onehead);
            endcase
            for (int t = 0; t < T; t++) rows[h][t] = onehead[t];
        end

        q_valid = 0; tab_commit = 0; row_valid = 0; out_ready = 1;
        q_vec = '0; row_data = '0;
        repeat (4) @(posedge clk);
        rstn = 1;
        tick;

        for (int h = 0; h < HEADS; h++) begin
            // Build head h's table into the spare bank.  For h > 0 this
            // overlaps head h-1's scan, which is the steady state.
            q_vec = qvec[h];
            q_valid = 1;
            while (!q_ready) tick;
            tick;
            q_valid = 0;

            while (!tab_ready) tick;
            // Commit only once the previous scan has fully drained -- swapping
            // mid-scan would rewrite the query under a half-finished sum.
            while (out_valid) tick;
            tab_commit = 1; tick; tab_commit = 0;

            scan(h, h >= HEADS-4);
            while (head_out <= h) tick;
        end

        // -- the norms a real cache never produces -------------------------
        //
        // Query head 0's table is still committed, and these rows carry norms
        // at 0, 1, 2**15-1, 2**15 and 2**16-1. The last two are NEGATIVE once
        // `_unpack_word` sign-extends them, which is the rule the 2,080 real
        // rows above cannot test: every norm `isqrt` produced is far below
        // 2**15, so an unsigned lane agrees with all of them and is still wrong.
        q_vec = qvec[0];
        q_valid = 1; while (!q_ready) tick; tick; q_valid = 0;
        while (!tab_ready) tick;
        while (out_valid) tick;
        tab_commit = 1; tick; tab_commit = 0;

        edge_phase = 1;
        for (int i = 0; i < EDGE; i++) begin
            row_valid = 1;
            row_data  = edge_rows[i];
            @(posedge clk);
            while (!row_ready) begin tick; @(posedge clk); end
            tick;
        end
        row_valid = 0;
        while (edge_out < EDGE) tick;

        check(head_out == HEADS && tok_out == 0,
              "every head scored exactly ctx+1 tokens");
        check(clips == 0, "no score saturated at these widths");
        check(stalls > 50, "the stalled heads really were held by back-pressure");
        check(edge_out == EDGE, "every constructed norm row was scored");
        $display("  %0d heads x %0d tokens = %0d scores + %0d edge rows; %0d stalls",
                 HEADS, T, HEADS*T, EDGE, stalls);

        if (errors == 0) $display("=== tb_score_lane: ALL TESTS PASSED ===");
        else $display("=== tb_score_lane: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
