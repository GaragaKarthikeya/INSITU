// RMS of a rotated vector, at wire precision.  `ops/quantize.py::_norm_wire`.
//
// The norm is quantized before the thresholds are built from it
// -------------------------------------------------------------
// The decoder only ever sees the norm at Q(NORM_FRAC).  If `rot_encode`
// thresholded against a wider norm than the one written to the cache row, the
// encoder and the decoder would be working from different numbers and a
// channel near a boundary would decode into the wrong bin.  So this block is
// the only producer of the norm: it rounds to the wire format, and everything
// downstream -- the thresholds included -- consumes what it emitted.
//
// Rounding is half-to-even, not half-up.  Norms are always positive, so a
// floor or a half-up biases every one of them the same way and the bias
// accumulates across a whole cache instead of averaging out.
`timescale 1ns/1ps
module rot_norm #(
    parameter int D         = 64,
    parameter int IN_BITS   = 24,   // Q(QK_FRAC) lanes, straight off rot_fwht
    parameter int QK_FRAC   = 16,
    parameter int NORM_BITS = 16,
    parameter int NORM_FRAC = 9,
    // Squarers instantiated.  D/LANES cycles per vector, and the only
    // area/throughput knob in the block.  Phase A runs once per token against
    // a Phase B that runs once per cached token, so this is deliberately not
    // 64: 64 squarers would buy back cycles that nothing is waiting on.
    parameter int LANES     = 8
) (
    input  logic clk,
    input  logic rstn,
    input  logic in_valid,
    output logic in_ready,
    input  logic signed [D*IN_BITS-1:0] in_vec,
    output logic out_valid,
    input  logic out_ready,
    output logic [NORM_BITS-1:0] out_norm,
    // The Q-format clamp fired.  `Q.clamp` saturates silently in Python, so
    // this is not an error -- it is the counter that says the format is being
    // used at its edge, which is what the saturation gate measures.
    output logic out_saturated
);
    localparam int LOG2D  = $clog2(D);
    localparam int SQ_W   = 2*IN_BITS + LOG2D;      // 64 * (2**23)**2 = 2**52
    localparam int ROOT_W = (SQ_W - LOG2D)/2 + 1;   // floor(sqrt(2**46)) needs 24 b
    localparam int RAD_W  = 2*ROOT_W;
    localparam int REM_W  = ROOT_W + 3;
    localparam int SHIFT  = QK_FRAC - NORM_FRAC;
    localparam int STEPS  = D/LANES;
    localparam logic [ROOT_W:0] NORM_MAX = (1 << (NORM_BITS-1)) - 1;

    generate
        if (D % LANES != 0) initial $fatal(1, "LANES must divide D");
        if (SHIFT <= 0)
            initial $fatal(1, "NORM_FRAC must be below QK_FRAC: a left shift has no rounding rule to get wrong");
    endgenerate

    // One enable for the whole block.  The square root is a fixed-latency
    // pipeline with no internal elasticity, so it freezes as a unit or drops a
    // vector the cycle a downstream block deasserts out_ready.
    wire advance = !out_valid || out_ready;

    // -- accumulate --------------------------------------------------------
    // Three registered stages, not one: square -> sum -> mean.  Fusing them
    // put a 64:8 lane mux, a DSP multiply, a 53-bit adder tree and the first
    // square-root subtract in the same cycle, which measured -0.990 ns at
    // 250 MHz.  Splitting costs two cycles of latency on a block that runs
    // once per token and nothing is waiting on.
    logic [$clog2(STEPS):0] step;
    logic busy;
    logic [2*IN_BITS-1:0] prod [0:LANES-1];
    // The selected lanes, registered.  `step` fans out across all 64 words of
    // `held` and the squarers are DSPs, so reading through that mux and into
    // the multiply in one cycle left 20 endpoints at -0.011 ns once the whole
    // block was placed.  One more cycle on a unit that runs once per token.
    logic signed [IN_BITS-1:0] lane_q [0:LANES-1];
    logic p_valid, p_first, p_last;
    logic q_valid, q_first, q_last;
    logic [SQ_W-1:0] acc, tree, sum_next;
    logic [RAD_W-1:0] mean;
    logic mean_valid;
    logic signed [D*IN_BITS-1:0] held;
    wire  acc_done = busy && (step == STEPS-1);

    always_comb begin
        tree = '0;
        for (int l = 0; l < LANES; l++)
            tree = tree + SQ_W'(prod[l]);
        // `p_first` and not `!busy`: by the time the first group's squares
        // reach this adder the accumulator is already several cycles into the
        // Next vector's schedule, and restarting from a stale acc there is the
        // kind of bug that only appears on the second vector.
        sum_next = (p_first ? '0 : acc) + tree;
    end

    assign in_ready = advance && !busy;

    logic signed [IN_BITS-1:0] lane_x [0:LANES-1];
    always_comb
        for (int l = 0; l < LANES; l++)
            lane_x[l] = held[(step*LANES + l)*IN_BITS +: IN_BITS];

    always_ff @(posedge clk) if (advance) begin
        for (int l = 0; l < LANES; l++) begin
            lane_q[l] <= lane_x[l];
            prod[l]   <= lane_q[l] * lane_q[l];
        end
        if (p_valid) begin
            acc <= sum_next;
            if (p_last) mean <= RAD_W'(sum_next >> LOG2D);   // D is 2**LOG2D
        end
    end

    // -- pipelined integer square root -------------------------------------
    // Digit by digit, MSB first, two operand bits per stage: exactly the
    // `floor(sqrt)` of `ops/quantize.py::isqrt`, which is why no float seed and
    // no correction step appear here.  One stage per output bit, so a vector
    // retires every STEPS+1 cycles rather than every ROOT_W.
    logic [ROOT_W-1:0] root [1:ROOT_W];
    logic [REM_W-1:0]  rem  [1:ROOT_W];
    logic [RAD_W-1:0]  rad  [1:ROOT_W];
    logic [ROOT_W-1:0] sq_valid;

    // Stage 0 reads the registered mean, so the only logic between two flops
    // here is one compare and one subtract.
    wire [RAD_W-1:0] rad0 = mean;

    generate
        genvar st;
        for (st = 0; st < ROOT_W; st = st + 1) begin : sqrt_stage
            // `prev` keeps the index inside the array on stage 0, whose real
            // inputs are the seed wires; a bare rad[st] there reads element 0,
            // which does not exist.
            localparam int PREV = (st == 0) ? 1 : st;
            wire [RAD_W-1:0]  rad_in  = (st == 0) ? rad0 : rad[PREV];
            wire [REM_W-1:0]  rem_in  = (st == 0) ? {REM_W{1'b0}} : rem[PREV];
            wire [ROOT_W-1:0] root_in = (st == 0) ? {ROOT_W{1'b0}} : root[PREV];
            wire [REM_W-1:0] shifted = {rem_in[REM_W-3:0], rad_in[RAD_W-1 -: 2]};
            wire [REM_W-1:0] trial   = {root_in, 2'b01};
            wire take = shifted >= trial;
            always_ff @(posedge clk) if (advance) begin
                rem[st+1]  <= take ? (shifted - trial) : shifted;
                root[st+1] <= {root_in[ROOT_W-2:0], take};
                rad[st+1]  <= rad_in << 2;
            end
        end
    endgenerate

    // -- requantize to Q(NORM_BITS, NORM_FRAC) -----------------------------
    wire [ROOT_W-1:0] r        = root[ROOT_W];
    wire [ROOT_W-1:0] q_up     = (r + (1 << (SHIFT-1))) >> SHIFT;
    wire tie                   = r[SHIFT-1:0] == (1 << (SHIFT-1));
    wire [ROOT_W-1:0] rounded  = tie ? {q_up[ROOT_W-1:1], 1'b0} : q_up;
    wire sat                   = {1'b0, rounded} > NORM_MAX;

    always_ff @(posedge clk) begin
        if (!rstn) begin
            busy <= 1'b0; step <= '0; sq_valid <= '0;
            q_valid <= 1'b0; q_first <= 1'b0; q_last <= 1'b0;
            p_valid <= 1'b0; p_first <= 1'b0; p_last <= 1'b0; mean_valid <= 1'b0;
        end else if (advance) begin
            if (!busy) begin
                busy <= in_valid;
                step <= '0;
                if (in_valid) held <= in_vec;
            end else begin
                step <= step + 1'b1;
                if (acc_done) busy <= 1'b0;
            end
            // Two stages, matching the select-then-square pipe above: these
            // mark what `prod` holds, and `prod` is now one cycle further back.
            q_valid <= busy;
            q_first <= busy && (step == 0);
            q_last  <= acc_done;
            p_valid <= q_valid;
            p_first <= q_first;
            p_last  <= q_last;
            mean_valid <= p_valid && p_last;
            sq_valid <= {sq_valid[ROOT_W-2:0], mean_valid};
        end
    end

    assign out_valid     = sq_valid[ROOT_W-1];
    assign out_norm      = sat ? NORM_BITS'(NORM_MAX) : NORM_BITS'(rounded);
    assign out_saturated = out_valid && sat;
endmodule
