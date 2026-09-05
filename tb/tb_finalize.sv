// Self-checking testbench for finalize.sv.
//
// THE GOLDENS ARE `reciprocal` AND `_finalize`
// --------------------------------------------
// `fin_acc.hex` and `fin_l.hex` are the accumulator and denominator each of
// the 32 query heads actually ended its scan on, tapped from `attend_online`
// by an observer. `fin_recip.hex` and `fin_out.hex` are what the kernel's own
// two lines produced from them. Nothing here recomputes a division.
//
// THE REAL DENOMINATORS DO NOT EXERCISE THE DIVIDER
// -------------------------------------------------
// All 32 heads end a 65-token scan with `l` near 65 * 32768, so all 32 land on
// the same normalised quotient to within 0.25% and on the SAME shift. A
// divider wrong at any other magnitude would agree with every real head. So
// the last 17 rows are constructed.
//
// WHAT THEY STRESS IS THE SHIFT, NOT THE QUOTIENT'S WIDTH
// -------------------------------------------------------
// They used to span "the reciprocal's whole range from 2**31 down to zero",
// which was the old fixed-Q16 divide: there the quotient's WIDTH tracked the
// context, and that is precisely the defect the normalised divide removes.
// Every quotient is now 31 bits. The degree of freedom is `lead`, the position
// of `l`'s leading one, which sets both the normalising shift and the output
// shift -- so the constructed rows sweep it from 0 to L_WIDTH-1 and sit on the
// `shr`/`shl` boundary at 31 where the normaliser changes direction.
//
// The last three are the denominators of the contexts the rewrite was made
// for: l = T * 2**15 at ctx 32,768, 131,072 and 1,048,576. At the middle one
// the old divide returned zero and took every output channel with it.
//
// Channel 0 of each is a saturated accumulator and channel 1 its negative, so
// the widest product this block can form is formed on every one.
//
// l = 0 IS NOT AN ERROR CASE
// --------------------------
// A masked or padded position scores nothing and its denominator is zero.
// `reciprocal` answers zero rather than dividing; so must this.
`timescale 1ns/1ps
module tb_finalize;
    localparam int D   = 64;
    localparam int AW  = 32;
    localparam int LW  = 48;
    localparam int OB  = 48;
    localparam int RB  = 32;
    localparam int SEAM = 24;
    localparam int N    = 49;      // 32 real heads + 17 constructed
    localparam int REAL = 32;

    logic clk = 0, rstn = 0;
    always #2 clk = ~clk;

    logic [D*AW-1:0] g_acc   [0:N-1];
    logic [LW-1:0]   g_l     [0:N-1];
    logic [RB-1:0]   g_recip [0:N-1];
    logic [7:0]      g_shift [0:N-1];
    logic [D*OB-1:0] g_out   [0:N-1];

    int errors = 0, seen = 0, range_seen = 0;

    logic in_valid, out_ready;
    logic in_ready, out_valid, range_error;
    logic [D*AW-1:0] in_acc;
    logic [LW-1:0] in_l;
    logic [D*OB-1:0] out_vec;
    logic [RB-1:0] out_recip;
    logic [$clog2(LW)+1:0] out_shift;

    finalize dut (
        .clk, .rstn, .in_valid, .in_ready, .in_acc, .in_l,
        .out_valid, .out_ready, .out_vec, .out_recip, .out_shift, .range_error);

    task check(input logic cond, input string what);
        if (!cond) begin
            $display("  FAIL: %s", what);
            errors = errors + 1;
        end
    endtask

    task automatic tick; @(negedge clk); endtask

    // A row is in range exactly when every channel of its golden fits the seam.
    function automatic logic fits_seam(input int row);
        logic signed [OB-1:0] v;
        fits_seam = 1'b1;
        for (int ch = 0; ch < D; ch++) begin
            v = $signed(g_out[row][ch*OB +: OB]);
            if (v >= (OB'(1) << (SEAM-1)) || v < -(OB'(1) << (SEAM-1)))
                fits_seam = 1'b0;
        end
    endfunction

    int cycles, cyc_min = 9999, cyc_max = 0;
    int sh_min = 9999, sh_max = 0;
    initial begin
        $readmemh("tb/vectors/fin_acc.hex", g_acc);
        $readmemh("tb/vectors/fin_l.hex", g_l);
        $readmemh("tb/vectors/fin_recip.hex", g_recip);
        $readmemh("tb/vectors/fin_shift.hex", g_shift);
        $readmemh("tb/vectors/fin_out.hex", g_out);

        in_valid = 0; out_ready = 1; in_acc = '0; in_l = '0;
        repeat (4) @(posedge clk);
        rstn = 1;

        for (int r = 0; r < N; r++) begin
            tick;
            in_acc = g_acc[r]; in_l = g_l[r]; in_valid = 1;
            @(posedge clk);
            while (!in_ready) begin tick; @(posedge clk); end
            tick; in_valid = 0;

            cycles = 0;
            while (!out_valid) begin tick; cycles = cycles + 1; end
            if (cycles < cyc_min) cyc_min = cycles;
            if (cycles > cyc_max) cyc_max = cycles;

            check(out_recip === g_recip[r],
                  $sformatf("row %0d (l=%0d): recip %0d != %0d", r, g_l[r],
                            out_recip, g_recip[r]));
            check(out_shift === g_shift[r],
                  $sformatf("row %0d (l=%0d): shift %0d != %0d", r, g_l[r],
                            out_shift, g_shift[r]));
            if (out_shift < sh_min) sh_min = out_shift;
            if (out_shift > sh_max) sh_max = out_shift;
            for (int ch = 0; ch < D; ch++)
                check(out_vec[ch*OB +: OB] === g_out[r][ch*OB +: OB],
                      $sformatf("row %0d ch %0d: %0d != %0d", r, ch,
                                $signed(out_vec[ch*OB +: OB]),
                                $signed(g_out[r][ch*OB +: OB])));
            // The flag is a claim about the seam, so it is checked against the
            // golden's own width -- not merely observed.
            check(range_error === !fits_seam(r),
                  $sformatf("row %0d: range_error %0d but fits_seam %0d",
                            r, range_error, fits_seam(r)));
            if (range_error) range_seen = range_seen + 1;
            if (r < REAL)
                check(!range_error,
                      $sformatf("head %0d is a real head and must fit the seam", r));
            seen = seen + 1;
            tick; @(posedge clk);      // consume
        end

        check(seen == N, "every row was finalized");
        check(range_seen > 0, "the constructed rows must overflow the seam");
        // The normalising shift must actually have been swept, or the rows
        // above stopped testing the thing they were rewritten to test.
        check(sh_min <= 16 && sh_max >= 60,
              $sformatf("shifts only spanned %0d..%0d", sh_min, sh_max));
        $display("  %0d rows (%0d real, %0d constructed), %0d..%0d cycles",
                 N, REAL, N-REAL, cyc_min, cyc_max);
        // NOT a range: the normalised divide makes every quotient 31 bits, so
        // what varies row to row is the SHIFT, not the reciprocal. Printing two
        // reciprocals as "from X to Y" was true of the old fixed-Q16 divide and
        // would now print the same number twice.
        $display("  shifts %0d..%0d, reciprocals all %0d b; %0d rows exceeded the 24-bit seam",
                 sh_min, sh_max, $clog2(g_recip[0]) + 1, range_seen);

        if (errors == 0) $display("=== tb_finalize: ALL TESTS PASSED ===");
        else $display("=== tb_finalize: %0d TESTS FAILED ===", errors);
        $finish;
    end
endmodule
