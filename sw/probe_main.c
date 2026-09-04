/*
 * DDR bandwidth sweep -- baremetal A53, implementation-plan step 1.
 *
 * Answers the one question the whole attention design is sized from:
 *
 *     does PS DDR4-2400 sustain the 13.0 GB/s that PARALLEL_KV=1 needs?
 *
 * 52 B/cycle at 250 MHz is the appetite of one score-lane group. If the
 * measured number is materially below it, PARALLEL_KV=1 is starved and every
 * throughput figure in the plan is optimistic. Nothing downstream should be
 * built until this prints.
 *
 * The access pattern deliberately mirrors a KV-cache scan: long sequential
 * reads, disjoint per port, no writes. It is NOT a general memcpy benchmark
 * and its number should not be quoted as one.
 *
 * Build with the Vitis SDT flow (see scripts/build_vitis_probe.py), run with
 *     xsdb host/run_probe.tcl
 * and capture UART0 on /dev/ttyUSB1.
 */
#include <stdio.h>
#include "xil_printf.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"

/* Set from the block design's address map. Override at build time if the
 * ADDRESS MAP printed by create_bd_probe.tcl says otherwise. */
#ifndef PROBE_BASE
#define PROBE_BASE 0xA0000000u
#endif

#define R_CTRL     0x00u   /* W: bit0 start.  R: bit0 busy, bit1 run */
#define R_MASK     0x04u
#define R_BASE_LO  0x08u
#define R_BASE_HI  0x0Cu
#define R_STRIDE   0x10u
#define R_BEATS    0x14u
#define R_BURST    0x18u
#define R_OUTST    0x1Cu
#define R_CYCLES   0x20u
#define R_GOT0     0x24u
#define R_GOT1     0x28u
#define R_GOT2     0x2Cu
#define R_MAGIC    0x30u
#define R_GOT3     0x34u
#define R_NPORT    0x38u

#define WR(o, v) Xil_Out32(PROBE_BASE + (o), (u32)(v))
#define RD(o)    Xil_In32(PROBE_BASE + (o))

/* Region the probe reads. Well above the ~2 MB the application itself uses,
 * and 4 KB aligned so no burst crosses a page boundary. Four ports x 64 MB
 * stride = 256 MB, comfortably inside the 2 GB of PS DDR4. */
#define REGION_BASE   0x10000000u
#define REGION_STRIDE 0x04000000u
#define BEAT_BYTES    16u              /* 128-bit HP port */

/* PL clock. MUST match the "actual" figure create_bd_probe.tcl printed, not
 * the requested one -- the PS PLL rounds, and reporting GB/s against a clock
 * the design is not running at is how a probe lies. */
#ifndef PL_CLK_HZ
#define PL_CLK_HZ 250000000u
#endif

static int popcount4(unsigned m)
{
    int n = 0;
    for (int i = 0; i < 4; i++) if (m & (1u << i)) n++;
    return n;
}

/* Returns MB/s * 100 (fixed point, to avoid pulling in soft float printf). */
static unsigned long run_one(unsigned mask, unsigned beats_per_port,
                             unsigned burst, unsigned outst,
                             unsigned *cycles_out, unsigned *beats_out)
{
    WR(R_MASK,    mask);
    WR(R_BASE_LO, REGION_BASE);
    WR(R_BASE_HI, 0);
    WR(R_STRIDE,  REGION_STRIDE);
    WR(R_BEATS,   beats_per_port);
    WR(R_BURST,   burst);
    WR(R_OUTST,   outst);
    WR(R_CTRL,    1);

    /* The hardware clears bit1 when the last enabled engine goes idle. */
    unsigned guard = 0;
    while (RD(R_CTRL) & 0x2u) {
        if (++guard > 200000000u) { xil_printf("  HUNG\r\n"); break; }
    }

    unsigned cyc = RD(R_CYCLES);
    unsigned got = RD(R_GOT0) + RD(R_GOT1) + RD(R_GOT2) + RD(R_GOT3);
    *cycles_out = cyc;
    *beats_out  = got;
    if (cyc == 0) return 0;

    /* bytes/s = got*16 * f / cyc.  Rearranged to stay in 64-bit integers and
     * to avoid overflow: (got * 16) * (f/1e6) / cyc  gives MB/s directly. */
    unsigned long long bytes = (unsigned long long)got * BEAT_BYTES;
    unsigned long long mbps  = bytes * (PL_CLK_HZ / 1000000ull) / cyc;
    return (unsigned long)mbps;
}

static void banner(void)
{
    xil_printf("\r\n=== ZCU104 DDR bandwidth probe ===\r\n");
    xil_printf("PL clock %u Hz, beat %u B\r\n", PL_CLK_HZ, BEAT_BYTES);
    xil_printf("target: 13000 MB/s (52 B/cyc at 250 MHz, PARALLEL_KV=1)\r\n");
    xil_printf("one 128-bit HP port ceiling: %u MB/s;  4 ports: %u MB/s\r\n\r\n",
               (unsigned)((unsigned long long)BEAT_BYTES * PL_CLK_HZ / 1000000ull),
               (unsigned)(4ull * BEAT_BYTES * PL_CLK_HZ / 1000000ull));
}

int main(void)
{
    /* The probe masters read DDR directly. Anything the A53 has dirty in its
     * caches for that region would not be visible to them, and stale lines
     * could be written back underneath the run. Flush once, up front. */
    Xil_DCacheFlush();

    banner();

    unsigned magic = RD(R_MAGIC);
    if (magic != 0x44445250u) {
        xil_printf("BAD MAGIC %08x at %08x -- check PROBE_BASE against the\r\n"
                   "ADDRESS MAP printed by create_bd_probe.tcl\r\n",
                   magic, PROBE_BASE);
        return 1;
    }
    xil_printf("magic ok, %u ports\r\n\r\n", RD(R_NPORT));

    const unsigned beats  = 1u << 20;          /* 16 MB per port */
    const unsigned bursts[]  = {16, 64, 128, 256};
    const unsigned outsts[]  = {1, 2, 4, 8, 16, 32};
    const unsigned masks[]   = {0x1, 0x3, 0x7, 0xF};

    /* EVERY CONFIG IS REPEATED AND THE MEDIAN REPORTED.
     *
     * The first pass of this sweep produced one 4-port point at 14678 MB/s
     * whose immediate neighbours (same ports, same burst, deeper queue) all sat
     * near 11800. A deeper queue cannot physically reduce throughput here, so
     * that reading was an artifact -- DRAM refresh or bank/row alignment
     * happening to fall well for one run.
     *
     * Reporting the MAXIMUM over a sweep systematically selects for exactly
     * that kind of luck, and the whole design would then be sized from a number
     * the hardware cannot reproduce. The median over repeats is the honest
     * statistic; min..max is printed alongside so the spread stays visible
     * rather than being hidden by the summary. */
#define REPS 5
    xil_printf("ports burst outst  medMB/s  perport  eff%%      min      max\r\n");
    xil_printf("-------------------------------------------------------------\r\n");

    unsigned long best = 0;
    unsigned best_p = 0, best_b = 0, best_o = 0;

    for (unsigned mi = 0; mi < sizeof(masks)/sizeof(masks[0]); mi++) {
        int np = popcount4(masks[mi]);
        unsigned long ceil_mbps =
            (unsigned long)((unsigned long long)np * BEAT_BYTES * PL_CLK_HZ / 1000000ull);
        for (unsigned bi = 0; bi < sizeof(bursts)/sizeof(bursts[0]); bi++) {
            for (unsigned oi = 0; oi < sizeof(outsts)/sizeof(outsts[0]); oi++) {
                unsigned long s[REPS];
                int bad = 0;
                for (int r = 0; r < REPS; r++) {
                    unsigned cyc = 0, got = 0;
                    s[r] = run_one(masks[mi], beats, bursts[bi], outsts[oi],
                                   &cyc, &got);
                    if (got != beats * (unsigned)np) {
                        xil_printf("%5d %5u %5u   BEAT MISMATCH got %u want %u\r\n",
                                   np, bursts[bi], outsts[oi], got, beats * np);
                        bad = 1;
                        break;
                    }
                }
                if (bad) continue;

                for (int a = 0; a < REPS; a++)          /* insertion sort */
                    for (int b = a + 1; b < REPS; b++)
                        if (s[b] < s[a]) { unsigned long t = s[a]; s[a] = s[b]; s[b] = t; }

                unsigned long med = s[REPS/2];
                unsigned eff = ceil_mbps ? (unsigned)(med * 100ul / ceil_mbps) : 0;
                xil_printf("%5d %5u %5u %8lu %8lu %5u %8lu %8lu\r\n",
                           np, bursts[bi], outsts[oi], med,
                           med / (unsigned long)np, eff, s[0], s[REPS-1]);
                if (med > best) {
                    best = med; best_p = np; best_b = bursts[bi], best_o = outsts[oi];
                }
            }
        }
    }

    xil_printf("\r\nBEST MEDIAN %lu MB/s at %u ports, burst %u, outstanding %u\r\n",
               best, best_p, best_b, best_o);
    if (best >= 13000ul) {
        xil_printf("GATE PASS: >= 13000 MB/s sustained. PARALLEL_KV=1 is fed.\r\n");
    } else {
        xil_printf("GATE FAIL: %lu MB/s < 13000 needed.\r\n", best);
        xil_printf("Options, in order of preference:\r\n");
        xil_printf("  a) drop the PL clock to %lu MHz -- 52 B/cyc then needs\r\n",
                   best * 1000000ul / 52ul / 1000000ul);
        xil_printf("     exactly what DDR delivers, and PARALLEL_KV=1 stays fed\r\n");
        xil_printf("  b) narrow the KV row: 3b/2b is 44 B/cyc = %lu MB/s at 250MHz\r\n",
                   44ul * (PL_CLK_HZ / 1000000ul));
        xil_printf("  c) accept a %lu%% starve at 250 MHz\r\n",
                   (13000ul - best) * 100ul / 13000ul);
    }
    xil_printf("=== done ===\r\n");
    return 0;
}
