/*
 * Step 13: the attention block driven by the A53, with no JTAG in the loop.
 *
 * WHAT CHANGED FROM STEP 12, AND WHAT DELIBERATELY DID NOT
 * --------------------------------------------------------
 * Step 12 ran one decode step from `xsdb` over the JTAG cable, ~1 ms per
 * transaction, and made no claim about time. Everything below the shim is
 * unchanged here -- same `attn_ctrl`, same four read masters, same write
 * master, same register map. The only thing replaced is the master driving
 * it: `jtag_axi` becomes the PS through M_AXI_HPM0_FPD. That is the whole
 * reason the shim was built as addressed memory rather than a FIFO window.
 *
 * WHAT THIS RUN IS FOR
 * --------------------
 * Three things step 12 could not answer:
 *
 *  1. MULTI-TOKEN. Each case loads a cache image holding tokens 0..ctx0-1 and
 *     NOTHING ELSE, then runs four decode steps on it. Token ctx0+n scores
 *     against n rows that exist only because the block's own AXI write master
 *     put them in DDR and the next scan's address generator found them again.
 *     A write path off by one plane or one token answers step 0 correctly and
 *     every step after it wrongly. Step 12, which ran one step, could not see
 *     that; `tb_attn`, which blanks one token, could not either.
 *
 *  2. MULTI-CONTEXT. 64, 256 and 1,024 cached tokens, so the plane spans and
 *     head strides the host programs are three different sets of numbers
 *     rather than the one set that happens to be all-4096 at ctx 64.
 *
 *  3. A DEVICE-SIDE TIME THAT MEANS SOMETHING. See below.
 *
 * TIME IS REPORTED IN THREE PARTS, AND ONLY ONE OF THEM IS THE BLOCK
 * ------------------------------------------------------------------
 * The A53 writes 2,304 words into the shim one store at a time over a 32-bit
 * AXI-Lite path, starts the block, then reads 1,536 words back the same way.
 * At ctx 64 that load is comparable to the scan itself. Reporting a single
 * "microseconds per token" here would be reporting the shim's ingress, not
 * attention, so the three are printed separately: load, run, read.
 *
 * And the run is timed TWICE, by two clocks that share nothing. `XTime` is the
 * A53's counter, read across a bus the block is using; `busy_cycles` and
 * `scan_cycles` are counted in the PL at the clock the DDR reads were issued
 * on. The bandwidth number may only be divided by the second -- the first
 * cannot resolve the scan inside a step -- and the two together give the PL
 * clock back, which is how a bitstream built at the wrong frequency announces
 * itself instead of quietly making every GB/s figure 40% too high.
 *
 * CACHE DISCIPLINE
 * ----------------
 * The block reaches DDR through S_AXI_HP, which is NOT coherent with the A53
 * caches. The image is memcpy'd with the D-cache on and then flushed, or the
 * block reads whatever DDR held before and every score is garbage from
 * perfectly correct hardware. Flush, not just clean-on-eviction: a dirty line
 * evicted later would land on top of a row the PL wrote itself.
 */
#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
/* 2026.1 (SDT flow) dropped xtime_l.h; XTime/XTime_GetTime/COUNTS_PER_SECOND
 * now come from xiltimer.h + xtimer_config.h. */
#if __has_include("xtime_l.h")
#include "xtime_l.h"
#else
#include "xiltimer.h"
#include "xtimer_config.h"
#endif

#include "xaxidma.h"

#include "attn_vectors.h"
#include "attn_server.h"
#include "attn_eth.h"
#include "attn_proto.h"

/* Pinned by `scripts/create_bd_attn_ps.tcl`, exactly as the JTAG design pinned
 * it, so this constant and that script are the same number in two files and
 * `MAGIC` is what says so. */
#define SHIM            0xA0000000UL
#define REG_CTRL        0x0000
#define REG_MAGIC       0x0004
#define REG_NTOK        0x0008
#define REG_BASE_LO     0x000C
#define REG_BASE_HI     0x0010
#define REG_STRIDE      0x0014
#define REG_SPAN        0x0018
#define REG_STARVE      0x001C
#define REG_CLIPS       0x0020
#define REG_OVF         0x0024
#define REG_IN_W        0x0028
#define REG_OUT_W       0x002C
#define REG_SCAN_CYC    0x0030
#define REG_BUSY_CYC    0x0034
#define REG_MODE        0x0038      /* 0 = buffers, 1 = the DMA stream */

/* Pinned by `create_bd_attn_dma.tcl`. 0xA0100000 and not 0xA0010000: the
 * shim's window is 128 KB, so anything below 0xA0020000 is inside it. */
#define DMA_BASE        0xA0100000UL

#define TOKEN_BYTES     (ATTN_IN_WORDS * 4)    /* 9,216 */
#define RESULT_BYTES    (ATTN_OUT_WORDS * 4)   /* 6,144 */
#define IN_BASE         0x2000
#define OUT_BASE        0x8000
#define MAGIC           0xA77E0001U

#define ST_BUSY         0x1
#define ST_DONE         0x2
#define ST_RANGE        0x4
#define ST_NORMSAT      0x8

/* Where the cache image goes. Clear of the ELF, which the linker puts at the
 * bottom of DDR, and 4 KB aligned because every plane base must be -- the
 * layout guarantees alignment RELATIVE to the base, so a misaligned base
 * misaligns all thirty-two planes at once. */
#define CACHE_BASE      0x10000000UL

/* What the bitstream was built for. Not trusted: `implied_hz` below is
 * measured, and a mismatch is printed rather than assumed away. */
#define PL_HZ_NOMINAL   250000000ULL

static inline u32 rd(u32 off)          { return Xil_In32(SHIM + off); }
static inline void wr(u32 off, u32 v)  { Xil_Out32(SHIM + off, v); }

static u32 outbuf[ATTN_OUT_WORDS];

/* DMA buffers, 64-byte aligned so cache maintenance covers exactly them. */
static u8 dma_tx[TOKEN_BYTES]  __attribute__((aligned(64)));
static u8 dma_rx[RESULT_BYTES] __attribute__((aligned(64)));
static XAxiDma dma;
static int dma_ok;

/* One decode step through the DMA. Returns wrong words, or a negative code.
 *
 * THE RECEIVE SIDE IS ARMED FIRST, AND THE BLOCK IS STARTED LAST.
 * The core emits its first egress beat as soon as group 0 has drained, so an
 * unarmed S2MM would stall the stream mid-step. Arming MM2S before the block
 * is started is safe on purpose: `attn_ctrl` holds `s_axis_tready` low until
 * `busy`, because `attn_ingress` clears its beat counter on `start` and a beat
 * accepted before that is swallowed and thrown away.
 */
static int run_step_dma(const attn_case_t *c, unsigned step,
                        u64 *us_load, u64 *us_run, u64 *us_read,
                        u32 *scan_cyc, u32 *busy_cyc, u32 *status)
{
    const u32 *in   = c->ingress + (u64)step * ATTN_IN_WORDS;
    const u32 *gold = c->golden  + (u64)step * ATTN_OUT_WORDS;
    XTime t0, t1;
    u32 st;

    XTime_GetTime(&t0);
    memcpy(dma_tx, in, TOKEN_BYTES);
    /* HPC0 is the coherent port, so this flush is belt and braces rather than
     * load-bearing -- but the standalone BSP maps DDR non-shareable, which
     * means the snoop is not something to rely on without checking. Flushing
     * 15 KB costs microseconds and removes the question. */
    Xil_DCacheFlushRange((UINTPTR)dma_tx, TOKEN_BYTES);
    Xil_DCacheFlushRange((UINTPTR)dma_rx, RESULT_BYTES);
    XTime_GetTime(&t1);
    *us_load = ((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND;

    wr(REG_NTOK,    c->n_tokens[step]);
    wr(REG_BASE_LO, (u32)CACHE_BASE);
    wr(REG_BASE_HI, (u32)(CACHE_BASE >> 32));
    wr(REG_STRIDE,  c->head_stride);
    wr(REG_SPAN,    c->plane_span);

    XTime_GetTime(&t0);
    if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)dma_rx, RESULT_BYTES,
                               XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
        return -4;
    if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)dma_tx, TOKEN_BYTES,
                               XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
        return -5;

    wr(REG_CTRL, 1);
    {
        u32 guard = 100000000;
        do {
            st = rd(REG_CTRL);
        } while (!(st & ST_DONE) && --guard);
        if (!guard) return -2;
        while ((XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA) ||
                XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE)) && --guard) { }
        if (!guard) return -6;
    }
    XTime_GetTime(&t1);
    /* One number, not three: with the DMA the transfer IS the step -- it
     * overlaps the scan instead of bracketing it, so there is no separate
     * load and read to report. */
    *us_run = ((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND;
    *us_read = 0;
    *status = st;
    *scan_cyc = rd(REG_SCAN_CYC);
    *busy_cyc = rd(REG_BUSY_CYC);

    Xil_DCacheInvalidateRange((UINTPTR)dma_rx, RESULT_BYTES);
    memcpy(outbuf, dma_rx, RESULT_BYTES);

    int bad = 0;
    for (unsigned i = 0; i < ATTN_OUT_WORDS; i++)
        if (outbuf[i] != gold[i]) {
            if (bad < 4)
                xil_printf("    dma word %4u got %08x exp %08x\r\n",
                           i, (unsigned)outbuf[i], (unsigned)gold[i]);
            bad++;
        }
    return bad;
}

/* One decode step. Returns the number of wrong words, or a negative code. */
static int run_step(const attn_case_t *c, unsigned step,
                    u64 *us_load, u64 *us_run, u64 *us_read,
                    u32 *scan_cyc, u32 *busy_cyc, u32 *status)
{
    const u32 *in   = c->ingress + (u64)step * ATTN_IN_WORDS;
    const u32 *gold = c->golden  + (u64)step * ATTN_OUT_WORDS;
    XTime t0, t1;
    u32 st;

    /* -- load the token ------------------------------------------------- */
    XTime_GetTime(&t0);
    for (unsigned i = 0; i < ATTN_IN_WORDS; i++)
        Xil_Out32(SHIM + IN_BASE + i * 4, in[i]);
    XTime_GetTime(&t1);
    *us_load = ((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND;

    /* Read the window back before starting. The host that cannot trust what it
     * loaded cannot interpret what it gets out -- and unlike JTAG, a dropped
     * store here would mean the AXI-Lite path itself is wrong, which is worth
     * finding on step 0 rather than in a wrong answer on step 3. */
    if (step == 0) {
        for (unsigned i = 0; i < ATTN_IN_WORDS; i++)
            if (Xil_In32(SHIM + IN_BASE + i * 4) != in[i]) {
                xil_printf("    input word %u read back %08x, wrote %08x\r\n",
                           i, (unsigned)Xil_In32(SHIM + IN_BASE + i * 4),
                           (unsigned)in[i]);
                return -1;
            }
    }

    /* -- the geometry of this step -------------------------------------- */
    wr(REG_NTOK,    c->n_tokens[step]);
    wr(REG_BASE_LO, (u32)CACHE_BASE);
    wr(REG_BASE_HI, (u32)(CACHE_BASE >> 32));
    wr(REG_STRIDE,  c->head_stride);
    wr(REG_SPAN,    c->plane_span);

    /* -- go -------------------------------------------------------------- */
    XTime_GetTime(&t0);
    wr(REG_CTRL, 1);
    {
        /* `done` is latched, so this loop cannot miss it; the guard is against
         * a block that never finishes, which is what an uninitialised DRAM
         * controller looks like. */
        u32 guard = 100000000;
        do {
            st = rd(REG_CTRL);
        } while (!(st & ST_DONE) && --guard);
        if (!guard) return -2;
    }
    XTime_GetTime(&t1);
    *us_run = ((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND;
    *status = st;

    *scan_cyc = rd(REG_SCAN_CYC);
    *busy_cyc = rd(REG_BUSY_CYC);

    /* -- read the answer -------------------------------------------------- */
    XTime_GetTime(&t0);
    for (unsigned i = 0; i < ATTN_OUT_WORDS; i++)
        outbuf[i] = Xil_In32(SHIM + OUT_BASE + i * 4);
    XTime_GetTime(&t1);
    *us_read = ((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND;

    int bad = 0;
    for (unsigned i = 0; i < ATTN_OUT_WORDS; i++)
        if (outbuf[i] != gold[i]) {
            if (bad < 4)
                xil_printf("    word %4u got %08x exp %08x\r\n",
                           i, (unsigned)outbuf[i], (unsigned)gold[i]);
            bad++;
        }
    return bad;
}

/* One case: load the image, then run every step on it without reloading. */
static int run_case(const attn_case_t *c, int use_dma)
{
    u64 tot_scan = 0, tot_busy = 0, tot_rows = 0, tot_us_run = 0;
    u64 tot_us_load = 0, tot_us_read = 0, tot_starve = 0;
    int fails = 0;

    xil_printf("--- %s [%s]: %u cached tokens, %u steps, "
               "image %u B (stride %u, span %u)\r\n",
               c->name, use_dma ? "DMA" : "AXI-Lite", c->ctx0, c->steps,
               c->image_bytes, c->head_stride, c->plane_span);
    wr(REG_MODE, use_dma ? 1 : 0);
    if ((rd(REG_MODE) & 1u) != (u32)(use_dma ? 1 : 0)) {
        xil_printf("    MODE did not take\r\n");
        return 1;
    }

    memcpy((void *)CACHE_BASE, c->image, c->image_bytes);
    Xil_DCacheFlushRange((UINTPTR)CACHE_BASE, c->image_bytes);

    /* The image really did reach DDR. A memcpy into an unbrought-up controller
     * succeeds silently and the first symptom is a wrong answer five steps
     * later. Reading back through the same cached mapping would agree with
     * itself, so this reads the last word of the last plane -- the address
     * furthest from anything the copy touched first. */
    {
        volatile u32 *tail = (volatile u32 *)(CACHE_BASE + c->image_bytes - 4);
        u32 want = ((const u32 *)c->image)[c->image_bytes / 4 - 1];
        Xil_DCacheInvalidateRange((UINTPTR)tail, 4);
        if (*tail != want) {
            xil_printf("    DDR readback %08x, wrote %08x -- the image is not "
                       "in memory\r\n", (unsigned)*tail, (unsigned)want);
            return 1;
        }
    }

    for (unsigned s = 0; s < c->steps; s++) {
        u64 ul = 0, ur = 0, ud = 0;
        u32 scan = 0, busy = 0, st = 0;
        int bad = use_dma ? run_step_dma(c, s, &ul, &ur, &ud, &scan, &busy, &st)
                          : run_step(c, s, &ul, &ur, &ud, &scan, &busy, &st);
        u32 starve = rd(REG_STARVE), clips = rd(REG_CLIPS), ovf = rd(REG_OVF);

        if (bad < 0) {
            xil_printf("  step %u  ERROR %d (status %08x)\r\n", s, bad,
                       (unsigned)st);
            fails++;
            continue;
        }
        /* Rows this step's scan read: eight KV heads over ctx+1 tokens. */
        u64 rows = (u64)c->n_kv_heads * c->n_tokens[s];
        u64 bytes = rows * c->row_bytes;
        u32 mbps = scan ? (u32)((bytes * PL_HZ_NOMINAL) / scan / 1000000ULL) : 0;

        xil_printf("  step %u  ctx %4u  %s  scan %6u cyc  busy %6u cyc  "
                   "%4u MB/s  starve %4u  clips %u  ovf %u  "
                   "load %3u us  run %3u us  read %3u us\r\n",
                   s, (unsigned)c->n_tokens[s], bad ? "FAIL" : "ok ",
                   (unsigned)scan, (unsigned)busy, (unsigned)mbps,
                   (unsigned)starve, (unsigned)clips, (unsigned)ovf,
                   (unsigned)ul, (unsigned)ur, (unsigned)ud);

        if (bad) {
            xil_printf("    %d of %u words wrong\r\n", bad, ATTN_OUT_WORDS);
            fails++;
        }
        if (st & ST_RANGE) {
            xil_printf("    range_error: a channel escaped the 24-bit seam\r\n");
            fails++;
        }
        if (st & ST_NORMSAT)
            xil_printf("    WARNING: a Q7.9 norm hit its clamp\r\n");
        if (clips)
            xil_printf("    WARNING: %u scores saturated\r\n", (unsigned)clips);
        if (ovf) {
            xil_printf("    overflow: %u accumulator channels wrapped\r\n",
                       (unsigned)ovf);
            fails++;
        }

        tot_scan += scan; tot_busy += busy; tot_rows += rows;
        tot_us_run += ur; tot_us_load += ul; tot_us_read += ud;
        tot_starve += starve;
    }

    if (tot_scan) {
        u64 bytes = tot_rows * c->row_bytes;
        u32 mbps  = (u32)((bytes * PL_HZ_NOMINAL) / tot_scan / 1000000ULL);
        /* The PL clock, back out of two clocks that share nothing. If this is
         * not near 250 the bitstream is not the one the MB/s above assume. */
        u32 implied = tot_us_run ? (u32)(tot_busy / tot_us_run) : 0;
        /* Cycles per row: 1.00 is one row retired per cycle, which is what
         * `kv_store_ddr` was sized for. Printed in hundredths -- xil_printf has
         * no %f. */
        u32 cpr100 = (u32)((tot_scan * 100) / tot_rows);
        xil_printf("  %s TOTAL: %u rows, %u scan cyc (%u.%02u cyc/row), "
                   "%u MB/s, PL clock ~%u MHz\r\n",
                   c->name, (unsigned)tot_rows, (unsigned)tot_scan,
                   (unsigned)(cpr100 / 100), (unsigned)(cpr100 % 100),
                   (unsigned)mbps, (unsigned)implied);
        xil_printf("  %s TIME:  load %u us, run %u us, read %u us "
                   "over %u steps\r\n", c->name, (unsigned)tot_us_load,
                   (unsigned)tot_us_run, (unsigned)tot_us_read, c->steps);
        /* THE NUMBER THAT ACTUALLY TESTS DDR.
         *
         * The MB/s above is what the ENGINE ASKED FOR, and it is capped at
         * 13.0 GB/s by the datapath: `PARALLEL_KV = 1` retires one row per
         * cycle and cannot consume more. It can never read back the 14.7 GB/s
         * step 1 measured with four synthetic read engines, and a run that
         * claimed to would be measuring something else.
         *
         * What says whether DDR keeps up is the STARVE FRACTION -- cycles the
         * scan spent with no row to score. Step 1's gate was "DDR sustains
         * more than the engine eats"; the falsifiable form of it here is that
         * starving does not GROW with context. A supply that could not keep up
         * would stall a proportional share of every scan, so the fraction
         * would be flat or rising across 64, 256 and 1,024 cached tokens
         * rather than falling towards the fixed round trip paid once per
         * scan. Printed in tenths of a percent. */
        u32 starve_permille = (u32)((tot_starve * 1000) / tot_scan);
        xil_printf("  %s DDR:   starve %u of %u scan cycles (%u.%u%% of the "
                   "scan)\r\n", c->name, (unsigned)tot_starve,
                   (unsigned)tot_scan, (unsigned)(starve_permille / 10),
                   (unsigned)(starve_permille % 10));
    }
    return fails;
}

/* --------------------------------------------------------------------------
 * The long-context probe: address pattern only, NOT a correctness result.
 *
 * WHY THIS EXISTS
 * ---------------
 * Step 14 found that per-group starve goes 8.0, 8.8, 9.6 and then 154.5 cycles
 * at ctx 8,192 -- 16x for an 8x context. Two components, not one: a fixed round
 * trip per group that amortises, and a per-row term that does not. Only the
 * second scales, and the question the whole 32k headline rests on is whether it
 * keeps climbing.
 *
 * Answering it bit-exactly needs a 16.9 MB cache image in the ELF and an
 * O(T^2) numpy prefill. Neither is needed to measure an ADDRESS PATTERN. So
 * the A53 fills DDR itself and the block scans it, and NOTHING here is
 * compared against a golden -- the answers are wrong by construction and are
 * never looked at.
 *
 * WHAT THE SYNTHETIC IMAGE BIASES, AND IN WHICH DIRECTION
 * ------------------------------------------------------
 * The scan is not quite data-independent: a new running maximum costs the
 * online softmax a rescale, and a rescale is 15 cycles of bubble. Those bubbles
 * hand DDR extra slack. A pseudo-random image produces a different number of
 * them than real activations do, so `cyc/row` here is not the real one.
 *
 * `starve` is the number this is for, and the bias runs the safe way: fewer
 * bubbles means less slack, so a scan with few rescales STRESSES the memory
 * harder. `probe8192` runs alongside the bit-exact ctx-8,192 case for exactly
 * this reason -- if the two disagree wildly, these numbers mean nothing.
 * -------------------------------------------------------------------------- */
typedef struct { const char *name; unsigned ctx; unsigned steps; } probe_t;

static const probe_t probes[] = {
    { "probe8192",   8192, 4 },     /* the control: same ctx as a checked case */
    { "probe16384", 16384, 4 },
    { "probe32768", 32768, 4 },
};

static unsigned round_up(unsigned x, unsigned a) { return (x + a - 1) / a * a; }

static int run_probe(const probe_t *pr, const attn_case_t *token_src)
{
    /* The layout, as `hw/ddr_layout.py` computes it: every plane takes the
     * WIDEST plane's span, because `attn_top` walks them as
     * `cache_base + p*plane_span` with one register. */
    unsigned cap    = pr->ctx + pr->steps;
    unsigned span   = round_up(cap * 16u, 4096u);
    unsigned stride = 4u * span;
    u64      bytes  = (u64)8u * stride;

    xil_printf("--- %s [DMA, TIMING ONLY -- answers not checked]: "
               "%u tokens, span %u, stride %u, image %u KB\r\n",
               pr->name, pr->ctx, span, stride, (unsigned)(bytes >> 10));

    /* Fill it. The content is irrelevant to the address pattern; it is not
     * zeroed because an all-zero cache makes every score identical, which is
     * the one input guaranteed to produce no rescales at all. */
    {
        u32 *p = (u32 *)CACHE_BASE, r = 0x1234567u;
        for (u64 i = 0; i < bytes / 4; i++) {
            r = r * 1103515245u + 12345u;
            p[i] = r;
        }
        Xil_DCacheFlushRange((UINTPTR)CACHE_BASE, bytes);
    }

    wr(REG_MODE, 1);
    u64 tot_scan = 0, tot_rows = 0, tot_starve = 0, tot_us = 0;
    for (unsigned s = 0; s < pr->steps; s++) {
        XTime t0, t1;
        u32 st, guard = 100000000;

        memcpy(dma_tx, token_src->ingress, TOKEN_BYTES);
        Xil_DCacheFlushRange((UINTPTR)dma_tx, TOKEN_BYTES);
        Xil_DCacheFlushRange((UINTPTR)dma_rx, RESULT_BYTES);

        wr(REG_NTOK,    pr->ctx + 1 + s);
        wr(REG_BASE_LO, (u32)CACHE_BASE);
        wr(REG_BASE_HI, (u32)(CACHE_BASE >> 32));
        wr(REG_STRIDE,  stride);
        wr(REG_SPAN,    span);

        XTime_GetTime(&t0);
        if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)dma_rx, RESULT_BYTES,
                                   XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) return 1;
        if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)dma_tx, TOKEN_BYTES,
                                   XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) return 1;
        wr(REG_CTRL, 1);
        do { st = rd(REG_CTRL); } while (!(st & ST_DONE) && --guard);
        if (!guard) { xil_printf("    hung at step %u\r\n", s); return 1; }
        while ((XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA) ||
                XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE)) && --guard) { }
        XTime_GetTime(&t1);

        u32 scan = rd(REG_SCAN_CYC), starve = rd(REG_STARVE);
        u64 rows = 8ull * (pr->ctx + 1 + s);
        u32 us = (u32)(((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND);
        u32 mbps = scan ? (u32)((rows * 52ull * PL_HZ_NOMINAL) / scan / 1000000ULL) : 0;
        xil_printf("  step %u  ctx %6u  scan %7u cyc  %5u MB/s  starve %5u  "
                   "run %5u us\r\n", s, pr->ctx + 1 + s, (unsigned)scan,
                   (unsigned)mbps, (unsigned)starve, (unsigned)us);
        tot_scan += scan; tot_rows += rows; tot_starve += starve; tot_us += us;
    }

    /* Per-group starve is the number step 14 asked for: the fixed round trip
     * shows up here as a constant and the per-row term as growth. */
    u32 cpr100 = (u32)((tot_scan * 100) / tot_rows);
    u32 permille = (u32)((tot_starve * 1000) / tot_scan);
    u32 per_grp10 = (u32)((tot_starve * 10) / (8ull * pr->steps));
    u32 per_row10k = (u32)((tot_starve * 10000) / tot_rows);
    xil_printf("  %s TOTAL: %u.%02u cyc/row, %u MB/s, starve %u.%u%% of the scan, "
               "%u.%u cyc/group, %u.%04u cyc/row\r\n",
               pr->name, (unsigned)(cpr100 / 100), (unsigned)(cpr100 % 100),
               (unsigned)((tot_rows * 52ull * PL_HZ_NOMINAL) / tot_scan / 1000000ULL),
               (unsigned)(permille / 10), (unsigned)(permille % 10),
               (unsigned)(per_grp10 / 10), (unsigned)(per_grp10 % 10),
               (unsigned)(per_row10k / 10000), (unsigned)(per_row10k % 10000));
    return 0;
}

/* --------------------------------------------------------------------------
 * The mailbox server: real inference, with the host in charge.
 *
 * The host runs the other fifteen layers and this runs one layer's attention,
 * per token, live. It never returns -- the session ends when the host writes
 * MBX_REQ_QUIT or simply stops the processor.
 *
 * Everything the host wrote is INVALIDATED before it is read and everything
 * written back is FLUSHED, because JTAG reaches DDR without going through
 * these caches. The doorbell is invalidated on every poll: cached, it reads
 * zero forever while the new value sits in memory a metre away.
 * -------------------------------------------------------------------------- */
static void serve(void)
{
    volatile u32 *m = (volatile u32 *)MBX_BASE;
    u8 *tok = (u8 *)(MBX_BASE + MBX_TOKEN_OFF);
    u8 *res = (u8 *)(MBX_BASE + MBX_RESULT_OFF);
    u32 served = 0;

    /* Clear it before announcing readiness, so a doorbell left over from a
     * previous session cannot be mistaken for a request in this one. */
    for (unsigned i = 0; i < MBX_CTRL_BYTES / 4; i++) m[i] = 0;
    Xil_DCacheFlushRange((UINTPTR)m, MBX_CTRL_BYTES);
    xil_printf("SERVER READY mailbox %08x token +%04x result +%04x\r\n",
               (unsigned)MBX_BASE, MBX_TOKEN_OFF, MBX_RESULT_OFF);

    for (;;) {
        u32 db;
        do {
            Xil_DCacheInvalidateRange((UINTPTR)m, MBX_CTRL_BYTES);
            db = m[MBX_DOORBELL];
        } while (db == MBX_IDLE);

        if (db == MBX_REQ_QUIT) {
            xil_printf("SERVER QUIT after %u steps\r\n", (unsigned)served);
            return;
        }
        if (db != MBX_REQ_STEP) {
            m[MBX_STATUS] = MBX_ERR; m[MBX_DOORBELL] = MBX_IDLE;
            Xil_DCacheFlushRange((UINTPTR)m, MBX_CTRL_BYTES);
            continue;
        }

        u32 ntok = m[MBX_NTOK], stride = m[MBX_STRIDE], span = m[MBX_SPAN];
        u32 blo = m[MBX_BASE_LO], bhi = m[MBX_BASE_HI], seq = m[MBX_SEQ];

        /* The token came in over JTAG; the DMA is about to read it. */
        Xil_DCacheInvalidateRange((UINTPTR)tok, TOKEN_BYTES);
        Xil_DCacheFlushRange((UINTPTR)res, RESULT_BYTES);

        XTime t0, t1;
        u32 st = 0, guard = 100000000, ok = 1;   /* st: read below even if the DMA never armed */
        XTime_GetTime(&t0);

        wr(REG_MODE, 1);
        wr(REG_NTOK, ntok);
        wr(REG_BASE_LO, blo);
        wr(REG_BASE_HI, bhi);
        wr(REG_STRIDE, stride);
        wr(REG_SPAN, span);

        if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)res, RESULT_BYTES,
                                   XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) ok = 0;
        if (ok && XAxiDma_SimpleTransfer(&dma, (UINTPTR)tok, TOKEN_BYTES,
                                         XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) ok = 0;
        if (ok) {
            wr(REG_CTRL, 1);
            do { st = rd(REG_CTRL); } while (!(st & ST_DONE) && --guard);
            if (!guard) ok = 0;
            while (ok && (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA) ||
                          XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE)) && --guard) { }
            if (!guard) ok = 0;
        }
        XTime_GetTime(&t1);

        /* The DMA wrote `res` through HPC0, so the A53's view of it may be
         * stale even though the memory is right. */
        Xil_DCacheInvalidateRange((UINTPTR)res, RESULT_BYTES);

        m[MBX_SCAN_CYC] = rd(REG_SCAN_CYC);
        m[MBX_BUSY_CYC] = rd(REG_BUSY_CYC);
        m[MBX_STARVE]   = rd(REG_STARVE);
        m[MBX_CLIPS]    = rd(REG_CLIPS);
        m[MBX_OVF]      = rd(REG_OVF);
        m[MBX_FLAGS]    = st & (ST_RANGE | ST_NORMSAT);
        m[MBX_DEV_US]   = (u32)(((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND);
        m[MBX_SEQ]      = seq;
        m[MBX_NSTEPS]   = ++served;
        m[MBX_STATUS]   = ok ? MBX_ACK : MBX_ERR;

        /* The result FIRST, then the doorbell. The host polls the doorbell, so
         * a doorbell that became visible before the data it announces would
         * hand back the previous token's answer. */
        Xil_DCacheFlushRange((UINTPTR)res, RESULT_BYTES);
        Xil_DCacheFlushRange((UINTPTR)m, MBX_CTRL_BYTES);
        m[MBX_DOORBELL] = MBX_IDLE;
        Xil_DCacheFlushRange((UINTPTR)m, MBX_CTRL_BYTES);
    }
}

/* --------------------------------------------------------------------------
 * The Ethernet server. Step 14's transport, and the one the design is FOR.
 *
 * Reassembly is BY OFFSET, not by arrival order: each frame carries its
 * fragment index and the total, so a reordered or duplicated frame lands where
 * it belongs instead of where it arrived. `seq` travels in every fragment so a
 * straggler from an abandoned message is dropped rather than spliced into the
 * current one -- on a direct cable that should never happen, and "should never
 * happen" is exactly the class of thing that produces a wrong answer instead
 * of an error when it does.
 *
 * The payload is reassembled STRAIGHT INTO the DMA's transmit buffer, so the
 * bytes off the wire are the bytes the core consumes. That is what
 * `attn_proto.h` is built around.
 * -------------------------------------------------------------------------- */
static u8 eth_frame[1600];
static u8 eth_msg[64 + ATTN_TOKEN_BYTES];

static void serve_eth(void)
{
    u32 cur_seq = 0xFFFFFFFFu, have = 0, want = 0, msg_len = 0;
    u32 served = 0;

    xil_printf("ETH SERVER READY\r\n");

    u32 spin = 0;
    for (;;) {
        int n = attn_eth_recv(eth_frame, sizeof(eth_frame));
        if (n <= (int)ATTN_FRAG_HDR_BYTES) {
            /* A heartbeat, so an idle server is visibly idle rather than
             * indistinguishable from a hung one. Every ~30 s of spinning. */
            if (++spin >= 200000000u) {
                u32 ra = 0, ro = 0, tf = 0;
                attn_eth_stats(&ra, &ro, &tf);
                xil_printf("ETH: waiting. rx %u (%u ours), tx %u, served %u\r\n",
                           (unsigned)ra, (unsigned)ro, (unsigned)tf,
                           (unsigned)served);
                spin = 0;
            }
            continue;
        }
        spin = 0;

        attn_frag_hdr fh;
        memcpy(&fh, eth_frame, ATTN_FRAG_HDR_BYTES);
        const u8 *body = eth_frame + ATTN_FRAG_HDR_BYTES;
        u32 blen = (u32)n - ATTN_FRAG_HDR_BYTES;

        if (fh.seq != cur_seq) {          /* a new message begins */
            cur_seq = fh.seq;
            have = 0;
            want = fh.nfrag;
            msg_len = 0;
        }
        if (fh.nfrag != want || fh.frag >= want)
            continue;                     /* a straggler; drop it */

        u32 off = (u32)fh.frag * ATTN_MTU_PAYLOAD;
        if (off + blen > sizeof(eth_msg))
            continue;
        memcpy(eth_msg + off, body, blen);
        if (off + blen > msg_len) msg_len = off + blen;
        have++;
        if (have < want)
            continue;

        /* -- a whole request ------------------------------------------- */
        attn_req_hdr rq;
        if (msg_len < sizeof(rq)) continue;
        memcpy(&rq, eth_msg, sizeof(rq));

        /* The first few requests, as PARSED. Two wrong guesses about this
         * path have already cost a rebuild each; what the board actually
         * decoded is cheaper than another theory. */
        if (served < 3)
            xil_printf("ETH: req seq %u frags %u/%u msg %u B | magic %08x "
                       "ver %u cmd %u payload %u ntok %u stride %u span %u\r\n",
                       (unsigned)fh.seq, (unsigned)have, (unsigned)want,
                       (unsigned)msg_len, (unsigned)rq.magic,
                       (unsigned)rq.version, (unsigned)rq.cmd,
                       (unsigned)rq.payload, (unsigned)rq.n_tokens,
                       (unsigned)rq.head_stride, (unsigned)rq.plane_span);

        attn_resp_hdr rs;
        memset(&rs, 0, sizeof(rs));
        rs.magic = ATTN_MAGIC;
        rs.seq = rq.seq;

        if (rq.magic != ATTN_MAGIC)            rs.status = ATTN_EBADMAGIC;
        else if (rq.version != ATTN_VERSION)   rs.status = ATTN_EBADVER;
        else if (rq.cmd == ATTN_CMD_PING)      rs.status = ATTN_OK;
        else if (rq.cmd == ATTN_CMD_LOAD) {
            u64 base = ((u64)rq.cache_base_hi << 32) | rq.cache_base_lo;
            memcpy((void *)(UINTPTR)base, eth_msg + sizeof(rq), rq.payload);
            Xil_DCacheFlushRange((UINTPTR)base, rq.payload);
            rs.status = ATTN_OK;
        } else if (rq.cmd != ATTN_CMD_STEP)    rs.status = ATTN_EBADCMD;
        else if (rq.payload != ATTN_TOKEN_BYTES ||
                 msg_len < sizeof(rq) + ATTN_TOKEN_BYTES)
            rs.status = ATTN_EBADLEN;
        else {
            XTime t0, t1;
            u32 st = 0, guard = 100000000, ok = 1;

            memcpy(dma_tx, eth_msg + sizeof(rq), TOKEN_BYTES);
            Xil_DCacheFlushRange((UINTPTR)dma_tx, TOKEN_BYTES);
            Xil_DCacheFlushRange((UINTPTR)dma_rx, RESULT_BYTES);

            XTime_GetTime(&t0);
            wr(REG_MODE, 1);
            wr(REG_NTOK,    rq.n_tokens);
            wr(REG_BASE_LO, rq.cache_base_lo);
            wr(REG_BASE_HI, rq.cache_base_hi);
            wr(REG_STRIDE,  rq.head_stride);
            wr(REG_SPAN,    rq.plane_span);

            if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)dma_rx, RESULT_BYTES,
                                       XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) ok = 0;
            if (ok && XAxiDma_SimpleTransfer(&dma, (UINTPTR)dma_tx, TOKEN_BYTES,
                                             XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) ok = 0;
            if (ok) {
                wr(REG_CTRL, 1);
                do { st = rd(REG_CTRL); } while (!(st & ST_DONE) && --guard);
                while (ok && guard &&
                       (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA) ||
                        XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE)) && --guard) { }
                if (!guard) ok = 0;
            }
            XTime_GetTime(&t1);
            Xil_DCacheInvalidateRange((UINTPTR)dma_rx, RESULT_BYTES);

            rs.status = ok ? ((st & ST_RANGE) ? ATTN_ERANGE : ATTN_OK) : ATTN_EDMA;
            rs.payload = ok ? ATTN_RESULT_BYTES : 0;
            rs.dev_us = (u32)(((t1 - t0) * 1000000ULL) / COUNTS_PER_SECOND);
            rs.scan_cycles = rd(REG_SCAN_CYC);
            rs.busy_cycles = rd(REG_BUSY_CYC);
            rs.starve_cycles = rd(REG_STARVE);
            rs.clips = rd(REG_CLIPS);
            rs.overflows = rd(REG_OVF);
            served++;
        }

        /* -- reply, fragmented the same way -------------------------- */
        u32 total = sizeof(rs) + rs.payload;
        memcpy(eth_msg, &rs, sizeof(rs));
        if (rs.payload)
            memcpy(eth_msg + sizeof(rs), dma_rx, rs.payload);

        u32 nf = (total + ATTN_MTU_PAYLOAD - 1) / ATTN_MTU_PAYLOAD;
        if (nf == 0) nf = 1;
        for (u32 f = 0; f < nf; f++) {
            u32 o = f * ATTN_MTU_PAYLOAD;
            u32 c = total - o;
            if (c > ATTN_MTU_PAYLOAD) c = ATTN_MTU_PAYLOAD;
            attn_frag_hdr oh = { rq.seq, (u16)f, (u16)nf };
            memcpy(eth_frame, &oh, ATTN_FRAG_HDR_BYTES);
            memcpy(eth_frame + ATTN_FRAG_HDR_BYTES, eth_msg + o, c);
            if (attn_eth_send(eth_frame, (int)(ATTN_FRAG_HDR_BYTES + c)) < 0)
                xil_printf("ETH: send failed on fragment %u\r\n", (unsigned)f);
        }
        cur_seq = 0xFFFFFFFFu;            /* done; the next seq starts fresh */
        if ((served % 64) == 1)
            xil_printf("ETH: %u steps served\r\n", (unsigned)served);
    }
}

int main(void)
{
    int fails = 0;

    Xil_DCacheEnable();
    xil_printf("\r\n=== ZCU104 attention block, driven by the A53 (step 13) ===\r\n");

    u32 magic = rd(REG_MAGIC);
    if (magic != MAGIC) {
        xil_printf("MAGIC reads %08x, expected %08x -- this is not the "
                   "bitstream this ELF targets\r\n",
                   (unsigned)magic, (unsigned)MAGIC);
        return 1;
    }
    xil_printf("MAGIC ok (%08x), shim windows %u in / %u out words\r\n",
               (unsigned)magic, (unsigned)rd(REG_IN_W), (unsigned)rd(REG_OUT_W));
    if (rd(REG_IN_W) != ATTN_IN_WORDS || rd(REG_OUT_W) != ATTN_OUT_WORDS) {
        xil_printf("window size disagrees with the vectors\r\n");
        return 1;
    }

    /* DDR, before anything depends on it. Two patterns: a stuck bus matches
     * one by accident. */
    {
        volatile u32 *p = (volatile u32 *)(CACHE_BASE);
        *p = 0x5A5AA5A5; Xil_DCacheFlushRange((UINTPTR)p, 4);
        if (*p != 0x5A5AA5A5) { xil_printf("DDR round trip failed (1)\r\n"); return 1; }
        *p = 0xA5A55A5A; Xil_DCacheFlushRange((UINTPTR)p, 4);
        if (*p != 0xA5A55A5A) { xil_printf("DDR round trip failed (2)\r\n"); return 1; }
    }
    xil_printf("DDR round trip ok at %08x\r\n", (unsigned)CACHE_BASE);

    /* The DMA. A failure here is reported and the run continues on the
     * AXI-Lite path rather than aborting: a bring-up that can still produce
     * the right 2,048 numbers slowly is worth more than one that stops. */
    {
        XAxiDma_Config *cfg = XAxiDma_LookupConfig(DMA_BASE);
        dma_ok = 0;
        if (!cfg)
            xil_printf("no DMA config at %08x\r\n", (unsigned)DMA_BASE);
        else if (XAxiDma_CfgInitialize(&dma, cfg) != XST_SUCCESS)
            xil_printf("DMA init failed\r\n");
        else if (XAxiDma_HasSg(&dma))
            xil_printf("DMA built for scatter-gather, expected simple mode\r\n");
        else {
            XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
            XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
            dma_ok = 1;
            xil_printf("DMA ready at %08x\r\n", (unsigned)DMA_BASE);
        }
    }

    /* Every case both ways. The DMA path is new and the AXI-Lite path is what
     * step 13 ran on this board, so the fast one is measured against a slow one
     * that has already been checked against the numpy kernel -- and both are
     * checked against the goldens, so agreeing with each other is not enough
     * to pass. */
    for (unsigned i = 0; i < attn_n_cases; i++) {
        fails += run_case(&attn_cases[i], 0);
        if (dma_ok)
            fails += run_case(&attn_cases[i], 1);
    }
    if (!dma_ok)
        xil_printf("!!! the DMA did not initialise; only the AXI-Lite path ran\r\n");

    /* The long-context probe. Last, so a failure here cannot cost the
     * bit-exact results above -- and it overwrites the cache region, which is
     * why nothing after it may assume the image is still there. */
    if (dma_ok && attn_n_cases > 0) {
        xil_printf("=== long-context probe: address pattern only ===\r\n");
        for (unsigned i = 0; i < sizeof(probes) / sizeof(probes[0]); i++)
            if (run_probe(&probes[i], &attn_cases[0]))
                xil_printf("!!! %s did not complete\r\n", probes[i].name);
    }

    xil_printf("=== %s ===\r\n", fails ? "FAILURES PRESENT" : "ALL TESTS PASSED");

    /* Self-test done; hand the board to the host. Ethernet if the link comes
     * up, and the JTAG mailbox if it does not -- a board that cannot reach the
     * network is still a board that can do inference, just slowly, and falling
     * back is better than a session that ends at a dead PHY. */
    if (dma_ok) {
        if (attn_eth_init() == 0)
            serve_eth();
        else {
            xil_printf("ETH unavailable; falling back to the JTAG mailbox\r\n");
            serve();
        }
    }
    Xil_DCacheDisable();
    return 0;
}
