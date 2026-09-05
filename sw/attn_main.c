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

#include "attn_vectors.h"

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
static int run_case(const attn_case_t *c)
{
    u64 tot_scan = 0, tot_busy = 0, tot_rows = 0, tot_us_run = 0;
    u64 tot_us_load = 0, tot_us_read = 0;
    int fails = 0;

    xil_printf("--- %s: %u cached tokens, %u steps, "
               "image %u B (stride %u, span %u)\r\n",
               c->name, c->ctx0, c->steps, c->image_bytes,
               c->head_stride, c->plane_span);

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
        int bad = run_step(c, s, &ul, &ur, &ud, &scan, &busy, &st);
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
    }
    return fails;
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

    for (unsigned i = 0; i < attn_n_cases; i++)
        fails += run_case(&attn_cases[i]);

    xil_printf("=== %s ===\r\n", fails ? "FAILURES PRESENT" : "ALL TESTS PASSED");
    Xil_DCacheDisable();
    return 0;
}
