/*
 * The mailbox the host and the A53 share, in DDR.
 *
 * WHY A MAILBOX AND NOT THE SHIM'S OWN REGISTERS
 * ----------------------------------------------
 * The host could drive `attn_ctrl` directly over JTAG -- step 12 did exactly
 * that -- and it costs one JTAG transaction per 32-bit word. A decode step is
 * 2,304 words in and 1,536 out, and at roughly a millisecond apiece that is
 * four seconds a token. Inference at four seconds a token is a demonstration
 * of patience.
 *
 * DDR is different: `xsdb` writes it with `mwr -bin -file`, which moves a whole
 * buffer through the DAP in one operation rather than one word at a time. So
 * the host puts the token in DDR, rings a doorbell, and the A53 -- which is on
 * the right side of the bus -- does the 15 KB of AXI traffic through the DMA at
 * 5 us. The slow link carries bulk transfers only.
 *
 * COHERENCY, WHICH IS THE WHOLE TRAP HERE
 * ---------------------------------------
 * JTAG writes DDR directly and knows nothing about the A53's caches. The A53
 * must therefore INVALIDATE before every read of anything the host wrote --
 * the doorbell included, on every poll -- or it spins forever on a cached zero
 * while the new value sits in memory. And it must FLUSH after writing the
 * result, or the host reads stale DDR behind a dirty line. Both directions,
 * every time.
 */
#ifndef ATTN_SERVER_H
#define ATTN_SERVER_H

/* Clear of the ELF (bottom of DDR) and clear of the cache image at
 * 0x10000000. 64-byte aligned so a cache operation covers exactly it. */
#define MBX_BASE        0x30000000UL

/* Doorbell values. The host writes REQ; the A53 answers ACK or ERR. It is not
 * a boolean, because a boolean cannot tell "not started" from "finished". */
#define MBX_IDLE        0x00000000u
#define MBX_REQ_STEP    0xA77E0001u
#define MBX_REQ_QUIT    0xA77E00FFu
#define MBX_ACK         0x5A5A0000u
#define MBX_ERR         0xDEAD0000u

/* Word offsets in the control block. */
#define MBX_DOORBELL    0
#define MBX_STATUS      1
#define MBX_NTOK        2
#define MBX_BASE_LO     3
#define MBX_BASE_HI     4
#define MBX_STRIDE      5
#define MBX_SPAN        6
#define MBX_SEQ         7
#define MBX_SCAN_CYC    8
#define MBX_BUSY_CYC    9
#define MBX_STARVE      10
#define MBX_CLIPS       11
#define MBX_OVF         12
#define MBX_FLAGS       13      /* range_error | norm_saturated */
#define MBX_DEV_US      14
#define MBX_NSTEPS      15      /* steps served, so the host can see progress */

#define MBX_CTRL_BYTES  0x40
#define MBX_TOKEN_OFF   0x1000  /* 9,216 B of ingress, host -> board */
#define MBX_RESULT_OFF  0x4000  /* 6,144 B of result,  board -> host */

#endif
