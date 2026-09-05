/*
 * Wire protocol between the x86 host and the ZCU104, for the attention block.
 *
 * THE PAYLOAD IS THE BEAT STREAM, UNCHANGED
 * -----------------------------------------
 * A request's payload is EXACTLY the 512-bit beat stream `attn_top` consumes,
 * in the order `hw/vectors.py::ingress_bytes` packs it, so the A53 hands the
 * received bytes to the DMA without touching them and the DMA hands them to
 * the core without converting a width. That is the same rule
 * `systolic_zcu104/host/sa_proto.h` follows and the reason `plan.MD` asked for
 * it: any repacking on the PS is a memcpy of the whole token on the critical
 * path, and step 13 measured what that costs -- 366 us of a 433 us step.
 *
 * All fields little-endian; both ends are LE.
 *
 * WHAT THE HOST OWNS
 * ------------------
 * The cache lives in the board's DDR and the HOST decides where: `cache_base`,
 * `head_stride` and `plane_span` come down with every step. The board keeps no
 * map of its own. That is deliberate -- the layout is computed by
 * `hw/ddr_layout.py`, which is also what built the image, so there is exactly
 * one place that knows it. A board holding its own copy would be a second
 * place for it to be wrong.
 */
#ifndef ATTN_PROTO_H
#define ATTN_PROTO_H
#include <stdint.h>

#define ATTN_MAGIC    0x314E5441u   /* "ATTN" with a version nibble: 'A','T','T','1' */
#define ATTN_VERSION  1
#define ATTN_PORT     7001          /* the TCP mock; raw Ethernet uses ETHERTYPE */
#define ATTN_ETHERTYPE 0x88B5       /* IEEE 802 local experimental 1 */

#define ATTN_CMD_PING 0
#define ATTN_CMD_STEP 1             /* one decode step: payload is the token */
#define ATTN_CMD_LOAD 2             /* write payload into the cache at cache_base */

/* The board reassembles a message into a buffer sized for one token, so a
 * request's payload cannot exceed this. A CMD_LOAD larger than it is dropped
 * fragment by fragment and never answered -- the host splits instead. */
#define ATTN_MAX_PAYLOAD  9216

#define ATTN_BEAT       64          /* bytes per 512-bit beat */
#define ATTN_TOKEN_BYTES  9216      /* 8 groups x 6 vectors x 3 beats x 64 B */
#define ATTN_RESULT_BYTES 6144      /* 2,048 lanes x 24 b */

/* Status codes. Anything non-zero leaves the payload undefined. */
#define ATTN_OK          0
#define ATTN_EBADMAGIC   1
#define ATTN_EBADVER     2
#define ATTN_EBADCMD     3
#define ATTN_EBADLEN     4
#define ATTN_ETIMEOUT    5
#define ATTN_EDMA        6
#define ATTN_ERANGE      7          /* a channel escaped the 24-bit seam */

/* THERE IS NO 64-BIT FIELD, AND THAT IS NOT A STYLE CHOICE.
 * `cache_base` is a DDR byte address and wants to be a `uint64_t`. A u64 in
 * this struct forces 8-byte alignment, and 8-byte alignment pads TWICE: four
 * bytes before the field if it is not already aligned, and four at the END,
 * because `sizeof` must be a multiple of the alignment. Either way C says 40
 * bytes and a packed Python `struct` says 36, both ends agree with themselves,
 * and they disagree on the wire. `tests/test_proto.py` caught it twice --
 * once for each kind of padding -- which is the argument for having that test
 * at all.
 * So the address is split, exactly as `attn_ctrl`'s own register map splits it
 * into BASE_LO at 0x0C and BASE_HI at 0x10. Every field here is 4 bytes or
 * less and nothing is padded. */
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t cmd;
    uint32_t cache_base_lo; /* byte address in the board's DDR, low 32 */
    uint32_t cache_base_hi; /* and high 32 */
    uint32_t n_tokens;      /* ctx + 1, this token included */
    uint32_t head_stride;   /* bytes per KV head */
    uint32_t plane_span;    /* bytes per plane; the same for all four */
    uint32_t payload;       /* bytes following this header */
    uint32_t seq;           /* echoed back; a reply to the wrong request is
                             * otherwise indistinguishable from a wrong answer */
} attn_req_hdr;

typedef struct {
    uint32_t magic;
    uint16_t status;
    uint16_t rsvd;
    uint32_t payload;
    uint32_t seq;           /* the request's, echoed */
    /* The device's own counters. `dev_us` is the A53's wall clock across the
     * step; `scan_cycles` and `busy_cycles` are counted in the PL at the clock
     * the DDR reads were issued on, and are the only ones a bandwidth may be
     * divided by. See `sw/attn_main.c`. */
    uint32_t dev_us;
    uint32_t scan_cycles;
    uint32_t busy_cycles;
    uint32_t starve_cycles;
    uint32_t clips;
    uint32_t overflows;
} attn_resp_hdr;

/* THE FRAGMENT HEADER, in front of every Ethernet frame's slice of a message.
 * 9,216 bytes of token does not fit a 1,500-byte frame, so a message arrives
 * as a run of frames and is reassembled BY OFFSET rather than by arrival
 * order. `host/attn_client.py::RawEthClient` packs this as "<IHH".
 *
 * `seq` is repeated in every fragment on purpose: it is what lets a straggler
 * from an abandoned message be dropped instead of being spliced into the
 * current one. */
typedef struct {
    uint32_t seq;
    uint16_t frag;
    uint16_t nfrag;
} attn_frag_hdr;

#define ATTN_FRAG_HDR_BYTES 8
#define ATTN_MTU_PAYLOAD    (1500 - ATTN_FRAG_HDR_BYTES)

#endif
