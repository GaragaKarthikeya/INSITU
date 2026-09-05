/*
 * Raw Ethernet server on the A53. No IP, no lwIP, no OS. Step 14's transport.
 *
 * WHY RAW FRAMES AND NOT A SOCKET STACK
 * -------------------------------------
 * `plan.MD` budgets 35 us of round trip and 123 us of wire time for a 15 KB
 * token at ctx 32k, against 927 us of DDR -- so the network is a minor term
 * and the only thing that would make it a major one is a stack. lwIP on a
 * baremetal A53 costs more in copies and timers than the wire costs in
 * nanoseconds. A frame with a custom ethertype needs neither.
 *
 * The link is a metre of cable between two devices with nothing else on it, so
 * there is no retransmit and no congestion control. A dropped frame is a bug
 * to find, not a condition to recover from -- and the host's sequence number
 * is what turns "we lost one" into an error rather than into a wrong answer.
 *
 * FRAGMENTATION, BECAUSE 9,216 BYTES DOES NOT FIT IN A FRAME
 * ----------------------------------------------------------
 * A request is the header plus up to 9,216 bytes of token, and an untagged
 * frame carries 1,500. So a request arrives as a run of frames each carrying
 * `{seq, frag, nfrag}` and its slice of the byte stream, and the server
 * reassembles by OFFSET rather than by arrival order. `host/attn_client.py`'s
 * `RawEthClient` is the other half and the two share `attn_proto.h`.
 *
 * THE PAYLOAD IS NEVER TOUCHED
 * ----------------------------
 * Fragments are reassembled straight into the DMA's transmit buffer, so the
 * bytes that came off the wire are the bytes the core consumes. That is the
 * rule `attn_proto.h` is built around: any repacking here is a memcpy of the
 * whole token on the critical path, and step 13 measured that at 366 us
 * against 67 us of block.
 */
#include <string.h>

#include "xemacps.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xparameters.h"
#if __has_include("xtime_l.h")
#include "xtime_l.h"
#else
#include "xiltimer.h"
#include "xtimer_config.h"
#endif

#include "attn_eth.h"
#include "attn_proto.h"

/* GEM3 is the ZCU104's RJ45. Enabled in `create_bd_attn_dma.tcl`; the device
 * tree carried `status = "disabled"` before that and nothing here would have
 * worked. */
#define GEM_BASE        0xFF0E0000U

#define RXBD_COUNT      64
#define TXBD_COUNT      16
#define FRAME_MAX       1536
#define BD_ALIGN        64

/* Our MAC. Locally administered (bit 1 of the first octet), so it cannot
 * collide with a real vendor's, and fixed so the host can address us without
 * discovery. */
static const u8 BOARD_MAC[6] = {0x02, 0x00, 0x5A, 0x77, 0xE0, 0x01};

static XEmacPs emac;

/* BD rings and buffers, uncached-by-flushing. The GEM masters DDR directly and
 * is NOT coherent with these caches, so every descriptor and buffer is flushed
 * before the hardware reads it and invalidated after the hardware writes it.
 * This is the same discipline the KV image needs, and the same bug if missed:
 * correct hardware, stale memory. */
static u8 rx_bd_space[RXBD_COUNT * sizeof(XEmacPs_Bd) * 2]
    __attribute__((aligned(BD_ALIGN)));
static u8 tx_bd_space[TXBD_COUNT * sizeof(XEmacPs_Bd) * 2]
    __attribute__((aligned(BD_ALIGN)));
static u8 rx_buf[RXBD_COUNT][FRAME_MAX] __attribute__((aligned(BD_ALIGN)));
static u8 tx_buf[FRAME_MAX] __attribute__((aligned(BD_ALIGN)));

static u8 host_mac[6];
static int host_known;

/* Counted and printed, because a silent receiver and a receiver that drops
 * everything look identical from the far end. `rx_any` is every frame the MAC
 * accepted; `rx_ours` is those with our ethertype. If the first is climbing
 * and the second is not, the filter or the ethertype is wrong -- which is a
 * different bug from a dead link, and the UART should say which. */
static u32 rx_any, rx_ours, tx_frames;

/* --------------------------------------------------------------------------
 * PHY
 *
 * The address is SCANNED, not assumed. Boards move the DP83867 between MDIO
 * addresses and a wrong constant looks exactly like a dead link -- which is
 * the least diagnosable failure in this whole file.
 * -------------------------------------------------------------------------- */
static int phy_find(XEmacPs *e)
{
    for (u32 a = 0; a < 32; a++) {
        u16 id1 = 0, id2 = 0;
        XEmacPs_PhyRead(e, a, 2, &id1);
        XEmacPs_PhyRead(e, a, 3, &id2);
        if (id1 != 0xFFFF && id1 != 0x0000) {
            xil_printf("ETH: PHY at MDIO %u (id %04x:%04x)\r\n",
                       (unsigned)a, id1, id2);
            return (int)a;
        }
    }
    return -1;
}

static int phy_wait_link(XEmacPs *e, u32 phy, u32 timeout_ms)
{
    /* Status register bit 2 is link, and it is LATCHING LOW: it must be read
     * twice to clear a stale down-event, or a link that came up before this
     * ran reports itself as down forever. */
    for (u32 i = 0; i < timeout_ms; i++) {
        u16 st = 0;
        XEmacPs_PhyRead(e, phy, 1, &st);
        XEmacPs_PhyRead(e, phy, 1, &st);
        if (st & 0x0004U)
            return 1;
        for (volatile int d = 0; d < 100000; d++) { }
    }
    return 0;
}

/* Link speed from the DP83867's status register 0x11, which is where it
 * reports the NEGOTIATED rate. Reading the control register instead gives what
 * was ADVERTISED, which is not the same thing on a link that fell back. */
static u32 phy_speed(XEmacPs *e, u32 phy)
{
    u16 s = 0;
    XEmacPs_PhyRead(e, phy, 0x11, &s);
    switch ((s >> 14) & 0x3U) {
    case 0: return 10;
    case 1: return 100;
    case 2: return 1000;
    default: return 0;
    }
}

/* --------------------------------------------------------------------------
 * bring-up
 * -------------------------------------------------------------------------- */
int attn_eth_init(void)
{
    XEmacPs_Config *cfg = XEmacPs_LookupConfig(GEM_BASE);
    if (!cfg) {
        xil_printf("ETH: no GEM config at %08x -- is ENET3 enabled in the BD?\r\n",
                   (unsigned)GEM_BASE);
        return -1;
    }
    if (XEmacPs_CfgInitialize(&emac, cfg, cfg->BaseAddress) != XST_SUCCESS) {
        xil_printf("ETH: CfgInitialize failed\r\n");
        return -1;
    }
    XEmacPs_SetMacAddress(&emac, (void *)BOARD_MAC, 1);

    int phy = phy_find(&emac);
    if (phy < 0) {
        xil_printf("ETH: no PHY answered on MDIO 0..31\r\n");
        return -1;
    }
    if (!phy_wait_link(&emac, (u32)phy, 5000)) {
        xil_printf("ETH: no link. Is the cable in?\r\n");
        return -1;
    }
    u32 mbps = phy_speed(&emac, (u32)phy);
    xil_printf("ETH: link up at %u Mb/s\r\n", (unsigned)mbps);
    if (mbps == 0)
        return -1;
    XEmacPs_SetOperatingSpeed(&emac, (u16)mbps);
    /* The GEM's TX clock divisor must match the negotiated speed, and it is
     * set by the platform's clock code rather than by the driver. */
    XEmacPs_SetMdioDivisor(&emac, MDC_DIV_224);

    /* -- BD rings ------------------------------------------------------- */
    XEmacPs_BdRing *rxr = &XEmacPs_GetRxRing(&emac);
    XEmacPs_BdRing *txr = &XEmacPs_GetTxRing(&emac);
    XEmacPs_Bd tmpl;

    XEmacPs_BdClear(&tmpl);
    if (XEmacPs_BdRingCreate(rxr, (UINTPTR)rx_bd_space, (UINTPTR)rx_bd_space,
                             BD_ALIGN, RXBD_COUNT) != XST_SUCCESS ||
        XEmacPs_BdRingClone(rxr, &tmpl, XEMACPS_RECV) != XST_SUCCESS) {
        xil_printf("ETH: rx ring setup failed\r\n");
        return -1;
    }
    XEmacPs_BdClear(&tmpl);
    XEmacPs_BdSetStatus(&tmpl, XEMACPS_TXBUF_USED_MASK);
    if (XEmacPs_BdRingCreate(txr, (UINTPTR)tx_bd_space, (UINTPTR)tx_bd_space,
                             BD_ALIGN, TXBD_COUNT) != XST_SUCCESS ||
        XEmacPs_BdRingClone(txr, &tmpl, XEMACPS_SEND) != XST_SUCCESS) {
        xil_printf("ETH: tx ring setup failed\r\n");
        return -1;
    }

    /* Hand every receive buffer to the hardware. */
    for (int i = 0; i < RXBD_COUNT; i++) {
        XEmacPs_Bd *bd;
        if (XEmacPs_BdRingAlloc(rxr, 1, &bd) != XST_SUCCESS) return -1;
        XEmacPs_BdSetAddressRx(bd, (UINTPTR)rx_buf[i]);
        Xil_DCacheInvalidateRange((UINTPTR)rx_buf[i], FRAME_MAX);
        if (XEmacPs_BdRingToHw(rxr, 1, bd) != XST_SUCCESS) return -1;
    }

    XEmacPs_SetQueuePtr(&emac, emac.RxBdRing.BaseBdAddr, 0, XEMACPS_RECV);
    XEmacPs_SetQueuePtr(&emac, emac.TxBdRing.BaseBdAddr, 0, XEMACPS_SEND);
    XEmacPs_Start(&emac);

    xil_printf("ETH: ready, MAC %02x:%02x:%02x:%02x:%02x:%02x, ethertype %04x\r\n",
               BOARD_MAC[0], BOARD_MAC[1], BOARD_MAC[2],
               BOARD_MAC[3], BOARD_MAC[4], BOARD_MAC[5],
               (unsigned)ATTN_ETHERTYPE);
    return 0;
}

/* --------------------------------------------------------------------------
 * one frame in, one frame out
 * -------------------------------------------------------------------------- */
int attn_eth_recv(u8 *dst, int max)
{
    XEmacPs_BdRing *rxr = &XEmacPs_GetRxRing(&emac);
    XEmacPs_Bd *bd;

    if (XEmacPs_BdRingFromHwRx(rxr, 1, &bd) == 0)
        return 0;

    u32 len = XEmacPs_BdGetLength(bd);
    UINTPTR addr = XEmacPs_BdGetBufAddr(bd);
    Xil_DCacheInvalidateRange(addr, len);

    int taken = 0;
    const u8 *f = (const u8 *)addr;
    rx_any++;
    if (rx_any <= 4)
        xil_printf("ETH: rx #%u len %u type %04x from "
                   "%02x:%02x:%02x:%02x:%02x:%02x\r\n",
                   (unsigned)rx_any, (unsigned)len,
                   (unsigned)((f[12] << 8) | f[13]),
                   f[6], f[7], f[8], f[9], f[10], f[11]);
    /* 12 bytes of MAC, 2 of ethertype. Anything else on the wire -- and on a
     * direct cable there should be nothing -- is dropped here. */
    if (len > 14 && ((f[12] << 8) | f[13]) == ATTN_ETHERTYPE) {
        rx_ours++;
        if (!host_known) {
            memcpy(host_mac, f + 6, 6);      /* reply to whoever asked */
            host_known = 1;
            xil_printf("ETH: host is %02x:%02x:%02x:%02x:%02x:%02x\r\n",
                       f[6], f[7], f[8], f[9], f[10], f[11]);
        }
        taken = (int)len - 14;
        if (taken > max) taken = max;
        memcpy(dst, f + 14, (size_t)taken);
    }

    /* Give the buffer straight back, whether it was ours or not: a ring that
     * leaks descriptors stops receiving after RXBD_COUNT frames, which looks
     * like the host going quiet. */
    XEmacPs_BdRingFree(rxr, 1, bd);
    if (XEmacPs_BdRingAlloc(rxr, 1, &bd) == XST_SUCCESS) {
        XEmacPs_BdSetAddressRx(bd, addr);
        Xil_DCacheInvalidateRange(addr, FRAME_MAX);
        XEmacPs_BdRingToHw(rxr, 1, bd);
    }
    return taken;
}

int attn_eth_send(const u8 *payload, int len)
{
    if (!host_known || len > FRAME_MAX - 14)
        return -1;

    XEmacPs_BdRing *txr = &XEmacPs_GetTxRing(&emac);
    XEmacPs_Bd *bd;

    memcpy(tx_buf, host_mac, 6);
    memcpy(tx_buf + 6, BOARD_MAC, 6);
    tx_buf[12] = (u8)(ATTN_ETHERTYPE >> 8);
    tx_buf[13] = (u8)(ATTN_ETHERTYPE & 0xFF);
    memcpy(tx_buf + 14, payload, (size_t)len);

    int total = len + 14;
    /* The minimum Ethernet frame is 60 bytes before the FCS. Short frames are
     * padded by the MAC on this part, but the padding is not zeroed -- so it
     * is zeroed here rather than sending whatever was in the buffer. */
    if (total < 60) {
        memset(tx_buf + total, 0, (size_t)(60 - total));
        total = 60;
    }
    Xil_DCacheFlushRange((UINTPTR)tx_buf, (u32)total);

    if (XEmacPs_BdRingAlloc(txr, 1, &bd) != XST_SUCCESS)
        return -1;
    XEmacPs_BdSetAddressTx(bd, (UINTPTR)tx_buf);
    XEmacPs_BdSetLength(bd, (u32)total);
    XEmacPs_BdClearTxUsed(bd);
    XEmacPs_BdSetLast(bd);
    if (XEmacPs_BdRingToHw(txr, 1, bd) != XST_SUCCESS)
        return -1;
    XEmacPs_Transmit(&emac);

    /* Polled: wait for the descriptor to come back before reusing `tx_buf`. */
    u32 guard = 10000000;
    while (XEmacPs_BdRingFromHwTx(txr, 1, &bd) == 0 && --guard) { }
    if (!guard)
        return -1;
    XEmacPs_BdRingFree(txr, 1, bd);
    tx_frames++;
    return len;
}

void attn_eth_forget_host(void)
{
    host_known = 0;
}

void attn_eth_stats(u32 *any, u32 *ours, u32 *tx)
{
    if (any)  *any  = rx_any;
    if (ours) *ours = rx_ours;
    if (tx)   *tx   = tx_frames;
}
