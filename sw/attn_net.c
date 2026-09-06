/*
 * The attention server on lwIP, over UDP. Step 14's transport, second attempt.
 *
 * Why this REPLACES `attn_eth.c`
 * ------------------------------
 * The hand-rolled polled GEM driver produced five real bugs -- descriptor cache
 * maintenance the driver does none of, `RXBUF_NEW` never cleared on re-arm,
 * four descriptors sharing a cache line so a flush erased its neighbours, a
 * ring array a quarter of the size its alignment demanded, a frame buffer left
 * at the old MTU -- and a sixth that was never found: the receiver stops after
 * one, four or nine decode steps depending on the run, with no error status
 * set. A varying failure point is a race, and racing a DMA engine by hand from
 * a poll loop is not a good use of anyone's evening.
 *
 * Xilinx maintains a port of lwIP for this exact controller
 * (`xemacpsif_dma.c`, `xemacpsif_physpeed.c`). It owns the descriptors, the
 * cache maintenance, the status acknowledgement and the PHY. This file owns a
 * UDP callback.
 *
 * And it deletes the fragmentation layer
 * --------------------------------------
 * A request is 9,252 bytes and a reply 6,188, neither of which fits an
 * Ethernet frame -- so the raw version carried a `{seq, frag, nfrag}` header
 * and reassembled by offset. UDP over IP does that in the stack: a 9 KB
 * datagram is fragmented and reassembled by code that has been correct for
 * twenty years. The protocol stops caring about the MTU, the host stops
 * needing raw sockets, and `attn_frag_hdr` is gone.
 *
 * The payload is still handed to the DMA untouched, which is the one rule the
 * wire format exists to preserve.
 */
#include <string.h>

#include "lwip/init.h"
#include "lwip/udp.h"
#include "lwip/inet.h"
#include "lwip/ip4_frag.h"
#include "netif/xadapter.h"
#include "xil_cache.h"
#include "xil_printf.h"

#include "attn_net.h"
#include "attn_proto.h"

/* Static addressing: this is a metre of cable between two devices and there is
 * nothing on it to serve DHCP. The host is 192.168.10.1 on enp4s0. */
#define BOARD_IP   "192.168.10.2"
#define BOARD_MASK "255.255.255.0"
#define BOARD_GW   "192.168.10.1"

static const unsigned char BOARD_MAC[6] = {0x02, 0x00, 0x5A, 0x77, 0xE0, 0x01};

static struct netif netif_storage;
struct netif *echo_netif = &netif_storage;

static attn_net_handler_t handler;
static void *handler_ctx;

/* Reassembled request in, response out. lwIP hands us a pbuf chain, because a
 * 9 KB datagram arrives as several IP fragments; `pbuf_copy_partial` walks it. */
static u8 rx_msg[64 + ATTN_TOKEN_BYTES];
static u8 tx_msg[64 + ATTN_RESULT_BYTES];

static u32 served, dropped;

static void on_udp(void *arg, struct udp_pcb *pcb, struct pbuf *p,
                   const ip_addr_t *addr, u16_t port)
{
    (void)arg;
    if (p == NULL)
        return;

    u16_t n = pbuf_copy_partial(p, rx_msg, sizeof(rx_msg), 0);
    u16_t total = p->tot_len;
    pbuf_free(p);

    if (n != total) {
        /* The datagram was larger than this buffer. Dropping it loudly beats
         * running a step on a truncated token. */
        if (++dropped <= 4)
            xil_printf("NET: dropped a %u B datagram (buffer is %u)\r\n",
                       (unsigned)total, (unsigned)sizeof(rx_msg));
        return;
    }

    int out = handler(handler_ctx, rx_msg, n, tx_msg, sizeof(tx_msg));
    if (out <= 0)
        return;

    struct pbuf *r = pbuf_alloc(PBUF_TRANSPORT, (u16_t)out, PBUF_RAM);
    if (r == NULL) {
        xil_printf("NET: out of pbufs for a %d B reply\r\n", out);
        return;
    }
    memcpy(r->payload, tx_msg, (size_t)out);
    /* Reply to whoever asked, on the port they asked from -- no configuration
     * and no discovery. */
    udp_sendto(pcb, r, addr, port);
    pbuf_free(r);

    if ((++served % 256U) == 1U)
        xil_printf("NET: %u served, %u dropped\r\n",
                   (unsigned)served, (unsigned)dropped);
}

int attn_net_init(attn_net_handler_t fn, void *ctx)
{
    ip_addr_t ip, mask, gw;
    struct udp_pcb *pcb;

    handler = fn;
    handler_ctx = ctx;

    lwip_init();

    inet_aton(BOARD_IP, &ip);
    inet_aton(BOARD_MASK, &mask);
    inet_aton(BOARD_GW, &gw);

    if (!xemac_add(echo_netif, &ip, &mask, &gw, (unsigned char *)BOARD_MAC,
                   XPAR_XEMACPS_0_BASEADDR)) {
        xil_printf("NET: xemac_add failed -- is ENET3 enabled in the BD?\r\n");
        return -1;
    }
    netif_set_default(echo_netif);
    netif_set_up(echo_netif);

    pcb = udp_new();
    if (pcb == NULL) {
        xil_printf("NET: udp_new failed\r\n");
        return -1;
    }
    if (udp_bind(pcb, IP_ADDR_ANY, ATTN_PORT) != ERR_OK) {
        xil_printf("NET: could not bind port %u\r\n", (unsigned)ATTN_PORT);
        return -1;
    }
    udp_recv(pcb, on_udp, NULL);

    xil_printf("NET: lwIP up, %s:%u, MAC %02x:%02x:%02x:%02x:%02x:%02x\r\n",
               BOARD_IP, (unsigned)ATTN_PORT, BOARD_MAC[0], BOARD_MAC[1],
               BOARD_MAC[2], BOARD_MAC[3], BOARD_MAC[4], BOARD_MAC[5]);
    return 0;
}

void attn_net_poll(void)
{
    /* Everything the stack needs, from a bare loop: hand it the frames the
     * adapter has received. No interrupts, no OS, and no descriptor of mine
     * anywhere near it.
     *
     * `sys_check_timeouts` does not exist in this configuration -- LWIP_TIMERS
     * is off -- and for a UDP server only one timer matters: a 9 KB request
     * arrives as several IP fragments, and if one is lost the partial
     * reassembly must eventually expire or its pbufs are never returned.
     * `ip_reass_tmr` is that timer, and lwIP wants it about once a second. */
    xemacif_input(echo_netif);
#if IP_REASSEMBLY
    static u32 tick;
    if ((++tick & 0xFFFFFU) == 0U)
        ip_reass_tmr();
#endif
}

u32 attn_net_served(void) { return served; }
