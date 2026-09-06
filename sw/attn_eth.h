/* The GEM3 raw-Ethernet transport. See attn_eth.c. */
#ifndef ATTN_ETH_H
#define ATTN_ETH_H
#include "xil_types.h"

/* 0 on success. Prints why it failed -- link, PHY or config -- because a
 * silent Ethernet failure is indistinguishable from a quiet host. */
int  attn_eth_init(void);

/* Bytes of PAYLOAD (after the 14-byte Ethernet header) copied into `dst`, or
 * 0 if no frame of ours was waiting. Non-blocking. */
int  attn_eth_recv(u8 *dst, int max);

/* Send `len` payload bytes to whoever last talked to us. Blocks until the
 * descriptor is retired, so `payload` may be reused on return. */
int  attn_eth_send(const u8 *payload, int len);

void attn_eth_forget_host(void);

/* Frames the MAC accepted, those with our ethertype, and those sent. A
 * receiver that drops everything and one that receives nothing look the same
 * from the far end; these tell them apart. */
void attn_eth_stats(u32 *any, u32 *ours, u32 *tx);

/* How many times the receive status had to be cleared. Non-zero means the ring
 * ran dry at least once; a receiver that never recovers is one that was never
 * acknowledged. */
u32 attn_eth_stalls(void);
#endif
