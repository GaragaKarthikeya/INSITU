/* The attention server's transport, on lwIP over UDP. See attn_net.c. */
#ifndef ATTN_NET_H
#define ATTN_NET_H
#include "xil_types.h"

/* Called with a whole reassembled request; returns the number of bytes written
 * into `out`, or <= 0 to send nothing. The handler owns the protocol; this
 * file owns the wire. */
typedef int (*attn_net_handler_t)(void *ctx, const u8 *req, u32 req_len,
                                  u8 *out, u32 out_max);

int  attn_net_init(attn_net_handler_t fn, void *ctx);
void attn_net_poll(void);
u32  attn_net_served(void);
#endif
