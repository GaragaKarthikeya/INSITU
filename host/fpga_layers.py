"""Every layer's attention on the FPGA, and no prefill anywhere.

WHAT CHANGED FROM ONE-LAYER-WITH-A-PUSHED-CACHE
-----------------------------------------------
The first version ran one layer and pushed a cache image the HOST had built in
numpy. Two things were wrong with that as a demonstration.

  * One layer of sixteen meant every throughput figure was one measurement
    multiplied by sixteen -- an extrapolation wearing a measurement's clothes.
  * A pushed cache meant the rows the block scanned were rows the host had
    computed. The hardware's write path was exercised for the four tokens of a
    test case and for nothing else.

Here the block serves all sixteen layers, and NOTHING is ever loaded. The
prompt is fed one token at a time through the same decode path a generated
token takes, so **every row in every cache was written by the block's own AXI
write master**, from the first token of the prompt to the last token generated.
By the end of a 200-token run that is 3,200 decode steps and 3,200 rows the
host never wrote.

SIXTEEN CACHES, ONE BLOCK
-------------------------
`cache_base` already travels with every step -- `attn_ctrl` has no map of its
own, deliberately -- so sixteen caches cost sixteen base addresses and no
hardware at all. Layer L lives at `base + L * layer_stride`, and the stride is
`8 heads * 4 planes * plane_span` for the chosen capacity.

The regions ARE zeroed before the first token. The argument that they need not
be -- a scan reads exactly `n_tokens` rows, and row `t` is written by step `t`
before step `t` scans it -- is a claim about the block's indices, and leaving
16.9 MB of the self-test probe's pseudo-random bytes underneath it means a
mistake there reads plausible noise rather than zeros. It costs one CMD_LOAD
per layer, once.

WHY THE NUMPY KERNEL STILL RUNS
-------------------------------
`collect()` is what produces the ingress the board is sent -- it does the
projections, RoPE and the rotation taps -- so its answer is a free by-product,
not a second computation. It is used as a per-token agreement check. It is NOT
what the model consumes: the board's numbers go through `to_float` and `W_o'`
and become the layer output.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import QuantConfig
from ..hw.ddr_layout import DdrLayout
from ..hw.vectors import collect, unpack_lanes
from ..ops.project import project


# WHAT THE BITSTREAM IS BUILT FOR. `attn_top` is parameterised at synthesis:
# KEY_BITS = 4, VAL_BITS = 2, so a cache row is 8*4 + 8*2 + 4 = 52 bytes.
#
# `QuantConfig()` DEFAULTS TO THREE-BIT KEYS, and `graft()` takes that default
# unless told otherwise. The host then computed 3-bit key codes for hardware
# that decodes 4-bit ones, and the stored rows could not agree.
#
# It hid beautifully. At T = 1 the softmax is over one element, so the output is
# the decoded VALUE alone and the key cannot affect it -- token 1 matched every
# time. Values and norms matched throughout because both sides use 2-bit values.
# Only tokens 2 onward differed, identically over both transports, which looked
# like a hardware fault for a day. The self-test never saw it because
# `board_vectors` passes key_bits=4 explicitly.
BOARD_KEY_BITS = 4
BOARD_VALUE_BITS = 2
BOARD_ROW_BYTES = 52


class FpgaLayers:
    """Routes every grafted layer's attention to one board."""

    def __init__(self, model, client, layers, capacity: int,
                 cache_base: int = 0x10000000, verify: bool = True,
                 quiet: bool = False, trace: bool = False, zero: bool = True,
                 key_bits: int = BOARD_KEY_BITS,
                 value_bits: int = BOARD_VALUE_BITS):
        from ..adapters.torch_llama import graft

        self.client = client
        self.capacity = capacity
        self.verify = verify
        self.quiet = quiet
        self.trace = trace
        self.wire_s = 0.0
        self.layers = list(layers)

        graft(model, layers=self.layers, capacity=capacity,
              quant=QuantConfig(key_bits=key_bits, value_bits=value_bits))
        self.mods = {L: model.model.layers[L].self_attn for L in self.layers}

        k0 = self.mods[self.layers[0]].kernel
        m = k0.cfg.model
        lay = DdrLayout.for_quant(k0.cfg.quant, m.head_dim, m.num_kv_heads,
                                  capacity)
        # ASSERTED, NOT ASSUMED. A quantisation the bitstream was not built for
        # produces rows of the wrong length, and the failure that follows looks
        # like a datapath bug rather than a configuration one.
        if lay.row_bytes != BOARD_ROW_BYTES:
            raise ValueError(
                f"the cache row is {lay.row_bytes} B for {key_bits}b/{value_bits}b "
                f"keys/values, but this bitstream is built for "
                f"{BOARD_KEY_BITS}b/{BOARD_VALUE_BITS}b and a {BOARD_ROW_BYTES} B "
                f"row. Rebuild the RTL or pass matching key_bits/value_bits.")
        self.head_stride = lay.describe()["head_stride"]
        self.plane_span = lay.plane_span(0)
        self.layer_stride = m.num_kv_heads * self.head_stride
        self.bases = {L: cache_base + i * self.layer_stride
                      for i, L in enumerate(self.layers)}

        # Counters, so a run can say what actually happened rather than that it
        # finished.
        # ZERO EVERY REGION BEFORE THE FIRST TOKEN.
        #
        # The claim that these do not need clearing -- "a scan reads exactly
        # n_tokens rows and row t is written by step t" -- is a claim about the
        # block's indices, and it was resting on whatever DDR happened to hold.
        # The self-test's long-context probe leaves 16.9 MB of pseudo-random
        # bytes at exactly this address, so a read that strays past n_tokens
        # returns plausible noise instead of zeros.
        #
        # That is also the ONE difference between the self-test's `fresh` case,
        # which passes at T = 1..6 with every row hardware-written, and live
        # inference, which fails from T = 2 with the same geometry and the same
        # code: `fresh` runs before the probe, on a zeroed region.
        #
        # Zeroing costs one CMD_LOAD per layer, once, and removes the variable.
        if zero:
            blank = bytes(self.layer_stride)
            for L in self.layers:
                self.client.load_cache(blank, self.bases[L])
            if not quiet:
                print(f"  zeroed {len(self.layers)} x {self.layer_stride} B "
                      f"of cache")

        self.steps = 0
        self.agree = 0
        self.disagree = 0
        self.dev_us = 0
        self.scan_cycles = 0
        self.starve_cycles = 0
        self.clips = 0
        self.overflows = 0
        self.flagged = 0

        for L in self.layers:
            self._patch(L)

    @property
    def bytes_used(self) -> int:
        return len(self.layers) * self.layer_stride

    def _patch(self, L: int) -> None:
        attn = self.mods[L]
        kern = attn.kernel
        base = self.bases[L]
        outer = self

        # The TYPE, not the instance: `KernelAttention` assigns
        # `__call__ = forward` at class scope and Python resolves dunders on the
        # type, so an instance attribute is ignored and the original numpy path
        # keeps running -- silently, because it produces the same text.
        # `KernelAttention.__new__` builds a fresh subclass per instance, so
        # this is per-layer and not global.
        cls = type(attn)
        orig = cls.forward

        def board_forward(self, hidden_states, *args, **kwargs):
            import torch

            x = hidden_states
            squeeze = x.dim() == 2
            if squeeze:
                x = x.unsqueeze(0)
            if x.shape[1] != 1:
                # The block is decode-only by construction: one token against a
                # cache. Nothing should reach here -- the driver feeds tokens
                # one at a time on purpose -- so it is loud rather than a quiet
                # fallback to numpy.
                raise RuntimeError(
                    f"layer {L} got {x.shape[1]} tokens at once; this path is "
                    f"one token at a time, and a silent numpy fallback would "
                    f"produce the same text with the board idle")

            row = x[0].detach().cpu().float().numpy()
            n_tokens = kern.cache.length + 1

            vs = collect(kern, row)
            t0 = time.time()
            got, c = outer.client.step(vs.ingress_bytes(), n_tokens, base,
                                       outer.head_stride, outer.plane_span)
            dt = time.time() - t0
            outer.steps += 1
            outer.wire_s += dt
            if outer.trace:
                print(f"    L{L:<2} tok {n_tokens:<4} {dt * 1000:7.1f} ms wire, "
                      f"{c.get('dev_us', 0):5d} us device")
            elif dt > 0.25:
                # A step should be a few milliseconds. Anything near the socket
                # timeout means frames are being lost or ignored, and saying so
                # per-step localises it to a layer and a token.
                print(f"    slow: L{L} token {n_tokens} took {dt * 1000:.0f} ms")
            outer.dev_us += c.get("dev_us", 0)
            outer.scan_cycles += c.get("scan_cycles", 0)
            outer.starve_cycles += c.get("starve_cycles", 0)
            outer.clips += c.get("clips", 0)
            outer.overflows += c.get("overflows", 0)
            if c.get("flags", 0):
                outer.flagged += 1

            mm = kern.cfg.model
            board = unpack_lanes(got, kern.cfg.fmt.qk_width,
                                 mm.num_heads * mm.head_dim)
            if outer.verify:
                if np.array_equal(board, vs.out_online.reshape(-1)):
                    outer.agree += 1
                else:
                    outer.disagree += 1
                    n_bad = int(np.count_nonzero(board != vs.out_online.reshape(-1)))
                    if not outer.quiet:
                        print(f"    !! layer {L} token {n_tokens}: {n_bad} of "
                              f"{board.size} channels differ from the host")

            acc = kern.qk_q.to_float(board[None])
            y = project(kern.weights.o, acc, unit="o_array",
                        bias=kern.weights.o_bias)
            o = torch.from_numpy(np.ascontiguousarray(y)).to(
                device=x.device, dtype=hidden_states.dtype)
            self._advance_hf_cache(kwargs.get("past_key_values"), x)
            return (o if squeeze else o.unsqueeze(0)), None

        cls.forward = board_forward
        cls.__call__ = board_forward
        cls._numpy_forward = orig

    # -- reporting ----------------------------------------------------------
    def summary(self) -> str:
        if self.steps == 0:
            return ("!!! THE BOARD WAS NEVER ASKED. Every number above is the "
                    "host's.")
        lines = [
            f"decode steps on the FPGA: {self.steps} "
            f"({len(self.layers)} layers x {self.steps // max(len(self.layers), 1)} tokens)",
        ]
        if self.verify:
            lines.append(f"  agreement with the host: {self.agree} exact, "
                         f"{self.disagree} different")
        lines.append(f"  device time: {self.dev_us / 1000:.1f} ms total, "
                     f"{self.dev_us / self.steps:.0f} us/step")
        lines.append(f"  wire time:   {self.wire_s * 1000:.0f} ms total, "
                     f"{self.wire_s / self.steps * 1000:.1f} ms/step "
                     f"({self.dev_us / 1000 / max(self.wire_s * 1000, 1e-9) * 100:.1f}% "
                     f"of it is the block)")
        c = self.client
        if getattr(c, "rx_frames", None) is not None:
            lines.append(f"  frames seen {c.rx_frames}, dropped as ours "
                         f"{c.rx_dropped_self}, dropped by seq {c.rx_dropped_seq}"
                         f"{'' if getattr(c, '_ignore_outgoing', False) else ' (no PACKET_IGNORE_OUTGOING)'}")
        lines.append(f"  scan: {self.scan_cycles} cycles, "
                     f"{self.starve_cycles} starving "
                     f"({100.0 * self.starve_cycles / max(self.scan_cycles, 1):.2f}%)")
        lines.append(f"  clips {self.clips}, overflows {self.overflows}, "
                     f"steps with a range/saturation flag {self.flagged}")
        return "\n".join(lines)
