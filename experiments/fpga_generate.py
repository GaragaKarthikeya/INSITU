"""Generate text with one layer's attention running on the ZCU104. Live.

WHAT IS ACTUALLY ON THE FPGA
---------------------------
One decoder layer's attention: rotate, quantise, cache write, score, online
softmax, accumulate, and the final divide. The board holds that layer's KV
cache in its own DDR and scans it for every token. The host runs everything
else -- the other fifteen layers, all four projections, RoPE, the MLPs -- which
is the split `plan.MD` specifies.

The FPGA's numbers are USED, not checked against. `kernel.py`'s seam is:

    merged = merge_heads(out)          <- this is what the FPGA returns
    acc    = qk_q.to_float(merged)
    y      = project(W_o', acc)

so the board's 2,048 lanes go through `to_float` and `W_o'` and become the
layer's output, and the next token is produced from them. A wrong answer on the
board changes the text.

The numpy kernel runs alongside anyway, because `collect` is what produces the
ingress the board is sent, and its answer is therefore free. It is reported as
a per-token agreement check -- but it is not what the model consumes.

PREFILL IS ON THE HOST, AND THAT IS NOT A CHEAT
-----------------------------------------------
`attn_top` is a DECODE block: one token against a cache, by construction. So
the prompt is prefilled by the numpy kernel, the resulting cache image is
pushed to the board's DDR once, and every generated token from then on is the
board's. That is the same division of labour the design was built for -- decode
is where the memory wall is, and prefill is compute-bound and not what this
project is about.

    python -m kernel.experiments.fpga_generate --layer 10 --tokens 16
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernel.host.attn_jtag import JtagClient          # noqa: E402
from kernel.hw.ddr_layout import DdrLayout            # noqa: E402
from kernel.hw.vectors import collect, unpack_lanes   # noqa: E402
from kernel.ops.project import project                # noqa: E402

MODEL = str(pathlib.Path(__file__).resolve().parents[1] / "models" / "Llama-3.2-1B")
PROMPT = "The history of computing hardware spans centuries, from mechanical"
CACHE_BASE = 0x10000000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cpu-only", action="store_true",
                    help="do not touch the board; run the numpy kernel alone")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kernel.adapters.torch_llama import graft

    print(f"loading {a.model}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()

    ids = tok(a.prompt, return_tensors="pt")["input_ids"]
    n_prompt = ids.shape[1]
    # The cache must hold the prompt AND everything generated. `capacity` is a
    # graft argument, so it is set here rather than by rebuilding the cache
    # afterwards -- which is what the DDR image's plane spans are computed from
    # and therefore has to be right before the first token, not after.
    capacity = n_prompt + a.tokens + 1
    graft(model, layers=[a.layer], capacity=capacity)
    attn = model.model.layers[a.layer].self_attn
    k = attn.kernel
    m = k.cfg.model

    print(f"prefilling {n_prompt} tokens on the host (the block is decode-only)")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())

    lay = DdrLayout.for_quant(k.cfg.quant, m.head_dim, m.num_kv_heads, capacity)
    ctx0 = k.cache.length
    image = lay.image(k.cache.buf[:ctx0])
    print(f"  cache: {ctx0} tokens, {image.size} B, stride {lay.describe()['head_stride']}, "
          f"span {lay.plane_span(0)}")

    client = None
    if not a.cpu_only:
        print("connecting to the board over JTAG")
        client = JtagClient()
        t0 = time.time()
        client.load_cache(image.tobytes(), CACHE_BASE)
        print(f"  pushed the cache image in {time.time() - t0:.1f} s")

    # ---- the decode loop, with the board in it ----------------------------
    text_ids = list(ids[0].tolist()) + [nxt]
    agree = disagree = 0
    dev_us = 0

    orig_forward = attn.forward

    def board_forward(hidden_states, *args, **kwargs):
        """One decode step: ask the FPGA, use what it says."""
        nonlocal agree, disagree, dev_us
        x = hidden_states
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        if x.shape[1] != 1 or client is None:
            return orig_forward(hidden_states, *args, **kwargs)

        row = x[0].detach().cpu().float().numpy()
        n_tokens = k.cache.length + 1

        # `collect` runs the numpy step AND taps the wire vectors, so it both
        # advances this side's cache and produces exactly the bytes the board
        # consumes. The board advances its own cache the same way, by writing
        # its row to DDR before it scans.
        vs = collect(k, row)
        got, counters = client.step(vs.ingress_bytes(), n_tokens, CACHE_BASE,
                                    lay.describe()["head_stride"], lay.plane_span(0))
        dev_us += counters["dev_us"]

        board = unpack_lanes(got, k.cfg.fmt.qk_width, m.num_heads * m.head_dim)
        if np.array_equal(board, vs.out_online.reshape(-1)):
            agree += 1
        else:
            disagree += 1
            bad = int(np.count_nonzero(board != vs.out_online.reshape(-1)))
            print(f"    !! token {len(text_ids)}: {bad} of {board.size} channels "
                  f"differ from the host")
        if counters["flags"]:
            print(f"    !! flags {counters['flags']:#x} (range/norm saturation)")
        if counters["clips"] or counters["overflows"]:
            print(f"    !! clips {counters['clips']} overflows {counters['overflows']}")

        # THE BOARD'S NUMBERS, through the rest of the seam.
        acc = k.qk_q.to_float(board[None])
        y = project(k.weights.o, acc, unit="o_array", bias=k.weights.o_bias)
        import torch as _t
        o = _t.from_numpy(np.ascontiguousarray(y)).to(
            device=x.device, dtype=hidden_states.dtype)
        attn._advance_hf_cache(kwargs.get("past_key_values"), x)
        return (o if squeeze else o.unsqueeze(0)), None

    attn.forward = board_forward
    attn.__call__ = board_forward

    print(f"generating {a.tokens} tokens"
          + ("" if client else " (numpy only, --cpu-only)"))
    t0 = time.time()
    with torch.no_grad():
        for i in range(a.tokens):
            step = torch.tensor([[text_ids[-1]]])
            out = model(input_ids=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
            text_ids.append(int(out.logits[0, -1].argmax()))
            print(f"  [{i + 1}/{a.tokens}] {tok.decode(text_ids[n_prompt:])!r}")
    wall = time.time() - t0

    print()
    print("=" * 70)
    print(tok.decode(text_ids, skip_special_tokens=True))
    print("=" * 70)
    if client:
        print(f"tokens on the FPGA: {agree} agreed with the host, {disagree} did not")
        print(f"device time: {dev_us} us total, {dev_us / max(a.tokens, 1):.0f} us/token "
              f"(the block itself)")
        print(f"wall: {wall:.1f} s, {wall / max(a.tokens, 1):.2f} s/token "
              f"(JTAG transport, not the block)")
        client.quit()
    return 0 if disagree == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
