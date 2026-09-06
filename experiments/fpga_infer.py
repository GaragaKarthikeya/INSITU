"""Llama 3.2 1B with all sixteen layers' attention on the ZCU104. No prefill.

    python -m kernel.experiments.fpga_infer --eth enp4s0 --tokens 24
    python -m kernel.experiments.fpga_infer --jtag --tokens 8
    python -m kernel.experiments.fpga_infer --cpu-only          # the control

There is no prefill stage, and that is the point
------------------------------------------------
The prompt is not run as a batch. Its tokens go through the same one-at-a-time
decode path a generated token takes, so every row of every one of the sixteen
caches is written by the block's own AXI write master. Nothing is loaded from
the host, and by the end of a run the board has done
`16 x (prompt + generated)` decode steps.

That also makes the throughput figure a measurement rather than an
extrapolation: the earlier one-layer runs reported "x16" for a layer count
nothing had executed.

What is measured
----------------
Per step, from the board's own counters: device microseconds, scan cycles,
starve cycles, clips, accumulator overflows, and whether any channel escaped
the 24-bit seam. Plus, with `--verify` (the default), an exactness check
against the numpy kernel.

That check is not free. `collect()` has to run to produce the bytes the board
is sent, but only its ingress half -- the projections, RoPE and the rotation,
all O(1) in context. The numpy attention underneath it is O(context) and
exists only to be compared against: at ctx 4,096 it is 675 ms a layer-step
against 9 ms for the ingress. `--no-verify` stops the kernel at the seam and
skips it, which is where a long run gets its wall clock back.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernel.host.fpga_layers import FpgaLayers            # noqa: E402

MODEL = str(pathlib.Path(__file__).resolve().parents[1] / "models" / "Llama-3.2-1B")
PROMPT = "The history of computing hardware spans centuries, from mechanical"
CACHE_BASE = 0x10000000


def open_client(a):
    if a.cpu_only:
        return None
    if getattr(a, "udp", None):
        from kernel.host.attn_client import UdpClient
        print(f"transport: UDP to {a.udp}:7001 (lwIP on the board)")
        c = UdpClient(a.udp, timeout=a.timeout)
        print(f"  ping: status {c.ping()['status']}")
        return c
    if a.eth:
        from kernel.host.attn_client import RawEthClient
        mac = bytes(int(b, 16) for b in a.mac.split(":"))
        print(f"transport: raw Ethernet on {a.eth} -> {a.mac}")
        c = RawEthClient(a.eth, mac, timeout=a.timeout)
        print(f"  ping: status {c.ping()['status']}")
        return c
    from kernel.host.attn_jtag import JtagClient
    print("transport: JTAG (slow; --eth is the real path)")
    return JtagClient()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", default="", help="default: every layer")
    ap.add_argument("--udp", metavar="IP", nargs="?", const="192.168.10.2",
                    help="UDP to the board's lwIP server (default 192.168.10.2)")
    ap.add_argument("--eth", metavar="IFACE")
    ap.add_argument("--mac", default="02:00:5a:77:e0:01")
    ap.add_argument("--jtag", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--no-zero", action="store_true",
                    help="do not clear the cache regions first (they hold the "
                         "self-test probe's pseudo-random bytes)")
    ap.add_argument("--trace", action="store_true",
                    help="print every board round trip, layer by layer")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-step comparison against numpy. The "
                         "check is O(context) numpy and dominates the wall "
                         "clock of a long run; the ingress the board is sent "
                         "is unchanged, byte for byte.")
    ap.add_argument("--device", default="auto",
                    help="cuda | cpu | auto. The 15/16 of the model that is "
                         "not attention is pure GPU work.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="fp32 by default: the seam converts through numpy "
                         "float32, so bf16 moves the numbers the board is sent")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = ("cuda" if (a.device == "auto" and torch.cuda.is_available())
           else ("cpu" if a.device == "auto" else a.device))
    dt = getattr(torch, a.dtype)
    print(f"loading {a.model} onto {dev} ({a.dtype})")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt)
    model.eval().to(dev)
    if dev.startswith("cuda"):
        print(f"  {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.memory_allocated() / 1e9:.1f} GB resident")

    n_layers = model.config.num_hidden_layers
    layers = ([int(v) for v in a.layers.split(",")] if a.layers
              else list(range(n_layers)))

    ids = tok(a.prompt, return_tensors="pt")["input_ids"][0].tolist()
    capacity = len(ids) + a.tokens + 1

    client = open_client(a)
    if client is None:
        print("--cpu-only: the board is not touched")
        from kernel.adapters.torch_llama import graft
        graft(model, layers=layers, capacity=capacity)
        fpga = None
    else:
        fpga = FpgaLayers(model, client, layers, capacity,
                          cache_base=CACHE_BASE, verify=not a.no_verify,
                          trace=a.trace, zero=not a.no_zero)
        print(f"{len(layers)} layers on the board, capacity {capacity}, "
              f"{fpga.layer_stride} B each, {fpga.bytes_used / 1e6:.1f} MB of DDR "
              f"from {CACHE_BASE:#x}")

    # -- the prompt, one token AT A time -----------------------------------
    print(f"feeding {len(ids)} prompt tokens through the pipeline "
          f"(no prefill; the hardware writes every row)")
    past = None
    t0 = time.time()
    with torch.no_grad():
        for i, t in enumerate(ids):
            out = model(input_ids=torch.tensor([[t]], device=dev),
                        past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            if (i + 1) % 4 == 0 or i + 1 == len(ids):
                print(f"  prompt {i + 1}/{len(ids)}")
    t_prompt = time.time() - t0
    nxt = int(out.logits[0, -1].argmax())

    # -- generation ---------------------------------------------------------
    print(f"generating {a.tokens} tokens")
    gen = [nxt]
    t0 = time.time()
    with torch.no_grad():
        for i in range(a.tokens):
            out = model(input_ids=torch.tensor([[gen[-1]]], device=dev),
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            gen.append(int(out.logits[0, -1].argmax()))
            print(f"  [{i + 1}/{a.tokens}] {tok.decode(gen)!r}")
    t_gen = time.time() - t0

    print()
    print("=" * 72)
    print(tok.decode(ids + gen, skip_special_tokens=True))
    print("=" * 72)
    print(f"prompt: {len(ids)} tokens in {t_prompt:.1f} s")
    print(f"generate: {a.tokens} tokens in {t_gen:.1f} s "
          f"({t_gen / max(a.tokens, 1) * 1000:.0f} ms/token wall)")
    if fpga:
        print(fpga.summary())
        per_tok = fpga.dev_us / max(len(ids) + a.tokens, 1)
        print(f"  device time per TOKEN across all {len(layers)} layers: "
              f"{per_tok:.0f} us -> {1e6 / max(per_tok, 1):.0f} tok/s if the "
              f"transport were free")
        if hasattr(client, "quit"):
            client.quit()
        if fpga.steps == 0 or (not a.no_verify and fpga.disagree):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
