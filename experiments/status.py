"""Two-second snapshot of a running sweep. `kernel/experiments/status.sh`

Reads status.json and events.jsonl and prints what is happening, whether it is
alive, and what has been measured so far. Reads only -- it never touches the
run, so it is safe to call at any time from any shell.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

C = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "b": "\033[1m",
     "d": "\033[2m", "x": "\033[0m"}


def bar(done: int, total: int, width: int = 34) -> str:
    if total <= 0:
        return " " * width
    fill = int(width * done / total)
    return "█" * fill + "·" * (width - fill)


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
    else:
        # Newest run wins. With several experiments in flight, "show me the one
        # that just wrote something" is almost always what is meant, and having
        # to remember a path is exactly the friction that stops anyone looking.
        found = sorted(Path("kernel/experiments/results").rglob("status.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            print("no runs found under kernel/experiments/results/")
            return 1
        d = found[0].parent
        if len(found) > 1:
            others = ", ".join(str(p.parent.name) for p in found[1:5])
            print(f"{C['d']}showing {d}  (also: {others}){C['x']}")
    sp, ep = d / "status.json", d / "events.jsonl"
    if not sp.exists():
        print(f"no run at {d} (status.json absent)")
        return 1
    s = json.loads(sp.read_text())

    # LIVENESS FIRST. Everything else is meaningless if it is wedged: the
    # heartbeat fires once per transformer layer, so a stale stamp is a hang,
    # not slowness.
    age = time.time() - s.get("updated_at", 0)
    phase = s.get("phase", "?")
    if phase in ("done", "aborted"):
        tag = f"{C['g' if phase == 'done' else 'r']}{phase.upper()}{C['x']}"
    elif age < 90:
        tag = f"{C['g']}ALIVE{C['x']}"
    elif age < 600:
        tag = f"{C['y']}STALLED {human(age)}{C['x']}"
    else:
        tag = f"{C['r']}NO HEARTBEAT {human(age)}{C['x']}"

    shape = (f"{s.get('layers')}L {s.get('kv_heads')}KV d{s.get('head_dim')}"
             if s.get("layers") else f"group '{s.get('group','?')}'")
    print(f"{C['b']}{s.get('model', d.name)}{C['x']}  ctx {s.get('context','?')}  "
          f"{s.get('dtype','')}  {shape}   {tag}")
    print(f"{C['d']}pid {s.get('pid')}  running {human(s.get('elapsed_h', 0) * 3600)}"
          f"  last tick {human(age)} ago{C['x']}\n")

    done, total = s.get("done_windows", 0), s.get("total_windows", 0)
    pct = 100 * done / total if total else 0
    print(f"  {bar(done, total)}  {done}/{total} windows  {pct:5.1f}%"
          f"   ETA {s.get('eta_h', '?')} h")
    extra = (f"round {s.get('round')} chunk {s.get('chunk')}" if s.get("round")
             else f"variant {s.get('variant_i','-')}/{s.get('total_variants','-')}")
    print(f"  phase {phase}  {extra}"
          f"   now: {C['b']}{s.get('config','-')}{C['x']} "
          f"window {s.get('window','-')}/{s.get('window_target','-')} "
          f"layer {s.get('layer','-')}/{s.get('layers_total','-')} "
          f"({s.get('activity','-')})")

    rss, avail = s.get("rss_gb", 0), s.get("mem_available_gb", 0)
    col = "r" if rss > 44 else ("y" if rss > 38 else "g")
    print(f"  memory {C[col]}{rss:.1f} GB{C['x']} resident "
          f"(peak {s.get('rss_peak_gb', 0):.1f}, {avail:.1f} GB free)\n")

    res = d / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        base = r.get("baseline", {}).get("ppl")
        print(f"  {'config':<9} {'ppl':>9} {'vs base':>9} {'windows':>9} "
              f"{'B/tok/L':>9} {'hours':>6}")
        for k, v in r.items():
            if k == "meta" or "ppl" not in v:
                continue
            ratio = f"{v['ppl'] / base:8.4f}x" if base else "        -"
            print(f"  {k:<9} {v['ppl']:9.4f} {ratio} {v['windows']:9d} "
                  f"{str(v.get('bytes_per_token_per_layer','-')):>9} "
                  f"{v['seconds'] / 3600:6.2f}")

    for a in s.get("alarms", []):
        col = "r" if a["level"] == "abort" else "y"
        print(f"\n  {C[col]}{a['level'].upper()} [{a['code']}]{C['x']} {a['message']}")

    if ep.exists():
        evs = [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
        interesting = [e for e in evs if e["kind"] != "window"][-4:]
        if interesting:
            print(f"\n  {C['d']}recent events{C['x']}")
            for e in interesting:
                print(f"  {C['d']}{e['elapsed_h']:6.2f}h  {e['kind']}"
                      f"{'  ' + str(e.get('message', e.get('config', '')))}{C['x']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
