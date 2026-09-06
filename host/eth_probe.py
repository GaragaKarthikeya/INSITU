"""Say what is actually wrong with the Ethernet link, in one command.

    sudo <venv>/bin/python -m kernel.host.eth_probe --eth enp4s0

Why this exists
---------------
`RawEthClient.ping()` either works or raises `TimeoutError`, and a timeout is
consistent with at least six different faults: the board is running an older
ELF, its PHY never came up, the MAC filter rejects our frames, our ethertype is
wrong, the interface is down on this side, or the reply is being sent and
dropped. Reading a Python traceback distinguishes none of them.

So this listens promiscuously, on every ethertype, and reports what it saw
rather than what it expected. The three questions it answers, in order:

  1. Is this side able to transmit at all?
  2. Does anything arrive from the board's MAC -- on any ethertype?
  3. Does a reply arrive with our ethertype, and does it parse?

A "no" at each step has a different next action, and the verdict says which.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from kernel.host import attn_proto as P            # noqa: E402

ETH_P_ALL = 0x0003


def iface_state(name: str) -> dict:
    def read(p, default="?"):
        try:
            with open(f"/sys/class/net/{name}/{p}") as f:
                return f.read().strip()
        except OSError:
            return default
    return {
        "exists": os.path.isdir(f"/sys/class/net/{name}"),
        "operstate": read("operstate"),
        "carrier": read("carrier"),
        "speed": read("speed"),
        "mac": read("address"),
        "mtu": read("mtu"),
    }


def probe(iface: str, peer: bytes, seconds: float = 3.0, tries: int = 5) -> int:
    st = iface_state(iface)
    print(f"interface {iface}: operstate={st['operstate']} carrier={st['carrier']} "
          f"speed={st['speed']} mac={st['mac']} mtu={st['mtu']}")
    if not st["exists"]:
        print(f"\nVERDICT: no interface called {iface}. `ip -br link` lists them.")
        return 2
    if st["carrier"] != "1":
        print("\nVERDICT: NO CARRIER. The cable is out, or the board's PHY is not "
              "driving the link. Nothing software-side can fix this.")
        return 2

    # Promiscuous, every ethertype -- so a reply with the wrong ethertype is
    # visible as a reply rather than as silence.
    rx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETH_P_ALL))
    rx.bind((iface, 0))
    rx.settimeout(0.25)
    src = rx.getsockname()[4][:6]
    print(f"listening promiscuously on {iface} (host MAC "
          f"{':'.join(f'{b:02x}' for b in src)})")

    tx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(P.ETHERTYPE))
    tx.bind((iface, P.ETHERTYPE))

    frag = struct.Struct("<IHH")
    eth = peer + src + struct.pack("!H", P.ETHERTYPE)

    from_board = 0
    ours = 0
    parsed = 0
    seen_types: dict[int, int] = {}
    sample = []

    for attempt in range(1, tries + 1):
        req = P.ping_request(seq=attempt)
        tx.send(eth + frag.pack(attempt, 0, 1) + req)
        print(f"  ping {attempt}: sent {len(req)} B "
              f"({len(eth) + frag.size + len(req)} B on the wire)")

        end = time.time() + seconds / tries
        while time.time() < end:
            try:
                f = rx.recv(2048)
            except socket.timeout:
                continue
            if len(f) < 14:
                continue
            dst, s6 = f[:6], f[6:12]
            et = (f[12] << 8) | f[13]
            seen_types[et] = seen_types.get(et, 0) + 1
            if s6 != peer:
                continue
            from_board += 1
            if len(sample) < 4:
                sample.append((et, len(f)))
            if et != P.ETHERTYPE:
                continue
            ours += 1
            body = f[14:]
            if len(body) < frag.size:
                continue
            _seq, _fr, _nf = frag.unpack_from(body)
            try:
                h, _ = P.parse_response(body[frag.size:])
            except ValueError as e:
                print(f"    a frame with our ethertype did not parse: {e}")
                continue
            parsed += 1
            print(f"    REPLY: status {h['status']} "
                  f"({P.STATUS_NAME.get(h['status'], '?')}) seq {h['seq']}")

    print()
    print(f"frames from the board's MAC: {from_board}")
    print(f"  of those, ethertype {P.ETHERTYPE:#06x}: {ours}, parsed as a "
          f"response: {parsed}")
    if seen_types:
        types = ", ".join(f"{t:#06x}x{n}" for t, n in
                          sorted(seen_types.items(), key=lambda kv: -kv[1])[:6])
        print(f"  all ethertypes seen on the wire: {types}")
    if sample:
        print(f"  first frames from the board: {sample}")

    print()
    if parsed:
        print("VERDICT: the link works and the board answers. If the client still "
              "times out, the fault is above the transport.")
        return 0
    if ours:
        print("VERDICT: frames come back with the right ethertype but do not parse "
              "as a response.\n"
              "  The board is alive and replying -- this is a FORMAT mismatch.\n"
              "  Check attn_proto.h against attn_proto.py "
              "(tests/test_proto.py compares them).")
        return 1
    if from_board:
        print("VERDICT: the board transmits, but not with our ethertype.\n"
              "  It is running SOMETHING -- most likely an older ELF, or its\n"
              "  Ethernet init failed and it fell back to the JTAG mailbox.\n"
              "  Look at the UART for 'ETH: link up' and 'ETH SERVER READY'.")
        return 1
    print("VERDICT: nothing at all from the board.\n"
          "  Carrier is up, so the PHY is talking to this NIC -- the board is not\n"
          "  sending. In order of likelihood:\n"
          "    1. the A53 is not running the current ELF\n"
          "         -> xsdb host/run_attn_server.tcl, and watch the UART\n"
          "    2. attn_eth_init() failed and it fell back to the JTAG mailbox\n"
          "         -> the UART says which, and why\n"
          "    3. it never received our frame, so it has no MAC to reply to\n"
          "         -> the board learns the host MAC from the first frame it\n"
          "            accepts; if its RX filter drops us it can never answer")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eth", default="enp4s0")
    ap.add_argument("--mac", default="02:00:5a:77:e0:01")
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args(argv)
    if os.geteuid() != 0:
        print("raw sockets need root: run under sudo with the venv's python")
        return 2
    return probe(a.eth, bytes(int(b, 16) for b in a.mac.split(":")), a.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
