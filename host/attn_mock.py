"""An x86 stand-in for the board, so the wire format can be wrong on a laptop.

WHAT THIS PROVES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
It proves the FORMAT and the CLIENT: that a request the host builds parses at
the other end, that the payload survives as the byte stream the core consumes,
that the sequence number comes back, that a short or malformed frame is
refused rather than half-read, and that the client reassembles a reply
correctly. `plan.MD` asks for exactly that -- "validate the wire format and the
host client before board-side software exists".

It does NOT recompute attention. It answers from the (ingress, golden) pairs
`hw/board_vectors.py` already produces, looked up by the token it was sent.

That is a deliberate refusal. Answering properly would mean unpacking the wire
q/k/v, rotating, quantising, appending to a cache and running the online scan
-- which is the kernel's own internals written a second time, in the one place
in this project that has no test of its own. A second golden model that agreed
with the first would prove nothing and would drift the first time either
changed. The semantics are checked where they are already checked: the board
against `out.online.hex`, and `out.online.hex` against the numpy kernel.

So a mismatch here means the transport is wrong. It cannot mean attention is.

    python -m kernel.host.attn_mock [--port 7001]
"""

from __future__ import annotations

import argparse
import socketserver
import struct

from . import attn_proto as P


class Table:
    """The (token -> result) pairs, keyed by the token's bytes."""

    def __init__(self, cases=None):
        self.by_token: dict[bytes, bytes] = {}
        self.meta: dict[bytes, dict] = {}
        self.loads: list[tuple[int, int]] = []      # (base, length)
        if cases:
            for c in cases:
                self.add_case(c)

    def add_case(self, case: dict) -> None:
        ing = case["ingress"].tobytes()
        gold = case["golden"].tobytes()
        for i, n in enumerate(case["n_tokens"]):
            tok = ing[i * P.TOKEN_BYTES:(i + 1) * P.TOKEN_BYTES]
            self.by_token[tok] = gold[i * P.RESULT_BYTES:(i + 1) * P.RESULT_BYTES]
            self.meta[tok] = {
                "n_tokens": int(n),
                "head_stride": case["head_stride"],
                "plane_span": case["plane_span"],
            }


def serve_one(table: Table, buf: bytes) -> bytes:
    """One request in, one response out. The whole protocol, as a function.

    Written as a pure function so it can be tested without a socket -- the
    socket is the part least likely to be wrong and the most annoying to set
    up in a test.
    """
    try:
        h, body = P.parse_request(buf)
    except ValueError:
        return P.response(P.EBADLEN)

    if h["magic"] != P.MAGIC:
        return P.response(P.EBADMAGIC, seq=h.get("seq", 0))
    if h["version"] != P.VERSION:
        return P.response(P.EBADVER, seq=h["seq"])

    if h["cmd"] == P.CMD_PING:
        return P.response(P.OK, seq=h["seq"])

    if h["cmd"] == P.CMD_LOAD:
        table.loads.append((h["cache_base"], len(body)))
        return P.response(P.OK, seq=h["seq"])

    if h["cmd"] != P.CMD_STEP:
        return P.response(P.EBADCMD, seq=h["seq"])

    if len(body) != P.TOKEN_BYTES:
        return P.response(P.EBADLEN, seq=h["seq"])

    got = table.by_token.get(bytes(body))
    if got is None:
        # An unknown token is a transport failure here, not a compute one: the
        # host sent bytes the table was not built from, which usually means the
        # payload was mangled on the way.
        return P.response(P.ETIMEOUT, seq=h["seq"])

    m = table.meta[bytes(body)]
    if h["n_tokens"] != m["n_tokens"] or h["head_stride"] != m["head_stride"] \
            or h["plane_span"] != m["plane_span"]:
        # The geometry travelled with the token and must match the token. A
        # host that sent the right bytes with the wrong stride would get the
        # wrong answer from real hardware and silence from a mock that ignored
        # this.
        return P.response(P.EBADLEN, seq=h["seq"])

    return P.response(P.OK, got, seq=h["seq"],
                      dev_us=0, scan_cycles=0, busy_cycles=0)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            head = _recv_exactly(self.request, P.REQ.size)
            if head is None:
                return
            n = struct.unpack_from("<I", head, P.REQ.size - 8)[0]
            body = _recv_exactly(self.request, n) if n else b""
            if body is None:
                return
            self.request.sendall(serve_one(self.server.table, head + body))


def _recv_exactly(sock, n: int):
    """TCP is a stream: a `recv` that returns fewer bytes is normal, not an error.

    Treating a short read as a whole message is the classic way a protocol that
    works on a loopback fails on a real network, where the segmentation is
    different.
    """
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, addr, table: Table):
        super().__init__(addr, _Handler)
        self.table = table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=P.PORT)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args(argv)

    from ..hw.board_vectors import build_case

    print(f"building the answer table: ctx {a.ctx}, {a.steps} steps")
    table = Table([build_case(a.ctx, a.steps)])
    print(f"{len(table.by_token)} tokens; listening on {a.port}")
    Server(("0.0.0.0", a.port), table).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
