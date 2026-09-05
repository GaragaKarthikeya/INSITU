"""The host's side: send a decode step, get 2,048 channels back.

TWO TRANSPORTS, ONE CALLER
--------------------------
`TcpClient` talks to `attn_mock` on this machine. `RawEthClient` talks to the
board over a raw Ethernet frame, which is what `plan.MD` sizes the system
against -- it takes the round trip from ~250 us to ~35 us, worth about 17% at
ctx 32k. They present the same `step()` so the thing above them cannot tell
which it is holding, and the mock is therefore a real rehearsal rather than a
different code path that happens to look similar.

WHAT THE HOST SENDS IS WHAT THE CORE EATS
-----------------------------------------
`step()` takes `VectorSet.ingress_bytes()` and puts it on the wire unchanged.
There is no encoding step here and there must not be one: the A53 hands the
received bytes to the DMA without touching them, and a transform on this side
would have to be undone on that one, on the critical path.
"""

from __future__ import annotations

import socket
import struct

from . import attn_proto as P


class _Base:
    def step(self, token: bytes, n_tokens: int, cache_base: int,
             head_stride: int, plane_span: int) -> tuple[bytes, dict]:
        """One decode step. Returns (6,144 result bytes, the device's counters)."""
        self._seq = getattr(self, "_seq", 0) + 1
        req = P.step_request(token, n_tokens, cache_base, head_stride,
                             plane_span, seq=self._seq)
        h, body = self._exchange(req)
        if h["status"] != P.OK:
            raise RuntimeError(
                f"device returned {h['status']} "
                f"({P.STATUS_NAME.get(h['status'], 'unknown')})")
        # The sequence number is checked, not assumed. A reply to the PREVIOUS
        # request is otherwise indistinguishable from a wrong answer to this
        # one -- and on a lossy link that is the failure that actually happens.
        if h["seq"] != self._seq:
            raise RuntimeError(f"reply to seq {h['seq']}, expected {self._seq}")
        if len(body) != P.RESULT_BYTES:
            raise RuntimeError(f"result is {len(body)} B, expected {P.RESULT_BYTES}")
        return body, h

    def load(self, image: bytes, cache_base: int) -> None:
        self._seq = getattr(self, "_seq", 0) + 1
        h, _ = self._exchange(P.load_request(image, cache_base, seq=self._seq))
        if h["status"] != P.OK:
            raise RuntimeError(f"load returned {h['status']}")

    # `JtagClient` calls it `load_cache`; the name is aliased rather than
    # picked, so a caller can hold any transport without knowing which.
    def load_cache(self, image: bytes, cache_base: int) -> None:
        self.load(image, cache_base)

    def ping(self) -> dict:
        self._seq = getattr(self, "_seq", 0) + 1
        h, _ = self._exchange(P.ping_request(seq=self._seq))
        return h


class TcpClient(_Base):
    """For `attn_mock`, and for a board running a TCP stack during bring-up."""

    def __init__(self, host="127.0.0.1", port=P.PORT, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _exchange(self, req: bytes) -> tuple[dict, bytes]:
        self.sock.sendall(req)
        head = self._recv_exactly(P.RESP.size)
        n = struct.unpack_from("<I", head, 8)[0]      # `payload`
        body = self._recv_exactly(n) if n else b""
        return P.parse_response(head + body)

    def _recv_exactly(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError(f"connection closed with {len(buf)} of {n} B")
            buf += chunk
        return buf


class RawEthClient(_Base):
    """Raw Ethernet frames, no IP, no OS on the far end.

    A 9,216-byte token does not fit a 1,500-byte frame, so a request is
    FRAGMENTED: the header travels in the first frame and the payload follows
    in as many as it takes, each carrying its offset. There is no retransmit --
    the link is a metre of cable between two devices with nothing else on it,
    and a checksum failure is a bug to find rather than a condition to recover
    from. `plan.MD` budgets 35 us of round trip on that basis.

    Requires CAP_NET_RAW (run as root, or `setcap cap_net_raw+ep`).
    """

    #  dst(6) src(6) ethertype(2) | seq(4) frag(2) nfrag(2)
    #
    # `attn_frag_hdr` in `host/attn_proto.h` is the same eight bytes, and
    # `ATTN_MTU_PAYLOAD` there is this same 1,492. Two definitions again, and
    # `tests/test_proto.py` checks them against each other the same way it
    # checks the request header.
    FRAG_HDR = struct.Struct("<IHH")
    MTU_PAYLOAD = 1500 - FRAG_HDR.size

    def __init__(self, iface: str, peer_mac: bytes, timeout=2.0):
        if len(peer_mac) != 6:
            raise ValueError("peer_mac must be 6 bytes")
        self.iface, self.peer = iface, peer_mac
        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                                  socket.htons(P.ETHERTYPE))
        self.sock.bind((iface, P.ETHERTYPE))
        self.sock.settimeout(timeout)
        self.src = self.sock.getsockname()[4][:6]
        # AN AF_PACKET SOCKET SEES ITS OWN TRANSMISSIONS.
        # `eth_probe` proved it: five pings sent, ten frames of our ethertype
        # observed. Without this the receive loop reassembles the REQUEST it
        # just sent as if it were the reply -- and a request parsed as a
        # response is not obviously wrong, because the magic matches and the
        # `version` field lands where `status` is read.
        try:
            self.sock.setsockopt(socket.SOL_PACKET, 23, 1)   # PACKET_IGNORE_OUTGOING
            self._ignore_outgoing = True
        except OSError:
            self._ignore_outgoing = False    # pre-4.20 kernel; the MAC filter covers it
        self.rx_frames = 0
        self.rx_dropped_self = 0
        self.rx_dropped_seq = 0

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _exchange(self, req: bytes) -> tuple[dict, bytes]:
        seq = struct.unpack_from("<I", req, P.REQ.size - 4)[0]
        chunks = [req[i:i + self.MTU_PAYLOAD]
                  for i in range(0, len(req), self.MTU_PAYLOAD)] or [b""]
        eth = self.peer + self.src + struct.pack("!H", P.ETHERTYPE)
        for i, c in enumerate(chunks):
            self.sock.send(eth + self.FRAG_HDR.pack(seq, i, len(chunks)) + c)

        parts: dict[int, bytes] = {}
        nfrag = None
        while nfrag is None or len(parts) < nfrag:
            try:
                frame = self.sock.recv(2048)
            except socket.timeout:
                # A bare TimeoutError is consistent with six different faults
                # and distinguishes none of them.
                raise TimeoutError(
                    f"no reply from {':'.join(f'{b:02x}' for b in self.peer)} "
                    f"on {self.iface} after {len(parts)} of "
                    f"{nfrag if nfrag else '?'} fragments.\n"
                    f"Run the probe, which says WHICH fault this is:\n"
                    f"  sudo <venv>/bin/python -m kernel.host.eth_probe "
                    f"--eth {self.iface}") from None
            body = frame[14:]
            if len(body) < self.FRAG_HDR.size:
                continue
            self.rx_frames += 1
            # BY SOURCE MAC, always -- PACKET_IGNORE_OUTGOING is a recent
            # kernel's convenience and this is the property that must hold.
            if frame[6:12] != self.peer:
                self.rx_dropped_self += 1
                continue
            rseq, frag, nf = self.FRAG_HDR.unpack_from(body)
            if rseq != seq:
                self.rx_dropped_seq += 1
                continue            # a straggler from an earlier request
            nfrag = nf
            parts[frag] = body[self.FRAG_HDR.size:]
        return P.parse_response(b"".join(parts[i] for i in range(nfrag)))
