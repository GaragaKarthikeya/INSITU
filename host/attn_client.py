"""The host's side: send a decode step, get 2,048 channels back.

Two transports, one caller
--------------------------
`TcpClient` talks to `attn_mock` on this machine. `RawEthClient` talks to the
board over a raw Ethernet frame, which is what `plan.MD` sizes the system
against -- it takes the round trip from ~250 us to ~35 us, worth about 17% at
ctx 32k. They present the same `step()` so the thing above them cannot tell
which it is holding, and the mock is therefore a real rehearsal rather than a
different code path that happens to look similar.

What the host sends is what the core eats
-----------------------------------------
`step()` takes `VectorSet.ingress_bytes()` and puts it on the wire unchanged.
There is no encoding step here and there must not be one: the A53 hands the
received bytes to the DMA without touching them, and a transform on this side
would have to be undone on that one, on the critical path.
"""

from __future__ import annotations

import socket
import struct
import time

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
        # The sequence number is checked, not assumed. A reply to the previous
        # request is otherwise indistinguishable from a wrong answer to this
        # one -- and on a lossy link that is the failure that actually happens.
        if h["seq"] != self._seq:
            raise RuntimeError(f"reply to seq {h['seq']}, expected {self._seq}")
        if len(body) != P.RESULT_BYTES:
            raise RuntimeError(f"result is {len(body)} B, expected {P.RESULT_BYTES}")
        return body, h

    # The board reassembles into a buffer sized for a token -- 64 +
    # ATTN_TOKEN_BYTES = 9,280 bytes -- and silently drops any fragment that
    # would run past it. A load bigger than that never completes, never
    # replies, and times out with "0 of ? fragments", which says nothing about
    # the actual limit. So the host does the splitting.
    LOAD_CHUNK = 8192

    def load(self, image: bytes, cache_base: int) -> None:
        for off in range(0, len(image), self.LOAD_CHUNK):
            part = image[off:off + self.LOAD_CHUNK]
            self._seq = getattr(self, "_seq", 0) + 1
            h, _ = self._exchange(
                P.load_request(part, cache_base + off, seq=self._seq))
            if h["status"] != P.OK:
                raise RuntimeError(
                    f"load of {len(part)} B at {cache_base + off:#x} returned "
                    f"{h['status']} ({P.STATUS_NAME.get(h['status'], '?')})")

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


class UdpClient(_Base):
    """The board over UDP, with lwIP on the far end.

    The simplest of the three, and that is the point.
    `RawEthClient` fragments by hand because a 9 KB message does not fit an
    Ethernet frame; `JtagClient` drives xsdb over pipes. Here the stack does
    the fragmenting and reassembly, so this is a `sendto` and a `recvfrom`.
    No root, no raw sockets, no MTU to match, and no descriptor ring on either
    side that this project wrote.
    """

    def __init__(self, host: str = "192.168.10.2", port: int = P.PORT,
                 timeout: float = 5.0):
        self.peer = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        # A 9 KB datagram is several IP fragments; the buffers need room for the
        # reassembled whole, not for one fragment.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _exchange(self, req: bytes) -> tuple[dict, bytes]:
        seq = struct.unpack_from("<I", req, P.REQ.size - 4)[0]
        self.sock.sendto(req, self.peer)
        deadline = time.time() + self.sock.gettimeout()
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                raise TimeoutError(
                    f"no reply from {self.peer[0]}:{self.peer[1]}. Is the board "
                    f"running the lwIP server (UART says 'NET SERVER READY')?"
                ) from None
            h, body = P.parse_response(data)
            # A reply to an earlier request is indistinguishable from a wrong
            # answer to this one unless the sequence number is checked.
            if h["seq"] == seq:
                return h, body
            if time.time() > deadline:
                raise TimeoutError(f"only saw replies to other requests, last "
                                   f"seq {h['seq']}, wanted {seq}")


class RawEthClient(_Base):
    """Raw Ethernet frames, no IP, no OS on the far end.

    A 9,216-byte token does not fit a 1,500-byte frame, so a request is
    Fragmented: the header travels in the first frame and the payload follows
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
    # Jumbo: matches ATTN_ETH_MTU in attn_proto.h. A reply is one frame at this
    # size instead of five, and five-frame replies are what the board's
    # receiver could not survive.
    ETH_MTU = 9000
    MTU_PAYLOAD = ETH_MTU - FRAG_HDR.size

    def __init__(self, iface: str, peer_mac: bytes, timeout=2.0):
        if len(peer_mac) != 6:
            raise ValueError("peer_mac must be 6 bytes")
        self.iface, self.peer = iface, peer_mac
        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                                  socket.htons(P.ETHERTYPE))
        self.sock.bind((iface, P.ETHERTYPE))
        self.sock.settimeout(timeout)
        self.src = self.sock.getsockname()[4][:6]
        # A jumbo frame that the interface will not carry is silently dropped by
        # the kernel, so the MTU is checked rather than assumed.
        try:
            with open(f"/sys/class/net/{iface}/mtu") as f:
                mtu = int(f.read().strip())
            if mtu < self.ETH_MTU:
                raise RuntimeError(
                    f"{iface} has MTU {mtu}; this protocol needs at least "
                    f"{self.ETH_MTU}. Run: sudo ip link set {iface} mtu "
                    f"{self.ETH_MTU}")
        except FileNotFoundError:
            pass
        # An AF_PACKET socket sees its own transmissions.
        # `eth_probe` proved it: five pings sent, ten frames of our ethertype
        # observed. Without this the receive loop reassembles the request it
        # just sent as if it were the reply -- and a request parsed as a
        # response is not obviously wrong, because the magic matches and the
        # `version` field lands where `status` is read.
        # SOL_PACKET is 263 and PACKET_IGNORE_OUTGOING is 23. Neither is
        # exposed by every Python build, so they are written as numbers, and
        # the guard catches AttributeError as well as OSError -- an older
        # kernel refuses the option and an older Python has no name for it.
        try:
            self.sock.setsockopt(getattr(socket, "SOL_PACKET", 263), 23, 1)
            self._ignore_outgoing = True
        except (OSError, AttributeError):
            self._ignore_outgoing = False    # the MAC filter below is the real guard
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
            # By source MAC, always. PACKET_IGNORE_OUTGOING is a recent
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
