"""The wire format exists in C and in Python, so it is checked against itself.

`host/attn_proto.h` is compiled by the A53's toolchain and `host/attn_proto.py`
runs on the machine with the GPU. Two definitions of one format is the shape of
bug that survives every unit test on both sides, because each end agrees with
itself: a field added to one, a `uint16_t` widened in the other, and the only
symptom is a board that returns plausible garbage.

So this compiles the header with the host compiler and asks IT for the offsets
and sizes, rather than reading them off the source. If `cc` is not available the
checks that need it skip -- loudly -- rather than passing vacuously.
"""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

from kernel.host import attn_proto as P

HDR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "host", "attn_proto.h")

_PROBE = r"""
#include <stdio.h>
#include <stddef.h>
#include "attn_proto.h"
int main(void) {
    printf("req %zu\n", sizeof(attn_req_hdr));
    printf("resp %zu\n", sizeof(attn_resp_hdr));
#define F(s, f) printf("%s.%s %zu\n", #s, #f, offsetof(s, f));
    F(attn_req_hdr, magic) F(attn_req_hdr, version) F(attn_req_hdr, cmd)
    F(attn_req_hdr, cache_base_lo) F(attn_req_hdr, cache_base_hi)
    F(attn_req_hdr, n_tokens)
    F(attn_req_hdr, head_stride) F(attn_req_hdr, plane_span)
    F(attn_req_hdr, payload) F(attn_req_hdr, seq)
    F(attn_resp_hdr, magic) F(attn_resp_hdr, status) F(attn_resp_hdr, rsvd)
    F(attn_resp_hdr, payload) F(attn_resp_hdr, seq) F(attn_resp_hdr, dev_us)
    F(attn_resp_hdr, scan_cycles) F(attn_resp_hdr, busy_cycles)
    F(attn_resp_hdr, starve_cycles) F(attn_resp_hdr, clips)
    F(attn_resp_hdr, overflows)
    printf("magic %u\n", (unsigned)ATTN_MAGIC);
    printf("version %u\n", (unsigned)ATTN_VERSION);
    printf("token %u\n", (unsigned)ATTN_TOKEN_BYTES);
    printf("result %u\n", (unsigned)ATTN_RESULT_BYTES);
    printf("beat %u\n", (unsigned)ATTN_BEAT);
    printf("erange %u\n", (unsigned)ATTN_ERANGE);
    printf("frag %zu\n", sizeof(attn_frag_hdr));
    printf("fraghdr %u\n", (unsigned)ATTN_FRAG_HDR_BYTES);
    printf("mtu %u\n", (unsigned)ATTN_MTU_PAYLOAD);
    F(attn_frag_hdr, seq) F(attn_frag_hdr, frag) F(attn_frag_hdr, nfrag)
    return 0;
}
"""


def _ask_the_compiler():
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.c")
        exe = os.path.join(d, "probe")
        with open(src, "w") as f:
            f.write(_PROBE)
        r = subprocess.run([cc, "-I", os.path.dirname(HDR), src, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"attn_proto.h does not compile:\n{r.stderr}")
        out = subprocess.run([exe], capture_output=True, text=True, check=True)
    d = {}
    for line in out.stdout.split("\n"):
        if line.strip():
            k, v = line.rsplit(" ", 1)
            d[k] = int(v)
    return d


def check_the_c_header_and_the_python_struct_are_the_same_format():
    """Sizes, field offsets and the constants, from the compiler itself."""
    c = _ask_the_compiler()
    if c is None:
        raise AssertionError("no C compiler found; this check cannot be skipped "
                             "silently because it is the only thing tying the "
                             "two definitions together")

    assert c["req"] == P.REQ.size, (c["req"], P.REQ.size)
    assert c["resp"] == P.RESP.size, (c["resp"], P.RESP.size)

    # Offsets, field by field, in the order the struct strings declare them.
    def offsets(fmt, fields):
        out, off = {}, 0
        for code, name in zip(_codes(fmt), fields):
            size = struct_size(code)
            off = (off + size - 1) // size * size      # natural alignment
            out[name] = off
            off += size
        return out

    for name, off in offsets(P.REQ.format, P.REQ_FIELDS).items():
        assert c[f"attn_req_hdr.{name}"] == off, (name, c[f"attn_req_hdr.{name}"], off)
    for name, off in offsets(P.RESP.format, P.RESP_FIELDS).items():
        assert c[f"attn_resp_hdr.{name}"] == off, (name, c[f"attn_resp_hdr.{name}"], off)

    assert c["magic"] == P.MAGIC, (c["magic"], P.MAGIC)
    assert c["version"] == P.VERSION
    assert c["token"] == P.TOKEN_BYTES
    assert c["result"] == P.RESULT_BYTES
    assert c["beat"] == P.BEAT
    assert c["erange"] == P.ERANGE

    # The fragment header, which the Ethernet path reassembles by. A mismatch
    # here splices frames at the wrong offset and produces a token that is the
    # right length and the wrong bytes.
    from kernel.host.attn_client import RawEthClient as R
    assert c["frag"] == R.FRAG_HDR.size == c["fraghdr"], (c, R.FRAG_HDR.size)
    assert c["mtu"] == R.MTU_PAYLOAD, (c["mtu"], R.MTU_PAYLOAD)
    # Jumbo, and the two ends must agree on it or fragments land at different
    # offsets and reassemble into a token of the right length and wrong bytes.
    assert R.MTU_PAYLOAD > 1500, R.MTU_PAYLOAD
    assert P.RESULT_BYTES + P.RESP.size <= R.MTU_PAYLOAD, (
        "a reply must fit in ONE frame; that is the entire point")
    assert c["attn_frag_hdr.seq"] == 0
    assert c["attn_frag_hdr.frag"] == 4
    assert c["attn_frag_hdr.nfrag"] == 6


def _codes(fmt):
    return [ch for ch in fmt if ch not in "<>=!@"]


def struct_size(code):
    import struct as _s
    return _s.calcsize("<" + code)


def check_a_step_request_round_trips():
    """What goes on the wire comes back off it, payload byte-for-byte."""
    token = bytes((i * 7 + 3) & 0xFF for i in range(P.TOKEN_BYTES))
    buf = P.step_request(token, n_tokens=1025, cache_base=0x10000000,
                         head_stride=81920, plane_span=20480, seq=42)
    h, body = P.parse_request(buf)
    assert h["magic"] == P.MAGIC and h["cmd"] == P.CMD_STEP
    assert h["n_tokens"] == 1025 and h["cache_base"] == 0x10000000
    assert h["head_stride"] == 81920 and h["plane_span"] == 20480
    assert h["seq"] == 42
    assert body == token


def check_the_payload_is_the_beat_stream_and_not_a_re_encoding():
    """The bytes on the wire ARE `ingress_bytes()`, at the header's own offset.

    The whole point of the format: the A53 hands the received bytes to the DMA
    without touching them. If this ever needs a transform, the transform is on
    the PS critical path and step 13 measured what that costs.
    """
    from kernel.hw.vectors import build

    vs, _ = build(ctx=32)
    token = vs.ingress_bytes()
    assert len(token) == P.TOKEN_BYTES
    buf = P.step_request(token, 33, 0, 16384, 4096)
    assert buf[P.REQ.size:] == token
    # And a whole number of the core's own beats, so the DMA never sees a
    # partial one.
    assert len(token) % P.BEAT == 0


def check_a_response_carries_the_counters_a_bandwidth_needs():
    r = P.response(P.OK, b"\x01\x02\x03\x04", seq=7, dev_us=67,
                   scan_cycles=10354, busy_cycles=16617, starve_cycles=76)
    h, body = P.parse_response(r)
    assert h["status"] == P.OK and h["seq"] == 7
    assert h["scan_cycles"] == 10354 and h["busy_cycles"] == 16617
    assert h["starve_cycles"] == 76 and h["dev_us"] == 67
    assert body == b"\x01\x02\x03\x04"


def check_a_short_buffer_is_an_error_and_not_a_silent_truncation():
    token = bytes(P.TOKEN_BYTES)
    buf = P.step_request(token, 1, 0, 0, 0)
    for n in (0, 4, P.REQ.size - 1, P.REQ.size + 10):
        try:
            P.parse_request(buf[:n])
        except ValueError:
            continue
        raise AssertionError(f"a {n}-byte request parsed without complaint")


def check_the_wrong_size_token_is_refused_before_it_reaches_the_wire():
    for n in (0, P.TOKEN_BYTES - 1, P.TOKEN_BYTES + 1):
        try:
            P.step_request(bytes(n), 1, 0, 0, 0)
        except ValueError:
            continue
        raise AssertionError(f"a {n}-byte token was accepted")


# --------------------------------------------------------------------------
# the mock, and the client that will talk to the board
# --------------------------------------------------------------------------

def _table():
    from kernel.host.attn_mock import Table
    from kernel.hw.board_vectors import build_case
    return Table([build_case(32, 3)]), build_case(32, 3)


def check_a_step_survives_the_round_trip_through_the_mock():
    """The transport returns the golden the board would have to return.

    Not a claim about attention -- the mock answers from `board_vectors`'
    own table. It is a claim that the bytes the host builds arrive as the
    bytes the core would consume, which is the half that has no other test.
    """
    from kernel.host.attn_mock import serve_one

    table, case = _table()
    ing = case["ingress"].tobytes()
    gold = case["golden"].tobytes()
    for i, n in enumerate(case["n_tokens"]):
        tok = ing[i * P.TOKEN_BYTES:(i + 1) * P.TOKEN_BYTES]
        req = P.step_request(tok, int(n), 0x10000000, case["head_stride"],
                             case["plane_span"], seq=100 + i)
        h, body = P.parse_response(serve_one(table, req))
        assert h["status"] == P.OK, h
        assert h["seq"] == 100 + i
        assert body == gold[i * P.RESULT_BYTES:(i + 1) * P.RESULT_BYTES]


def check_the_mock_refuses_what_the_board_would_have_to_refuse():
    """Every rejection path, so none of them is a silent success."""
    from kernel.host.attn_mock import serve_one

    table, case = _table()
    tok = case["ingress"].tobytes()[:P.TOKEN_BYTES]
    good = P.step_request(tok, int(case["n_tokens"][0]), 0, case["head_stride"],
                          case["plane_span"], seq=1)

    def status(buf):
        return P.parse_response(serve_one(table, buf))[0]["status"]

    assert status(good) == P.OK
    assert status(b"\x00" * 4) == P.EBADLEN
    assert status(b"\xff\xff\xff\xff" + good[4:]) == P.EBADMAGIC
    assert status(good[:4] + b"\x09\x00" + good[6:]) == P.EBADVER
    assert status(good[:6] + b"\x7f\x00" + good[8:]) == P.EBADCMD
    # The geometry travels with the token and must match it: right bytes, wrong
    # stride is a wrong answer on real hardware.
    bad_stride = P.step_request(tok, int(case["n_tokens"][0]), 0,
                                case["head_stride"] + 4096, case["plane_span"])
    assert status(bad_stride) == P.EBADLEN
    # A token the table never saw means the payload was mangled in transit.
    assert status(P.step_request(bytes(P.TOKEN_BYTES), int(case["n_tokens"][0]),
                                 0, case["head_stride"], case["plane_span"])) \
        == P.ETIMEOUT


def check_the_client_and_the_mock_agree_over_a_real_socket():
    """Through an actual TCP connection, because a stream can split anywhere.

    `serve_one` is a function and cannot get framing wrong. The socket path
    can: a `recv` that returns half a header is normal on a real network and
    is the classic way a protocol that works on loopback fails on a wire.
    """
    import threading

    from kernel.host.attn_client import TcpClient
    from kernel.host.attn_mock import Server

    table, case = _table()
    srv = Server(("127.0.0.1", 0), table)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        ing = case["ingress"].tobytes()
        gold = case["golden"].tobytes()
        with TcpClient("127.0.0.1", port, timeout=20.0) as c:
            assert c.ping()["status"] == P.OK
            for i, n in enumerate(case["n_tokens"]):
                tok = ing[i * P.TOKEN_BYTES:(i + 1) * P.TOKEN_BYTES]
                body, h = c.step(tok, int(n), 0x10000000,
                                 case["head_stride"], case["plane_span"])
                assert body == gold[i * P.RESULT_BYTES:(i + 1) * P.RESULT_BYTES]
                assert h["seq"] == i + 2      # ping took 1
    finally:
        srv.shutdown()
        srv.server_close()


def check_the_client_rejects_a_reply_to_the_wrong_request():
    """A stale reply is otherwise indistinguishable from a wrong answer."""
    from kernel.host.attn_client import _Base

    class Stale(_Base):
        def _exchange(self, req):
            return P.parse_response(P.response(P.OK, bytes(P.RESULT_BYTES), seq=999))

    try:
        Stale().step(bytes(P.TOKEN_BYTES), 1, 0, 0, 0)
    except RuntimeError as e:
        assert "seq" in str(e), e
        return
    raise AssertionError("a reply to seq 999 was accepted")
