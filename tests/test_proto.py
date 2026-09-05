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
