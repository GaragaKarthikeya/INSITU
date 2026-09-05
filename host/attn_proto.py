"""The Python side of `host/attn_proto.h`, and the packing that feeds it.

TWO DEFINITIONS OF ONE WIRE FORMAT
----------------------------------
The board's server is C and the host is Python, because the host is where torch
runs. So the format exists twice, and two definitions of one format is exactly
the shape of bug that survives every unit test on both sides: each end agrees
with itself. `tests/test_proto.py` compiles the header and compares the offsets
and sizes it reports against the `struct` strings here, so a field added to one
and not the other fails a test rather than a board run.

THE PAYLOAD IS NOT ENCODED HERE
-------------------------------
`step_request` takes the ingress bytes as `hw/vectors.py` already packs them
and puts them on the wire unchanged. There is no second packer: a request's
payload is the beat stream the core consumes, byte for byte, which is what lets
the A53 hand it to the DMA without touching it.
"""

from __future__ import annotations

import struct

MAGIC = 0x314E5441
VERSION = 1
PORT = 7001
ETHERTYPE = 0x88B5

CMD_PING = 0
CMD_STEP = 1
CMD_LOAD = 2

BEAT = 64
TOKEN_BYTES = 9216
# The board's reassembly buffer holds one token; a larger request is dropped
# fragment by fragment and never answered. `RawEthClient.load` splits to fit.
MAX_PAYLOAD = 9216
RESULT_BYTES = 6144

OK, EBADMAGIC, EBADVER, EBADCMD, EBADLEN, ETIMEOUT, EDMA, ERANGE = range(8)

STATUS_NAME = {
    OK: "ok", EBADMAGIC: "bad magic", EBADVER: "bad version",
    EBADCMD: "bad command", EBADLEN: "bad length", ETIMEOUT: "timeout",
    EDMA: "dma error", ERANGE: "a channel escaped the 24-bit seam",
}

# `<` is little-endian AND no padding, and the C header has no field wider
# than 4 bytes so that nothing is padded there either. A `uint64_t` for
# `cache_base` pads twice -- before the field AND at the end of the struct,
# because `sizeof` must be a multiple of the alignment -- and `tests/
# test_proto.py` caught both. The address is split the way `attn_ctrl`'s own
# register map splits it, into BASE_LO and BASE_HI.
REQ = struct.Struct("<IHHIIIIIII")
RESP = struct.Struct("<IHHIIIIIIII")

REQ_FIELDS = ("magic", "version", "cmd", "cache_base_lo", "cache_base_hi",
              "n_tokens", "head_stride", "plane_span", "payload", "seq")
RESP_FIELDS = ("magic", "status", "rsvd", "payload", "seq", "dev_us",
               "scan_cycles", "busy_cycles", "starve_cycles", "clips",
               "overflows")


def step_request(token: bytes, n_tokens: int, cache_base: int,
                 head_stride: int, plane_span: int, seq: int = 0) -> bytes:
    """One decode step. `token` is `VectorSet.ingress_bytes()`, unchanged."""
    if len(token) != TOKEN_BYTES:
        raise ValueError(f"token is {len(token)} B, expected {TOKEN_BYTES}")
    return REQ.pack(MAGIC, VERSION, CMD_STEP, cache_base & 0xFFFFFFFF,
                    cache_base >> 32, n_tokens, head_stride, plane_span,
                    len(token), seq) + token


def load_request(image: bytes, cache_base: int, seq: int = 0) -> bytes:
    """Write `image` into the board's DDR at `cache_base`."""
    return REQ.pack(MAGIC, VERSION, CMD_LOAD, cache_base & 0xFFFFFFFF,
                    cache_base >> 32, 0, 0, 0, len(image), seq) + bytes(image)


def ping_request(seq: int = 0) -> bytes:
    return REQ.pack(MAGIC, VERSION, CMD_PING, 0, 0, 0, 0, 0, 0, seq)


def parse_request(buf: bytes) -> tuple[dict, bytes]:
    if len(buf) < REQ.size:
        raise ValueError(f"{len(buf)} B is shorter than the {REQ.size} B header")
    h = dict(zip(REQ_FIELDS, REQ.unpack_from(buf)))
    h["cache_base"] = h["cache_base_lo"] | (h["cache_base_hi"] << 32)
    body = buf[REQ.size:REQ.size + h["payload"]]
    if len(body) != h["payload"]:
        raise ValueError(f"payload is {len(body)} B, header says {h['payload']}")
    return h, body


def response(status: int, payload: bytes = b"", seq: int = 0, **counters) -> bytes:
    return RESP.pack(MAGIC, status, 0, len(payload), seq,
                     counters.get("dev_us", 0),
                     counters.get("scan_cycles", 0),
                     counters.get("busy_cycles", 0),
                     counters.get("starve_cycles", 0),
                     counters.get("clips", 0),
                     counters.get("overflows", 0)) + payload


def parse_response(buf: bytes) -> tuple[dict, bytes]:
    if len(buf) < RESP.size:
        raise ValueError(f"{len(buf)} B is shorter than the {RESP.size} B header")
    h = dict(zip(RESP_FIELDS, RESP.unpack_from(buf)))
    if h["magic"] != MAGIC:
        raise ValueError(f"magic {h['magic']:#x}, expected {MAGIC:#x}")
    body = buf[RESP.size:RESP.size + h["payload"]]
    if len(body) != h["payload"]:
        raise ValueError(f"payload is {len(body)} B, header says {h['payload']}")
    return h, body
