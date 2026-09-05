"""Drive the board over JTAG, one decode step at a time. Real inference.

WHY THIS IS FAST ENOUGH AND THE OBVIOUS THING IS NOT
----------------------------------------------------
The host could poke `attn_ctrl`'s registers over JTAG directly -- step 12 did
-- and it costs one JTAG transaction per 32-bit word. A decode step is 2,304
words in and 1,536 out, and at roughly a millisecond apiece that is four
seconds a token.

So the slow link carries BULK transfers only. `xsdb` writes DDR with
`mwr -bin -file`, which moves a whole buffer through the DAP in one operation,
and the A53 -- already on the right side of the bus -- does the 15 KB of AXI
traffic through the DMA in about 5 us. The host writes a token, rings a
doorbell, and reads 6,144 bytes back.

This is not the transport the design is FOR: `plan.MD` sizes the system against
raw Ethernet at ~35 us of round trip. It is the transport that exists today,
and it is enough to put the silicon in the loop of a real model rather than
replaying a recording at it.

HOW IT TALKS TO xsdb
--------------------
`xsdb` is a TCL REPL, so it is driven as a subprocess over pipes with a
sentinel printed after every command -- reading until a prompt is guesswork,
and guessing wrong desynchronises the stream in a way that looks like the board
returning garbage. Every exchange is bounded by a marker this side chose.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import time

# Must match `sw/attn_server.h`.
MBX_BASE = 0x30000000
MBX_IDLE = 0x00000000
MBX_REQ_STEP = 0xA77E0001
MBX_REQ_QUIT = 0xA77E00FF
MBX_ACK = 0x5A5A0000
MBX_ERR = 0xDEAD0000

W_DOORBELL, W_STATUS, W_NTOK, W_BASE_LO, W_BASE_HI = 0, 1, 2, 3, 4
W_STRIDE, W_SPAN, W_SEQ, W_SCAN, W_BUSY = 5, 6, 7, 8, 9
W_STARVE, W_CLIPS, W_OVF, W_FLAGS, W_DEV_US, W_NSTEPS = 10, 11, 12, 13, 14, 15

TOKEN_OFF = 0x1000
RESULT_OFF = 0x4000
TOKEN_BYTES = 9216
RESULT_BYTES = 6144

MARK = "___ATTN_DONE___"


class XsdbError(RuntimeError):
    pass


class Xsdb:
    """A bounded-exchange wrapper around the xsdb REPL."""

    def __init__(self, xsdb: str | None = None, timeout: float = 120.0):
        exe = xsdb or shutil.which("xsdb")
        if exe is None:
            raise XsdbError("xsdb not on PATH; source Vitis settings64.sh first")
        self.timeout = timeout
        self.p = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._drain_banner()

    def _drain_banner(self):
        self.cmd("puts hello")

    def cmd(self, tcl: str) -> str:
        """Run one command, return its output. Bounded by a marker, not a prompt."""
        if self.p.poll() is not None:
            raise XsdbError("xsdb exited")
        self.p.stdin.write(tcl + f"\nputs {MARK}\n")
        self.p.stdin.flush()
        out, deadline = [], time.time() + self.timeout
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise XsdbError("xsdb closed its output:\n" + "".join(out))
            if MARK in line:
                return "".join(out)
            out.append(line)
            if time.time() > deadline:
                raise XsdbError(f"xsdb timed out on {tcl!r}:\n" + "".join(out))

    def close(self):
        try:
            self.p.stdin.write("exit\n")
            self.p.stdin.flush()
            self.p.wait(timeout=10)
        except Exception:                                   # noqa: BLE001
            self.p.kill()


class JtagClient:
    """The same `step()` as `TcpClient` and `RawEthClient`.

    Everything above this cannot tell which transport it is holding, which is
    the point -- when the Ethernet server exists it replaces this and nothing
    else changes.
    """

    def __init__(self, xsdb: Xsdb | None = None, mbx: int = MBX_BASE,
                 verbose: bool = False):
        self.x = xsdb or Xsdb()
        self.mbx = mbx
        self.verbose = verbose
        self.seq = 0
        self._tmp = tempfile.mkdtemp(prefix="attn_jtag_")
        self._connect()

    def _connect(self):
        self.x.cmd("connect")
        self.x.cmd('targets -set -nocase -filter {name =~ "*A53*#0*"}')
        # A round trip before anything depends on one. `mwr` prints nothing
        # whether it worked or not, so without this the first evidence that the
        # link is dead is a step that times out several seconds later -- or, as
        # happened here, a cache image "pushed" in 0.6 s that never arrived.
        probe = self.mbx + 0x20
        for pattern in (0x5A5AA5A5, 0xA5A55A5A):
            self._wr_words(probe, [pattern])
            got = self._rd_words(probe, 1)[0]
            if got != pattern:
                raise XsdbError(
                    f"JTAG round trip failed at {probe:#x}: wrote {pattern:#010x}, "
                    f"read {got:#010x}. Is the board programmed and the A53 running?")

    # -- memory --------------------------------------------------------------
    def _wr_words(self, addr: int, words: list[int]) -> None:
        self.x.cmd(f"mwr -force {addr:#x} {{{' '.join(str(w) for w in words)}}}")

    def _rd_words(self, addr: int, n: int) -> list[int]:
        # `puts [mrd ...]`, NOT bare `mrd`. A TCL REPL echoes each command's
        # result only when its input is a terminal; driven from a pipe it
        # prints nothing at all, so a bare `mrd` returns an empty string and
        # every read looks like a board that answered with silence.
        out = self.x.cmd(f"puts [mrd -force {addr:#x} {n}]")
        vals = []
        for line in out.strip().split("\n"):
            if ":" not in line:
                continue
            for tokn in line.split(":", 1)[1].split():
                vals.append(int(tokn, 16))
        if len(vals) != n:
            raise XsdbError(f"read {len(vals)} of {n} words at {addr:#x}:\n{out}")
        return vals

    def _wr_bin(self, addr: int, data: bytes) -> None:
        """The bulk path: one DAP operation for the whole buffer."""
        path = os.path.join(self._tmp, "tx.bin")
        with open(path, "wb") as f:
            f.write(data)
        self.x.cmd(f"mwr -force -bin -file {path} {addr:#x} {len(data) // 4}")

    def _rd_bin(self, addr: int, n_bytes: int) -> bytes:
        path = os.path.join(self._tmp, "rx.bin")
        self.x.cmd(f"mrd -force -bin -file {path} {addr:#x} {n_bytes // 4}")
        with open(path, "rb") as f:
            return f.read()

    # -- the interface ------------------------------------------------------
    def load_cache(self, image: bytes, cache_base: int) -> None:
        """The image, once, before decoding. Megabytes, so this is the slow part."""
        self._wr_bin(cache_base, image)
        # Verified, because `mwr -bin -file` reports nothing. The LAST words
        # are checked rather than the first: a transfer that starts and dies
        # part way is the failure a head-of-buffer check cannot see.
        tail = len(image) - 16
        want = list(struct.unpack("<4I", image[tail:tail + 16]))
        got = self._rd_words(cache_base + tail, 4)
        if got != want:
            raise XsdbError(
                f"the cache image did not land: at +{tail:#x} wrote "
                f"{[f'{w:#010x}' for w in want]}, read {[f'{g:#010x}' for g in got]}")

    def step(self, token: bytes, n_tokens: int, cache_base: int,
             head_stride: int, plane_span: int, timeout: float = 30.0):
        if len(token) != TOKEN_BYTES:
            raise ValueError(f"token is {len(token)} B, expected {TOKEN_BYTES}")
        self.seq += 1
        self._wr_bin(self.mbx + TOKEN_OFF, token)
        # Geometry first, doorbell last: the board polls the doorbell, so a
        # doorbell visible before the parameters it announces would run the
        # step with the previous token's geometry.
        self._wr_words(self.mbx + 4 * W_NTOK, [
            n_tokens, cache_base & 0xFFFFFFFF, cache_base >> 32,
            head_stride, plane_span, self.seq])
        self._wr_words(self.mbx + 4 * W_DOORBELL, [MBX_REQ_STEP])

        deadline = time.time() + timeout
        while True:
            db, status = self._rd_words(self.mbx, 2)
            if db == MBX_IDLE and status in (MBX_ACK, MBX_ERR):
                break
            if time.time() > deadline:
                raise XsdbError(f"the board did not answer (doorbell {db:#x}, "
                                f"status {status:#x})")
        if status == MBX_ERR:
            raise RuntimeError("the board reported an error for this step")

        ctrl = self._rd_words(self.mbx, 16)
        if ctrl[W_SEQ] != self.seq:
            raise RuntimeError(f"answer to seq {ctrl[W_SEQ]}, expected {self.seq}")
        counters = {
            "scan_cycles": ctrl[W_SCAN], "busy_cycles": ctrl[W_BUSY],
            "starve_cycles": ctrl[W_STARVE], "clips": ctrl[W_CLIPS],
            "overflows": ctrl[W_OVF], "flags": ctrl[W_FLAGS],
            "dev_us": ctrl[W_DEV_US], "steps": ctrl[W_NSTEPS],
        }
        return self._rd_bin(self.mbx + RESULT_OFF, RESULT_BYTES), counters

    def quit(self):
        try:
            self._wr_words(self.mbx + 4 * W_DOORBELL, [MBX_REQ_QUIT])
        finally:
            self.x.close()
