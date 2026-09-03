"""Progress and health for a multi-day run.

WHAT THIS IS FOR
----------------
The sweep is roughly 57 hours. Nobody watches that, so the question is not
"how do I display progress" but "what would make me stop the run early" --
and everything here is built around answering that in two seconds, from any
shell, without attaching to the process.

Three outputs, deliberately separate:

  status.json   a small snapshot, overwritten. What is happening RIGHT NOW.
                Its `updated_at` is the liveness signal: a heartbeat fires once
                per transformer layer, so a stamp older than about a minute
                means the run is wedged, not slow.
  events.jsonl  append-only history. Every window, every alarm, every phase
                change, in order. This is the audit trail.
  results.json  the checkpointed measurement itself, written by the caller.

ALARMS ABORT, THEY DO NOT WARN
------------------------------
An alarm that only prints is an alarm nobody sees at hour 30. The ones that
mean the results are worthless (the dense control disagreeing with the
baseline, a non-finite loss) stop the run. The ones that mean it is about to
die (resident memory near the ceiling) stop it while there is still a usable
checkpoint. Anything softer is recorded and carried on from.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def rss_gb() -> float:
    """Resident set size, GB. Read from /proc so it costs nothing to poll."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return 0.0


def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float("inf")


ALARM_EXIT = 3      # "stop, this is broken" -- do not retry
STOPPED_EXIT = 4    # "you asked me to stop" -- do not retry, but nothing is wrong
                    # Both distinct from 0 (finished) and 1 (crashed), so a
                    # supervisor can tell the four apart -- and so a log at hour
                    # 30 does not say COMPLETED when someone pressed Ctrl-C.


class Alarm(SystemExit):
    """Raised to stop the run deliberately. The checkpoint on disk stays valid.

    Exits with ALARM_EXIT rather than 1 so a restart wrapper does NOT relaunch:
    an alarm means the results are wrong or the machine is out of room, and
    retrying either just burns hours reproducing the same failure.
    """

    def __init__(self, message: str) -> None:
        print(f"\nALARM: {message}", flush=True)
        super().__init__(ALARM_EXIT)


def write_json_atomic(path: Path, obj, keep_backup: bool = True) -> None:
    """Write JSON so an interrupted write cannot destroy the previous one.

    Write to a temporary file in the same directory, fsync it, then rename --
    rename within a filesystem is atomic, so a reader (or a crash) sees either
    the old file or the new one, never a truncated one. On a run measured in
    days, a checkpoint corrupted by a kill at the wrong microsecond is the
    difference between losing one window and losing everything.

    The previous good copy is COPIED alongside as `.bak`, not moved. Moving it
    would leave an instant with no file at `path` at all -- harmless for the
    resilient reader below, but every other reader (the status viewer, a shell,
    a person) sees the checkpoint briefly vanish. A copy costs a few kilobytes
    of I/O per window and keeps the primary continuously present.
    """
    import json as _json
    import shutil
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        _json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    if keep_backup and path.exists():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        except OSError:
            pass
    tmp.replace(path)


def read_json_resilient(path: Path):
    """Load a checkpoint, falling back to the backup if the primary is broken."""
    import json as _json
    for candidate in (Path(path), Path(str(path) + ".bak")):
        if not candidate.exists():
            continue
        try:
            return _json.loads(candidate.read_text()), candidate
        except (ValueError, OSError):
            continue
    return {}, None


@dataclass
class RunLog:
    directory: Path
    heartbeat_interval: float = 1.0      # seconds between status.json writes

    started: float = field(default_factory=time.time)
    _last_write: float = 0.0
    _state: dict = field(default_factory=dict)
    _alarms: list = field(default_factory=list)
    _rss_peak: float = 0.0

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.status_path = self.directory / "status.json"
        self.events_path = self.directory / "events.jsonl"
        self._state.update(pid=os.getpid(), started_at=self.started, phase="starting")

    # -- writing -----------------------------------------------------------

    def event(self, kind: str, **fields) -> None:
        """Append one line to the history. Never throttled -- events are rare."""
        rec = {"t": time.time(), "elapsed_h": (time.time() - self.started) / 3600,
               "kind": kind, **fields}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._write(force=True)

    def set(self, **fields) -> None:
        """Update the snapshot without forcing a write."""
        self._state.update(fields)

    def heartbeat(self, **fields) -> None:
        """Called once per transformer layer. Throttled to one write a second.

        The point is not the fields; it is `updated_at`. A run that has stopped
        moving looks exactly like a slow one in any log that only prints per
        window, and 14 minutes of silence is indistinguishable from a hang.
        """
        self._state.update(fields)
        self._write(force=False)

    def _write(self, force: bool) -> None:
        now = time.time()
        if not force and now - self._last_write < self.heartbeat_interval:
            return
        self._last_write = now
        r = rss_gb()
        self._rss_peak = max(self._rss_peak, r)
        snap = dict(self._state)
        snap.update(updated_at=now, elapsed_h=(now - self.started) / 3600,
                    rss_gb=round(r, 2), rss_peak_gb=round(self._rss_peak, 2),
                    mem_available_gb=round(mem_available_gb(), 2),
                    alarms=self._alarms)
        # No backup for the status snapshot: it is regenerated every second and
        # a stale copy would be worse than an absent one.
        write_json_atomic(self.status_path, snap, keep_backup=False)

    # -- alarms ------------------------------------------------------------

    def warn(self, code: str, message: str, **fields) -> None:
        self._alarms.append({"level": "warn", "code": code, "message": message,
                             "t": time.time()})
        self.event("warn", code=code, message=message, **fields)

    def abort(self, code: str, message: str, **fields) -> None:
        self._alarms.append({"level": "abort", "code": code, "message": message,
                             "t": time.time()})
        self.set(phase="aborted")
        self.event("abort", code=code, message=message, **fields)
        raise Alarm(f"[{code}] {message}")

    def preflight(self, need_gb: float, need_disk_gb: float = 2.0) -> None:
        """Refuse to start a multi-day run that obviously cannot finish.

        Cheap, and it turns two classes of overnight failure into an immediate
        error message: not enough memory for the model, and not enough disk for
        the checkpoints.
        """
        import shutil
        avail = mem_available_gb()
        if avail < need_gb:
            self.abort("insufficient_memory",
                       f"{avail:.1f} GB available, this run needs about "
                       f"{need_gb:.1f} GB")
        free = shutil.disk_usage(self.directory).free / 1e9
        if free < need_disk_gb:
            self.abort("insufficient_disk",
                       f"{free:.1f} GB free at {self.directory}, need "
                       f"{need_disk_gb:.1f} GB")
        self.event("preflight_ok", mem_available_gb=round(avail, 1),
                   disk_free_gb=round(free, 1))

    # -- the checks that are worth stopping for ----------------------------

    def check_memory(self, abort_gb: float, warn_gb: float) -> None:
        r = rss_gb()
        if r >= abort_gb:
            self.abort("rss_ceiling",
                       f"resident {r:.1f} GB >= {abort_gb} GB; stopping while the "
                       f"checkpoint is still good rather than being OOM-killed",
                       rss_gb=r)
        if r >= warn_gb and not any(a["code"] == "rss_high" for a in self._alarms):
            self.warn("rss_high", f"resident {r:.1f} GB (warn at {warn_gb})")

    def check_loss(self, config: str, window: int, nll: float) -> None:
        import math
        if not math.isfinite(nll):
            self.abort("nonfinite_loss",
                       f"{config} window {window} produced a non-finite loss; "
                       f"every number after this would be meaningless",
                       config=config, window=window)

    def check_dense_control(self, baseline_per_window, dense_per_window,
                            tol: float) -> None:
        """The dense control must reproduce the baseline, ON THE SAME WINDOWS.

        Takes per-window values and compares only the prefix both have covered.
        Comparing two CUMULATIVE perplexities over different window counts is
        meaningless and this check used to do exactly that: window-to-window
        perplexity on real text swings by 50% or more, so a baseline two windows
        ahead of the control can differ by several percent with nothing wrong at
        all. It fired at hour 17 of a 100-hour run, on a kernel that was
        reproducing the model to eight decimal places.

        A safety check that stops a correct run is not a cheap kind of wrong --
        it costs exactly as much as the failure it was guarding against.
        """
        n = min(len(baseline_per_window), len(dense_per_window))
        if n < 1:
            return
        import math
        b = sum(baseline_per_window[:n]) / n
        d = sum(dense_per_window[:n]) / n
        rel = abs(math.exp(d - b) - 1.0)
        if rel > tol:
            self.abort("dense_control_mismatch",
                       f"dense control {math.exp(d):.4f} vs baseline "
                       f"{math.exp(b):.4f} over the SAME {n} window(s) "
                       f"({rel:.2%} > {tol:.2%}). The kernel is not reproducing "
                       f"the model; compression numbers would be attributing a "
                       f"plumbing bug to quantisation.",
                       baseline=math.exp(b), dense=math.exp(d),
                       relative=rel, shared_windows=n)
