#!/usr/bin/env bash
# Run a long sweep so it survives the things that kill long sweeps.
#
#   kernel/experiments/supervise.sh [args passed to wikitext_ppl.py]
#
# What it handles:
#   terminal closed / SSH dropped   the run is setsid-detached (see launch.sh),
#                                   so it has no controlling terminal to lose
#   unexpected crash                relaunched, up to MAX_RESTARTS, after a
#                                   pause; the run resumes from its checkpoint
#   an alarm (exit 3)               NOT relaunched. An alarm means the results
#                                   are wrong or the machine is out of room;
#                                   retrying reproduces the same failure hours
#                                   later
#   clean stop (SIGTERM / Ctrl-C)   finishes the window in flight, checkpoints,
#                                   exits 4. NOT relaunched -- and the log says
#                                   STOPPED ON REQUEST, not COMPLETED
#   completion (exit 0)             stops
#   a second supervisor             refused, via a PID lockfile
#
# Not handled: a machine reboot. `launch.sh --install-reboot` prints the
# crontab line for that.

set -u
cd "$(dirname "$0")/../.." || exit 1

DIR="${DIR:-kernel/experiments/results/qwen3-8b}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY="${RESTART_DELAY:-60}"
LOG="$DIR/supervisor.log"
LOCK="$DIR/supervisor.pid"
mkdir -p "$DIR"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say()   { echo "[$(stamp)] $*" >> "$LOG"; }

# A PID lockfile, not a pgrep pattern. Matching command lines is fragile --
# any shell whose arguments happen to contain the script name and the output
# directory looks like a running sweep, which is exactly how the first version
# of this guard managed to refuse to start against itself.
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    say "ALREADY RUNNING as pid $old -- refusing to start a second copy"
    echo "already running as pid $old (lock: $LOCK)" >&2
    exit 1
  fi
  say "clearing a stale lock from pid ${old:-unknown}"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

say "supervisor starting: pid=$$ dir=$DIR max_restarts=$MAX_RESTARTS args=$*"

attempt=0
while :; do
  attempt=$((attempt + 1))
  say "launch attempt $attempt"
  .venv/bin/python kernel/experiments/wikitext_ppl.py --dir "$DIR" "$@" \
      >> "$DIR/run.log" 2>&1
  code=$?

  case $code in
    0) say "COMPLETED (exit 0) -- the whole sweep is done"; break ;;
    3) say "STOPPED BY ALARM (exit 3) -- not restarting. See $DIR/events.jsonl"; break ;;
    4) say "STOPPED ON REQUEST (exit 4) -- not restarting. Rerun launch.sh to resume"; break ;;
    *)
      if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
        say "DIED (exit $code) and hit the restart limit ($MAX_RESTARTS) -- giving up"
        break
      fi
      say "DIED (exit $code) -- restarting in ${RESTART_DELAY}s; resumes from the checkpoint"
      sleep "$RESTART_DELAY"
      ;;
  esac
done

say "supervisor exiting"
