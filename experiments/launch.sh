#!/usr/bin/env bash
# Start a supervised sweep, fully detached from this terminal.
#
#   kernel/experiments/launch.sh [args for wikitext_ppl.py]
#   kernel/experiments/launch.sh --install-reboot     print the crontab line
#
# setsid gives the run its own session, so closing the terminal or dropping an
# SSH connection cannot signal it.
cd "$(dirname "$0")/../.." || exit 1
DIR="${DIR:-kernel/experiments/results/qwen3-8b}"

if [ "${1:-}" = "--install-reboot" ]; then
  shift
  echo "# add this line with 'crontab -e' to resume the sweep after a reboot:"
  echo "@reboot cd $(pwd) && DIR=$DIR kernel/experiments/launch.sh $* >/dev/null 2>&1"
  exit 0
fi

mkdir -p "$DIR"

# Check the lock HERE, not only inside the supervisor. The supervisor's refusal
# goes to its own log, which is detached from this terminal -- so without this
# check a duplicate launch looks like it succeeded, and cheerfully prints the
# PID of the run that was already going.
LOCK="$DIR/supervisor.pid"
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "ALREADY RUNNING as pid $old for $DIR" >&2
    echo "  progress  kernel/experiments/status.sh" >&2
    echo "  stop      DIR=$DIR kernel/experiments/stop.sh" >&2
    exit 1
  fi
  echo "clearing a stale lock from pid ${old:-unknown}"
  rm -f "$LOCK"
fi

before=$(date +%s)
DIR="$DIR" setsid nohup kernel/experiments/supervise.sh "$@" >/dev/null 2>&1 < /dev/null &
disown 2>/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -f "$LOCK" ] && [ "$(stat -c %Y "$LOCK" 2>/dev/null || echo 0)" -ge "$before" ] && break
  sleep 1
done
if [ -f "$LOCK" ]; then
  echo "launched, detached.  pid $(cat "$LOCK")  dir=$DIR"
else
  echo "launch failed -- check $DIR/supervisor.log" >&2
  tail -3 "$DIR/supervisor.log" 2>/dev/null >&2
  exit 1
fi
echo "  progress    kernel/experiments/status.sh -w"
echo "  raw log     tail -f $DIR/run.log"
echo "  supervisor  tail -f $DIR/supervisor.log"
echo "  stop        DIR=$DIR kernel/experiments/stop.sh   (clean stop at a window boundary)"
