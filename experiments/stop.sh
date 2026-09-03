#!/usr/bin/env bash
# Stop a supervised sweep cleanly, by PID rather than by command-line pattern.
#
#   kernel/experiments/stop.sh
#
# The sweep finishes the window it is in, writes its checkpoint, and exits; the
# supervisor sees a requested stop and does not relaunch. Rerun launch.sh to
# resume from exactly there.
#
# Matching on `pkill -f` is deliberately NOT used: any shell whose arguments
# happen to contain the script name matches too, which in testing meant killing
# the terminal that issued the command.
cd "$(dirname "$0")/../.." || exit 1
DIR="${DIR:-kernel/experiments/results/qwen3-8b}"
LOCK="$DIR/supervisor.pid"

[ -f "$LOCK" ] || { echo "no supervisor lock at $LOCK -- nothing to stop"; exit 0; }
sup=$(cat "$LOCK")
kill -0 "$sup" 2>/dev/null || { echo "stale lock (pid $sup gone); removing"; rm -f "$LOCK"; exit 0; }

# The python child is the supervisor's only child; signal it, not the wrapper,
# so the supervisor stays alive to observe the clean stop and decline to retry.
child=$(pgrep -P "$sup" -f wikitext_ppl.py | head -1)
if [ -n "$child" ]; then
  echo "asking pid $child to stop at the next window boundary..."
  kill -TERM "$child"
else
  echo "no run in flight; stopping supervisor $sup"
  kill -TERM "$sup"
fi
