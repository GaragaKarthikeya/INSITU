#!/usr/bin/env bash
# Live status of whichever sweep or ablation most recently wrote a heartbeat.
#
#   status.sh          one snapshot
#   status.sh -w       refresh every 10 s
#   status.sh <dir>    a specific run
cd "$(dirname "$0")/../.." || exit 1
if [ "$1" = "-w" ]; then
  shift
  while true; do clear; .venv/bin/python kernel/experiments/status.py "$@"; sleep 10; done
else
  .venv/bin/python kernel/experiments/status.py "$@"
fi
