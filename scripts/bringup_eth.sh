#!/usr/bin/env bash
# Program the board, capture its UART, and print what it said. One command.
#
# The three-terminal dance -- capture in one, xsdb in another, client in a third
# -- has an ordering that is easy to get wrong and whose failure mode is an
# empty log. The most common mistake is running the client against a board that
# was never reprogrammed, which looks exactly like a dead link.
#
#   ./scripts/bringup_eth.sh [seconds]
set -uo pipefail
cd "$(dirname "$0")/.."
SECS="${1:-75}"
LOG=/tmp/bringup_eth.log
DEV="${UART:-/dev/ttyUSB1}"

: > "$LOG"
./host/uart_cap.py "$LOG" "$SECS" "$DEV" &
CAP=$!
sleep 1

source /home/digital3/2026.1/Vitis/settings64.sh 2>/dev/null
echo "### programming the board (this takes ~30 s)"
xsdb host/run_attn_server.tcl 2>&1 | sed 's/^/    /'

echo "### waiting for the self-test and Ethernet bring-up"
wait $CAP 2>/dev/null

echo
echo "================= UART ================="
cat "$LOG"
echo "========================================"
echo
if   grep -q "ETH SERVER READY"   "$LOG"; then
    echo "VERDICT: the Ethernet server is up. Run the probe, then fpga_infer."
elif grep -q "ETH unavailable"    "$LOG"; then
    echo "VERDICT: Ethernet init FAILED and it fell back to the JTAG mailbox."
    echo "         The 'ETH:' lines above say which step failed."
elif grep -q "SERVER READY"       "$LOG"; then
    echo "VERDICT: this is the JTAG-mailbox ELF -- it has no Ethernet server."
    echo "         The board is running an OLDER build than the repository."
elif grep -q "ALL TESTS PASSED"   "$LOG"; then
    echo "VERDICT: the self-test passed but no server started. The ELF is older"
    echo "         than sw/attn_main.c's serve()/serve_eth()."
elif [ ! -s "$LOG" ]; then
    echo "VERDICT: the UART said NOTHING. Wrong device? Try UART=/dev/ttyUSB2 $0"
else
    echo "VERDICT: the board did not finish its self-test. See above."
fi
