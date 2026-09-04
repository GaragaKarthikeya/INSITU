#!/usr/bin/env bash
# Run OOC synthesis for standalone RTL blocks.  Step 5 currently has rot_fwht.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-rot_fwht}"
case "$TARGET" in
    rot_fwht) TCL="scripts/synth_rot_fwht.tcl" ;;
    *) echo "usage: $0 [rot_fwht]" >&2; exit 2 ;;
esac

VIVADO_SETTINGS="${VIVADO_SETTINGS:-/home/digital3/2026.1/Vivado/settings64.sh}"
if [ ! -f "$VIVADO_SETTINGS" ]; then
    echo "Vivado settings script not found: $VIVADO_SETTINGS" >&2
    exit 2
fi
source "$VIVADO_SETTINGS"

# Lab installations commonly keep the floating-license endpoint in ~/.flexlmrc
# rather than exporting it from settings64.sh.  Respect an explicitly supplied
# environment first; otherwise load that local key without printing it.
if [ -z "${XILINXD_LICENSE_FILE:-}" ] && [ -f "$HOME/.flexlmrc" ]; then
    set -a
    source "$HOME/.flexlmrc"
    set +a
fi

OUT="build/synth/$TARGET"
mkdir -p "$OUT"
vivado -mode batch -nojournal -nolog -source "$TCL" -tclargs "$OUT"
