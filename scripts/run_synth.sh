#!/usr/bin/env bash
# Run OOC synthesis for standalone RTL blocks.  One block per invocation;
# `all` walks every block the datapath has so far.
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A SRCS=(
    [rot_fwht]="rtl/rot_fwht.sv"
    [rot_norm]="rtl/rot_norm.sv"
    [rot_encode]="rtl/rot_encode.sv"
    [qtab_build]="rtl/qtab_build.sv"
    [score_lane]="rtl/score_lane.sv rtl/qtab_build.sv"
    [exp_lut]="rtl/exp_lut.sv"
    [softmax_online]="rtl/softmax_online.sv rtl/exp_lut.sv"
    [accum]="rtl/accum.sv"
    [finalize]="rtl/finalize.sv"
    [kv_plane_rd]="rtl/kv_plane_rd.sv"
    [kv_store_ddr]="rtl/kv_store_ddr.sv rtl/kv_plane_rd.sv"
    [kv_write]="rtl/kv_write.sv"
    [attn_top]="rtl/attn_top.sv rtl/attn_ingress.sv rtl/rot_fwht.sv rtl/rot_norm.sv rtl/rot_encode.sv rtl/score_lane.sv rtl/qtab_build.sv rtl/softmax_online.sv rtl/exp_lut.sv rtl/accum.sv rtl/finalize.sv rtl/kv_store_ddr.sv rtl/kv_plane_rd.sv rtl/kv_write.sv"
)
TARGET="${1:-all}"
if [ "$TARGET" = "all" ]; then
    TARGETS="rot_fwht rot_norm rot_encode qtab_build score_lane exp_lut softmax_online accum finalize kv_plane_rd kv_store_ddr kv_write attn_top"
elif [ -n "${SRCS[$TARGET]:-}" ]; then
    TARGETS="$TARGET"
else
    echo "usage: $0 [all|${!SRCS[*]}]" >&2; exit 2
fi

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

# `IMPL=1` runs place and route instead of synthesis alone.  Step 11's exit
# criterion is a ROUTED 250 MHz, which is a different claim from a synthesised
# one and takes long enough that it is not the default.
TCL="scripts/synth_block.tcl"
SUB="synth"
if [ "${IMPL:-0}" = "1" ]; then
    TCL="scripts/impl_block.tcl"
    SUB="impl"
fi

for t in $TARGETS; do
    OUT="build/$SUB/$t"
    mkdir -p "$OUT"
    echo "--- $t"
    vivado -mode batch -nojournal -nolog -source "$TCL" \
        -tclargs "$OUT" "$t" ${SRCS[$t]}
done
