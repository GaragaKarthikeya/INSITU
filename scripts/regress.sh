#!/usr/bin/env bash
# Run every self-checking testbench in kernel/tb.
#
# Icarus lives in the ubuntu-work distrobox, not on the RHEL host. Vivado and
# Vitis are the other way round -- host only. Set KERNEL_NO_DISTROBOX=1 if you
# are already inside the container.
#
#   ./scripts/regress.sh            # all benches
#   ./scripts/regress.sh ddr        # only benches whose name contains "ddr"
set -uo pipefail
cd "$(dirname "$0")/.."
FILTER="${1:-}"

run() {   # run <name> <extra sources...>
    local name=$1; shift
    local vvp="/tmp/kernel_${name}.vvp"
    local cmd="iverilog -g2012 -o ${vvp} -s ${name} tb/${name}.sv $* && vvp ${vvp}"
    if [ "${KERNEL_NO_DISTROBOX:-0}" = "1" ]; then
        bash -lc "$cmd"
    else
        distrobox enter ubuntu-work -- bash -lc "cd $(pwd) && $cmd"
    fi
}

declare -a NAMES=(tb_ddr_probe tb_rot tb_rot_norm tb_rot_encode tb_qtab_build tb_score_lane tb_exp_lut tb_softmax_accum)
declare -a SRCS=("rtl/ddr_probe.sv rtl/axi_rd_engine.sv" "rtl/rot_fwht.sv" \
                 "rtl/rot_norm.sv" "rtl/rot_encode.sv" "rtl/qtab_build.sv" \
                 "rtl/score_lane.sv rtl/qtab_build.sv" "rtl/exp_lut.sv" \
                 "rtl/softmax_online.sv rtl/accum.sv rtl/exp_lut.sv")

fail=0
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]] && continue
    echo "--- $name"
    out=$(run "$name" ${SRCS[$i]} 2>&1)
    echo "$out" | sed 's/^/    /'
    # A bench that compiles but never reaches its verdict is a failure, so the
    # PASS banner must be present -- absence of "FAIL" is not enough.
    if ! echo "$out" | grep -q "ALL TESTS PASSED"; then
        echo "    ^^ $name FAILED"
        fail=1
    fi
done

if [ $fail -eq 0 ]; then echo "=== regress: all benches passed"; else echo "=== regress: FAILURES"; fi
exit $fail
