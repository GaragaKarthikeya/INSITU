#!/usr/bin/env bash
# Bring the PS up over JTAG so DDR answers, with no FSBL and no A53 software.
#
# `create_bd_attn.tcl` gives the PS no master port and step 12 runs nothing on
# the A53 -- but the DRAM controller still has to be initialised by somebody,
# and `psu_init` is that somebody. Without it the block's first read to DDR
# never returns and it sits at busy forever, which looks exactly like a
# datapath hang.
#
# This is register configuration, not software: no ELF is loaded and the A53
# stays in reset. Step 13 replaces it with a real FSBL.
#
# ORDER MATTERS: THE PL IS PROGRAMMED FIRST
# -----------------------------------------
# `psu_ps_pl_isolation_removal` opens the path between the PS and a PL that is
# expected to already hold a design, and an earlier version of this script ran
# `rst -system` first -- which clears the PL configuration. Vivado then found a
# device with no debug core in it and no `hw_axi` to talk to. So there is no
# system reset here, and `jtag_attn.tcl` calls this AFTER programming.
#
#   ./scripts/jtag_ps_init.sh [freq_mhz]
set -euo pipefail
cd "$(dirname "$0")/.."
FREQ="${1:-150}"

INIT="build/bd_attn_f${FREQ}/attn_zcu104.gen/sources_1/bd/attn_bd/ip/attn_bd_zynq_ultra_ps_e_0_0/psu_init.tcl"
if [ ! -f "$INIT" ]; then
    echo "psu_init.tcl not found: $INIT" >&2
    echo "Run scripts/create_bd_attn.tcl for this frequency first." >&2
    exit 2
fi

XSDB="${XSDB:-/home/digital3/2026.1/Vitis/bin/xsdb}"
[ -x "$XSDB" ] || { echo "xsdb not found at $XSDB" >&2; exit 2; }

"$XSDB" -eval "
    connect
    targets -set -filter {name =~ \"PSU\"}
    source $INIT
    psu_init
    after 200
    psu_ps_pl_isolation_removal
    psu_ps_pl_reset_config
    puts \"### PSU init done -- DDR is up\"
    disconnect
"
