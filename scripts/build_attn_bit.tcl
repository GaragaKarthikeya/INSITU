# Synthesise, implement and write the bitstream for the attention block (step 12).
#
# Usage: vivado -mode batch -source scripts/build_attn_bit.tcl -tclargs <jobs> <freq_mhz>
#
# The WNS check is not advisory. `attn_top` routes out of context at +0.053 ns,
# which is 1.3% of the period: in a full system, with the PS, four
# smartconnects and a JTAG bridge competing for the same fabric, it can go
# either way. A design that closes only at 200 MHz still produces the right
# 2,048 numbers over JTAG -- step 12 makes no timing claim -- but every
# throughput number from step 13 onward would be against a clock the design is
# not running at. So the slack is reported loudly and a negative one is an
# explicit warning, not a line in a log.
set J    [expr {[llength $argv] > 0 ? [lindex $argv 0] : 8}]
set FREQ [expr {[llength $argv] > 1 ? [lindex $argv 1] : 250}]
set root [file normalize [file dirname [info script]]/..]
set proj $root/build/bd_attn_f${FREQ}

open_project $proj/attn_zcu104.xpr
set_property top attn_bd_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "### TOP = [get_property top [get_filesets sources_1]]"

# RESET THE OUT-OF-CONTEXT CHILD RUNS FIRST
# ----------------------------------------
# Every cell in the block design gets its own synthesis run, and the module
# reference holding the whole datapath is one of them. `reset_run synth_1` does
# NOT touch those: the top level simply re-links their cached DCPs. An edit to
# `attn_top.sv` then builds the PREVIOUS netlist, reports the previous
# utilisation to the digit and the previous WNS to the picosecond, and there is
# nothing in the log that says so -- which is how a fixed design can be
# measured as still broken.
# A freshly created project has no child runs yet -- they appear when the block
# design is first generated -- and `launch_runs {}` is an error rather than a
# no-op, so the empty case is handled explicitly.
set ooc [get_runs -filter {IS_SYNTHESIS && NAME != "synth_1"}]
if {[llength $ooc] > 0} {
    foreach r $ooc { reset_run $r }
    launch_runs $ooc -jobs $J
    foreach r $ooc { wait_on_run $r }
    foreach r $ooc {
        if {[get_property PROGRESS [get_runs $r]] != "100%"} {
            puts "### OOC SYNTH FAILED: $r"; exit 1
        }
    }
    puts "### re-synthesised [llength $ooc] out-of-context runs"
} else {
    puts "### no out-of-context runs yet (fresh project)"
}

reset_run synth_1
launch_runs synth_1 -jobs $J
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { puts "### SYNTH FAILED"; exit 1 }

launch_runs impl_1 -to_step write_bitstream -jobs $J
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { puts "### IMPL FAILED"; exit 1 }

open_run impl_1
set wns [get_property SLACK [get_timing_paths -delay_type max]]
set whs [get_property SLACK [get_timing_paths -delay_type min]]
puts "### IMPL FREQ=$FREQ WNS=${wns}ns WHS=${whs}ns"
if {$wns < 0} {
    puts "### WARNING: negative setup slack -- the block is NOT running at ${FREQ}MHz."
    puts "### Step 12 is correctness only and is still valid, but no later step"
    puts "### may report a rate against this clock."
}
# Reported so the netlist that was BUILT can be compared against the one that
# was expected -- an unchanged flop count after an RTL edit means a stale DCP
# was linked, not that the edit did nothing. `PRIMITIVE_SUBGROUP == flop`
# matched nothing in 2026.1 and printed a confident 0; `REF_NAME =~ FD*` is
# what the registers actually carry.
puts "### FF [llength [get_cells -hier -filter {REF_NAME =~ FD*}]]"
report_utilization    -file $proj/impl_util.rpt
report_timing_summary -file $proj/impl_timing.rpt

set bit $proj/attn_zcu104.runs/impl_1/attn_bd_wrapper.bit
if {![file exists $bit]} { puts "### NO BITSTREAM at $bit"; exit 1 }
file copy -force $bit $root/build/attn_f${FREQ}.bit
write_hw_platform -fixed -include_bit -force $root/build/attn_f${FREQ}.xsa
puts "### BIT written: $root/build/attn_f${FREQ}.bit"
puts "### ATTN BIT OK"
