# Synthesise, implement and export the DDR probe (implementation-plan step 1).
#
# Usage: vivado -mode batch -source scripts/build_probe_bit.tcl -tclargs <jobs> <freq_mhz>
#
# The WNS report matters more than usual here: the probe must be running at the
# clock the bandwidth arithmetic assumes. A design that only closes at 200 MHz
# would still "work" and would report GB/s against a 250 MHz figure that is a
# fiction. If WNS is negative, fix it before believing any number this produces.
set J    [expr {[llength $argv] > 0 ? [lindex $argv 0] : 8}]
set FREQ [expr {[llength $argv] > 1 ? [lindex $argv 1] : 250}]
set root [file normalize [file dirname [info script]]/..]
set proj $root/build/bd_probe_f${FREQ}

open_project $proj/ddr_probe_zcu104.xpr
set_property top probe_bd_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "### TOP = [get_property top [get_filesets sources_1]]"

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
    puts "### WARNING: negative setup slack -- the probe is NOT running at ${FREQ}MHz,"
    puts "### so any GB/s it reports against that clock is fiction."
}
report_utilization    -file $proj/impl_util.rpt
report_timing_summary -file $proj/impl_timing.rpt

write_hw_platform -fixed -include_bit -force $root/build/ddr_probe_f${FREQ}.xsa
puts "### XSA written: $root/build/ddr_probe_f${FREQ}.xsa"
puts "### PROBE BIT OK"
