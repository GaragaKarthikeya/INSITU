# Out-of-context synthesis for the Step 5 randomized FWHT block.
# Usage: vivado -mode batch -source scripts/synth_rot_fwht.tcl -tclargs <out-dir>
if {[llength $argv] != 1} {
    error "usage: synth_rot_fwht.tcl <out-dir>"
}

set out [file normalize [lindex $argv 0]]
file mkdir $out
set root [file normalize [file join [file dirname [info script]] ..]]

read_verilog -sv [file join $root rtl rot_fwht.sv]
read_xdc [file join $root scripts rot_fwht_250mhz.xdc]
synth_design -top rot_fwht -part xczu7ev-ffvc1156-2-e -mode out_of_context
write_checkpoint -force [file join $out rot_fwht_synth.dcp]

report_utilization -hierarchical -file [file join $out utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $out timing.rpt]

set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths] != 1} {
    error "no constrained max-delay path was reported"
}
set wns [get_property SLACK [lindex $paths 0]]
puts [format "ROT_FWHT_OOC_WNS_NS %.3f" $wns]
if {$wns < 0.0} {
    error [format "rot_fwht misses the 250 MHz target (WNS %.3f ns)" $wns]
}
puts "ROT_FWHT_OOC_SYNTHESIS_PASSED"
