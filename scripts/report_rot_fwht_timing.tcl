# Report timing from the OOC synthesis checkpoint without re-running synthesis.
# Usage: vivado -mode batch -source report_rot_fwht_timing.tcl -tclargs <out-dir>
if {[llength $argv] != 1} {
    error "usage: report_rot_fwht_timing.tcl <out-dir>"
}
set out [file normalize [lindex $argv 0]]
open_checkpoint [file join $out rot_fwht_synth.dcp]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $out timing.rpt]
set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths] != 1} { error "no constrained max-delay path was reported" }
set wns [get_property SLACK [lindex $paths 0]]
puts [format "ROT_FWHT_OOC_WNS_NS %.3f" $wns]
if {$wns < 0.0} { error [format "rot_fwht misses 250 MHz (WNS %.3f ns)" $wns] }
puts "ROT_FWHT_OOC_TIMING_PASSED"
