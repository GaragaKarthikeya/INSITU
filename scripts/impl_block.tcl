# Out-of-context PLACE AND ROUTE for one block, with the timing check as an error.
#
# `synth_block.tcl` answers whether the logic can be described at 250 MHz.
# Step 11's exit criterion is stronger and needs this one: a top-level whose
# synthesis WNS is positive can still miss after placement, because the four
# score lanes, four softmaxes and four accumulators are the first thing in this
# design that has to be PLACED near each other. Post-route WNS is the number
# `plan.MD` records.
#
# Usage: vivado -mode batch -source scripts/impl_block.tcl \
#            -tclargs <out-dir> <top> <src.sv> [<src.sv> ...]
if {[llength $argv] < 3} {
    error "usage: impl_block.tcl <out-dir> <top> <src.sv> ..."
}

set out  [file normalize [lindex $argv 0]]
set top  [lindex $argv 1]
set srcs [lrange $argv 2 end]
file mkdir $out
set root [file normalize [file join [file dirname [info script]] ..]]

foreach s $srcs { read_verilog -sv [file join $root $s] }
read_xdc [file join $root scripts block_250mhz.xdc]
synth_design -top $top -part xczu7ev-ffvc1156-2-e -mode out_of_context

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force [file join $out ${top}_routed.dcp]
report_utilization -hierarchical -file [file join $out utilization_routed.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $out timing_routed.rpt]

set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths] != 1} {
    error "no constrained max-delay path was reported"
}
set wns [get_property SLACK [lindex $paths 0]]
puts [format "%s_ROUTED_WNS_NS %.3f" [string toupper $top] $wns]
if {$wns < 0.0} {
    error [format "%s misses 250 MHz after routing (WNS %.3f ns)" $top $wns]
}
puts [format "%s_PLACE_AND_ROUTE_PASSED" [string toupper $top]]
