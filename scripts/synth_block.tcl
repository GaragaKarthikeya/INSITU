# Out-of-context synthesis for one standalone datapath block.
#
# Usage: vivado -mode batch -source scripts/synth_block.tcl \
#            -tclargs <out-dir> <top> <src.sv> [<src.sv> ...]
#
# The WNS check is an ERROR, not a report: the plan's rule is that every RTL
# step ends with synthesis that MEETS 250 MHz, so a build that misses must fail
# the script rather than print a number somebody has to notice.
if {[llength $argv] < 3} {
    error "usage: synth_block.tcl <out-dir> <top> <src.sv> ..."
}

set out  [file normalize [lindex $argv 0]]
set top  [lindex $argv 1]
set srcs [lrange $argv 2 end]
file mkdir $out
set root [file normalize [file join [file dirname [info script]] ..]]

foreach s $srcs { read_verilog -sv [file join $root $s] }
read_xdc [file join $root scripts block_250mhz.xdc]
synth_design -top $top -part xczu7ev-ffvc1156-2-e -mode out_of_context
write_checkpoint -force [file join $out ${top}_synth.dcp]

report_utilization -hierarchical -file [file join $out utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 -file [file join $out timing.rpt]

set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths] != 1} {
    error "no constrained max-delay path was reported"
}
set wns [get_property SLACK [lindex $paths 0]]
puts [format "%s_OOC_WNS_NS %.3f" [string toupper $top] $wns]
if {$wns < 0.0} {
    error [format "%s misses the 250 MHz target (WNS %.3f ns)" $top $wns]
}
puts [format "%s_OOC_SYNTHESIS_PASSED" [string toupper $top]]
