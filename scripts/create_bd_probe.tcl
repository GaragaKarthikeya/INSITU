# ZCU104 system for the DDR bandwidth probe (implementation-plan step 1).
#
#   PS M_AXI_HPM0_FPD (AXI-Lite, 32b) -> ddr_probe control registers
#   ddr_probe m00..m03 (AXI4, 128b)   -> PS S_AXI_HP0..HP3_FPD -> DDR
#
# Four HP ports, not three: a 128-bit HP port moves 16 B per PL clock, so at a
# single 250 MHz domain three ports reach only 12.0 GB/s against the 13.0 GB/s
# the score engine needs. Four give 16.0 GB/s with no second clock domain and
# no CDC. See kernel/rtl/ddr_probe.sv.
#
# Interconnect is wired explicitly rather than via apply_bd_automation: the
# automation rules are version-sensitive and fail opaquely.
#
# Usage: vivado -mode batch -source scripts/create_bd_probe.tcl -tclargs <freq_mhz>

set FREQ [expr {[llength $argv] > 0 ? [lindex $argv 0] : 250}]
set root [file normalize [file dirname [info script]]/..]
set proj $root/build/bd_probe_f${FREQ}
set part xczu7ev-ffvc1156-2-e
set board xilinx.com:zcu104:part0:1.1

file delete -force $proj
create_project ddr_probe_zcu104 $proj -part $part
set_property board_part $board [current_project]
add_files -norecurse [list $root/rtl/ddr_probe.sv $root/rtl/axi_rd_engine.sv]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1

create_bd_design "probe_bd"

# ---------- Processing system ----------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
# Set individually so one bad key cannot silently void the whole dict.
# GP2..GP5 are S_AXI_HP0..HP3_FPD.
foreach {k v} [list \
    PSU__USE__M_AXI_GP0                   1 \
    PSU__USE__M_AXI_GP1                   0 \
    PSU__USE__S_AXI_GP2                   1 \
    PSU__USE__S_AXI_GP3                   1 \
    PSU__USE__S_AXI_GP4                   1 \
    PSU__USE__S_AXI_GP5                   1 \
    PSU__MAXIGP0__DATA_WIDTH              32 \
    PSU__SAXIGP2__DATA_WIDTH              128 \
    PSU__SAXIGP3__DATA_WIDTH              128 \
    PSU__SAXIGP4__DATA_WIDTH              128 \
    PSU__SAXIGP5__DATA_WIDTH              128 \
    PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ   $FREQ ] {
    if {[catch {set_property CONFIG.$k $v $ps} e]} { puts "WARN: CONFIG.$k -> $e" }
}
set clk [get_bd_pins $ps/pl_clk0]
# The PS PLL cannot hit every requested rate (ask for 200 MHz and you get
# 187.5). Read back what it actually produces and use THAT everywhere, or the
# declared FREQ_HZ will mismatch and validation fails. It is also the number
# the bandwidth arithmetic must use -- reporting GB/s against the REQUESTED
# clock when the PLL gave you something else is how a probe lies.
set ACT_HZ [get_property CONFIG.FREQ_HZ $clk]
puts "### requested ${FREQ}MHz -> actual [expr {$ACT_HZ/1000000.0}]MHz"
set rstn_src [get_bd_pins $ps/pl_resetn0]

# ---------- Reset ----------
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl]
connect_bd_net $clk [get_bd_pins $rst/slowest_sync_clk]
connect_bd_net $rstn_src [get_bd_pins $rst/ext_reset_in]
set arstn [get_bd_pins $rst/peripheral_aresetn]

# ---------- Probe ----------
set pr [create_bd_cell -type module -reference ddr_probe ddr_probe_0]
# Module-reference clocks carry no metadata; declare the rate so timing
# constraints and width checks propagate correctly.
#
# ASSOCIATED_BUSIF is deliberately NOT set: in 2026.1 it is read-only on a
# module-reference clock pin and setting it only produces a CRITICAL WARNING.
# Vivado infers the association from the s_axi/m00_axi..m03_axi name prefixes
# on its own -- confirmed by all four masters appearing in the address map.
catch {set_property CONFIG.FREQ_HZ $ACT_HZ [get_bd_pins $pr/clk]}
connect_bd_net $clk   [get_bd_pins $pr/clk]
connect_bd_net $arstn [get_bd_pins $pr/rstn]

# ---------- Control path: PS -> probe registers ----------
set scc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $scc
connect_bd_net $clk   [get_bd_pins $scc/aclk]
connect_bd_net $arstn [get_bd_pins $scc/aresetn]
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $scc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $scc/M00_AXI]       [get_bd_intf_pins $pr/s_axi]
connect_bd_net $clk [get_bd_pins $ps/maxihpm0_fpd_aclk]

# ---------- Data path: one master per HP port, NO shared interconnect ----------
# Each engine gets its own dedicated path to its own HP port. Funnelling them
# through one smartconnect would serialise them at the crossbar and measure the
# interconnect instead of the DRAM -- which is the one thing this probe must
# not do.
set hp_clk_pin {saxihp0_fpd_aclk saxihp1_fpd_aclk saxihp2_fpd_aclk saxihp3_fpd_aclk}
set hp_intf    {S_AXI_HP0_FPD S_AXI_HP1_FPD S_AXI_HP2_FPD S_AXI_HP3_FPD}
for {set i 0} {$i < 4} {incr i} {
    set m [format "m%02d_axi" $i]
    set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_data_$i]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $sc
    connect_bd_net $clk   [get_bd_pins $sc/aclk]
    connect_bd_net $arstn [get_bd_pins $sc/aresetn]
    connect_bd_intf_net [get_bd_intf_pins $pr/$m]      [get_bd_intf_pins $sc/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sc/M00_AXI] [get_bd_intf_pins $ps/[lindex $hp_intf $i]]
    connect_bd_net $clk [get_bd_pins $ps/[lindex $hp_clk_pin $i]]
}

regenerate_bd_layout
assign_bd_address
validate_bd_design
save_bd_design

puts "### ADDRESS MAP"
foreach s [get_bd_addr_segs] {
    puts [format "###   %-46s %s +%s" [file tail $s] \
        [get_property OFFSET $s] [get_property RANGE $s]]
}
make_wrapper -files [get_files probe_bd.bd] -top -import
set_property top probe_bd_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "### TOP = [get_property top [get_filesets sources_1]]"
puts "### PROBE BD OK FREQ=${FREQ}MHz ACTUAL=[expr {$ACT_HZ/1000000.0}]MHz"
