# ZCU104 system for the attention block over JTAG (implementation-plan step 12).
#
#   jtag_axi (AXI4, 32b)              -> attn_ctrl control + buffers  @ 0xA0000000
#                                     -> PS S_AXI_HP3_FPD -> DDR       @ 0x00000000
#   attn_ctrl m00..m03 (AXI4, 128b)   -> PS S_AXI_HP0..HP3_FPD -> DDR
#   attn_ctrl write master (AXI4)     -> PS S_AXI_HP0_FPD      -> DDR
#
# NO PS MASTER AT ALL. Step 12 is correctness only and the host is `xsdb`
# through the JTAG cable, so nothing runs on the A53 and there is no software
# to get wrong. That means JTAG must also be able to write the cache image into
# DDR, so it gets a second path through HP3 -- and the two are given explicit,
# far-apart addresses, because both the shim and DDR would otherwise be
# assigned 0x0 and one of them would silently win. Step 13 replaces this master with the PS and keeps everything
# below it identical -- which is the reason the shim is memory-mapped rather
# than a stream.
#
# The four read masters keep their own dedicated HP ports, exactly as
# `create_bd_probe.tcl` does and for the same reason: funnelling them through
# one interconnect serialises them at the crossbar, and the whole design is
# sized against four ports delivering 16.0 GB/s.
#
# Usage: vivado -mode batch -source scripts/create_bd_attn.tcl -tclargs <freq_mhz>

set FREQ [expr {[llength $argv] > 0 ? [lindex $argv 0] : 250}]
set root [file normalize [file dirname [info script]]/..]
set proj $root/build/bd_attn_f${FREQ}
set part xczu7ev-ffvc1156-2-e
set board xilinx.com:zcu104:part0:1.1

file delete -force $proj
create_project attn_zcu104 $proj -part $part
set_property board_part $board [current_project]
add_files -norecurse [list \
    $root/rtl/attn_ctrl.sv $root/rtl/attn_top.sv $root/rtl/attn_ingress.sv \
    $root/rtl/rot_fwht.sv $root/rtl/rot_norm.sv $root/rtl/rot_encode.sv \
    $root/rtl/score_lane.sv $root/rtl/qtab_build.sv $root/rtl/softmax_online.sv \
    $root/rtl/exp_lut.sv $root/rtl/accum.sv $root/rtl/finalize.sv \
    $root/rtl/kv_store_ddr.sv $root/rtl/kv_plane_rd.sv $root/rtl/kv_write.sv]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1

create_bd_design "attn_bd"

# ---------- Processing system: DDR and clock only ----------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
# GP2..GP5 are S_AXI_HP0..HP3_FPD. No M_AXI at all: JTAG is the master.
foreach {k v} [list \
    PSU__USE__M_AXI_GP0                   0 \
    PSU__USE__M_AXI_GP1                   0 \
    PSU__USE__S_AXI_GP2                   1 \
    PSU__USE__S_AXI_GP3                   1 \
    PSU__USE__S_AXI_GP4                   1 \
    PSU__USE__S_AXI_GP5                   1 \
    PSU__SAXIGP2__DATA_WIDTH              128 \
    PSU__SAXIGP3__DATA_WIDTH              128 \
    PSU__SAXIGP4__DATA_WIDTH              128 \
    PSU__SAXIGP5__DATA_WIDTH              128 \
    PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ   $FREQ ] {
    if {[catch {set_property CONFIG.$k $v $ps} e]} { puts "WARN: CONFIG.$k -> $e" }
}
set clk [get_bd_pins $ps/pl_clk0]
# The PS PLL cannot hit every requested rate. Read back what it actually
# produces and use THAT everywhere, or the declared FREQ_HZ mismatches and
# validation fails.
set ACT_HZ [get_property CONFIG.FREQ_HZ $clk]
puts "### requested ${FREQ}MHz -> actual [expr {$ACT_HZ/1000000.0}]MHz"
set rstn_src [get_bd_pins $ps/pl_resetn0]

# ---------- Reset ----------
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl]
connect_bd_net $clk [get_bd_pins $rst/slowest_sync_clk]
connect_bd_net $rstn_src [get_bd_pins $rst/ext_reset_in]
set arstn [get_bd_pins $rst/peripheral_aresetn]

# ---------- The block ----------
set at [create_bd_cell -type module -reference attn_ctrl attn_ctrl_0]
catch {set_property CONFIG.FREQ_HZ $ACT_HZ [get_bd_pins $at/clk]}
connect_bd_net $clk   [get_bd_pins $at/clk]
connect_bd_net $arstn [get_bd_pins $at/rstn]

# ---------- Control path: JTAG -> the shim ----------
#
# `jtag_axi` is an AXI4 master driven by the debug bridge, so `create_hw_axi_txn`
# in Vivado or xsdb reaches these registers with no software on the board at
# all. A smartconnect adapts its 32-bit AXI4 to the shim's AXI-Lite.
set jt [create_bd_cell -type ip -vlnv xilinx.com:ip:jtag_axi jtag_axi_0]
# PROTOCOL 2 is AXI4-Lite, which supports no bursts: every `create_hw_axi_txn`
# moves one word. `scripts/jtag_attn.tcl` detects that and issues single-word
# transactions, which costs seconds over a whole run and nothing in
# correctness. Full AXI4 here would allow bursts and is worth revisiting only
# if the load time ever matters -- and by step 13 the PS, not JTAG, is the
# master.
set_property -dict [list CONFIG.PROTOCOL {2} CONFIG.M_AXI_DATA_WIDTH {32}] $jt
connect_bd_net $clk   [get_bd_pins $jt/aclk]
connect_bd_net $arstn [get_bd_pins $jt/aresetn]

set scc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $scc
connect_bd_net $clk   [get_bd_pins $scc/aclk]
connect_bd_net $arstn [get_bd_pins $scc/aresetn]
connect_bd_intf_net [get_bd_intf_pins $jt/M_AXI]    [get_bd_intf_pins $scc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $scc/M00_AXI] [get_bd_intf_pins $at/s_axi]

# ---------- Data path: one read master per HP port ----------
set hp_clk_pin {saxihp0_fpd_aclk saxihp1_fpd_aclk saxihp2_fpd_aclk saxihp3_fpd_aclk}
set hp_intf    {S_AXI_HP0_FPD S_AXI_HP1_FPD S_AXI_HP2_FPD S_AXI_HP3_FPD}
for {set i 0} {$i < 4} {incr i} {
    set m [format "m%02d_axi" $i]
    set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_data_$i]
    # Port 0 also carries the block's write master; port 3 also carries JTAG's
    # path to DDR, which is used only to load the image before a run.
    set nsi [expr {($i == 0 || $i == 3) ? 2 : 1}]
    set_property -dict [list CONFIG.NUM_SI $nsi CONFIG.NUM_MI {1}] $sc
    connect_bd_net $clk   [get_bd_pins $sc/aclk]
    connect_bd_net $arstn [get_bd_pins $sc/aresetn]
    connect_bd_intf_net [get_bd_intf_pins $at/$m]      [get_bd_intf_pins $sc/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $sc/M00_AXI] [get_bd_intf_pins $ps/[lindex $hp_intf $i]]
    connect_bd_net $clk [get_bd_pins $ps/[lindex $hp_clk_pin $i]]
}

# ---------- The write path ----------
#
# 416 B per token against 13.0 GB/s of reads, so it shares port 0 rather than
# taking a fifth. It is a separate MASTER -- a write that stalled the read
# stream would stall the score pipe -- but not a separate PORT, because the
# traffic is 0.02% and the crossbar has the headroom.
connect_bd_intf_net [get_bd_intf_pins $at/m_wr_axi] [get_bd_intf_pins sc_data_0/S01_AXI]

# ---------- JTAG's own path to DDR ----------
#
# Only for loading the cache image and only between runs, so it shares HP3
# rather than taking a port of its own. It is idle while a scan is running.
connect_bd_intf_net [get_bd_intf_pins $scc/M01_AXI] [get_bd_intf_pins sc_data_3/S01_AXI]

regenerate_bd_layout
assign_bd_address

# The shim and DDR both want 0x0 from the JTAG master's point of view, and
# whichever `assign_bd_address` picked second would be unreachable. Put the
# shim at 0xA0000000 explicitly; `scripts/jtag_attn.tcl` uses the same number.
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces jtag_axi_0/Data]] {
    if {[string match "*attn_ctrl*" $seg]} {
        set_property OFFSET 0xA0000000 $seg
        puts "### shim at 0xA0000000 ($seg)"
    }
}
validate_bd_design
save_bd_design

puts "### ADDRESS MAP"
foreach s [get_bd_addr_segs] {
    puts [format "###   %-46s %s +%s" [file tail $s] \
        [get_property OFFSET $s] [get_property RANGE $s]]
}
make_wrapper -files [get_files attn_bd.bd] -top -import
set_property top attn_bd_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "### TOP = [get_property top [get_filesets sources_1]]"
puts "### ATTN BD OK FREQ=${FREQ}MHz ACTUAL=[expr {$ACT_HZ/1000000.0}]MHz"
