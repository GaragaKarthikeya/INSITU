# ZCU104 system for the attention block driven by the PS (step 13).
#
#   PS M_AXI_HPM0_FPD (AXI4, 32b)     -> attn_ctrl control + buffers  @ 0xA0000000
#   attn_ctrl m00..m03 (AXI4, 128b)   -> PS S_AXI_HP0..HP3_FPD -> DDR
#   attn_ctrl write master (AXI4)     -> PS S_AXI_HP0_FPD      -> DDR
#
# EVERYTHING BELOW THE SHIM IS `create_bd_attn.tcl` UNCHANGED. The only thing
# replaced is the master: `jtag_axi` becomes the PS. That was the plan for the
# shim from the start, and it is why it was built as addressed memory rather
# than a FIFO window -- a stream would have had to be rebuilt here.
#
# TWO THINGS THE JTAG DESIGN NEEDED AND THIS ONE DOES NOT
# -------------------------------------------------------
#  * JTAG's second path to DDR through HP3, which existed only so the cable
#    could write the cache image. The A53 writes it now, through the PS's own
#    port, so HP3 carries the block's fourth read master and nothing else.
#  * `scripts/jtag_ps_init.sh`. There is software on the A53 now, so the DRAM
#    controller is brought up by `psu_init` from `host/run_attn_a53.tcl`
#    BEFORE the ELF is downloaded -- which is the ordinary Vitis flow and the
#    reason step 12's most confusing failure cannot happen here.
#
# The four read masters keep their own dedicated HP ports, exactly as
# `create_bd_probe.tcl` does and for the same reason: funnelling them through
# one interconnect serialises them at the crossbar, and the whole design is
# sized against four ports delivering 16.0 GB/s.
#
# Usage: vivado -mode batch -source scripts/create_bd_attn_ps.tcl -tclargs <freq_mhz>

set FREQ [expr {[llength $argv] > 0 ? [lindex $argv 0] : 250}]
set root [file normalize [file dirname [info script]]/..]
set proj $root/build/bd_attnps_f${FREQ}
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

# ---------- Processing system: DDR, clock, and the master ----------
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
# GP2..GP5 are S_AXI_HP0..HP3_FPD. GP0 is M_AXI_HPM0_FPD, the PS's own path
# to the shim -- the master JTAG used to be.
foreach {k v} [list \
    PSU__USE__M_AXI_GP0                   1 \
    PSU__MAXIGP0__DATA_WIDTH              32 \
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

# ---------- Control path: PS -> the shim ----------
#
# One 32-bit AXI4 master out of the PS, adapted to the shim's AXI-Lite by a
# smartconnect. Full AXI4 rather than AXI-Lite at the PS end, so the A53's
# stores are not each a separate address phase on the way in -- which is the
# whole 9,216-byte token load, and the part of a decode step the host pays for.
set scc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $scc
connect_bd_net $clk   [get_bd_pins $scc/aclk]
connect_bd_net $arstn [get_bd_pins $scc/aresetn]
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $scc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $scc/M00_AXI]       [get_bd_intf_pins $at/s_axi]
connect_bd_net $clk [get_bd_pins $ps/maxihpm0_fpd_aclk]

# ---------- Data path: one read master per HP port ----------
set hp_clk_pin {saxihp0_fpd_aclk saxihp1_fpd_aclk saxihp2_fpd_aclk saxihp3_fpd_aclk}
set hp_intf    {S_AXI_HP0_FPD S_AXI_HP1_FPD S_AXI_HP2_FPD S_AXI_HP3_FPD}
for {set i 0} {$i < 4} {incr i} {
    set m [format "m%02d_axi" $i]
    set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_data_$i]
    # Port 0 also carries the block's write master. Port 3 carried JTAG's path
    # to DDR in step 12 and carries nothing extra now: the A53 loads the image
    # through the PS's own port.
    set nsi [expr {$i == 0 ? 2 : 1}]
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

regenerate_bd_layout
assign_bd_address

# The shim's address is pinned rather than left to `assign_bd_address`, because
# `sw/attn_main.c` has it as a constant: the two are the same number in two
# files, and the `MAGIC` register is what says so at run time. 0xA0000000 is
# also where the JTAG design put it, so the register map is unchanged.
foreach seg [get_bd_addr_segs -of_objects \
                [get_bd_addr_spaces zynq_ultra_ps_e_0/Data]] {
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
puts "### ATTN PS BD OK FREQ=${FREQ}MHz ACTUAL=[expr {$ACT_HZ/1000000.0}]MHz"
