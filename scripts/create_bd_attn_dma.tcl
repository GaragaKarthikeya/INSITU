# ZCU104 system for the attention block driven by the PS (step 13).
#
#   PS M_AXI_HPM0_FPD (AXI4, 32b)     -> attn_ctrl control + buffers  @ 0xA0000000
#                                     -> axi_dma registers            @ 0xA0100000
#   axi_dma MM2S -> attn_ctrl s_axis; attn_ctrl m_axis -> axi_dma S2MM
#   axi_dma MM/SM masters             -> PS S_AXI_HPC0_FPD -> DDR
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
# THE DMA IS HERE BECAUSE STEP 13 MEASURED WHY
# --------------------------------------------
# 366 us of every 433 us decode step was the A53 pushing 15 KB through the
# 32-bit AXI-Lite window one store at a time -- 85% of the step against 67 us
# of block. The shim keeps its buffers (MODE 0) and gains a 512-bit stream
# (MODE 1); this adds the DMA that drives it.
#
# IT GETS ITS OWN COHERENT PORT, NOT A SHARE OF AN HP PORT.
# All four HP ports carry a KV read master and the design is sized against them
# delivering 16.0 GB/s; putting 15 KB per token onto one of them would tax the
# exact resource the whole architecture is short of. HPC0 is otherwise unused,
# and being coherent it also means the token buffers need no cache maintenance
# -- which removes a whole class of "correct hardware, stale data" bug that the
# image copy still has to be careful about.
#
# Usage: vivado -mode batch -source scripts/create_bd_attn_dma.tcl -tclargs <freq_mhz>

set FREQ [expr {[llength $argv] > 0 ? [lindex $argv 0] : 250}]
set root [file normalize [file dirname [info script]]/..]
set proj $root/build/bd_attndma_f${FREQ}
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
    PSU__USE__S_AXI_GP0                   1 \
    PSU__SAXIGP0__DATA_WIDTH              128 \
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
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $scc
connect_bd_net $clk   [get_bd_pins $scc/aclk]
connect_bd_net $arstn [get_bd_pins $scc/aresetn]
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $scc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $scc/M00_AXI]       [get_bd_intf_pins $at/s_axi]
connect_bd_net $clk [get_bd_pins $ps/maxihpm0_fpd_aclk]

# ---------- The DMA ----------
#
# Simple mode: one descriptor per transfer, which is exactly one token in and
# one result out. Scatter-gather would buy nothing -- there is no list to walk.
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg               {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_mm2s_dre         {0} \
    CONFIG.c_include_s2mm_dre         {0} \
    CONFIG.c_m_axi_mm2s_data_width    {512} \
    CONFIG.c_m_axis_mm2s_tdata_width  {512} \
    CONFIG.c_m_axi_s2mm_data_width    {512} \
    CONFIG.c_s_axis_s2mm_tdata_width  {512} \
    CONFIG.c_mm2s_burst_size          {16} \
    CONFIG.c_s2mm_burst_size          {16} \
    CONFIG.c_sg_length_width          {26} \
] $dma
connect_bd_net $clk [get_bd_pins $dma/s_axi_lite_aclk] \
                    [get_bd_pins $dma/m_axi_mm2s_aclk] \
                    [get_bd_pins $dma/m_axi_s2mm_aclk]
connect_bd_net $arstn [get_bd_pins $dma/axi_resetn]

# The streams. 512 bits is the core's own beat, so nothing converts widths on
# the way in or out -- the payload the host builds IS the beat stream the block
# consumes, which is what `plan.MD` asked the protocol to preserve.
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $at/s_axis]
connect_bd_intf_net [get_bd_intf_pins $at/m_axis]       [get_bd_intf_pins $dma/S_AXIS_S2MM]

# The DMA's own registers, on the same control path as the shim.
connect_bd_intf_net [get_bd_intf_pins $scc/M01_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]

# ---------- The DMA's path to DDR: HPC0, its own port ----------
set scdma [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_dma]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $scdma
connect_bd_net $clk   [get_bd_pins $scdma/aclk]
connect_bd_net $arstn [get_bd_pins $scdma/aresetn]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $scdma/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $scdma/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $scdma/M00_AXI]  [get_bd_intf_pins $ps/S_AXI_HPC0_FPD]
connect_bd_net $clk [get_bd_pins $ps/saxihpc0_fpd_aclk]

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
    # 0xA0100000, not 0xA0010000: the shim's window is 128 KB wide (LAW = 17),
    # so anything below 0xA0020000 lands inside it and `assign_bd_address`
    # rejects the overlap. `sw/attn_main.c` has the same number.
    if {[string match "*axi_dma*" $seg]} {
        set_property OFFSET 0xA0100000 $seg
        puts "### dma at 0xA0100000 ($seg)"
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
puts "### ATTN DMA BD OK FREQ=${FREQ}MHz ACTUAL=[expr {$ACT_HZ/1000000.0}]MHz"
