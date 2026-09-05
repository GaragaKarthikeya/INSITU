# Run the attention block from the A53, over JTAG but with no JTAG in the loop.
#
#   xsdb host/run_attn_a53.tcl
# Capture UART0 separately:  ./host/uart_cap.py /tmp/attn.log 300 &
#
# The cable programs the PL and downloads the ELF and then gets out of the way:
# every AXI transaction of the run is issued by the A53. That is the difference
# from `scripts/jtag_attn.tcl`, where the cable WAS the master and a decode step
# took minutes.
#
# ORDER MATTERS, AND STEP 12 LEARNED WHY THE HARD WAY
# ---------------------------------------------------
#  * `rst -system` first, or `psu_init` runs against a half-initialised PS.
#  * `psu_init` before `fpga`, unlike step 12. There the PL had to be
#    programmed first because `psu_ps_pl_isolation_removal` wanted a configured
#    PL and Vivado needed its debug core to survive; here nothing reads a debug
#    core, so this is the ordinary Vitis order and `psu_init` brings DDR up
#    before anything can touch it.
#  * The A53 must be HALTED and its registers cleared before `dow`, or the
#    still-running previous ELF wedges the core ("EDITR not ready").
#
# The ELF carries ~1.2 MB of goldens, so `dow` is not instant. That is a
# one-time load and no part of any number the run reports.
set ROOT [file dirname [file dirname [file normalize [info script]]]]
set FREQ [expr {[info exists ::env(ATTN_FREQ)] ? $::env(ATTN_FREQ) : 250}]
set VAR  [expr {[info exists ::env(ATTN_VARIANT)] ? $::env(ATTN_VARIANT) : "attndma"}]
set WS   $ROOT/build/vitis_ws_${VAR}_f${FREQ}
set PLAT attn_plat_f${FREQ}
set APP  attn_test_f${FREQ}
# GLOBBED, NOT SPELLED OUT. Vitis names the bitstream after the XSA and puts
# copies of it in three places; hard-coding one of them is a path that breaks
# silently on the next tool version, and "file not found" three minutes into a
# board session is the worst time to learn it.
proc only {what pattern} {
    set hits [glob -nocomplain $pattern]
    if {[llength $hits] == 0} {
        error "no $what matching $pattern -- run scripts/build_vitis_attn.py first"
    }
    return [lindex [lsort $hits] 0]
}
set BIT  [only bitstream  $WS/$PLAT/hw/sdt/*.bit]
set INIT [only psu_init   $WS/$APP/_ide/psinit/psu_init.tcl]
set ELF  [only ELF        $WS/$APP/build/*.elf]
puts "### bit  $BIT"
puts "### init $INIT"
puts "### elf  $ELF"

connect

targets -set -nocase -filter {name =~ "*PSU*"}
rst -system
after 4000

targets -set -nocase -filter {name =~ "*PSU*"}
mwr 0xffca0038 0x1ff
after 500

puts "### psu_init"
source $INIT
psu_init
after 1000
catch {psu_ps_pl_isolation_removal}
catch {psu_ps_pl_reset_config}
after 500

puts "### fpga $BIT"
targets -set -nocase -filter {name =~ "*PL*"}
fpga -file $BIT
after 500

targets -set -nocase -filter {name =~ "*A53*#0*"}
catch {stop}
after 300
rst -processor -clear-registers
after 500
catch {stop}

puts "### dow $ELF"
dow $ELF
puts "### con"
con

# NO `stop`, AND NO WAIT. This launcher exists to leave the A53 RUNNING: it
# finishes its self-test, prints SERVER READY, and then sits in `serve()`
# polling the mailbox for work. Halting the processor here -- which is what
# `run_attn_a53.tcl` does, correctly, for a self-contained test run -- would
# kill the server before the host could use it.
#
# xsdb exits and releases the cable; `host/attn_jtag.py` opens its own
# connection through the same hw_server. The self-test takes a few seconds, so
# give it a moment before the first token.
after 20000
puts "### SERVER RUNNING -- the A53 is polling its mailbox"
puts "### now run: python -m kernel.experiments.fpga_generate --tokens 16"
