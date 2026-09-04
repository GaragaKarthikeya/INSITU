# Bring up the DDR probe on the A53 over JTAG (no SD card / boot image).
#   xsdb host/run_probe.tcl
# Capture UART0 separately:  ./host/uart_cap.py /tmp/probe.log 300 &
#
# Order matters, and every line of it was learned the hard way in
# fpga_work/systolic_zcu104:
#  * rst -system first, or psu_init fails against a half-initialised PS.
#  * the A53 must be HALTED and its registers cleared before `dow`, otherwise
#    the still-running previous ELF wedges the core ("EDITR not ready").
set FREQ 250
set ROOT [file dirname [file dirname [file normalize [info script]]]]
set WS   $ROOT/build/vitis_ws_probe_f$FREQ
set BIT  $WS/probe_plat_f$FREQ/hw/sdt/ddr_probe_f$FREQ.bit
set INIT $WS/probe_app_f$FREQ/_ide/psinit/psu_init.tcl
set ELF  $WS/probe_app_f$FREQ/build/probe_app_f$FREQ.elf

foreach f [list $BIT $INIT $ELF] {
    if {![file exists $f]} { puts "### MISSING: $f"; exit 1 }
}

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

puts "### fpga"
targets -set -nocase -filter {name =~ "*PL*"}
fpga -file $BIT
after 500

targets -set -nocase -filter {name =~ "*A53*#0*"}
catch {stop}
after 300
rst -processor -clear-registers
after 500
catch {stop}
after 300

puts "### dow"
dow $ELF
puts "### con"
con
# The sweep is 4 masks x 4 bursts x 6 depths = 96 runs of 16 MB/port. Give it
# room; the capture script is what actually bounds the wall time.
after 60000
catch {stop}
puts "### DONE"
