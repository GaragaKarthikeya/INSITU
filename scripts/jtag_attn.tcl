# Run one decode step on the board over JTAG, and check it (step 12).
#
#   vivado -mode batch -source scripts/jtag_attn.tcl -tclargs <bitstream> [vecdir]
#
# There is no software on the A53. `jtag_axi` is the only master, so this
# script IS the host: it writes the cache image into DDR, loads the token,
# starts the block, polls, reads the answer back and compares it to
# `out.online.hex` -- the same file `tb_attn` checks against, which is what the
# unmodified numpy kernel produced.
#
# ~1 ms per JTAG round trip and ~4,000 transactions, so expect minutes and make
# NO throughput claim from this. Step 13's DMA path is where a device-side time
# means anything.
#
# DDR MUST BE BROUGHT UP FIRST, AND NOTHING HERE CAN DO IT
# --------------------------------------------------------
# The PS DRAM controller is dead until someone runs `psu_init`. There is no
# FSBL in this design and no software on the A53, so `scripts/jtag_ps_init.sh`
# does it over JTAG with `xsdb` before this script runs -- otherwise the block
# starts, issues its first AR to a controller that never answers, and sits at
# busy forever. That is exactly what an unbrought-up board looks like: status
# 0x00000001 and no progress, which reads like a datapath hang and is not one.
# The round-trip check below is what tells the two apart.
#
# THE CACHE GOES TO DDR FIRST, AND TOKEN 64 IS LEFT OUT
# -----------------------------------------------------
# `ddr_image.hex` holds all 65 tokens because the kernel had already appended
# this one when the vectors were collected. Loading it whole would let a
# design whose write path did nothing still answer correctly, since this token
# attends to itself. So the image is written with token 64's slot ZEROED in
# every plane of every head, exactly as `tb_attn` does, and the only way it
# comes back is through the block's own AXI write master.

if {[llength $argv] < 1} {
    error "usage: jtag_attn.tcl <bitstream> \[vector-dir\]"
}
set bit  [file normalize [lindex $argv 0]]
set root [file normalize [file join [file dirname [info script]] ..]]
set vec  [expr {[llength $argv] > 1 ? [file normalize [lindex $argv 1]] \
                                    : [file join $root tb vectors]}]

# ---------- the register map, from rtl/attn_ctrl.sv ----------
# The shim's base in the JTAG master's address space, set explicitly by
# `create_bd_attn.tcl`. DDR is at 0x0 in that same space, which is why the shim
# is not.
set SHIM         0xA0000000
set REG_CTRL     0x00000
set REG_MAGIC    0x00004
set REG_NTOK     0x00008
set REG_BASE_LO  0x0000C
set REG_BASE_HI  0x00010
set REG_STRIDE   0x00014
set REG_SPAN     0x00018
set REG_STARVE   0x0001C
set REG_CLIPS    0x00020
set REG_OVF      0x00024
set IN_BASE      0x02000
set OUT_BASE     0x08000
set MAGIC        "a77e0001"

set CTX          65
set HEAD_STRIDE  16384
set PLANE_SPAN   4096
set DDR_BASE     0x00000000
set IN_WORDS     2304
set OUT_WORDS    1536

# ---------- helpers ----------
# Every shim access is relative to SHIM; DDR is absolute.
proc reg {off} {
    global SHIM
    return [expr {$SHIM + $off}]
}
# The JTAG master's core name is discovered, not assumed: `hw_axi_1` is only
# the name it happens to get, and after `xsdb` has been on the chain the whole
# core list has to be re-enumerated anyway.
set AXI ""
proc rd {addr {len 1}} {
    global AXI
    create_hw_axi_txn -force tx [get_hw_axis $AXI] \
        -address [format %08x $addr] -len $len -type read
    run_hw_axi -quiet [get_hw_axi_txns tx]
    return [get_property DATA [get_hw_axi_txns tx]]
}
proc wr {addr data} {
    global AXI
    create_hw_axi_txn -force tx [get_hw_axis $AXI] \
        -address [format %08x $addr] -len [expr {[llength $data]}] \
        -data $data -type write
    run_hw_axi -quiet [get_hw_axi_txns tx]
}

# THE BURST WORD ORDER IS DETECTED, NOT ASSUMED
# ---------------------------------------------
# `create_hw_axi_txn -len N` takes and returns N words in ONE list, and whether
# that list is ascending or descending in address is a convention this script
# must not guess: `systolic_zcu104` only ever issued single-word transactions,
# so there is no precedent to copy. Guessing wrong scrambles the 16 words of
# every 512-bit beat, which looks exactly like a datapath that permutes
# channels -- and a readback check cannot catch it, because writing and reading
# with the same wrong transform agrees with itself.
#
# `detect_word_order` writes a burst of known words and reads them back with
# SINGLE-word reads, which have no ordering to get wrong.
set BURST_DESC 1
proc wr_words {addr words} {
    global BURST_DESC
    wr $addr [expr {$BURST_DESC ? [lreverse $words] : $words}]
}
proc rd_words {addr n} {
    global BURST_DESC
    set d [rd $addr $n]
    return [expr {$BURST_DESC ? [lreverse $d] : $d}]
}

proc detect_word_order {base} {
    global BURST_DESC
    set probe {}
    for {set i 0} {$i < 16} {incr i} { lappend probe [format %08x [expr {0xC0DE0000 + $i}]] }
    # Ascending first; the single-word reads below say whether it landed that
    # way. `jtag_axi` configured as AXI4-Lite refuses any LEN but 1, and that
    # is a Tcl error rather than a bad result -- so it is caught here and the
    # whole run drops to one word per transaction. ~11,000 transactions at
    # roughly a millisecond each, which is seconds, not the hour a rebuild of
    # the bitstream as full AXI4 would cost for no correctness difference.
    set BURST_DESC 0
    if {[catch {wr_words $base $probe} e]} {
        puts "### bursts unavailable ($e)"
        puts "### falling back to one word per transaction"
        set BURST_DESC -1
        return
    }
    set first [string tolower [rd $base]]
    set last  [string tolower [rd [expr {$base + 15*4}]]]
    if {$first eq "c0de0000" && $last eq "c0de000f"} {
        puts "### burst word order: ascending"
        return
    }
    if {$first eq "c0de000f" && $last eq "c0de0000"} {
        set BURST_DESC 1
        puts "### burst word order: descending (list is reversed)"
        return
    }
    # Bursts are not behaving as either convention -- fall back to one word per
    # transaction. Slower by ~16x and unambiguous.
    puts "### burst probe read back $first .. $last -- falling back to single-word writes"
    set BURST_DESC -1
}

# With BURST_DESC = -1 every access is a single word, which is the convention
# `systolic_zcu104` used and the only one with no ordering question in it.
proc wr_words_safe {addr words} {
    global BURST_DESC
    if {$BURST_DESC == -1} {
        set a $addr
        foreach w $words { wr $a $w; set a [expr {$a + 4}] }
    } else {
        wr_words $addr $words
    }
}
proc rd_words_safe {addr n} {
    global BURST_DESC
    if {$BURST_DESC == -1} {
        set out {}
        for {set i 0} {$i < $n} {incr i} { lappend out [rd [expr {$addr + $i*4}]] }
        return $out
    }
    return [rd_words $addr $n]
}

# ---------- connect ----------
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE $bit [current_hw_device]
program_hw_devices [current_hw_device]
refresh_hw_device [current_hw_device]

# ---------- bring the PS up, now that the PL holds a design ----------
#
# Called from here rather than left to the caller, because the ordering is not
# obvious and getting it wrong produces two different unhelpful failures: a
# device with no debug core (PS first), or a block stuck at busy (no PS init at
# all).
set freq 150
if {[regexp {attn_f(\d+)\.bit} [file tail $bit] -> f]} { set freq $f }
puts "### bringing the PS up (psu_init, ${freq} MHz project)"
if {[catch {exec [file join $root scripts jtag_ps_init.sh] $freq} out]} {
    puts $out
    error "psu_init failed -- DDR will not answer, so the run would hang at busy"
}
puts $out

# `xsdb` has been on the JTAG chain since the device was programmed, so the
# debug cores must be enumerated AGAIN -- the handle taken before it ran is
# stale, and using it fails with "Invalid hw_axi handle".
refresh_hw_device [current_hw_device]
set AXI [lindex [get_hw_axis] 0]
if {$AXI eq ""} {
    error "no hw_axi found after psu_init -- is jtag_axi in this bitstream, and did programming survive?"
}
puts "### hw_axi: $AXI"

set magic [rd [reg $REG_MAGIC]]
if {![string equal -nocase $magic $MAGIC]} {
    error "MAGIC reads $magic, expected $MAGIC -- this is not the bitstream this script targets"
}
puts "### MAGIC ok ($magic)"

# In the input window, which the token overwrites a moment later.
detect_word_order [expr {$SHIM + $IN_BASE}]

# ---------- is DDR actually there? ----------
#
# Writing 6,784 words into a controller that was never initialised succeeds
# silently -- JTAG acknowledges the write, the data goes nowhere, and the first
# symptom is the block sitting at busy forever several minutes later. One word
# out and back says so in a second, and names the cause instead of the symptom.
set probe_addr [expr {$DDR_BASE + 0x00100000}]
wr $probe_addr 5a5aa5a5
set back [string tolower [rd $probe_addr]]
if {$back ne "5a5aa5a5"} {
    error "DDR round trip failed: wrote 5a5aa5a5, read $back -- run scripts/jtag_ps_init.sh first (psu_init has not been run, so the DRAM controller is dead)"
}
wr $probe_addr a5a55a5a
set back [string tolower [rd $probe_addr]]
if {$back ne "a5a55a5a"} {
    error "DDR round trip failed on the second pattern: read $back (a stuck bus can match one pattern by accident)"
}
puts "### DDR round trip ok"

# ---------- load the cache image into DDR, token 64 blanked ----------
#
# 8,192 beats of 128 b = 128 KB. JTAG is 32 bits wide, so this is the slow part
# and it is written in bursts of 16 words rather than one at a time.
set fh [open [file join $vec ddr_image.hex] r]
set img [split [string trim [read $fh]] "\n"]
close $fh

# ONLY THE BEATS THE SCAN READS
# -----------------------------
# The image is 128 KB but the scan reads exactly tokens 0..64 of each plane of
# each head: 1,040 B of a 4,096 B code plane and 260 B of the norms plane. At
# ~1 ms per JTAG transaction the difference is 32,768 words against 6,784, so
# the rest is left as whatever DDR already held -- addresses the design never
# issues a read to.
set beats {}
for {set h 0} {$h < 8} {incr h} {
    for {set p 0} {$p < 4} {incr p} {
        set b0 [expr {($h*$HEAD_STRIDE + $p*$PLANE_SPAN)/16}]
        # 65 tokens x 16 B is 65 beats; x 4 B is 17 (the last is partial).
        set n  [expr {$p < 3 ? 65 : 17}]
        for {set b 0} {$b < $n} {incr b} { lappend beats [expr {$b0 + $b}] }
    }
}

# Token 64's slot, zeroed: three code/norm planes at beat 64 of their span, and
# the norms plane where token 64 is the LOW FOUR BYTES of beat 16. Blanking the
# whole norms beat would take tokens 61..63 with it, and those are cached
# tokens the golden depends on.
array set blank {}
for {set h 0} {$h < 8} {incr h} {
    for {set p 0} {$p < 3} {incr p} {
        set blank([expr {($h*$HEAD_STRIDE + $p*$PLANE_SPAN)/16 + 64}]) all
    }
    set blank([expr {($h*$HEAD_STRIDE + 3*$PLANE_SPAN)/16 + 16}]) low32
}

# A beat is four 32-bit words, low word LAST in the MSB-first hex line.
proc beat_words {line} {
    set out {}
    for {set w 3} {$w >= 0} {incr w -1} {
        lappend out [string range $line [expr {$w*8}] [expr {$w*8+7}]]
    }
    return $out
}

set nwritten 0
set run_addr -1
set run_words {}
foreach b $beats {
    set line [format %032s [string trim [lindex $img $b]]]
    if {[info exists blank($b)]} {
        if {$blank($b) eq "all"} {
            set line [string repeat 0 32]
        } else {
            set line "[string range $line 0 23]00000000"
        }
    }
    set addr [expr {$DDR_BASE + $b*16}]
    # Flush when this beat is not contiguous with the run, or the run is full.
    if {$run_addr >= 0 && ($addr != [expr {$run_addr + [llength $run_words]*4}] \
                           || [llength $run_words] >= 64)} {
        wr_words_safe $run_addr $run_words
        incr nwritten [llength $run_words]
        set run_addr -1
        set run_words {}
    }
    if {$run_addr < 0} { set run_addr $addr }
    set run_words [concat $run_words [beat_words $line]]
}
if {$run_addr >= 0} {
    wr_words_safe $run_addr $run_words
    incr nwritten [llength $run_words]
}

puts "### cache image loaded: $nwritten words over [llength $beats] beats, token 64 blanked"

# ---------- load the token ----------
set fh [open [file join $vec ingress.hex] r]
set beats [split [string trim [read $fh]] "\n"]
close $fh
if {[llength $beats] != 144} {
    error "ingress.hex has [llength $beats] beats, expected 144"
}
set words {}
foreach line $beats {
    set hexline [format %0128s [string trim $line]]
    for {set w 15} {$w >= 0} {incr w -1} {
        lappend words [string range $hexline [expr {$w*8}] [expr {$w*8+7}]]
    }
}
if {[llength $words] != $IN_WORDS} {
    error "assembled [llength $words] input words, expected $IN_WORDS"
}
for {set i 0} {$i < $IN_WORDS} {incr i 16} {
    wr_words_safe [expr {$SHIM + $IN_BASE + $i*4}] [lrange $words $i [expr {$i+15}]]
}

# Read it back before starting. A host that cannot trust what it loaded cannot
# interpret what it gets out, and on JTAG a dropped write is a real possibility.
set bad 0
for {set i 0} {$i < $IN_WORDS} {incr i 256} {
    set got [rd_words_safe [expr {$SHIM + $IN_BASE + $i*4}] 16]
    for {set k 0} {$k < 16} {incr k} {
        if {![string equal -nocase [lindex $got $k] [lindex $words [expr {$i+$k}]]]} {
            incr bad
        }
    }
}
if {$bad} { error "$bad input words did not read back" }
puts "### token loaded and verified: $IN_WORDS words"

# ---------- go ----------
wr [reg $REG_NTOK]    [format %08x $CTX]
wr [reg $REG_BASE_LO] [format %08x $DDR_BASE]
wr [reg $REG_BASE_HI] 00000000
wr [reg $REG_STRIDE]  [format %08x $HEAD_STRIDE]
wr [reg $REG_SPAN]    [format %08x $PLANE_SPAN]
wr [reg $REG_CTRL]    00000001

set polls 0
while {1} {
    set st [rd [reg $REG_CTRL]]
    incr polls
    # bit 1 is `done`, and it is LATCHED -- a pulse would be missed between
    # two JTAG reads a millisecond apart.
    # `0x$st` unquoted is a parse error in `expr`: the hex prefix must be
    # part of a STRING it converts, which is why the quotes are load-bearing.
    if {[expr {("0x$st" & 0x2) != 0}]} { break }
    if {$polls > 200} { error "the step never reported done (status $st)" }
}
set st [rd [reg $REG_CTRL]]
if {[expr {("0x$st" & 0x4) != 0}]} { puts "### WARNING: range_error -- a channel escaped the 24-bit seam" }
puts "### done after $polls polls, status $st"
puts "### starve [rd [reg $REG_STARVE]]  clips [rd [reg $REG_CLIPS]]  overflows [rd [reg $REG_OVF]]"

# ---------- read the answer and compare ----------
set got {}
for {set i 0} {$i < $OUT_WORDS} {incr i 16} {
    set got [concat $got [rd_words_safe [expr {$SHIM + $OUT_BASE + $i*4}] 16]]
}

set fh [open [file join $vec out.online.hex] r]
set goldhex [string trim [read $fh]]
close $fh
# 2,048 lanes of 24 b, lane 0 LAST in the file, packed LSB-first on the wire --
# `hw/vectors.py::pack_lanes` and `beats`. Rebuild the same word stream.
set nlane 2048
set acc 0
set nacc 0
set gold {}
for {set l 0} {$l < $nlane} {incr l} {
    set h [string range $goldhex [expr {($nlane-1-$l)*6}] [expr {($nlane-1-$l)*6+5}]]
    scan $h %x v
    set acc [expr {$acc | ($v << $nacc)}]
    incr nacc 24
    while {$nacc >= 32} {
        lappend gold [format %08x [expr {$acc & 0xFFFFFFFF}]]
        set acc [expr {$acc >> 32}]
        incr nacc -32
    }
}
if {$nacc > 0} { lappend gold [format %08x [expr {$acc & 0xFFFFFFFF}]] }

if {[llength $gold] != $OUT_WORDS} {
    error "golden assembled to [llength $gold] words, expected $OUT_WORDS"
}
set bad 0
for {set i 0} {$i < $OUT_WORDS} {incr i} {
    if {![string equal -nocase [lindex $got $i] [lindex $gold $i]]} {
        if {$bad < 8} {
            puts "###   word $i: [lindex $got $i] != [lindex $gold $i]"
        }
        incr bad
    }
}
if {$bad} {
    error "JTAG_ATTN_FAILED: $bad of $OUT_WORDS words differ from out.online.hex"
}
puts "### all $OUT_WORDS words match out.online.hex"
puts "### JTAG_ATTN_PASSED"
close_hw_manager
