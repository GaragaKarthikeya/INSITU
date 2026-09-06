#!/usr/bin/env python3
"""Capture ZCU104 UART to a log file. Line-buffered, flushes every line.

  ./host/uart_cap.py <logfile> [seconds] [device] [baud]
"""
import os, sys, time, termios

log  = sys.argv[1]
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
dev  = sys.argv[3] if len(sys.argv) > 3 else "/dev/ttyUSB1"
baud = int(sys.argv[4]) if len(sys.argv) > 4 else 115200

fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
sp = getattr(termios, "B%d" % baud)
a  = termios.tcgetattr(fd)
# raw 8N1, no flow control, no echo, blocking-read semantics via select
a[0] = 0                      # iflag
a[1] = 0                      # oflag
a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
a[3] = 0                      # lflag
a[4] = a[5] = sp              # ispeed/ospeed
a[6] = list(a[6])
a[6][termios.VMIN]  = 0
a[6][termios.VTIME] = 0
termios.tcsetattr(fd, termios.TCSANOW, a)
termios.tcflush(fd, termios.TCIFLUSH)

import select
end = time.time() + secs
with open(log, "wb", buffering=0) as f:
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.5)
        if not r:
            continue
        try:
            data = os.read(fd, 4096)
        except BlockingIOError:
            continue
        if data:
            f.write(data)
os.close(fd)
