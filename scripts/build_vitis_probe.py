#!/usr/bin/env python3
"""
Build the baremetal Vitis platform + application for the ZCU104 DDR probe.

2026.1 removed xsct, so this uses the Vitis Python API:
    vitis -s scripts/build_vitis_probe.py

PL_CLK_HZ is passed to the compiler rather than hardcoded in the C, because the
PS PLL rounds: ask for 200 MHz and you get 187.5. Reporting GB/s against the
requested clock when the design is running at another is how a probe lies. Take
the number create_bd_probe.tcl printed as "actual".
"""
import os, shutil, sys, vitis

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREQ   = int(os.environ.get("PROBE_FREQ", "250"))
CLK_HZ = int(os.environ.get("PL_CLK_HZ", str(FREQ * 1000000)))
XSA    = os.path.join(ROOT, "build", f"ddr_probe_f{FREQ}.xsa")
WS     = os.path.join(ROOT, "build", f"vitis_ws_probe_f{FREQ}")
PLAT   = f"probe_plat_f{FREQ}"
APP    = f"probe_app_f{FREQ}"
CPU    = "psu_cortexa53_0"
DOMAIN = f"standalone_{CPU}"

if not os.path.isfile(XSA):
    sys.exit(f"XSA not found: {XSA} -- run scripts/build_probe_bit.tcl first")
if os.path.isdir(WS):
    shutil.rmtree(WS)
os.makedirs(WS, exist_ok=True)

client = vitis.create_client()
client.set_workspace(WS)

print(f"### creating platform from {XSA}")
plat = client.create_platform_component(
    name=PLAT, hw_design=XSA, os="standalone", cpu=CPU, domain_name=DOMAIN)
plat.build()
xpfm = client.find_platform_in_repos(PLAT)
print(f"### platform: {xpfm}")

print("### creating application")
client.create_app_component(name=APP, platform=xpfm, domain=DOMAIN)
app = client.get_component(name=APP)
app.import_files(from_loc=os.path.join(ROOT, "sw"), files=["probe_main.c"])

# The control block's address comes from the BD address map, which
# create_bd_probe.tcl prints. 0xA0000000 is what it produced; overriding here
# rather than in the C keeps one source of truth.
base = os.environ.get("PROBE_BASE", "0xA0000000")
for cfg in ("Release", "Debug"):
    try:
        app.set_app_config(key="USER_COMPILE_FLAGS",
                           values=f"-DPL_CLK_HZ={CLK_HZ}u -DPROBE_BASE={base}u -O2")
        break
    except Exception as e:      # API name varies across 2026.x point releases
        print(f"### set_app_config({cfg}) -> {e}")

app.build()

elf = None
for r, _, fs in os.walk(os.path.join(WS, APP)):
    for f in fs:
        if f.endswith(".elf"):
            elf = os.path.join(r, f)
print(f"### ELF: {elf}")
print(f"### PL_CLK_HZ={CLK_HZ} PROBE_BASE={base}")
print("### VITIS OK")
vitis.dispose()
