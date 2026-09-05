#!/usr/bin/env python3
"""Build the baremetal A53 platform and application for step 13.

2026.1 removed xsct, so this uses the Vitis Python API:
    vitis -s scripts/build_vitis_attn.py

`sw/attn_vectors.c` is GENERATED and is not in the repository -- it is 6 MB of
numbers and it is reproducible from a seed in seven seconds. This script
regenerates it if it is missing, so a fresh checkout cannot build an ELF whose
goldens are older than the kernel that is supposed to produce them.
"""
import os
import subprocess
import shutil
import sys

import vitis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREQ = os.environ.get("ATTN_FREQ", "250")
VAR = os.environ.get("ATTN_VARIANT", "attndma")
XSA = os.path.join(ROOT, "build", f"{VAR}_f{FREQ}.xsa")
WS = os.path.join(ROOT, "build", f"vitis_ws_{VAR}_f{FREQ}")
PLAT = f"attn_plat_f{FREQ}"
APP = f"attn_test_f{FREQ}"
CPU = "psu_cortexa53_0"
DOMAIN = f"standalone_{CPU}"

if not os.path.isfile(XSA):
    sys.exit(f"XSA not found: {XSA}\n"
             f"  build it with: vivado -mode batch -source scripts/build_attn_bit.tcl "
             f"-tclargs 8 {FREQ} {VAR}")

# The goldens, if they are not already here. `python -m kernel.hw.board_vectors`
# is the only thing that may write these two files.
vec = os.path.join(ROOT, "sw", "attn_vectors.c")
if not os.path.isfile(vec):
    print("### regenerating sw/attn_vectors.c")
    subprocess.run([sys.executable, "-m", "kernel.hw.board_vectors",
                    "--out", os.path.join(ROOT, "sw")],
                   cwd=os.path.dirname(ROOT), check=True)

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
app.import_files(from_loc=os.path.join(ROOT, "sw"),
                 files=["attn_main.c", "attn_vectors.c", "attn_vectors.h"])
app.build()

elf = None
for r, _, fs in os.walk(os.path.join(WS, APP)):
    for f in fs:
        if f.endswith(".elf"):
            elf = os.path.join(r, f)
print(f"### ELF: {elf}")
if elf:
    print(f"### ELF size: {os.path.getsize(elf)} B")
print("### VITIS OK")
vitis.dispose()
