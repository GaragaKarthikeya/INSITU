# Standalone timing contract for Step 5.  The integrated design will derive
# this 250 MHz clock from the ZCU104 clock wizard; OOC synthesis needs the
# equivalent constraint explicitly.
create_clock -name clk_250mhz -period 4.000 [get_ports clk]
