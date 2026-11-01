"""
Validates saleh_amam() / saleh_ampm(), chained after iq_imbalance() and
group_delay(), same order as the MATLAB script. Compare "AMAM Radius"
and "AMPM Radius" against your MATLAB disp() output.
"""

import numpy as np
from impairments.modulation import bits_to_psk_symbols
from impairments.iq_imbalance import iq_imbalance
from impairments.group_delay import group_delay
from impairments.saleh_nonlinearity import saleh_amam, saleh_ampm

# ---- Parameters (match your MATLAB script) ----
M = 16
ampImb_dB = 0.5
phaseImb_deg = 3
GDmax_ns = 300

np.random.seed(0)

numBits = 1000
bits = np.random.randint(0, 2, numBits)

symbols_ideal = bits_to_psk_symbols(bits, M)
symbols = iq_imbalance(symbols_ideal, ampImb_dB, phaseImb_deg)
gd_symbols = group_delay(symbols, GDmax_ns * 1e-9)
amam_symbols = saleh_amam(gd_symbols)
ampm_symbols = saleh_ampm(amam_symbols)

print("GD Radius")
print(np.abs(gd_symbols[:10]))

print("\nAMAM Radius")
print(np.abs(amam_symbols[:10]))

print("\nAMPM Radius")
print(np.abs(ampm_symbols[:10]))
