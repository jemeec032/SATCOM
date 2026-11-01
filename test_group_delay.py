"""
Validates group_delay() by chaining it after iq_imbalance(), same order
as the MATLAB script. Compare "GD Radius" output against your MATLAB
disp() at GD = 300ns (the m==7 case in your GD_values_ns list).
"""

import numpy as np
from impairments.modulation import bits_to_psk_symbols
from impairments.iq_imbalance import iq_imbalance
from impairments.group_delay import group_delay

# ---- Parameters (match your MATLAB script) ----
M = 16
ampImb_dB = 0.5
phaseImb_deg = 3
GDmax_ns = 300  # worst case, matches MATLAB's m==7 snapshot condition

np.random.seed(0)

numBits = 1000
bits = np.random.randint(0, 2, numBits)

symbols_ideal = bits_to_psk_symbols(bits, M)
symbols = iq_imbalance(symbols_ideal, ampImb_dB, phaseImb_deg)
gd_symbols = group_delay(symbols, GDmax_ns * 1e-9)

print("IQ Imbalance Radius")
print(np.abs(symbols[:10]))

print("\nGD Radius")
print(np.abs(gd_symbols[:10]))
