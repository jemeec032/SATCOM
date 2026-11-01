"""
Quick validation script for the ported IQ imbalance + modulation functions.
Run this and compare the printed radii against your MATLAB disp() output
for the same M, ampImb_dB, and phaseImb_deg.
"""

import numpy as np
from impairments.modulation import bits_to_psk_symbols
from impairments.iq_imbalance import iq_imbalance

# ---- Parameters (match your MATLAB script) ----
M = 16
ampImb_dB = 0.5
phaseImb_deg = 3

np.random.seed(0)  # fixed seed so results are repeatable

# ---- Generate bits (small number just for this test) ----
numBits = 1000
bits = np.random.randint(0, 2, numBits)

# ---- Modulate ----
symbols_ideal = bits_to_psk_symbols(bits, M)

# ---- Apply IQ imbalance ----
symbols = iq_imbalance(symbols_ideal, ampImb_dB, phaseImb_deg)

# ---- Print first 10 radii, same style as your MATLAB disp() ----
print("Original Radius")
print(np.abs(symbols_ideal[:10]))

print("\nIQ Imbalance Radius")
print(np.abs(symbols[:10]))
