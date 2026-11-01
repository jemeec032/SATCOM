"""
Validates phase_noise(), awgn_channel(), and carrier_phase_recovery(),
chained after all previous impairments, same order as the MATLAB script.
Compare against your MATLAB disp()/subplot outputs at the same SNR.
"""

import numpy as np
from impairments.modulation import bits_to_psk_symbols
from impairments.iq_imbalance import iq_imbalance
from impairments.group_delay import group_delay
from impairments.saleh_nonlinearity import saleh_amam, saleh_ampm
from impairments.phase_noise_awgn import phase_noise, awgn_channel, carrier_phase_recovery

# ---- Parameters (match your MATLAB script) ----
M = 16
ampImb_dB = 0.5
phaseImb_deg = 3
GDmax_ns = 300
sigma_deg = 3
SNR_dB = 13

np.random.seed(0)

numBits = 1000
bits = np.random.randint(0, 2, numBits)

symbols_ideal = bits_to_psk_symbols(bits, M)
symbols = iq_imbalance(symbols_ideal, ampImb_dB, phaseImb_deg)
gd_symbols = group_delay(symbols, GDmax_ns * 1e-9)
amam_symbols = saleh_amam(gd_symbols)
ampm_symbols = saleh_ampm(amam_symbols)
pn_symbols = phase_noise(ampm_symbols, sigma_deg)
rx_symbols = awgn_channel(pn_symbols, SNR_dB)
rx_symbols = carrier_phase_recovery(rx_symbols, symbols)  # reference = IQ-imbalanced tx

print("AMPM Radius")
print(np.abs(ampm_symbols[:10]))

print("\nAfter Phase Noise Radius")
print(np.abs(pn_symbols[:10]))

print("\nAfter AWGN + Recovery Radius")
print(np.abs(rx_symbols[:10]))
