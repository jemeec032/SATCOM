"""
Sanity checks for run_simulation():
  1. No impairments, high SNR -> BER should be ~0 (near-error-free link).
  2. All impairments, moderate SNR -> BER should be clearly higher than (1).
"""

import numpy as np
from impairments.core import run_simulation

np.random.seed(0)

M = 16
numBits = 100_000
bits = np.random.randint(0, 2, numBits)

# ---- Case 1: clean link, high SNR ----
result_clean = run_simulation(bits, M, snr_db=25, impairments={})
print(f"Clean link, SNR=25dB  -> BER={result_clean['ber']:.6f}  EVM={result_clean['evm']:.6f}")

# ---- Case 2: all impairments, moderate SNR ----
impairments_all = {
    "iq_imbalance": {"amp_imb_db": 0.5, "phase_imb_deg": 3},
    "group_delay": {"gd_max_ns": 300},
    "saleh": True,
    "phase_noise": {"sigma_deg": 3},
}
result_impaired = run_simulation(bits, M, snr_db=13, impairments=impairments_all)
print(f"All impairments, SNR=13dB -> BER={result_impaired['ber']:.6f}  EVM={result_impaired['evm']:.6f}")

assert result_impaired["ber"] > result_clean["ber"], "Impaired link should have higher BER than clean link!"
print("\nSanity check passed: impaired BER > clean BER, as expected.")
