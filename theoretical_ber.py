"""
Theoretical BER for M-PSK over an AWGN-only channel (Gray-coded,
standard approximation). Used to validate that the simulator's
clean-link (no impairments) results match known communications theory
-- the single most convincing sanity check for a simulator like this.

    BER =~ (2/k) * Q( sqrt(2*Es/N0) * sin(pi/M) )     for M >= 4
    BER  = Q( sqrt(2*Es/N0) )                          for M = 2 (exact)

where k = log2(M), Es/N0 is the linear symbol SNR (matches how this
simulator defines its snr_db, since symbol energy is normalized to 1
for PSK), and Q(.) is the Gaussian Q-function.

This formula assumes Gray coding and is a standard approximation,
accurate at moderate-to-high SNR for M >= 4 (the union bound / nearest-
neighbor approximation -- it slightly overestimates true BER at very
low SNR, which is normal and expected).
"""

import numpy as np
from scipy.stats import norm


def theoretical_ber_mpsk(M: int, snr_db):
    """
    Theoretical BER for M-PSK, Gray-coded, AWGN-only channel.

    Parameters
    ----------
    M : int
        Modulation order.
    snr_db : float or array-like
        Symbol SNR (Es/N0) in dB.

    Returns
    -------
    np.ndarray or float
        Theoretical BER, same shape as snr_db.
    """
    snr_db_arr = np.atleast_1d(np.asarray(snr_db, dtype=float))
    es_n0 = 10 ** (snr_db_arr / 10)
    k = np.log2(M)

    if M == 2:
        ber = norm.sf(np.sqrt(2 * es_n0))  # exact BPSK BER
    else:
        q = norm.sf(np.sqrt(2 * es_n0) * np.sin(np.pi / M))
        ber = (2 / k) * q

    return ber if ber.size > 1 else float(ber[0])
