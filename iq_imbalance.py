"""
IQ Imbalance impairment.

Ported from MATLAB:
    g   = 10^(ampImb_dB/20)
    phi_iq = deg2rad(phaseImb_deg)
    I_out = I_in
    Q_out = g*(Q_in*cos(phi_iq) + I_in*sin(phi_iq))
    symbols = I_out + 1j*Q_out
"""

import numpy as np


def iq_imbalance(symbols: np.ndarray, amp_imb_db: float, phase_imb_deg: float) -> np.ndarray:
    """
    Apply transmitter IQ imbalance to a complex symbol array.

    Parameters
    ----------
    symbols : np.ndarray (complex)
        Ideal modulated symbols (e.g. output of PSK modulation).
    amp_imb_db : float
        Gain mismatch between I and Q paths, in dB.
    phase_imb_deg : float
        Phase error away from ideal 90 degrees, in degrees.

    Returns
    -------
    np.ndarray (complex)
        Symbols after IQ imbalance.
    """
    g = 10 ** (amp_imb_db / 20)
    phi_iq = np.deg2rad(phase_imb_deg)

    i_in = np.real(symbols)
    q_in = np.imag(symbols)

    i_out = i_in
    q_out = g * (q_in * np.cos(phi_iq) + i_in * np.sin(phi_iq))

    return i_out + 1j * q_out
