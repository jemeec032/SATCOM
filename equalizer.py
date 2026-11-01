"""
Zero-Forcing (ZF) and MMSE equalizers.

IMPORTANT SCOPE NOTE: these equalizers invert the known GROUP DELAY
channel response only (linear OR parabolic -- whichever gd_type was
actually applied). That's the impairment an equalizer is actually
designed for -- it's a linear, frequency-selective distortion, exactly
the kind of thing equalization corrects in real systems (this is why
equalizers exist: to undo multipath/ISI-like channel responses).

They do NOT fix:
    - AM-AM / AM-PM (nonlinear -- needs predistortion, not a linear filter)
    - IQ imbalance (a transmitter-side impairment with different structure)
    - Phase noise (random, not a fixed filter response)
    - Doppler (needs a tracking loop, not a static filter -- see carrier_recovery.py)

Comparing BER before/after equalization is most meaningful when Group
Delay is the (or a major) enabled impairment.
"""

import numpy as np

from impairments.group_delay import compute_gd_response


def zero_forcing_equalize(rx_symbols: np.ndarray, gd_max_seconds: float, gd_type: str = "linear") -> np.ndarray:
    """
    Invert the known group-delay channel response exactly (assuming it's
    the true channel). Amplifies noise at frequencies where |H| is small
    -- classic ZF weakness.

    Parameters
    ----------
    rx_symbols : np.ndarray (complex)
    gd_max_seconds : float
        The group delay parameter that was applied (must match, since
        ZF needs to know the channel it's inverting).
    gd_type : str
        'linear' (default) or 'parabolic' -- must match whichever GD
        model was actually applied to these symbols, or the inversion
        targets the wrong channel entirely.

    Returns
    -------
    np.ndarray (complex)
    """
    N = len(rx_symbols)
    H = compute_gd_response(N, gd_max_seconds, gd_type)

    X = np.fft.fftshift(np.fft.fft(rx_symbols))
    Y = X / H
    return np.fft.ifft(np.fft.ifftshift(Y))


def mmse_equalize(rx_symbols: np.ndarray, gd_max_seconds: float, snr_linear: float, gd_type: str = "linear") -> np.ndarray:
    """
    MMSE (Wiener) equalizer for the same known group-delay channel.
    Balances inverting the channel against not amplifying noise too much
    -- generally outperforms ZF at low/moderate SNR.

    Parameters
    ----------
    rx_symbols : np.ndarray (complex)
    gd_max_seconds : float
    snr_linear : float
        Linear (not dB) SNR estimate, used to set the noise-vs-inversion
        tradeoff.
    gd_type : str
        'linear' (default) or 'parabolic' -- must match whichever GD
        model was actually applied.

    Returns
    -------
    np.ndarray (complex)
    """
    N = len(rx_symbols)
    H = compute_gd_response(N, gd_max_seconds, gd_type)

    X = np.fft.fftshift(np.fft.fft(rx_symbols))
    W = np.conj(H) / (np.abs(H) ** 2 + 1.0 / snr_linear)
    Y = X * W
    return np.fft.ifft(np.fft.ifftshift(Y))
