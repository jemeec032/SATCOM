"""
Spectrum analysis for the symbol stream at any stage.

Uses scipy.signal.welch on the complex baseband symbol sequence, which
is a standard way to visualize:
    - Spectral regrowth from AM-AM (the spectrum widens/grows shoulders
      when the amplifier compresses peaks non-linearly)
    - Frequency shift from Doppler (the whole spectrum shifts sideways)
    - Noise floor from AWGN (the spectrum floor rises)
    - Phase noise sidebands (skirts appear around the carrier)

Note: since this operates on the symbol-rate sequence (not an
oversampled/pulse-shaped waveform), it shows the *baseband symbol
spectrum* -- a good, standard way to see these effects without needing
a full pulse-shaping filter chain.
"""

import numpy as np
from scipy import signal


def compute_psd(symbols: np.ndarray, fs: float = 1.0, nperseg: int = None):
    """
    Compute the power spectral density of a complex symbol sequence.

    Parameters
    ----------
    symbols : np.ndarray (complex)
    fs : float
        Sample rate to assign (symbol rate, normalized to 1.0 by default).
    nperseg : int, optional
        Segment length for Welch's method. Defaults to min(256, len(symbols)).

    Returns
    -------
    freqs : np.ndarray
        Frequency axis (same units as fs), centered at 0.
    psd_db : np.ndarray
        Power spectral density in dB.
    """
    if nperseg is None:
        nperseg = min(256, len(symbols))

    freqs, psd = signal.welch(
        symbols, fs=fs, nperseg=nperseg, return_onesided=False, scaling="density"
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    psd_db = 10 * np.log10(psd + 1e-16)
    return freqs, psd_db
