"""
Group Delay impairment: LINEAR and PARABOLIC variants.

Linear GD (original, ported from MATLAB):
    N = length(symbols)
    f = linspace(-1,1,N).'
    tau = (GD_scaleFactor*GDmax) .* f
    H = exp(-1j*2*pi*f.*tau)
    X = fftshift(fft(symbols))
    Y = X .* H
    gdSymbols = ifft(ifftshift(Y))

Parabolic GD (new):
    Same idea, but the delay-vs-frequency curve is quadratic instead of
    linear:
        tau(f) = (GD_scaleFactor*GDmax) .* f.^2
    Physically this models a channel/filter whose group delay is roughly
    flat near band-center and grows (symmetrically) toward the band
    edges -- a very common real-world impairment shape (e.g. a filter's
    passband-edge GD bump), as opposed to linear GD's edge-to-edge tilt.
    Because tau(f) is even in f, the resulting phase term H is *not*
    antisymmetric the way linear GD's is -- it distorts symbols
    symmetrically around band-center instead of "tilting" the band.

Both share the same GD_SCALE_FACTOR calibration (GDmax = 300ns -> 45
degree phase twist at the band edge, f = +-1), so a given gd_max_ns
value is comparably "severe" whichever type you pick.
"""
import numpy as np

# Calibrated so that GDmax = 300ns produces a 45 degree twist at f = +-1
# (same calibration as the MATLAB script; reused for both GD types)
TARGET_PHASE_DEG = 45
TARGET_PHASE_RAD = np.deg2rad(TARGET_PHASE_DEG)
GD_MAX_WORST = 300e-9
GD_SCALE_FACTOR = TARGET_PHASE_RAD / (2 * np.pi * GD_MAX_WORST)

# Valid values for the `gd_type` parameter used throughout this module
# and by anything downstream (core.py, equalizer.py, sweep.py, app.py).
GD_TYPES = ("linear", "parabolic")

# Human-friendly labels, handy for UI dropdowns / plot titles.
GD_TYPE_LABELS = {
    "linear": "Linear",
    "parabolic": "Parabolic",
}


def _validate_gd_type(gd_type: str) -> str:
    gd_type = (gd_type or "linear").lower()
    if gd_type not in GD_TYPES:
        raise ValueError(f"gd_type must be one of {GD_TYPES}, got {gd_type!r}")
    return gd_type


def compute_gd_response(N: int, gd_max_seconds: float, gd_type: str = "linear") -> np.ndarray:
    """
    Compute the frequency-domain response H used by group_delay(), so
    it can be reused by an equalizer that needs to know the channel
    it's trying to invert.

    Parameters
    ----------
    N : int
        Number of symbols (length of the frequency axis).
    gd_max_seconds : float
        Maximum group delay in seconds (e.g. 100e-9 for 100 ns).
    gd_type : str
        'linear' (default) or 'parabolic'. Selects whether tau(f) is
        proportional to f or f**2.

    Returns
    -------
    np.ndarray (complex)
        Frequency-domain response H(f).
    """
    gd_type = _validate_gd_type(gd_type)
    f = np.linspace(-1, 1, N)
    if gd_type == "parabolic":
        tau = (GD_SCALE_FACTOR * gd_max_seconds) * f ** 2
    else:
        tau = (GD_SCALE_FACTOR * gd_max_seconds) * f
    return np.exp(-1j * 2 * np.pi * f * tau)


def group_delay(symbols: np.ndarray, gd_max_seconds: float, gd_type: str = "linear") -> np.ndarray:
    """
    Apply a frequency-dependent group delay to a complex symbol array.

    Parameters
    ----------
    symbols : np.ndarray (complex)
    gd_max_seconds : float
        Maximum group delay in seconds (e.g. 100e-9 for 100 ns).
    gd_type : str
        'linear' (default) -- tau(f) proportional to f (band tilt).
        'parabolic' -- tau(f) proportional to f**2 (symmetric band-edge
        bump).

    Returns
    -------
    np.ndarray (complex)
        Symbols after group delay distortion.
    """
    N = len(symbols)
    H = compute_gd_response(N, gd_max_seconds, gd_type)
    X = np.fft.fftshift(np.fft.fft(symbols))
    Y = X * H
    gd_symbols = np.fft.ifft(np.fft.ifftshift(Y))
    return gd_symbols
