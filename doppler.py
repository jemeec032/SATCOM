"""
Doppler / Frequency Offset impairment.

Models a satellite-relative-motion frequency offset the way a
communications textbook writes it -- as a continuous-time phase
rotation of the carrier:

    Ts = 1 / symbol_rate_hz                  (symbol period)
    t  = n * Ts                              (time axis, n = 0, 1, 2, ...)
    phase(t) = phi0 + 2*pi * fd * t
    symbols_out[n] = symbols_in[n] * exp(1j * phase(t))

where fd is the Doppler frequency in Hz. Internally this is expressed
via freq_offset_norm = fd / symbol_rate_hz (cycles/symbol), which is
mathematically identical to fd*t at each symbol instant -- but building
the phase explicitly off an actual time axis (rather than a bare
sample index) keeps the model traceable to the physical Doppler
frequency and symbol rate, matching how it's usually written in a
communications text.

Two additions beyond the deterministic ramp, both physically motivated:

1. Random initial carrier phase (phi0 ~ Uniform(0, 2*pi)), on by
   default. A real receiver never starts phase-aligned with the
   transmitter; every run should look slightly different, the way a
   real acquisition would. Pass random_initial_phase=False (e.g. for a
   reproducible regression test) to force phi0 = 0.

2. Optional residual random-walk phase (a small Wiener process) on top
   of the deterministic Doppler ramp, via residual_walk_std_deg. This
   approximates what's left over after a PLL/Costas loop has tracked
   out most of the Doppler but a small residual + oscillator phase
   instability remains -- i.e. combined residual-Doppler-after-tracking
   + phase noise in one term. Off by default (0.0 = pure deterministic
   ramp); set a small value (e.g. 0.05-0.5 degrees per symbol) to
   enable it.

IMPORTANT -- units guard:
freq_offset_norm (cycles/symbol) must stay small (realistically ~1e-6
to ~1e-3). It is NOT the same kind of quantity as Group Delay's
gd_max_ns (which is legitimately swept over 0-300). If freq_offset_norm
is set to something like 50 or 300, the phase ramp wraps around dozens
to hundreds of full rotations *every single symbol* -- that's not
"severe Doppler", it's equivalent to replacing every symbol with an
independent uniformly-random phase, and the signal is destroyed
outright at every SNR. This module raises a clear error if
freq_offset_norm is passed a value outside a sane range.
"""
import warnings

import numpy as np

# Sane operating range for freq_offset_norm (cycles/symbol).
MAX_SANE_FREQ_OFFSET_NORM = 0.01

# Default symbol rate used to translate freq_offset_norm <-> an actual
# Doppler frequency in Hz for display/reporting purposes. Doesn't
# affect the resulting phase ramp (freq_offset_norm already fully
# determines it) -- only changes what fd_hz is reported as.
DEFAULT_SYMBOL_RATE_HZ = 1e6


def doppler_shift(
    symbols: np.ndarray,
    freq_offset_norm: float,
    symbol_rate_hz: float = DEFAULT_SYMBOL_RATE_HZ,
    random_initial_phase: bool = True,
    residual_walk_std_deg: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply a Doppler / frequency-offset phase rotation, built off an
    explicit time axis (t = n * Ts), plus an optional random initial
    phase and optional residual random-walk phase.

    Parameters
    ----------
    symbols : np.ndarray (complex)
    freq_offset_norm : float
        Frequency offset normalized to symbol rate (cycles/symbol).
        Must be small -- see module docstring. Values above 0.01 raise
        ValueError (near-certain units mistake).
    symbol_rate_hz : float, optional
        Symbol rate in Hz, used only to build the time axis t = n*Ts
        and to derive the corresponding Doppler frequency
        fd_hz = freq_offset_norm * symbol_rate_hz for reporting. Does
        not change the resulting phase ramp itself.
    random_initial_phase : bool, optional (default True)
        If True, adds a random phi0 ~ Uniform(0, 2*pi) on top of the
        deterministic ramp, so every run starts from a different carrier
        phase -- like a real receiver would see. Set False for a fully
        deterministic/reproducible run (e.g. regression tests).
    residual_walk_std_deg : float, optional (default 0.0)
        If > 0, adds a cumulative random-walk (Wiener process) phase
        component on top of the deterministic ramp, with each symbol's
        increment drawn from N(0, residual_walk_std_deg degrees).
        Models residual Doppler left after PLL tracking combined with
        oscillator phase instability. 0.0 (default) disables it -- pure
        deterministic ramp (+ optional random phi0).
    rng : np.random.Generator, optional
        Random generator to use for phi0 / the walk. Defaults to
        np.random's global generator if not given.

    Returns
    -------
    np.ndarray (complex)
    """
    if abs(freq_offset_norm) > MAX_SANE_FREQ_OFFSET_NORM:
        raise ValueError(
            f"freq_offset_norm={freq_offset_norm} is way outside the sane range "
            f"(+/-{MAX_SANE_FREQ_OFFSET_NORM} cycles/symbol). This almost always means "
            f"a units mix-up -- freq_offset_norm is cycles PER SYMBOL, not a "
            f"Group-Delay-style value in ns. Realistic values are typically "
            f"1e-6 to 1e-3. If you really intend this, raise "
            f"MAX_SANE_FREQ_OFFSET_NORM in impairments/doppler.py."
        )
    if abs(freq_offset_norm) > 0.001:
        warnings.warn(
            f"freq_offset_norm={freq_offset_norm} is unusually large for "
            f"cycles/symbol -- double check this is intentional.",
            stacklevel=2,
        )

    _rng = rng if rng is not None else np.random

    N = len(symbols)
    Ts = 1.0 / symbol_rate_hz
    t = np.arange(N) * Ts
    fd_hz = freq_offset_norm * symbol_rate_hz  # actual Doppler frequency, for reference

    phi0 = _rng.uniform(0, 2 * np.pi) if random_initial_phase else 0.0
    phase = phi0 + 2 * np.pi * fd_hz * t

    if residual_walk_std_deg > 0:
        step_std_rad = np.deg2rad(residual_walk_std_deg)
        walk = np.cumsum(_rng.normal(0.0, step_std_rad, size=N))
        phase = phase + walk

    return symbols * np.exp(1j * phase)
