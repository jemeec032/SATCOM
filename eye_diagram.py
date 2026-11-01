"""
Eye Diagram generation.

Symbol samples alone don't show an eye pattern -- you need a
pulse-shaped, oversampled waveform for that (this is what a real
transmitter sends: not instantaneous symbol values, but a continuous
waveform built from pulses). This module implements a standard
root-raised-cosine (RRC) pulse shape from scratch (no extra
dependency), oversamples the symbol stream, filters it, and then
slices the result into overlapping symbol-period segments so they can
be plotted on top of each other -- the classic "eye" shape.

Group delay is the clearest impairment to see here: it spreads energy
between adjacent symbols (inter-symbol interference), which closes the
eye -- much more intuitive than a constellation for showing ISI.
"""

import numpy as np


def rrc_filter_taps(beta: float, span_symbols: int, sps: int) -> np.ndarray:
    """
    Design a root-raised-cosine filter's time-domain taps.

    Parameters
    ----------
    beta : float
        Roll-off factor (0 to 1).
    span_symbols : int
        Filter length in symbol periods (total length = span_symbols * sps + 1).
    sps : int
        Samples per symbol (oversampling factor).

    Returns
    -------
    np.ndarray
        Filter taps, unit energy normalized.
    """
    N = span_symbols * sps
    t = (np.arange(-N / 2, N / 2 + 1)) / sps  # time in symbol periods
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            numerator = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = numerator / denominator

    h = h / np.sqrt(np.sum(h ** 2))  # normalize energy
    return h


def pulse_shape(symbols: np.ndarray, sps: int = 8, beta: float = 0.35, span: int = 8) -> np.ndarray:
    """
    Upsample and pulse-shape a symbol stream with an RRC filter.

    Parameters
    ----------
    symbols : np.ndarray (complex)
    sps : int
        Samples per symbol.
    beta : float
        RRC roll-off factor.
    span : int
        Filter span in symbols.

    Returns
    -------
    np.ndarray (complex)
        Oversampled, pulse-shaped waveform.
    """
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols

    taps = rrc_filter_taps(beta, span, sps)
    waveform = np.convolve(upsampled, taps, mode="same")
    return waveform


def eye_traces(waveform: np.ndarray, sps: int, num_traces: int = 150):
    """
    Slice a waveform into overlapping 2-symbol-period segments for an
    eye diagram plot.

    Parameters
    ----------
    waveform : np.ndarray (complex)
    sps : int
        Samples per symbol.
    num_traces : int
        Max number of traces to return (for plotting speed).

    Returns
    -------
    np.ndarray, shape (n_traces, 2*sps)
        Each row is one "eye" trace of the REAL part (commonly plotted
        for eye diagrams -- the imaginary part looks the same by symmetry
        for PSK, so real part alone conveys the eye clearly).
    """
    seg_len = 2 * sps
    n_segments = (len(waveform) - seg_len) // sps
    n_segments = min(n_segments, num_traces)

    traces = np.zeros((n_segments, seg_len))
    for i in range(n_segments):
        start = i * sps
        traces[i] = waveform.real[start : start + seg_len]

    return traces
