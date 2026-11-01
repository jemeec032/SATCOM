"""
Costas Loop (digital PLL) carrier recovery for M-PSK.

The simple carrier_phase_recovery() in phase_noise_awgn.py applies ONE
average phase correction to the whole block -- fine for a constant
offset, but it can't track a continuously drifting phase (like
Doppler), because the correction doesn't change over time.

A Costas loop tracks and corrects the phase SYMBOL BY SYMBOL using
feedback, which is what real receivers do to handle exactly this kind
of drift. This is a standard decision-directed M-th power Costas loop:
raising each symbol to the M-th power strips off the M-PSK modulation,
leaving just (M times) the residual phase error to feed the loop.
"""

import numpy as np


def costas_loop_recovery(rx_symbols: np.ndarray, M: int, loop_bandwidth: float = 0.02, damping: float = 0.707) -> np.ndarray:
    """
    Track and correct phase (including a drifting/Doppler-like phase
    ramp) using a 2nd-order digital Costas loop (M-th power method).

    NOTE: the M-th power method works well for M=4 and M=8, but suffers
    badly from noise enhancement at M=16 and above (raising a noisy
    symbol to the 16th power amplifies that noise enormously). For
    M >= 16, use decision_directed_costas() instead.

    Parameters
    ----------
    rx_symbols : np.ndarray (complex)
    M : int
        PSK modulation order (the loop uses the M-th power trick to
        strip modulation before measuring phase error).
    loop_bandwidth : float
        Normalized loop bandwidth (larger = tracks faster but noisier).
    damping : float
        Loop damping factor (0.707 is a common critically-damped choice).

    Returns
    -------
    np.ndarray (complex)
        Phase-corrected symbols.
    """
    N = len(rx_symbols)

    theta = loop_bandwidth
    denom = 1 + 2 * damping * theta + theta ** 2
    Kp = (4 * damping * theta) / denom
    Ki = (4 * theta ** 2) / denom

    phase_est = 0.0
    freq_est = 0.0
    corrected = np.zeros(N, dtype=complex)

    for n in range(N):
        corrected[n] = rx_symbols[n] * np.exp(-1j * phase_est)
        error = np.angle(corrected[n] ** M) / M
        freq_est += Ki * error
        phase_est += freq_est + Kp * error

    return corrected


def decision_directed_costas(rx_symbols: np.ndarray, M: int, loop_bandwidth: float = 0.02, damping: float = 0.707) -> np.ndarray:
    """
    Track and correct phase using a decision-directed Costas loop: instead
    of the M-th power trick, it makes a hard decision on the nearest
    PSK constellation point each symbol and uses the angle to that
    decision as the phase error. This avoids the M-th power noise
    enhancement, so it scales better to higher M (8, 16-PSK).

    LIMITATION: decision-directed loops need decisions to be mostly
    correct to work at all -- at very dense constellations (32-PSK) or
    very low SNR, early wrong decisions can prevent the loop from ever
    locking. Real systems solve this with a known training preamble;
    this simplified version has no preamble, so treat very dense
    constellations' results with that caveat in mind.

    Parameters
    ----------
    rx_symbols : np.ndarray (complex)
    M : int
    loop_bandwidth : float
    damping : float

    Returns
    -------
    np.ndarray (complex)
    """
    N = len(rx_symbols)
    step = 2 * np.pi / M

    theta = loop_bandwidth
    denom = 1 + 2 * damping * theta + theta ** 2
    Kp = (4 * damping * theta) / denom
    Ki = (4 * theta ** 2) / denom

    phase_est = 0.0
    freq_est = 0.0
    corrected = np.zeros(N, dtype=complex)

    for n in range(N):
        corrected[n] = rx_symbols[n] * np.exp(-1j * phase_est)
        decided_angle = np.round(np.angle(corrected[n]) / step) * step
        decision = np.exp(1j * decided_angle)
        error = np.angle(corrected[n] * np.conj(decision))
        freq_est += Ki * error
        phase_est += freq_est + Kp * error

    return corrected
