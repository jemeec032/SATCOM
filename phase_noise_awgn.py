"""
Phase Noise, AWGN Channel, and Carrier Phase Recovery.

Ported from MATLAB:

    % Phase Noise
    sigma = deg2rad(sigma_deg)
    phaseNoise = sigma * randn(N,1)
    pnSymbols = ampmSymbols .* exp(1j*phaseNoise)

    % AWGN
    signalPower = mean(abs(pnSymbols).^2)
    noisePower = signalPower/(10^(SNR/10))
    noise = sqrt(noisePower/2).*(randn(N,1) + 1j*randn(N,1))
    rxSymbols = pnSymbols + noise

    % Carrier Phase Recovery
    meanPhaseError = angle(mean(rxSymbols .* conj(symbols)))
    rxSymbols = rxSymbols .* exp(-1j*meanPhaseError)
"""

import numpy as np


def phase_noise(symbols: np.ndarray, sigma_deg: float) -> np.ndarray:
    """
    Apply random phase noise (Gaussian, per-symbol) to a complex symbol array.

    Parameters
    ----------
    symbols : np.ndarray (complex)
    sigma_deg : float
        Standard deviation of the phase noise, in degrees.
    """
    sigma = np.deg2rad(sigma_deg)
    noise_angles = sigma * np.random.randn(len(symbols))
    return symbols * np.exp(1j * noise_angles)


def awgn_channel(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add complex AWGN noise at the specified SNR (dB), scaled to the
    signal's own average power (matches MATLAB's signalPower calculation).
    """
    signal_power = np.mean(np.abs(symbols) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )
    return symbols + noise


def carrier_phase_recovery(rx_symbols: np.ndarray, tx_reference: np.ndarray) -> np.ndarray:
    """
    Remove the common (average) phase rotation between rx_symbols and a
    reference (the IQ-imbalanced but otherwise undistorted tx symbols),
    so only the impairment-induced variation remains.

    Parameters
    ----------
    rx_symbols : np.ndarray (complex)
        Symbols after all channel impairments.
    tx_reference : np.ndarray (complex)
        The transmitted symbols to compare against (typically the
        IQ-imbalanced symbols, matching the MATLAB script's `symbols`).
    """
    mean_phase_error = np.angle(np.mean(rx_symbols * np.conj(tx_reference)))
    return rx_symbols * np.exp(-1j * mean_phase_error)
