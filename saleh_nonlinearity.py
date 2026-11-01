"""
Saleh AM-AM / AM-PM amplifier nonlinearity.

Ported from MATLAB:
    % AM-AM
    A = (alpha_a * r) / (1 + beta_a * r^2)
    amamSymbols = A * exp(1j*theta)

    % AM-PM
    phi_pm = (alpha_phi * r^2) / (1 + beta_phi * r^2)
    ampmSymbols = abs(amamSymbols) * exp(1j*(angle(amamSymbols) + phi_pm))
"""

import numpy as np

# Same Saleh model coefficients as the MATLAB script
ALPHA_A = 2.1587
BETA_A = 1.1517
ALPHA_PHI = 4.0033
BETA_PHI = 9.1040


def saleh_amam(symbols: np.ndarray) -> np.ndarray:
    """Apply Saleh AM-AM (amplitude) distortion."""
    r = np.abs(symbols)
    theta = np.angle(symbols)

    A = (ALPHA_A * r) / (1 + BETA_A * r ** 2)

    return A * np.exp(1j * theta)


def saleh_ampm(symbols: np.ndarray) -> np.ndarray:
    """Apply Saleh AM-PM (phase) distortion. Call this AFTER saleh_amam()."""
    r = np.abs(symbols)

    phi_pm = (ALPHA_PHI * r ** 2) / (1 + BETA_PHI * r ** 2)

    return np.abs(symbols) * np.exp(1j * (np.angle(symbols) + phi_pm))
