"""
M-PSK modulation and demodulation, gray-coded, matching MATLAB's
pskmod(data, M, 0, 'gray') / pskdemod(data, M, 0, 'gray').
"""

import numpy as np


def _binary_to_gray(n: np.ndarray) -> np.ndarray:
    return n ^ (n >> 1)


def bits_to_psk_symbols(bits: np.ndarray, M: int):
    """
    Convert a 1D array of 0/1 bits into gray-coded M-PSK symbols.
    Pads bits with zeros if not a multiple of log2(M), same as MATLAB code.
    Returns (symbols, padded_bits) so callers can compute BER against the
    same padded bit array the MATLAB script uses.
    """
    k = int(np.log2(M))
    pad_rem = len(bits) % k
    if pad_rem != 0:
        bits = np.concatenate([bits, np.zeros(k - pad_rem, dtype=bits.dtype)])

    bits_reshaped = bits.reshape(-1, k)
    weights = 2 ** np.arange(k - 1, -1, -1)
    data = bits_reshaped @ weights

    gray_symbols = _binary_to_gray(np.arange(M))
    gray_to_natural = np.zeros(M, dtype=int)
    gray_to_natural[gray_symbols] = np.arange(M)
    natural_index = gray_to_natural[data]

    angles = 2 * np.pi * natural_index / M
    symbols = np.exp(1j * angles)
    return symbols, bits


def psk_symbols_to_bits(rx_symbols: np.ndarray, M: int) -> np.ndarray:
    """
    Demodulate M-PSK symbols back to bits (gray-coded), via nearest-angle
    detection. Inverse of bits_to_psk_symbols().
    """
    k = int(np.log2(M))

    angles = np.angle(rx_symbols) % (2 * np.pi)
    natural_index = np.round(angles / (2 * np.pi / M)).astype(int) % M

    gray_index = _binary_to_gray(natural_index)

    bit_positions = np.arange(k - 1, -1, -1)
    rx_bits = ((gray_index[:, None] >> bit_positions) & 1).astype(int)

    return rx_bits.reshape(-1)