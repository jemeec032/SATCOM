"""
Core simulation engine. One function that runs the full chain, applying
only the impairments the caller enables, in the same physical order as
the MATLAB script, extended with Doppler and Rain Fade:

    modulate -> IQ imbalance -> group delay -> AM-AM -> AM-PM
             -> Doppler shift -> phase noise -> [rain fade reduces SNR]
             -> AWGN -> carrier phase recovery -> demod

Also records the symbol array after EVERY stage that actually ran, so
the UI can show a separate constellation plot per enabled impairment
(matching the MATLAB script's subplot grid), not just one final
before/after pair.
"""

import time

import numpy as np

from impairments.modulation import bits_to_psk_symbols, psk_symbols_to_bits
from impairments.iq_imbalance import iq_imbalance
from impairments.group_delay import group_delay
from impairments.saleh_nonlinearity import saleh_amam, saleh_ampm
from impairments.phase_noise_awgn import phase_noise, awgn_channel, carrier_phase_recovery
from impairments.doppler import doppler_shift
from impairments.rain_attenuation import apply_rain_fade

VALID_M = (2, 4, 8, 16, 32, 64)


class SimulationInputError(ValueError):
    """Raised when run_simulation() is given parameters that don't make sense."""


def validate_inputs(bits: np.ndarray, M: int, snr_db: float, impairments: dict) -> None:
    """
    Basic sanity checks on simulation inputs, so bad parameter combos
    fail with a clear message instead of a confusing exception (or
    silently-wrong results) deep inside the impairment chain.
    """
    if M not in VALID_M or (M & (M - 1)) != 0:
        raise SimulationInputError(f"M={M} must be a power of 2 (got {M}).")
    if len(bits) < int(np.log2(M)):
        raise SimulationInputError(
            f"Need at least log2(M)={int(np.log2(M))} bits for M={M}-PSK, got {len(bits)}."
        )
    if not set(np.unique(bits)).issubset({0, 1}):
        raise SimulationInputError("bits array must contain only 0s and 1s.")
    if snr_db < -20 or snr_db > 60:
        raise SimulationInputError(f"SNR={snr_db}dB is outside a sane simulation range (-20 to 60 dB).")

    if "iq_imbalance" in impairments:
        p = impairments["iq_imbalance"]
        if p.get("amp_imb_db", 0) < 0:
            raise SimulationInputError("IQ amplitude imbalance (dB) cannot be negative.")
    if "group_delay" in impairments:
        p = impairments["group_delay"]
        if p.get("gd_max_ns", 0) < 0:
            raise SimulationInputError("Group delay (ns) cannot be negative.")
        if p.get("gd_type", "linear") not in ("linear", "parabolic"):
            raise SimulationInputError("gd_type must be 'linear' or 'parabolic'.")
    if "phase_noise" in impairments and impairments["phase_noise"].get("sigma_deg", 0) < 0:
        raise SimulationInputError("Phase noise std dev (deg) cannot be negative.")
    if "rain_fade" in impairments and impairments["rain_fade"].get("fade_db", 0) < 0:
        raise SimulationInputError("Rain fade (dB) cannot be negative.")


def run_simulation(bits: np.ndarray, M: int, snr_db: float, impairments: dict = None, progress_callback=None) -> dict:
    """
    Run one full simulation pass.

    Parameters
    ----------
    bits : np.ndarray (0/1 ints)
    M : int
    snr_db : float
        Clear-sky AWGN channel SNR in dB (AWGN is always applied -- it
        represents the channel itself, matching the MATLAB script).
    impairments : dict, optional
            'iq_imbalance': {'amp_imb_db': float, 'phase_imb_deg': float}
            'group_delay':  {'gd_max_ns': float, 'gd_type': 'linear'|'parabolic'}
            'amam':         True   (Saleh AM-AM amplitude distortion)
            'ampm':         True   (Saleh AM-PM phase distortion)
            'doppler':      {'freq_offset_norm': float}  (cycles/symbol)
            'phase_noise':  {'sigma_deg': float}
            'rain_fade':    {'fade_db': float}  (reduces effective SNR)
    progress_callback : callable, optional
        Called as progress_callback(fraction_complete) after each of the
        11 pipeline stages (modulate, IQ imbalance, group delay, AM-AM,
        AM-PM, Doppler, phase noise, rain fade, AWGN, carrier recovery,
        demodulate) -- fraction_complete goes from ~0.09 to 1.0, so a UI
        progress bar advances meaningfully through a single run instead
        of jumping straight from 0% to 100%. Optional -- pass None (the
        default) to skip progress reporting entirely, e.g. for sweeps
        that already report progress per-run rather than per-stage.

    Returns
    -------
    dict with keys:
        'ber', 'evm'
        'tx_ideal', 'tx_actual', 'rx_symbols'
        'effective_snr_db' -- SNR actually used for AWGN (after rain fade)
        'stages' -- list of (label, symbols) tuples, one entry per stage
                    that actually ran, in order.
        'timing_ms' -- dict of {stage_name: elapsed_ms}, for the
                       Performance Dashboard.
    """
    if impairments is None:
        impairments = {}
    validate_inputs(bits, M, snr_db, impairments)

    stages = []
    timing_ms = {}
    t_start = time.perf_counter()

    def _mark(label):
        now = time.perf_counter()
        timing_ms[label] = (now - _mark.last) * 1000.0
        _mark.last = now

    _mark.last = t_start

    _TOTAL_STAGES = 11
    _stage_count = [0]

    def _progress(label):
        if progress_callback is not None:
            _stage_count[0] += 1
            progress_callback(_stage_count[0] / _TOTAL_STAGES, label)

    # ---- Modulate ----
    symbols_ideal, padded_bits = bits_to_psk_symbols(bits, M)
    stages.append(("Ideal (Transmitted)", symbols_ideal))
    _mark("Modulation")
    _progress("Modulation")

    # ---- IQ imbalance (transmitter) ----
    if "iq_imbalance" in impairments:
        symbols = iq_imbalance(symbols_ideal, **impairments["iq_imbalance"])
        stages.append(("After IQ Imbalance", symbols))
    else:
        symbols = symbols_ideal.copy()
    _mark("IQ Imbalance")
    _progress("IQ Imbalance")

    tx_reference = symbols  # phase recovery reference, matches MATLAB's `symbols`

    sig = symbols

    # ---- Group delay ----
    if "group_delay" in impairments:
        gd_type = impairments["group_delay"].get("gd_type", "linear")
        sig = group_delay(sig, impairments["group_delay"]["gd_max_ns"] * 1e-9, gd_type=gd_type)
        stages.append((f"After Group Delay ({gd_type})", sig))
    _mark("Group Delay")
    _progress("Group Delay")

    # ---- Saleh AM-AM (independently toggleable) ----
    if impairments.get("amam"):
        sig = saleh_amam(sig)
        stages.append(("After AM-AM", sig))
    _mark("AM-AM")
    _progress("AM-AM")

    # ---- Saleh AM-PM (independently toggleable) ----
    if impairments.get("ampm"):
        sig = saleh_ampm(sig)
        stages.append(("After AM-PM", sig))
    _mark("AM-PM")
    _progress("AM-PM")

    # ---- Doppler / frequency offset ----
    if "doppler" in impairments:
        sig = doppler_shift(
            sig,
            impairments["doppler"]["freq_offset_norm"],
            residual_walk_std_deg=impairments["doppler"].get("residual_walk_std_deg", 0.0),
        )
        stages.append(("After Doppler Shift", sig))
    _mark("Doppler")
    _progress("Doppler")

    # ---- Phase noise ----
    if "phase_noise" in impairments:
        sig = phase_noise(sig, impairments["phase_noise"]["sigma_deg"])
        stages.append(("After Phase Noise", sig))
    _mark("Phase Noise")
    _progress("Phase Noise")

    # ---- Rain fade (reduces effective channel SNR, no shape change by itself) ----
    effective_snr_db = snr_db
    if "rain_fade" in impairments:
        effective_snr_db = apply_rain_fade(snr_db, impairments["rain_fade"]["fade_db"])
    _mark("Rain Fade")
    _progress("Rain Fade")

    # ---- AWGN channel (always applied) ----
    rx_symbols = awgn_channel(sig, effective_snr_db)
    _mark("AWGN")
    _progress("AWGN")

    # ---- Carrier phase recovery ----
    rx_symbols = carrier_phase_recovery(rx_symbols, tx_reference)
    stages.append((f"Received (AWGN {effective_snr_db:.1f}dB eff. + Recovery)", rx_symbols))
    _mark("Carrier Recovery")
    _progress("Carrier Recovery")

    # ---- Demodulate ----
    rx_bits = psk_symbols_to_bits(rx_symbols, M)
    _mark("Demodulation")
    _progress("Demodulation")

    # ---- BER ----
    ber = float(np.mean(padded_bits != rx_bits))

    # ---- EVM (referenced against ideal, pre-IQ-imbalance symbols) ----
    evm = float(np.sqrt(np.mean(np.abs(rx_symbols - symbols_ideal) ** 2)))

    timing_ms["Total"] = (time.perf_counter() - t_start) * 1000.0

    return {
        "ber": ber,
        "evm": evm,
        "tx_ideal": symbols_ideal,
        "tx_actual": symbols,
        "rx_symbols": rx_symbols,
        "effective_snr_db": effective_snr_db,
        "stages": stages,
        "timing_ms": timing_ms,
    }
