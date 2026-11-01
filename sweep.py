"""
Sweep engine for the comparison table + regression tab.

Runs run_simulation() across a grid of M x numBits x GD x SNR values
(holding other impairments fixed), same idea as the MATLAB script's
nested loops that filled BER_matrix / EVM_matrix -- returned as a tidy
table instead of a multi-D array, which is easier to show/filter/plot
in Streamlit.

gd_type ('linear' or 'parabolic') is fixed for the whole sweep, same as
the other impairment settings -- only GD_ns and SNR_dB vary per grid
point.
"""
import numpy as np
import pandas as pd

from impairments.core import run_simulation


def run_sweep(
    M_values: list,
    numBits_values: list,
    gd_values_ns: list,
    snr_values_db: list,
    iq_params: dict = None,
    amam: bool = False,
    ampm: bool = False,
    pn_params: dict = None,
    doppler_params: dict = None,
    rain_params: dict = None,
    gd_type: str = "linear",
    progress_callback=None,
) -> pd.DataFrame:
    """
    Sweep over M x numBits x GD x SNR, holding IQ imbalance / AM-AM /
    AM-PM / phase noise / Doppler / rain fade / GD-type settings fixed
    across the whole sweep.

    gd_type : str
        'linear' (default) or 'parabolic'. Applied to every GD value in
        gd_values_ns for this sweep.
    progress_callback : callable, optional
        Called as progress_callback(fraction_complete) after each
        simulation run, so the UI can show a live progress bar.

    Returns
    -------
    pd.DataFrame with columns: M, numBits, GD_ns, GD_type, SNR_dB, BER, EVM
    """
    rows = []
    total_runs = len(M_values) * len(numBits_values) * len(gd_values_ns) * len(snr_values_db)
    completed = 0

    for M in M_values:
        for numBits in numBits_values:
            bits = np.random.randint(0, 2, numBits)
            for gd in gd_values_ns:
                for snr in snr_values_db:
                    impairments = {}
                    if iq_params:
                        impairments["iq_imbalance"] = iq_params
                    if gd > 0:
                        impairments["group_delay"] = {"gd_max_ns": gd, "gd_type": gd_type}
                    if amam:
                        impairments["amam"] = True
                    if ampm:
                        impairments["ampm"] = True
                    if pn_params:
                        impairments["phase_noise"] = pn_params
                    if doppler_params:
                        impairments["doppler"] = doppler_params
                    if rain_params:
                        impairments["rain_fade"] = rain_params

                    result = run_simulation(bits, M, snr, impairments)
                    rows.append(
                        {
                            "M": M,
                            "numBits": numBits,
                            "GD_ns": gd,
                            "GD_type": gd_type if gd > 0 else "",
                            "SNR_dB": snr,
                            "BER": result["ber"],
                            "EVM": result["evm"],
                        }
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / total_runs)

    return pd.DataFrame(rows)


def fit_ber_vs_snr(df: pd.DataFrame, gd_ns: float, numBits: int, M: int):
    """
    Fit log10(BER) vs SNR (linear) for a single (GD, numBits, M) slice.
    Returns (slope, intercept, r_squared) or None if not enough valid points.
    """
    subset = df[(df["GD_ns"] == gd_ns) & (df["numBits"] == numBits) & (df["M"] == M)]
    ber = subset["BER"].values
    snr = subset["SNR_dB"].values
    valid = ber > 0  # log10(0) is undefined, drop zero-BER points
    if valid.sum() < 2:
        return None
    log_ber = np.log10(ber[valid])
    snr_valid = snr[valid]
    slope, intercept = np.polyfit(snr_valid, log_ber, 1)
    fitted = slope * snr_valid + intercept
    ss_res = np.sum((log_ber - fitted) ** 2)
    ss_tot = np.sum((log_ber - np.mean(log_ber)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r_squared


def fit_evm_vs_gd(df: pd.DataFrame, snr_db: float, numBits: int, M: int):
    """
    Fit EVM vs GD (linear) for a single (SNR, numBits, M) slice.
    Returns (slope, intercept, r_squared) or None if not enough points.
    """
    subset = df[(df["SNR_dB"] == snr_db) & (df["numBits"] == numBits) & (df["M"] == M)]
    gd = subset["GD_ns"].values
    evm = subset["EVM"].values
    if len(gd) < 2:
        return None
    slope, intercept = np.polyfit(gd, evm, 1)
    fitted = slope * gd + intercept
    ss_res = np.sum((evm - fitted) ** 2)
    ss_tot = np.sum((evm - np.mean(evm)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r_squared


# Suggested default Doppler sweep points (cycles/symbol). Deliberately
# tiny -- freq_offset_norm is cycles PER SYMBOL, not a Group-Delay-style
# ns value, so it needs to stay in this range or the phase ramp wraps
# around many full rotations every symbol and destroys the signal at
# every SNR (BER pins near 0.5 regardless of noise level).
DEFAULT_DOPPLER_SWEEP_VALUES = [0.0, 0.00001, 0.00002, 0.00005, 0.00008, 0.00010, 0.00015]


def run_doppler_sweep(
    M: int,
    numBits: int,
    doppler_values_norm: list,
    snr_values_db: list,
    iq_params: dict = None,
    gd_params: dict = None,
    amam: bool = False,
    ampm: bool = False,
    pn_params: dict = None,
    rain_params: dict = None,
    residual_walk_std_deg: float = 0.0,
    n_trials: int = 3,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Dedicated sweep over Doppler (freq_offset_norm, cycles/symbol) x SNR,
    holding M, numBits, and any other enabled impairments fixed.

    This is intentionally a separate function from run_sweep() rather
    than folding Doppler in as another axis there: Doppler's natural
    scale (cycles/symbol, ~1e-5 to ~1e-3) is orders of magnitude
    different from Group Delay's (nanoseconds, 0-300), and giving it
    its own sweep with its own correctly-scaled defaults makes that
    units mismatch impossible to reproduce by accident.

    Each (Doppler, SNR) point is run n_trials times (fresh random
    initial phase + fresh AWGN + fresh residual walk each trial, same
    bits) and the BER/EVM are averaged. A single trial's BER is a noisy
    Monte-Carlo estimate -- especially at low BER, where a handful of
    bit errors either way swings log10(BER) a lot -- so averaging
    multiple trials smooths the curve and firms up the BER-vs-SNR
    regression fit (higher R^2) without changing what's being modeled.

    Parameters
    ----------
    M : int
    numBits : int
    doppler_values_norm : list of float
        Doppler values to sweep, in cycles/symbol. Use small values --
        see DEFAULT_DOPPLER_SWEEP_VALUES for a sane default range.
    snr_values_db : list of float
    iq_params, gd_params, pn_params, rain_params : dict, optional
        Other impairments held fixed across the sweep (gd_params should
        be a dict like {"gd_max_ns": ..., "gd_type": ...}, matching
        run_simulation's expected format).
    amam, ampm : bool, optional
    residual_walk_std_deg : float, optional (default 0.0)
        Passed through to doppler_shift() as the residual random-walk
        phase (models leftover Doppler after PLL tracking + oscillator
        instability). 0.0 = pure deterministic ramp (+ random phi0).
    n_trials : int, optional (default 3)
        Number of independent trials averaged per (Doppler, SNR) point.
    progress_callback : callable, optional
        Called as progress_callback(fraction_complete) after each run.

    Returns
    -------
    pd.DataFrame with columns: Doppler_norm, SNR_dB, BER, EVM, BER_std, EVM_std, n_trials
    """
    rows = []
    total_runs = len(doppler_values_norm) * len(snr_values_db) * n_trials
    completed = 0
    bits = np.random.randint(0, 2, numBits)

    for dop in doppler_values_norm:
        for snr in snr_values_db:
            trial_bers = []
            trial_evms = []
            for _ in range(n_trials):
                impairments = {}
                if iq_params:
                    impairments["iq_imbalance"] = iq_params
                if gd_params:
                    impairments["group_delay"] = gd_params
                if amam:
                    impairments["amam"] = True
                if ampm:
                    impairments["ampm"] = True
                if dop != 0:
                    impairments["doppler"] = {
                        "freq_offset_norm": dop,
                        "residual_walk_std_deg": residual_walk_std_deg,
                    }
                if pn_params:
                    impairments["phase_noise"] = pn_params
                if rain_params:
                    impairments["rain_fade"] = rain_params

                result = run_simulation(bits, M, snr, impairments)
                trial_bers.append(result["ber"])
                trial_evms.append(result["evm"])
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_runs)

            rows.append(
                {
                    "Doppler_norm": dop,
                    "SNR_dB": snr,
                    "BER": float(np.mean(trial_bers)),
                    "EVM": float(np.mean(trial_evms)),
                    "BER_std": float(np.std(trial_bers)),
                    "EVM_std": float(np.std(trial_evms)),
                    "n_trials": n_trials,
                }
            )

    return pd.DataFrame(rows)


def fit_ber_vs_snr_doppler(df: pd.DataFrame, doppler_norm: float):
    """
    Fit log10(BER) vs SNR (linear) for a single Doppler value slice, from
    a run_doppler_sweep() result. Mirrors fit_ber_vs_snr() but keyed on
    Doppler_norm instead of GD_ns.
    Returns (slope, intercept, r_squared) or None if not enough valid points.
    """
    subset = df[df["Doppler_norm"] == doppler_norm]
    ber = subset["BER"].values
    snr = subset["SNR_dB"].values
    valid = ber > 0
    if valid.sum() < 2:
        return None
    log_ber = np.log10(ber[valid])
    snr_valid = snr[valid]
    slope, intercept = np.polyfit(snr_valid, log_ber, 1)
    fitted = slope * snr_valid + intercept
    ss_res = np.sum((log_ber - fitted) ** 2)
    ss_tot = np.sum((log_ber - np.mean(log_ber)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r_squared
