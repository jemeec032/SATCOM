"""
ITU-R P.618-13 / P.838-3 compliant rain attenuation model.

Replaces the earlier "pick a dB number" rain-fade slider with the actual
ITU-R rain attenuation prediction procedure: specific attenuation from
ITU-R P.838-3 (Recommendation P.838-3, 03/2005) combined with the slant-path
geometry and statistical time-percentage scaling of ITU-R P.618-13
(Recommendation P.618-13, 12/2017), Steps 2-10 of Annex 1.

One deliberate, clearly-labeled simplification: ITU-R P.839 (rain height)
and ITU-R P.837 (rain rate exceeded 0.01% of an average year) are themselves
derived from large, gridded global climate maps (bilinear-interpolated over
a 0-360 deg longitude x -90..90 deg latitude grid) that are impractical to
embed in a lightweight simulator. Rather than silently faking that lookup,
this module takes the rain height and rain rate as direct physical inputs --
exactly the fallback ITU-R P.618-13 itself documents ("if not provided, an
estimate is obtained from ITU-R P.837 ... or local measured rainfall rate
data"). Every other step of the official procedure -- specific attenuation,
slant path length, horizontal reduction factor, vertical adjustment factor,
effective path length, and the time-percentage scaling law -- is implemented
exactly as published, with no further approximation.

Coefficient tables and the Step 2-10 procedure were checked against the
open-source ITU-Rpy reference implementation
(https://github.com/inigodelportillo/ITU-Rpy), which is itself validated
against the ITU's own official validation example suite (see
itu-rpy.readthedocs.io/en/latest/validation/).

Public API
----------
RAIN_RATE_PRESETS_MM_PER_HR   -- dict of reference R_0.01 values (mm/h)
rain_specific_attenuation_coefficients(f_ghz, el_deg, tau_deg) -> (k, alpha)
rain_specific_attenuation(R_mm_per_hr, f_ghz, el_deg, tau_deg) -> gamma_R (dB/km)
slant_path_length(h_rain_km, h_station_km, el_deg)             -> Ls (km)
itu_r_p618_rain_attenuation(...)                                -> dict
"""

import numpy as np

# ---------------------------------------------------------------------
# ITU-R P.837-based reference rain rates R_0.01 (mm/h) exceeded for 0.01%
# of an average year -- typical values by rain intensity, for use when a
# location-specific value from ITU-R P.837 maps or local meteorological
# records is not available.
# ---------------------------------------------------------------------
RAIN_RATE_PRESETS_MM_PER_HR = {
    "Drizzle": 0.25,
    "Light rain": 2.5,
    "Medium rain": 12.5,
    "Heavy rain": 25.0,
    "Downpour": 50.0,
    "Tropical": 100.0,
    "Monsoon": 150.0,
}

# ---------------------------------------------------------------------
# ITU-R P.838-3 frequency-dependent curve-fit coefficients for specific
# attenuation (Recommendation ITU-R P.838-3, 03/2005), horizontal (H) and
# vertical (V) linear polarizations. Valid for 1-1000 GHz.
# ---------------------------------------------------------------------
_KH = {"aj": [-5.33980, -0.35351, -0.23789, -0.94158],
       "bj": [-0.10008, 1.26970, 0.86036, 0.64552],
       "cj": [1.13098, 0.45400, 0.15354, 0.16817],
       "mk": -0.18961, "ck": 0.71147}

_KV = {"aj": [-3.80595, -3.44965, -0.39902, 0.50167],
       "bj": [0.56934, -0.22911, 0.73042, 1.07319],
       "cj": [0.81061, 0.51059, 0.11899, 0.27195],
       "mk": -0.16398, "ck": 0.63297}

_ALPHA_H = {"aj": [-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
            "bj": [1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
            "cj": [-0.55187, 0.19822, 0.13164, 1.47828, 3.43990],
            "ma": 0.67849, "ca": -1.95537}

_ALPHA_V = {"aj": [-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
            "bj": [2.33840, 0.95545, 1.14520, 0.791669, 0.791459],
            "cj": [-0.76284, 0.54039, 0.26809, 0.116226, 0.116479],
            "ma": -0.053739, "ca": 0.83433}


def _curve(f_ghz, a, b, c):
    return a * np.exp(-((np.log10(f_ghz) - b) / c) ** 2)


def rain_specific_attenuation_coefficients(f_ghz: float, el_deg: float, tau_deg: float):
    """
    ITU-R P.838-3 coefficients k, alpha such that gamma_R = k * R^alpha.

    Parameters
    ----------
    f_ghz : frequency, GHz (Recommendation valid for 1-1000 GHz)
    el_deg : path elevation angle, degrees
    tau_deg : polarization tilt angle relative to the horizontal, degrees
              (0 = horizontal, 90 = vertical, 45 = circular)
    """
    KH = 10 ** (sum(_curve(f_ghz, _KH["aj"][j], _KH["bj"][j], _KH["cj"][j]) for j in range(4))
                + _KH["mk"] * np.log10(f_ghz) + _KH["ck"])
    KV = 10 ** (sum(_curve(f_ghz, _KV["aj"][j], _KV["bj"][j], _KV["cj"][j]) for j in range(4))
                + _KV["mk"] * np.log10(f_ghz) + _KV["ck"])
    alphaH = (sum(_curve(f_ghz, _ALPHA_H["aj"][j], _ALPHA_H["bj"][j], _ALPHA_H["cj"][j]) for j in range(5))
              + _ALPHA_H["ma"] * np.log10(f_ghz) + _ALPHA_H["ca"])
    alphaV = (sum(_curve(f_ghz, _ALPHA_V["aj"][j], _ALPHA_V["bj"][j], _ALPHA_V["cj"][j]) for j in range(5))
              + _ALPHA_V["ma"] * np.log10(f_ghz) + _ALPHA_V["ca"])

    cos_el2 = np.cos(np.deg2rad(el_deg)) ** 2
    cos_2tau = np.cos(np.deg2rad(2 * tau_deg))

    k = (KH + KV + (KH - KV) * cos_el2 * cos_2tau) / 2.0
    alpha = (KH * alphaH + KV * alphaV + (KH * alphaH - KV * alphaV) * cos_el2 * cos_2tau) / (2.0 * k)
    return k, alpha


def rain_specific_attenuation(R_mm_per_hr: float, f_ghz: float, el_deg: float, tau_deg: float) -> float:
    """Specific attenuation gamma_R (dB/km) -- ITU-R P.838-3 Eq. (1): gamma_R = k * R^alpha."""
    k, alpha = rain_specific_attenuation_coefficients(f_ghz, el_deg, tau_deg)
    return k * R_mm_per_hr ** alpha


def slant_path_length(h_rain_km: float, h_station_km: float, el_deg: float,
                       earth_radius_km: float = 8500.0) -> float:
    """ITU-R P.618-13 Step 2 (Eq. 1 for el >= 5 deg, Eq. 2 otherwise): slant path length, km."""
    delta_h = max(h_rain_km - h_station_km, 0.0)
    el_rad = np.deg2rad(el_deg)
    if el_deg >= 5:
        return delta_h / np.sin(el_rad)
    return 2 * delta_h / (np.sqrt(np.sin(el_rad) ** 2 + 2 * delta_h / earth_radius_km) + np.sin(el_rad))


def itu_r_p618_rain_attenuation(
    f_ghz: float,
    el_deg: float,
    tau_deg: float,
    lat_deg: float,
    h_station_km: float,
    h_rain_km: float,
    r001_mm_per_hr: float,
    p_percent: float = 0.01,
) -> dict:
    """
    Full ITU-R P.618-13 Annex 1 rain attenuation procedure, Steps 2-10.
    (Step 1 -- rain height from the ITU-R P.839 global isotherm grid -- is
    supplied directly as h_rain_km; see module docstring.)

    Parameters
    ----------
    f_ghz : frequency, GHz
    el_deg : elevation angle, degrees (>= 5 recommended)
    tau_deg : polarization tilt angle, degrees (0=H, 90=V, 45=circular)
    lat_deg : earth station latitude, degrees (used in Steps 7 and 10)
    h_station_km : earth station height above mean sea level, km
    h_rain_km : mean annual rain height above mean sea level, km
                (= 0 deg C isotherm height + 0.36 km, per ITU-R P.839)
    r001_mm_per_hr : point rainfall rate exceeded for 0.01% of an average
                     year, mm/h (ITU-R P.837, or a local/measured value)
    p_percent : percentage of time the returned attenuation is exceeded,
                valid range 0.001-5 (%). Default 0.01% is the standard
                SATCOM link-margin design point.

    Returns
    -------
    dict with the intermediate quantities (useful for display/debugging)
    and the final predicted attenuation "A_p_db" (dB) -- the value to
    subtract from clear-sky SNR.
    """
    Re = 8500.0

    # Step 2: slant path length
    Ls = slant_path_length(h_rain_km, h_station_km, el_deg, Re)

    # Step 3: horizontal projection of the slant path
    Lg = abs(Ls * np.cos(np.deg2rad(el_deg)))

    # Step 5 (R_0.01 itself -- "Step 4" -- is supplied directly, see above):
    # specific attenuation from ITU-R P.838-3
    gamma_r = rain_specific_attenuation(r001_mm_per_hr, f_ghz, el_deg, tau_deg)

    # Step 6: horizontal reduction factor for 0.01% of the time
    r001 = 1.0 / (1 + 0.78 * np.sqrt(Lg * gamma_r / f_ghz) - 0.38 * (1 - np.exp(-2 * Lg)))

    # Step 7: vertical adjustment factor for 0.01% of the time
    delta_h = h_rain_km - h_station_km
    delta_h_safe = delta_h if delta_h > 0 else 1e-9
    eta_deg = np.rad2deg(np.arctan2(delta_h, Lg * r001))
    if eta_deg > el_deg:
        Lr = Lg * r001 / np.cos(np.deg2rad(el_deg))
    else:
        Lr = delta_h_safe / np.sin(np.deg2rad(el_deg))
    xi = (36 - abs(lat_deg)) if abs(lat_deg) < 36 else 0.0
    v001 = 1.0 / (1 + np.sqrt(np.sin(np.deg2rad(el_deg))) *
                  (31 * (1 - np.exp(-el_deg / (1 + xi))) * np.sqrt(Lr * gamma_r) / f_ghz ** 2 - 0.45))

    # Step 8: effective path length
    Le = Lr * v001

    # Step 9: attenuation exceeded for 0.01% of an average year
    A001 = gamma_r * Le

    # Step 10: scale A(0.01%) to the requested percentage of time p
    if p_percent >= 1 or abs(lat_deg) >= 36:
        beta = 0.0
    elif el_deg > 25:
        beta = -0.005 * (abs(lat_deg) - 36)
    else:
        beta = -0.005 * (abs(lat_deg) - 36) + 1.8 - 4.25 * np.sin(np.deg2rad(el_deg))

    exponent = -(0.655 + 0.033 * np.log(p_percent) - 0.045 * np.log(max(A001, 1e-9)) -
                 beta * (1 - p_percent) * np.sin(np.deg2rad(el_deg)))
    A_p = A001 * (p_percent / 0.01) ** exponent
    A_p = max(A_p, 0.0)

    return {
        "slant_path_length_km": Ls,
        "horizontal_projection_km": Lg,
        "specific_attenuation_db_per_km": gamma_r,
        "horizontal_reduction_factor": r001,
        "vertical_adjustment_factor": v001,
        "effective_path_length_km": Le,
        "A_0p01_db": A001,
        "A_p_db": A_p,
    }
