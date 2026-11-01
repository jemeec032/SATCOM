"""
Impairment Severity Meter -- renders a simple text bar (e.g. "###...")
showing how severe each currently-enabled impairment's setting is,
relative to its slider's max value.
"""

FULL_BLOCK = "\u2588"  # █
EMPTY_BLOCK = "\u2591"  # ░


def severity_bar(value: float, max_value: float, length: int = 10) -> str:
    """
    Build a text severity bar like '\u2588\u2588\u2588\u2591\u2591\u2591'.

    Parameters
    ----------
    value : float
        Current parameter value.
    max_value : float
        The value that should render as a "full" bar.
    length : int
        Number of characters in the bar.

    Returns
    -------
    str
    """
    if max_value <= 0:
        fraction = 0.0
    else:
        fraction = max(0.0, min(1.0, value / max_value))

    filled = round(fraction * length)
    return FULL_BLOCK * filled + EMPTY_BLOCK * (length - filled)


# Reference max values used to normalize each impairment's severity bar.
# These match the slider maximums used elsewhere in the app.
SEVERITY_MAX = {
    "iq_imbalance": 3.0,       # amp_imb_db max
    "group_delay": 300.0,      # gd_max_ns max
    "doppler": 0.001,          # freq_offset_norm max
    "phase_noise": 15.0,       # sigma_deg max
    "rain_fade": 15.0,         # fade_db max
}


def impairment_severity(impairments: dict) -> dict:
    """
    Given an impairments dict (as built by the app's impairment_controls()),
    return {label: bar_string} for every enabled impairment.
    AM-AM / AM-PM are on/off (no intensity parameter), so they get a
    fixed full bar when enabled.
    """
    bars = {}
    if "iq_imbalance" in impairments:
        bars["IQ Imbalance"] = severity_bar(impairments["iq_imbalance"]["amp_imb_db"], SEVERITY_MAX["iq_imbalance"])
    if "group_delay" in impairments:
        bars["Group Delay"] = severity_bar(impairments["group_delay"]["gd_max_ns"], SEVERITY_MAX["group_delay"])
    if impairments.get("amam"):
        bars["AM-AM"] = FULL_BLOCK * 10
    if impairments.get("ampm"):
        bars["AM-PM"] = FULL_BLOCK * 10
    if "doppler" in impairments:
        bars["Doppler"] = severity_bar(impairments["doppler"]["freq_offset_norm"], SEVERITY_MAX["doppler"])
    if "phase_noise" in impairments:
        bars["Phase Noise"] = severity_bar(impairments["phase_noise"]["sigma_deg"], SEVERITY_MAX["phase_noise"])
    if "rain_fade" in impairments:
        bars["Rain Fade"] = severity_bar(impairments["rain_fade"]["fade_db"], SEVERITY_MAX["rain_fade"])
    return bars
