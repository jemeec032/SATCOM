"""
Rain Attenuation (Rain Fade) impairment.

At Ku/Ka-band, rain along the propagation path attenuates the signal,
directly eating into the link budget's SNR margin. Modeled here simply
as a dB reduction applied to the channel SNR before AWGN is added --
this is exactly how rain fade shows up in a link budget: it doesn't
distort the constellation shape by itself, it just leaves less SNR
margin to work with.
"""


def apply_rain_fade(snr_db: float, fade_db: float) -> float:
    """
    Reduce the effective channel SNR by the rain fade margin.

    Parameters
    ----------
    snr_db : float
        Clear-sky SNR (dB).
    fade_db : float
        Rain fade attenuation to apply (dB).

    Returns
    -------
    float
        Effective SNR (dB) after rain fade.
    """
    return snr_db - fade_db
