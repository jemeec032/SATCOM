"""
Satcom RF Impairment Simulator - Streamlit app.
Run with:
    streamlit run app.py
"""
import io
import time
import textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from impairments.core import run_simulation
from impairments.sweep import (
    run_sweep, fit_ber_vs_snr, fit_evm_vs_gd,
    run_doppler_sweep, fit_ber_vs_snr_doppler, DEFAULT_DOPPLER_SWEEP_VALUES,
)
from impairments.spectrum import compute_psd
from impairments.eye_diagram import pulse_shape, eye_traces
from impairments.equalizer import zero_forcing_equalize, mmse_equalize
from impairments.carrier_recovery import costas_loop_recovery, decision_directed_costas
from impairments.severity import impairment_severity
from impairments.theoretical_ber import theoretical_ber_mpsk
from impairments.modulation import bits_to_psk_symbols, psk_symbols_to_bits
from impairments.group_delay import group_delay
from impairments.doppler import doppler_shift
from impairments.phase_noise_awgn import awgn_channel, carrier_phase_recovery
from impairments.itu_rain_model import (
    itu_r_p618_rain_attenuation,
    RAIN_RATE_PRESETS_MM_PER_HR,
)
from auth import check_credentials, create_account

NUM_BITS_OPTIONS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]  # includes 1e7
st.set_page_config(page_title="Satcom RF Impairment Simulator", layout="wide", page_icon="🛰️")

HERO_CSS = textwrap.dedent("""
<style>
.hero {
    background:
        radial-gradient(1.5px 1.5px at 20px 30px, white, transparent),
        radial-gradient(1.5px 1.5px at 90px 120px, white, transparent),
        radial-gradient(1px 1px at 150px 40px, white, transparent),
        radial-gradient(1px 1px at 210px 150px, white, transparent),
        radial-gradient(1.5px 1.5px at 280px 60px, white, transparent),
        radial-gradient(1px 1px at 340px 190px, white, transparent),
        radial-gradient(1.5px 1.5px at 400px 100px, white, transparent),
        radial-gradient(1px 1px at 460px 30px, white, transparent),
        radial-gradient(1.5px 1.5px at 60px 200px, white, transparent),
        radial-gradient(1px 1px at 500px 220px, white, transparent),
        linear-gradient(135deg, #020111 0%, #001233 35%, #002664 65%, #023e7d 100%);
    background-size: 550px 250px, 550px 250px, 550px 250px, 550px 250px,
                     550px 250px, 550px 250px, 550px 250px, 550px 250px,
                     550px 250px, 550px 250px, cover;
    border-radius: 16px;
    padding: 60px 40px;
    text-align: center;
    color: #f0f4ff;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
.hero h1 { font-size: 2.4rem; margin-bottom: 0.3rem; letter-spacing: 0.5px; }
.hero p { font-size: 1.05rem; color: #a8c0e8; max-width: 700px; margin: 0.6rem auto; }
</style>
""")

# ====================================================================
# Global theme -- fonts, logo/title banner, feature cards, badges,
# tab styling. Injected once, applies to the login page and every tab.
# ====================================================================
GLOBAL_CSS = textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .fc-title { font-family: 'Poppins', sans-serif; }

/* ---- Persistent logo / title banner ---- */
.topbar {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 16px 26px;
    margin-bottom: 14px;
    border-radius: 16px;
    background: linear-gradient(120deg, #020111 0%, #001233 45%, #023e7d 120%);
    box-shadow: 0 6px 22px rgba(0,0,0,0.35);
}
.topbar .logo-badge {
    flex-shrink: 0;
    font-size: 2rem;
    width: 54px; height: 54px;
    display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle at 30% 30%, #5aa9ff, #023e7d 75%);
    border-radius: 50%;
    box-shadow: 0 0 16px rgba(90,169,255,0.55);
}
.topbar .title-block { line-height: 1.15; }
.topbar .title-block h1 {
    font-size: 1.45rem;
    margin: 0;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff, #9fc9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.3px;
}
.topbar .title-block p {
    margin: 3px 0 0 0;
    font-size: 0.82rem;
    color: #a8c0e8;
}
.topbar .user-chip {
    margin-left: auto;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.8rem;
    color: #e3ecff;
    white-space: nowrap;
}

/* ---- Feature / about cards ---- */
.feature-card {
    background: linear-gradient(160deg, rgba(2,62,125,0.10), rgba(2,38,100,0.02));
    border: 1px solid rgba(2,62,125,0.22);
    border-radius: 14px;
    padding: 18px 18px 16px 18px;
    height: 100%;
    margin-bottom: 14px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 22px rgba(2,62,125,0.18);
}
.feature-card .fc-icon { font-size: 1.7rem; margin-bottom: 6px; }
.feature-card .fc-title { margin: 0 0 6px 0; font-size: 1.02rem; font-weight: 600; }
.feature-card p { margin: 0; font-size: 0.86rem; opacity: 0.85; line-height: 1.4; }

/* ---- Stat strip ---- */
.stat-chip {
    text-align: center;
    padding: 14px 8px;
    border-radius: 12px;
    background: rgba(2,62,125,0.08);
    border: 1px solid rgba(2,62,125,0.18);
}
.stat-chip .stat-num {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #023e7d;
}
.stat-chip .stat-label {
    font-size: 0.78rem;
    opacity: 0.75;
    margin-top: 2px;
}

/* ---- Small badge pill ---- */
.badge-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    background: rgba(2,62,125,0.12);
    border: 1px solid rgba(2,62,125,0.3);
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 6px 2px 0;
}

/* ---- Tabs: slightly bolder + bigger ---- */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}
</style>
""")

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ====================================================================
# Color palette + small helpers for colorful cards/chips/badges.
# Every card-producing section cycles through this so the app doesn't
# read as a single-hue theme.
# ====================================================================
PALETTE = ["#7C3AED", "#EC4899", "#F97316", "#14B8A6", "#3B82F6", "#22C55E", "#F59E0B", "#EF4444"]


def feature_card_html(icon: str, title: str, desc: str, color: str) -> str:
    return f"""
    <div class="feature-card" style="border-left:4px solid {color};">
        <div class="fc-icon" style="background:{color}22;border:1px solid {color}55;
             width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
             justify-content:center;margin-bottom:8px;">{icon}</div>
        <div class="fc-title" style="color:{color};">{title}</div>
        <p>{desc}</p>
    </div>
    """


def stat_chip_html(num: str, label: str, color: str) -> str:
    return f"""
    <div class="stat-chip" style="background:{color}14;border-color:{color}55;">
        <div class="stat-num" style="color:{color};">{num}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def badge_html(text: str, color: str) -> str:
    return f'<span class="badge-pill" style="background:{color}22;border-color:{color}66;color:{color};">{text}</span>'


COLORFUL_EXTRAS_CSS = textwrap.dedent("""
<style>
/* Rainbow underline beneath the top banner */
.rainbow-bar {
    height: 4px;
    border-radius: 999px;
    margin: 0 0 18px 0;
    background: linear-gradient(90deg, #7C3AED, #EC4899, #F97316, #F59E0B, #22C55E, #14B8A6, #3B82F6);
    background-size: 200% 100%;
}

/* Colorful metric values */
[data-testid="stMetricValue"] { color: #7C3AED; font-weight: 700; }

/* Colored active-tab underlines, cycling through the palette */
div[data-baseweb="tab-list"] button:nth-child(1)[aria-selected="true"] { color:#7C3AED !important; border-bottom-color:#7C3AED !important; }
div[data-baseweb="tab-list"] button:nth-child(2)[aria-selected="true"] { color:#EC4899 !important; border-bottom-color:#EC4899 !important; }
div[data-baseweb="tab-list"] button:nth-child(3)[aria-selected="true"] { color:#F97316 !important; border-bottom-color:#F97316 !important; }
div[data-baseweb="tab-list"] button:nth-child(4)[aria-selected="true"] { color:#14B8A6 !important; border-bottom-color:#14B8A6 !important; }
div[data-baseweb="tab-list"] button:nth-child(5)[aria-selected="true"] { color:#3B82F6 !important; border-bottom-color:#3B82F6 !important; }
div[data-baseweb="tab-list"] button:nth-child(6)[aria-selected="true"] { color:#22C55E !important; border-bottom-color:#22C55E !important; }
div[data-baseweb="tab-list"] button:nth-child(7)[aria-selected="true"] { color:#F59E0B !important; border-bottom-color:#F59E0B !important; }
div[data-baseweb="tab-list"] button:nth-child(8)[aria-selected="true"] { color:#EF4444 !important; border-bottom-color:#EF4444 !important; }
</style>
""")
st.markdown(COLORFUL_EXTRAS_CSS, unsafe_allow_html=True)

# ====================================================================
# LOGIN GATE
# ====================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.markdown(HERO_CSS, unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent("""
        <div class="hero">
        <div style="font-size:2.8rem;">🛰️</div>
        <h1>Satcom RF Impairment Simulator</h1>
        <p>Modeling real-world RF impairments in satellite communication links.
        Sign in, create an account, or continue as guest to get started.</p>
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.write("")
    _, col_login, _ = st.columns([1, 1.3, 1])
    with col_login:
        login_tab, signup_tab = st.tabs(["Sign In", "Create Account"])
        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Sign In", type="primary", use_container_width=True):
                if check_credentials(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        with signup_tab:
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            if st.button("Create Account", type="primary", use_container_width=True):
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, message = create_account(new_username, new_password)
                    if success:
                        st.success(message + " Switch to the Sign In tab to log in.")
                    else:
                        st.error(message)
        st.divider()
        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.username = "Guest"
            st.rerun()
    st.stop()

# ====================================================================
# LOGGED IN: persistent logo/title banner + sidebar identity + Reset All
# ====================================================================
st.session_state.setdefault("history", [])

st.markdown(
    f"""
    <div class="topbar">
        <div class="logo-badge">🛰️</div>
        <div class="title-block">
            <h1>Satcom RF Impairment Simulator</h1>
            <p>Modeling real-world RF impairments in satellite communication links</p>
        </div>
        <div class="user-chip">👤 Logged in as {st.session_state.username}</div>
    </div>
    <div class="rainbow-bar"></div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <div style="font-size:1.6rem;">🛰️</div>
            <div style="font-weight:700;font-size:1rem;">Satcom RF Simulator</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Logged in as:** {st.session_state.username}")
    col_logout, col_reset = st.columns(2)
    with col_logout:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    with col_reset:
        if st.button("Reset All", use_container_width=True):
            keep = {
                "logged_in": st.session_state.logged_in,
                "username": st.session_state.username,
                "history": st.session_state.history,
            }
            st.session_state.clear()
            st.session_state.update(keep)
            st.rerun()
    st.divider()

def log_history(mode: str, params: dict, ber=None, evm=None):
    """Append a run to the session's history log."""
    st.session_state.history.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": mode,
            "params": str(params),
            "BER": f"{ber:.6f}" if ber is not None else "",
            "EVM": f"{evm:.6f}" if evm is not None else "",
        }
    )

def seed_control(key_prefix: str):
    """
    Render a 'Random seed (optional)' text input for reproducibility.
    Leave blank -> current behavior (a fresh random run every time).
    Enter an integer -> every random draw in that run (bit generation,
    AWGN noise, Doppler's random initial phase / residual walk, phase
    noise, etc.) becomes fully deterministic and can be regenerated
    exactly later -- e.g. to reproduce a specific figure for a report.

    Returns the raw text the user typed (pass straight to apply_seed()
    right before the run starts).
    """
    return st.text_input(
        "Random seed (optional)",
        value="",
        key=f"{key_prefix}_seed",
        placeholder="e.g. 42 -- leave blank for a new random run",
        help=(
            "Leave blank for normal random behavior. Set an integer to make this run fully "
            "reproducible -- same bits, same noise, same Doppler phase, every time. If you leave "
            "it blank, the app still picks a random seed internally and shows it to you after the "
            "run, so you can paste it back in later to regenerate that exact result for a report."
        ),
    )


def apply_seed(seed_text: str) -> int:
    """
    Seed numpy's global RNG for the upcoming run. If seed_text is a
    non-empty integer string, use it directly. If blank, auto-generate
    a fresh random seed from OS entropy (via SeedSequence) and seed
    with that instead -- so even an "unset" run is still exactly
    reproducible after the fact, as long as the effective seed is
    reported back to the user (which every call site here does).

    Returns the effective integer seed that was applied -- log this
    alongside the run's results.
    """
    if seed_text is not None and str(seed_text).strip() != "":
        seed = int(str(seed_text).strip())
    else:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    np.random.seed(seed)
    return seed


def seed_text_is_valid(seed_text: str) -> bool:
    """
    Cheap format check for a seed field's text -- blank or a valid
    integer -- WITHOUT calling apply_seed() (so it has no RNG side
    effects). Meant to be checked right after a Run button is pressed,
    so an invalid seed can cleanly stop just that button's own logic
    (by setting the button's own flag to False) instead of calling
    st.stop(), which would halt the ENTIRE script for that rerun --
    since Streamlit tabs all execute linearly regardless of which tab
    is visually selected, st.stop() here would blank out every other
    tab too, not just the one with the bad seed.
    """
    if seed_text is None or str(seed_text).strip() == "":
        return True
    try:
        int(str(seed_text).strip())
        return True
    except ValueError:
        return False

def reset_parameters_button(key_suffix: str):
    """
    A local 'Reset Parameters' button for a specific tab.
    Clears every widget's value back to default while retaining session state.

    Uses st.session_state.clear() (Streamlit's own full-reset method) rather
    than a manual key-by-key loop -- functionally the same, but is the
    officially documented way to guarantee every entry is gone. This only
    resets widgets that have an explicit key= (Streamlit doesn't expose
    unkeyed widgets' values through session_state at all, so give any
    widget an explicit key if it needs to respond to this reset).
    """
    if st.button("Reset Parameters", key=f"reset_params_{key_suffix}"):
        keep = {
            "logged_in": st.session_state.logged_in,
            "username": st.session_state.username,
            "history": st.session_state.history,
        }
        st.session_state.clear()
        st.session_state.update(keep)
        st.rerun()

def stages_for_display(stages: list, impairments: dict) -> list:
    """
    Rain Fade is modeled in impairments/core.py as a reduction of the
    effective SNR passed to the AWGN stage -- it never transforms the
    signal array itself, so run_simulation()'s stage list has nothing
    to show for it (every other impairment gets its own stage because
    each one does transform the signal).

    For DISPLAY ONLY (constellation grids and PSD plots), this inserts
    one extra, clearly-labeled illustrative stage right before
    "Received": the signal scaled by the fade's equivalent amplitude
    attenuation (10**(-fade_db/20)). This is the physically-correct way
    to *picture* what a rain fade actually does to the signal on its
    way in, even though the simulator's own math takes the equivalent
    (and for BER/EVM, identical) shortcut of adding extra noise instead.

    This never touches result["ber"]/result["evm"] or the real stage
    list used anywhere else -- it only affects what gets plotted here.
    """
    if not impairments or "rain_fade" not in impairments or len(stages) < 2:
        return stages
    fade_db = impairments["rain_fade"].get("fade_db", 0)
    if fade_db <= 0:
        return stages
    prior_label, prior_signal = stages[-2]
    atten = 10 ** (-fade_db / 20)
    illustrative_label = f"After Rain Fade ({fade_db:g} dB attenuation)"
    return stages[:-1] + [(illustrative_label, prior_signal * atten)] + [stages[-1]]

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plotly_constellation_grid(stages: list, plot_n: int = 1500, panel_height: int = 340, n_cols: int = None):
    """
    Interactive constellation grid, one panel per (label, complex_symbol_array)
    stage -- replaces the static Matplotlib PNG version. Hovering any point
    shows its exact I/Q coordinates, the equivalent of MATLAB's data cursor,
    which a plain image can never provide no matter how it's styled.

    Parameters
    ----------
    stages : list of (label, np.ndarray) tuples
    plot_n : int
        Max points plotted per panel (matches the existing point caps used
        elsewhere to keep figures responsive).
    panel_height : int
        Height per row, in pixels.
    n_cols : int, optional
        Force a specific column count (e.g. 2 for before/after pairs).
        Defaults to min(len(stages), 4), matching the old grid layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n = len(stages)
    if n_cols is None:
        n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[label for label, _ in stages])
    for i, (label, syms) in enumerate(stages):
        row = i // n_cols + 1
        col = i % n_cols + 1
        pts = np.asarray(syms)[:plot_n]
        fig.add_trace(
            go.Scattergl(
                x=pts.real, y=pts.imag, mode="markers",
                marker=dict(size=5, opacity=0.6, color="#4C78A8"),
                hovertemplate="I=%{x:.4f}<br>Q=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=row, col=col,
        )
        axis_num = i + 1
        x_id = "x" if axis_num == 1 else f"x{axis_num}"
        fig.update_xaxes(range=[-1.5, 1.5], zeroline=True, row=row, col=col)
        fig.update_yaxes(range=[-1.5, 1.5], zeroline=True, scaleanchor=x_id, scaleratio=1, row=row, col=col)
    fig.update_layout(
        height=panel_height * n_rows,
        margin=dict(t=40, b=10, l=10, r=10),
        template="plotly_dark",
    )
    return fig


def matplotlib_constellation_grid(stages: list, plot_n: int = 1500, n_cols: int = None, panel_size: float = 3.1):
    """
    Static Matplotlib constellation grid -- the DEFAULT renderer.

    One panel per (label, complex_symbol_array) stage, laid out in the same
    row/column shape as plotly_constellation_grid() so the two are visual
    drop-ins for each other -- only the rendering technology differs, never
    the data being shown. Matches the plain white-background style used
    for every other Matplotlib figure in this app (BER/EVM sweeps, eye
    diagrams, PSD) and in the project report.

    Parameters mirror plotly_constellation_grid(): plot_n caps points per
    panel, n_cols defaults to min(len(stages), 4).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(stages)
    if n == 0:
        fig, ax = plt.subplots(figsize=(panel_size, panel_size))
        ax.axis("off")
        return fig
    if n_cols is None:
        n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
    )
    for i, (label, syms) in enumerate(stages):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        pts = np.asarray(syms)[:plot_n]
        ax.scatter(pts.real, pts.imag, s=10, alpha=0.6, color="#4C78A8", edgecolors="none")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.75", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.3)
        # Plotly subplot titles use "<br>" for a line break -- normalize
        # to a real newline so multi-line labels (e.g. "Config A (...)\nBER=...")
        # render correctly here too.
        ax.set_title(label.replace("<br>", "\n"), fontsize=9)
        ax.tick_params(labelsize=7)
    # Blank out any leftover panels in an incomplete last row.
    for j in range(n, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis("off")
    fig.tight_layout()
    return fig


def render_constellation_grid(stages: list, plot_n: int = 1500, n_cols: int = None, key: str = "constellation"):
    """
    Single entry point used by every tab to display a constellation grid.

    Shows its OWN local "Interactive constellation (Plotly)" checkbox right
    above the plot, in that tab's output section -- not a global sidebar
    setting. Unchecked (default) renders the static Matplotlib grid, with a
    "Download constellation (PNG)" button beneath it (Plotly's own camera
    icon covers that need when the checkbox is on). Check it to switch just
    this result to the interactive Plotly version. Each call site gets an
    independent checkbox (keyed off `key`), so ticking it in Single Run has
    no effect on Compare Configurations, Equalizer, etc.

    Both branches are built from the exact same `stages` data, so the
    checkbox only changes *how* a result is shown, never *what* is shown.
    """
    use_plotly = st.checkbox(
        "\U0001F4CA Interactive constellation (Plotly)",
        value=False,
        key=f"{key}_interactive_toggle",
        help=(
            "Off (default): fast, static Matplotlib constellation grid -- matches the plots used "
            "throughout the rest of this app and in the project report. On: interactive Plotly "
            "version instead -- hover any point for its exact I/Q value and zoom/pan each panel."
        ),
    )
    if use_plotly:
        fig = plotly_constellation_grid(stages, plot_n=plot_n, n_cols=n_cols)
        st.plotly_chart(fig, use_container_width=True, key=key)
        st.caption("Hover any point to see its exact I/Q value. Use the camera icon (top-right on hover) to save as PNG.")
    else:
        fig = matplotlib_constellation_grid(stages, plot_n=plot_n, n_cols=n_cols)
        st.pyplot(fig)
        st.download_button(
            "\U0001F4E5 Download constellation (PNG)",
            fig_to_png_bytes(fig),
            f"{key}.png",
            "image/png",
            key=f"{key}_png_download",
        )
        plt.close(fig)
        st.caption("Static Matplotlib rendering (default). Check the box above for an interactive Plotly version with hover tooltips and per-panel zoom.")


def resolve_rotation_ambiguity(rx_symbols: np.ndarray, padded_bits: np.ndarray, M: int):
    """
    Blind M-th-power / decision-directed carrier recovery can only track
    phase up to an unknown multiple of 2*pi/M -- it strips exactly that
    much information to remove the modulation. Landing on the "wrong"
    one of the M rotational states flips bits consistently, which looks
    like the loop failed even though it tracked the drift correctly.

    A real receiver resolves this with a short known training preamble.
    We don't have one here, so we emulate the same effect for a fair
    BER comparison: try correcting by each of the M possible fixed
    offsets and report the one that actually matches the transmitted
    data -- i.e. what a receiver WOULD lock onto given a preamble.

    Returns (best_ber, best_corrected_symbols).
    """
    best_ber = None
    best_symbols = rx_symbols
    for m in range(M):
        offset = 2 * np.pi * m / M
        candidate = rx_symbols * np.exp(-1j * offset)
        ber = np.mean(padded_bits != psk_symbols_to_bits(candidate, M))
        if best_ber is None or ber < best_ber:
            best_ber = ber
            best_symbols = candidate
    return best_ber, best_symbols

# ====================================================================
# Impairment knowledge base -- plain-English description + governing
# equation for each impairment. Shown via an expander ("Know about this
# impairment") right under the control when the impairment is enabled,
# instead of a hover tooltip.
# ====================================================================
IMPAIRMENT_INFO = {
    "iq_imbalance": {
        "description": (
            "Happens at the transmitter's quadrature modulator when the I (in-phase) and Q "
            "(quadrature) paths don't have identical gain, and aren't perfectly 90 degrees apart. "
            "This warps the constellation into an ellipse instead of a circle."
        ),
        "equation": r"g = 10^{A_{dB}/20} \qquad I_{out} = I_{in} \qquad Q_{out} = g\left(Q_{in}\cos\phi + I_{in}\sin\phi\right)",
        "legend": "A_dB = amplitude imbalance (dB), φ = phase imbalance (deviation from 90°)",
    },
    "amam": {
        "description": (
            "Saleh AM-AM model of a Traveling Wave Tube Amplifier (TWTA) / Solid-State Power "
            "Amplifier (SSPA). As input amplitude r grows, the amplifier gain compresses, pulling "
            "high-radius (high-power) symbols inward -- the classic amplifier saturation curve."
        ),
        "equation": r"A(r) = \dfrac{\alpha_a\, r}{1 + \beta_a\, r^{2}}",
        "legend": "r = |input symbol|, α_a, β_a = Saleh AM-AM model coefficients",
    },
    "ampm": {
        "description": (
            "Saleh AM-PM model -- the same nonlinear amplifier also twists the phase of a symbol by "
            "an amount that grows with its input power, on top of the AM-AM amplitude compression."
        ),
        "equation": r"\Phi(r) = \dfrac{\alpha_{\phi}\, r^{2}}{1 + \beta_{\phi}\, r^{2}}",
        "legend": "r = |input symbol|, α_φ, β_φ = Saleh AM-PM model coefficients",
    },
    "doppler": {
        "description": (
            "Relative motion between the satellite and ground station shifts the carrier frequency "
            "(the Doppler effect). Modeled as a phase ramp that advances by a fixed amount every "
            "symbol -- a steady rotation of the whole constellation over time, unlike phase noise "
            "which jitters randomly."
        ),
        "equation": r"\theta_n = 2\pi\, \Delta f_{norm}\, n \qquad s_n^{out} = s_n^{in}\, e^{j\theta_n}",
        "legend": "Δf_norm = frequency offset normalized to symbol rate (cycles/symbol), n = symbol index",
    },
    "phase_noise": {
        "description": (
            "Local oscillator instability adds a small random phase error to every symbol, "
            "independently. This blurs the constellation points angularly (spreads them along an "
            "arc) rather than shifting them all the same way."
        ),
        "equation": r"\theta_n \sim \mathcal{N}(0,\ \sigma^2) \qquad s_n^{out} = s_n^{in}\, e^{j\theta_n}",
        "legend": "σ = phase noise standard deviation (radians), one random draw per symbol",
    },
    "rain_fade": {
        "description": (
            "At Ku/Ka-band, rain along the propagation path attenuates the signal. The attenuation "
            "A(p) shown above is computed from the actual ITU-R P.618-13 rain attenuation procedure "
            "(slant path length, horizontal reduction factor, vertical adjustment factor, effective "
            "path length) combined with ITU-R P.838-3 specific attenuation (frequency- and "
            "polarization-dependent k, \u03b1 coefficients) -- not a hand-picked number. Rain fade doesn't "
            "distort the constellation's shape by itself -- it simply eats into the link budget, "
            "leaving less SNR margin before AWGN is added."
        ),
        "equation": r"SNR_{eff}(dB) = SNR_{clear}(dB) - A(p) \qquad \gamma_R = k\,R^\alpha,\ \ A(p) = \gamma_R \cdot L_e",
        "legend": "A(p) = predicted rain attenuation exceeded for p% of an average year (dB, ITU-R P.618-13); γ_R = specific attenuation (dB/km, ITU-R P.838-3); L_e = effective path length (km)",
    },
}

# Short, consistent display names for use in plot titles/legends -- e.g.
# "AM-AM, Doppler" instead of a bare impairment count like "2 impairment(s)".
IMPAIRMENT_SHORT_NAME = {
    "iq_imbalance": "IQ Imbalance",
    "group_delay": "Group Delay",
    "amam": "AM-AM",
    "ampm": "AM-PM",
    "doppler": "Doppler",
    "phase_noise": "Phase Noise",
    "rain_fade": "Rain Fade",
}

def impairment_names_str(cfg: dict) -> str:
    """Human-readable, comma-joined list of enabled impairments for a config
    dict, e.g. 'AM-AM, Doppler'. Returns 'Clean (no impairments)' if empty."""
    if not cfg:
        return "Clean (no impairments)"
    return ", ".join(IMPAIRMENT_SHORT_NAME.get(k, k) for k in cfg.keys())

# Group Delay's equation depends on which model (Linear / Parabolic) is
# selected, so it's kept separate and picked dynamically.
GD_INFO = {
    "description": (
        "Frequency-dependent delay across the channel bandwidth smears symbols into their "
        "neighbors (inter-symbol interference). **Linear** GD tilts the delay steadily from one "
        "band edge to the other. **Parabolic** GD instead bumps the delay up symmetrically toward "
        "*both* band edges, leaving the band center relatively undistorted -- a different, and very "
        "common, real-filter signature."
    ),
    "equation": {
        "linear": r"\tau(f) = k \cdot GD_{max} \cdot f \qquad H(f) = e^{-j 2\pi f\, \tau(f)}",
        "parabolic": r"\tau(f) = k \cdot GD_{max} \cdot f^{2} \qquad H(f) = e^{-j 2\pi f\, \tau(f)}",
    },
    "legend": "f = normalized frequency (-1 to 1 across the band), GD_max = worst-case group delay, k = calibration constant",
}


def render_impairment_info(name: str, gd_type: str = None):
    """
    Render an expandable 'Know about this impairment' panel: a short
    plain-English description plus its governing equation. Used instead
    of a hover tooltip so the equation is easy to read and copy.
    """
    if name == "group_delay":
        info = GD_INFO
        equation = info["equation"].get((gd_type or "linear").lower(), info["equation"]["linear"])
    else:
        info = IMPAIRMENT_INFO.get(name)
        if not info:
            return
        equation = info["equation"]

    with st.expander("📖 Know about this impairment & its equation", expanded=False):
        st.markdown(info["description"])
        st.latex(equation)
        if info.get("legend"):
            st.caption(info["legend"])


tab_home, tab_about, tab_single, tab_sweep, tab_compare, tab_advanced, tab_dashboard, tab_history = st.tabs(
    ["Home", "About", "Single Run", "Sweep & Comparison", "Compare Configurations", "Advanced Analysis", "Performance Dashboard", "History"]
)

# ====================================================================
# TAB 0: HOME
# ====================================================================
with tab_home:
    st.markdown(HERO_CSS, unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent("""
        <div class="hero">
        <div style="font-size:3.2rem;">🛰️</div>
        <h1>Satcom RF Impairment Simulator</h1>
        <p>Modeling real-world RF impairments in satellite communication links --
        with live constellation views, BER / EVM sweeps, and regression fits.</p>
        </div>
        """),
        unsafe_allow_html=True,
    )
    st.write("")

    stat_cols = st.columns(4)
    home_stats = [
        ("7", "Impairments modeled"),
        ("5", "M-PSK orders (2 → 32)"),
        ("2", "Equalizer types (ZF / MMSE)"),
        ("6", "Analysis tools"),
    ]
    for col, (num, label), color in zip(stat_cols, home_stats, PALETTE):
        with col:
            st.markdown(stat_chip_html(num, label, color), unsafe_allow_html=True)

    st.write("")
    st.write("")

    home_cards = [
        ("🎛️", "Single Run", "Turn on individual impairments, see a separate constellation plot for every stage of the signal chain."),
        ("📊", "Sweep & Comparison", "Sweep across modulation order, bit count, group delay, and SNR. Export the comparison table as CSV."),
        ("⚖️", "Compare Configurations", "Put two different impairment combinations side by side -- e.g. 1 impairment vs 2."),
        ("🔬", "Advanced Analysis", "Spectrum, Eye Diagram, Equalizer, and Carrier Recovery tools for a closer look at any single impairment."),
        ("📈", "Performance Dashboard", "A live rollup of every run this session -- totals, BER trends, and best/worst results."),
        ("📖", "About", "Full write-up of the signal chain, every impairment's governing equation, and the metrics used."),
        ("🕘", "History", "Every run this session logged automatically, exportable as CSV."),
    ]
    grid_cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(home_cards):
        with grid_cols[i % 3]:
            st.markdown(feature_card_html(icon, title, desc, PALETTE[i % len(PALETTE)]), unsafe_allow_html=True)

    st.info(f"Welcome, {st.session_state.username}! Use the tabs above to get started.")

# ====================================================================
# TAB: ABOUT
# ====================================================================
with tab_about:
    st.markdown(
        f"""
        <div class="feature-card" style="margin-bottom:20px;border-left:4px solid {PALETTE[4]};">
            <div class="fc-icon" style="background:{PALETTE[4]}22;border:1px solid {PALETTE[4]}55;
                 width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
                 justify-content:center;margin-bottom:8px;">🛰️</div>
            <div class="fc-title" style="font-size:1.3rem;color:{PALETTE[4]};">About the Satcom RF Impairment Simulator</div>
            <p>
            This simulator models the real-world radio-frequency (RF) impairments that distort a signal on
            its way through a satellite communication (SATCOM) link -- from the modulator on the ground/on the
            satellite, through the amplifier and free-space channel, to the receiver. It lets you enable
            impairments individually or together, see exactly how each one reshapes the constellation, and
            quantify the damage in BER (Bit Error Rate) and EVM (Error Vector Magnitude).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🔗 Signal chain modeled")
    chain_steps = [
        "1. Modulate (M-PSK)", "2. IQ Imbalance", "3. Group Delay", "4. AM-AM", "5. AM-PM",
        "6. Doppler Shift", "7. Phase Noise", "8. Rain Fade", "9. AWGN", "10. Carrier Recovery", "11. Demodulate",
    ]
    chain_html = "".join(badge_html(step, PALETTE[i % len(PALETTE)]) for i, step in enumerate(chain_steps))
    st.markdown(
        f"""
Impairments are applied in the same physical order they'd occur in a real link:

<div style="margin: 10px 0;">{chain_html}</div>

Every stage that's actually enabled is recorded, so the Single Run tab can show you a separate
constellation snapshot after each one.
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🧩 Impairments modeled")
    st.caption("Open **Know about this impairment & its equation** under any impairment's controls to see its governing equation.")
    about_cards = [
        ("🌀", "IQ Imbalance", "Gain/phase mismatch in the Tx quadrature modulator"),
        ("📡", "Group Delay (Linear / Parabolic)", "Frequency-dependent channel delay → inter-symbol interference"),
        ("📉", "AM-AM", "Amplifier gain compression at high power (Saleh model)"),
        ("🔄", "AM-PM", "Amplifier-induced phase distortion (Saleh model)"),
        ("🛰️", "Doppler / Frequency Offset", "Carrier shift from satellite-ground relative motion"),
        ("📻", "Phase Noise", "Random oscillator phase jitter"),
        ("🌧️", "Rain Fade", "ITU-R P.618-13 / P.838-3 rain attenuation, reduces effective SNR"),
        ("🌫️", "AWGN", "Thermal/channel noise (always applied, like a real channel)"),
    ]
    about_cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(about_cards):
        with about_cols[i % 4]:
            st.markdown(feature_card_html(icon, name, desc, PALETTE[i % len(PALETTE)]), unsafe_allow_html=True)

    st.subheader("📐 Metrics")
    col_ber, col_evm = st.columns(2)
    with col_ber:
        st.markdown("**BER -- Bit Error Rate**")
        st.latex(r"BER = \dfrac{\text{number of bit errors}}{\text{total bits sent}}")
    with col_evm:
        st.markdown("**EVM -- Error Vector Magnitude**")
        st.latex(r"EVM = \sqrt{\dfrac{1}{N}\sum_{n=1}^{N} \left| s_n^{rx} - s_n^{ideal} \right|^2}")

    st.subheader("🗺️ What each tab does")
    st.markdown(
        """
- **Single Run** -- enable any mix of impairments, run once, see BER/EVM and a per-stage constellation grid.
- **Sweep & Comparison** -- sweep modulation order, bit count, group delay, and SNR over a grid; get a results table, BER/EVM curves, and regression fits.
- **Compare Configurations** -- put two impairment setups side by side (e.g. Config A vs Config B).
- **Advanced Analysis** -- Spectrum Analyzer (PSD per stage), Eye Diagram (ISI + SNR visualization), Equalizer (ZF/MMSE inversion of Group Delay), Carrier Recovery (Costas loop tracking of Doppler).
- **History** -- every run this session, exportable as CSV.
        """
    )

    st.subheader("📐 Theoretical Reference")
    st.markdown(
        """
This simulator has been validated in two ways:
- **MATLAB Verification** -- The implemented impairment models (IQ imbalance, group delay, Saleh AM-AM/AM-PM,
  phase noise, AWGN, and carrier phase recovery) were compared against the original MATLAB implementation
  using identical parameters and fixed random seeds. The simulated signal behaviour and performance metrics
  closely matched the MATLAB reference.
- **Communication Theory** -- Under clean AWGN-only conditions, the simulated BER is compared with the
  closed-form theoretical Gray-coded M-PSK BER curve in the **Sweep & Comparison** tab. The simulated and
  theoretical results closely agree at moderate-to-high SNR, confirming the correctness of the modulation,
  channel, and demodulation chain.
        """
    )
    st.latex(r"BER_{theory} \approx \dfrac{2}{k}\, Q\!\left(\sqrt{2\cdot E_s/N_0}\ \sin\dfrac{\pi}{M}\right), \quad k = \log_2 M")
    st.caption("Q(.) is the Gaussian Q-function (upper-tail probability of the standard normal distribution). Exact formula used for BPSK (M=2).")

    st.subheader("⚙️ Tech notes")
    st.markdown(
        """
- Modulation: Gray-coded M-PSK, matching MATLAB's `pskmod`/`pskdemod` conventions.
- Built with Python, NumPy, SciPy, Pandas, Matplotlib, and Streamlit.
        """
    )

# ====================================================================
# Shared impairment control builder (info panels instead of tooltips)
# ====================================================================
def impairment_controls(key_prefix: str, columns=None, exclude: list = None):
    exclude = exclude or []
    impairments = {}
    def _iq():
        if st.checkbox("IQ Imbalance", key=f"{key_prefix}_iq"):
            amp = st.slider("  Amp imbalance (dB)", 0.0, 3.0, 0.5, 0.1, key=f"{key_prefix}_iq_amp")
            phase = st.slider("  Phase imbalance (deg)", 0.0, 15.0, 3.0, 0.5, key=f"{key_prefix}_iq_phase")
            impairments["iq_imbalance"] = {"amp_imb_db": amp, "phase_imb_deg": phase}
            render_impairment_info("iq_imbalance")
    def _gd():
        if st.checkbox("Group Delay", key=f"{key_prefix}_gd"):
            gd_type = st.radio("  GD model", ["Linear", "Parabolic"], key=f"{key_prefix}_gd_type", horizontal=True)
            gd_val = st.slider("  Max group delay (ns)", 0, 300, 100, 10, key=f"{key_prefix}_gd_val")
            impairments["group_delay"] = {"gd_max_ns": gd_val, "gd_type": gd_type.lower()}
            render_impairment_info("group_delay", gd_type=gd_type.lower())
    def _amam():
        if st.checkbox("AM-AM (gain compression)", key=f"{key_prefix}_amam"):
            impairments["amam"] = True
            render_impairment_info("amam")
    def _ampm():
        if st.checkbox("AM-PM (phase distortion)", key=f"{key_prefix}_ampm"):
            impairments["ampm"] = True
            render_impairment_info("ampm")
    def _doppler():
        if st.checkbox("Doppler / Freq. Offset", key=f"{key_prefix}_dop"):
            fo = st.slider("  Freq offset (cycles/symbol)", 0.0, 0.001, 0.0001, 0.00005, key=f"{key_prefix}_dop_val", format="%.5f")
            impairments["doppler"] = {"freq_offset_norm": fo}
            render_impairment_info("doppler")
    def _pn():
        if st.checkbox("Phase Noise", key=f"{key_prefix}_pn"):
            sigma = st.slider("  Std dev (deg)", 0.0, 15.0, 3.0, 0.5, key=f"{key_prefix}_pn_val")
            impairments["phase_noise"] = {"sigma_deg": sigma}
            render_impairment_info("phase_noise")
    def _rain():
        if st.checkbox("Rain Fade (ITU-R P.618-13)", key=f"{key_prefix}_rain"):
            preset_names = list(RAIN_RATE_PRESETS_MM_PER_HR.keys()) + ["Custom"]
            preset = st.selectbox(
                "  Rain intensity (ITU-R P.837 reference values)",
                preset_names, index=3, key=f"{key_prefix}_rain_preset",
                help="Standard reference rain rates exceeded 0.01% of an average year. "
                     "Pick 'Custom' to enter a location-specific R\u2080.\u2080\u2081 from ITU-R P.837 maps or local met data.",
            )
            if preset == "Custom":
                r001 = st.slider("  Rain rate R\u2080.\u2080\u2081 (mm/h, exceeded 0.01% of year)", 0.1, 200.0, 25.0, 0.5, key=f"{key_prefix}_rain_r001")
            else:
                r001 = RAIN_RATE_PRESETS_MM_PER_HR[preset]
                st.caption(f"  R\u2080.\u2080\u2081 = {r001:g} mm/h")

            col_a, col_b = st.columns(2)
            with col_a:
                freq_ghz = st.number_input("  Frequency (GHz)", 1.0, 100.0, 12.0, 0.5, key=f"{key_prefix}_rain_freq")
                el_deg = st.slider("  Elevation angle (deg)", 5.0, 90.0, 45.0, 1.0, key=f"{key_prefix}_rain_el")
                lat_deg = st.slider("  Earth-station latitude (deg)", 0.0, 60.0, 23.0, 0.5, key=f"{key_prefix}_rain_lat")
            with col_b:
                tau_label = st.selectbox("  Polarization", ["Circular (45\u00b0)", "Horizontal (0\u00b0)", "Vertical (90\u00b0)"], key=f"{key_prefix}_rain_tau")
                tau_deg = {"Circular (45\u00b0)": 45.0, "Horizontal (0\u00b0)": 0.0, "Vertical (90\u00b0)": 90.0}[tau_label]
                h_station_km = st.number_input("  Station height a.s.l. (km)", 0.0, 5.0, 0.05, 0.05, key=f"{key_prefix}_rain_hs")
                h_rain_km = st.number_input("  Rain height (km, ITU-R P.839)", 1.0, 8.0, 4.8, 0.1, key=f"{key_prefix}_rain_hr",
                                            help="Mean annual rain height = 0\u00b0C isotherm height + 0.36 km. "
                                                 "~4-5 km is typical for tropical/subtropical latitudes.")
            p_percent = st.select_slider(
                "  Time percentage exceeded, p (%)",
                options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0], value=0.01,
                key=f"{key_prefix}_rain_p",
                help="0.01% is the standard SATCOM link-margin design point (≈53 minutes/year).",
            )

            itu = itu_r_p618_rain_attenuation(freq_ghz, el_deg, tau_deg, lat_deg, h_station_km, h_rain_km, r001, p_percent)
            fade = itu["A_p_db"]
            st.metric("  Predicted rain attenuation A(p)", f"{fade:.2f} dB")
            st.caption(
                f"  Specific attenuation \u03b3_R = {itu['specific_attenuation_db_per_km']:.3f} dB/km · "
                f"slant path = {itu['slant_path_length_km']:.2f} km · "
                f"effective path length = {itu['effective_path_length_km']:.2f} km"
            )
            impairments["rain_fade"] = {"fade_db": fade}
            render_impairment_info("rain_fade")

    all_builders = [
        ("iq_imbalance", _iq), ("group_delay", _gd), ("amam", _amam),
        ("ampm", _ampm), ("doppler", _doppler), ("phase_noise", _pn), ("rain_fade", _rain),
    ]
    builders = [b for name, b in all_builders if name not in exclude]
    if columns:
        for i, builder in enumerate(builders):
            with columns[i % len(columns)]:
                builder()
    else:
        for builder in builders:
            builder()
    return impairments

# ====================================================================
# TAB 1: SINGLE RUN
# ====================================================================
with tab_single:
    reset_parameters_button("single")
    st.header("Simulation Settings")
    M = st.selectbox("Modulation order (M-PSK)", [2, 4, 8, 16, 32], index=2, key="single_M")
    numBits = st.selectbox("Number of bits", NUM_BITS_OPTIONS, index=1, key="single_numBits")
    snr_db = st.slider("SNR (dB)", min_value=0, max_value=25, value=13, key="single_snr")
    st.header("Impairments")
    impairments = impairment_controls("single")
    severity = impairment_severity(impairments)
    if severity:
        st.markdown("**Impairment Severity**")
        severity_lines = "\n".join(f"{label:<12} {bar}" for label, bar in severity.items())
        st.code(severity_lines, language=None)
    output_choice = st.radio("Show metric(s)", ["BER", "EVM", "Both"], index=2, key="single_output_choice")
    seed_text_single = seed_control("single")
    run_button = st.button("Run Simulation", type="primary")
    if run_button and not seed_text_is_valid(seed_text_single):
        st.error("Random seed must be a whole number (e.g. 42) or left blank.")
        run_button = False

    if run_button:
        effective_seed = apply_seed(seed_text_single)

        progress = st.progress(0, text="Starting simulation...")

        def update_single_progress(frac, label):
            progress.progress(frac, text=f"{label}... {int(frac * 100)}%")

        t_start = time.time()
        bits = np.random.randint(0, 2, numBits)
        result = run_simulation(bits, M, snr_db, impairments, progress_callback=update_single_progress)
        elapsed = time.time() - t_start
        progress.empty()

        log_history("Single Run", {"M": M, "numBits": numBits, "SNR": snr_db, "impairments": list(impairments.keys()), "seed": effective_seed, "elapsed_s": round(elapsed, 2)}, result["ber"], result["evm"])

        # Cache everything needed to redisplay this result into session_state.
        # Without this, toggling the constellation's own "Interactive (Plotly)"
        # checkbox below -- or any other widget -- triggers a Streamlit rerun in
        # which `run_button` goes back to False, and an "if run_button:"-only
        # block like this would vanish entirely along with its checkbox. Caching
        # here and displaying from session_state (below, unconditionally) keeps
        # the result on screen across any such rerun.
        st.session_state["single_run_output"] = {
            "result": result,
            "M": M, "numBits": numBits, "snr_db": snr_db,
            "impairments": impairments, "output_choice": output_choice,
            "effective_seed": effective_seed, "elapsed": elapsed,
            "stages": stages_for_display(result["stages"], impairments),
            "plot_n": min(2000, numBits),
        }

    if "single_run_output" in st.session_state:
        o = st.session_state["single_run_output"]
        result = o["result"]
        st.success(f"Simulation completed in {o['elapsed']:.2f} seconds.")
        st.caption(f"🎲 Seed used: **{o['effective_seed']}** -- enter this value in the seed field above to reproduce this exact run later.")

        col1, col2 = st.columns(2)
        if o["output_choice"] in ("BER", "Both"):
            col1.metric("BER", f"{result['ber']:.6f}")
        if o["output_choice"] in ("EVM", "Both"):
            col2.metric("EVM", f"{result['evm']:.6f}")
        if result["effective_snr_db"] != o["snr_db"]:
            st.caption(f"Effective SNR after rain fade: {result['effective_snr_db']:.1f} dB")

        summary_df = pd.DataFrame([{"M": o["M"], "numBits": o["numBits"], "SNR_dB": o["snr_db"], "BER": result["ber"], "EVM": result["evm"]}])
        st.download_button("Download this result (CSV)", summary_df.to_csv(index=False).encode("utf-8"), "single_run_result.csv", "text/csv")

        st.subheader("Constellation (per impairment stage)")
        render_constellation_grid(o["stages"], plot_n=o["plot_n"], key="single_run_constellation")
        if not o["impairments"]:
            st.info("No impairments enabled -- this shows the clean AWGN-only link.")
    else:
        st.info("Set your parameters in the sidebar, then click **Run Simulation**.")

# ====================================================================
# TAB 2: SWEEP & COMPARISON
# ====================================================================
with tab_sweep:
    reset_parameters_button("sweep")
    st.subheader("Sweep across M x numBits x GD x SNR")
    st.caption("Reproduces the BER_matrix / EVM_matrix style sweep from the original MATLAB script.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sweep_M_values = st.multiselect("Modulation order(s) (M-PSK)", [2, 4, 8, 16, 32], default=[16], key="sweep_M")
        sweep_numBits_values = st.multiselect("Number of bits (can pick more than one)", NUM_BITS_OPTIONS, default=[10_000], key="sweep_numBits")
    with col_b:
        gd_min, gd_max = st.slider("Group Delay range (ns)", 0, 300, (0, 300), step=50, key="sweep_gd_range")
        gd_step = st.selectbox("GD step (ns)", [50, 100, 150], index=0, key="sweep_gd_step")
        sweep_gd_type = st.radio("GD model for this sweep", ["Linear", "Parabolic"], horizontal=True, key="sweep_range_gd_type")
    with col_c:
        snr_min, snr_max = st.slider("SNR range (dB)", 0, 25, (6, 18), step=1, key="sweep_snr_range")
        snr_step = st.selectbox("SNR step (dB)", [1, 2, 3], index=1, key="sweep_snr_step")
    
    st.markdown("**Fixed impairments applied across the whole sweep:**")
    st.caption("Doppler / frequency offset has its own dedicated sweep below (different natural scale from GD/SNR) -- set it there instead.")
    fixed_cols = st.columns(4)
    sweep_impairments_template = impairment_controls("sweep", columns=fixed_cols, exclude=["doppler"])
    seed_text_sweep = seed_control("sweep")
    run_sweep_button = st.button("Run Sweep", type="primary")
    if run_sweep_button and not seed_text_is_valid(seed_text_sweep):
        st.error("Random seed must be a whole number (e.g. 42) or left blank.")
        run_sweep_button = False

    if run_sweep_button:
        if not sweep_numBits_values or not sweep_M_values:
            st.error("Pick at least one modulation order and one numBits value.")
        else:
            effective_seed_sweep = apply_seed(seed_text_sweep)

            with st.spinner("Executing grid sweep analysis..."):
                t_start = time.time()
                gd_values = list(range(gd_min, gd_max + 1, gd_step))
                snr_values = list(range(snr_min, snr_max + 1, snr_step))
                total_runs = len(sweep_M_values) * len(sweep_numBits_values) * len(gd_values) * len(snr_values)
                total_bits_processed = total_runs * max(sweep_numBits_values)
                
                if total_bits_processed > 2_000_000_000:
                    st.warning("Large simulation detected -- execution might take longer.")
                
                progress = st.progress(0, text=f"Running 0/{total_runs} simulations...")
                def update_progress(frac):
                    progress.progress(min(frac, 1.0), text=f"Running simulations... {int(frac * 100)}%")

                df = run_sweep(
                    M_values=sweep_M_values,
                    numBits_values=sweep_numBits_values,
                    gd_values_ns=gd_values,
                    snr_values_db=snr_values,
                    iq_params=sweep_impairments_template.get("iq_imbalance"),
                    amam=bool(sweep_impairments_template.get("amam")),
                    ampm=bool(sweep_impairments_template.get("ampm")),
                    pn_params=sweep_impairments_template.get("phase_noise"),
                    rain_params=sweep_impairments_template.get("rain_fade"),
                    gd_type=sweep_gd_type.lower(),
                    progress_callback=update_progress,
                )
                progress.empty()
                elapsed = time.time() - t_start

            st.success(f"Sweep completed in {elapsed:.2f} seconds.")
            st.caption(f"🎲 Seed used: **{effective_seed_sweep}** -- enter this value in the seed field above to reproduce this exact sweep later.")
            st.session_state["sweep_df"] = df
            st.session_state["sweep_impairments_template"] = sweep_impairments_template
            log_history(
                "Sweep",
                {"M": sweep_M_values, "numBits": sweep_numBits_values, "GD_range": (gd_min, gd_max), "SNR_range": (snr_min, snr_max), "impairments": list(sweep_impairments_template.keys()), "seed": effective_seed_sweep, "elapsed_s": round(elapsed, 2)},
                ber=df["BER"].mean(),
                evm=df["EVM"].mean(),
            )

    if "sweep_df" in st.session_state:
        df = st.session_state["sweep_df"]
        st.subheader("Comparison Table")
        st.dataframe(df.style.format({"BER": "{:.6f}", "EVM": "{:.6f}"}), use_container_width=True)
        st.download_button("Download BER/EVM CSV", df.to_csv(index=False).encode("utf-8"), "sweep_results.csv", "text/csv")
        
        col_filt1, col_filt2 = st.columns(2)
        with col_filt1:
            table_numBits = st.selectbox("Plots below use numBits =", sorted(df["numBits"].unique()), key="plot_numBits")
        with col_filt2:
            table_M = st.selectbox("Plots below use M =", sorted(df["M"].unique()), key="plot_M")
            
        df_plot = df[(df["numBits"] == table_numBits) & (df["M"] == table_M)]
        st.subheader(f"BER vs SNR (by Group Delay), M={table_M}, numBits={table_numBits}")

        show_theory = st.checkbox(
            "📐 Overlay theoretical AWGN-only BER (reference)",
            value=True,
            help="Plots the known closed-form M-PSK BER formula (Gray-coded, AWGN-only, no other impairments) as a dashed reference line. "
                 "Your GD=0ns curve (if no other fixed impairments are enabled) should sit close to it -- that agreement is good supporting "
                 "evidence that the simulator's core math behaves as communications theory predicts.",
        )

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for gd in sorted(df_plot["GD_ns"].unique()):
            subset = df_plot[df_plot["GD_ns"] == gd]
            ax1.semilogy(subset["SNR_dB"], subset["BER"].clip(lower=1e-6), "-o", label=f"GD={gd}ns")

        if show_theory:
            theory_snr = np.array(sorted(df_plot["SNR_dB"].unique()), dtype=float)
            theory_ber = theoretical_ber_mpsk(table_M, theory_snr)
            ax1.semilogy(theory_snr, np.clip(theory_ber, 1e-6, None), "k--", linewidth=2, label="Theoretical AWGN")

        ax1.set_xlabel("SNR (dB)")
        ax1.set_ylabel("BER")
        ax1.legend()
        ax1.grid(True, which="both")
        st.pyplot(fig1)

        if show_theory:
            gd_zero_present = 0 in df_plot["GD_ns"].values
            fixed_impairments_on = bool(st.session_state.get("sweep_impairments_template"))
            if gd_zero_present and not fixed_impairments_on:
                theory_snr = np.array(sorted(df_plot["SNR_dB"].unique()), dtype=float)
                theory_ber_vals = theoretical_ber_mpsk(table_M, theory_snr)
                sim_ber_vals = df_plot[df_plot["GD_ns"] == 0].sort_values("SNR_dB")["BER"].values
                valid_mask = theory_ber_vals > 0
                if valid_mask.sum() > 0:
                    pct_diff = np.abs(sim_ber_vals[valid_mask] - theory_ber_vals[valid_mask]) / theory_ber_vals[valid_mask] * 100
                    avg_pct_diff = np.mean(pct_diff)
                    if avg_pct_diff < 15:
                        st.success(f"📐 Close agreement with theory: GD=0ns simulated BER is within {avg_pct_diff:.1f}% of the theoretical reference on average -- consistent with expected communications theory behavior.")
                    else:
                        st.warning(f"📐 GD=0ns simulated BER differs from the theoretical reference by {avg_pct_diff:.1f}% on average -- expected at low SNR/high M (the formula is an approximation there), worth a closer look otherwise.")
            elif fixed_impairments_on:
                st.caption("Note: fixed impairments are enabled for this sweep, so GD=0ns won't match the AWGN-only theory line -- that's expected, not an error. Turn off all fixed impairments and re-run to validate the clean baseline.")

        st.subheader(f"EVM vs Group Delay (by SNR), M={table_M}, numBits={table_numBits}")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for snr in sorted(df_plot["SNR_dB"].unique()):
            subset = df_plot[df_plot["SNR_dB"] == snr]
            ax2.plot(subset["GD_ns"], subset["EVM"], "-s", label=f"SNR={snr}dB")
        ax2.set_xlabel("Group Delay (ns)")
        ax2.set_ylabel("EVM")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        if len(df["M"].unique()) > 1:
            st.subheader(f"BER vs SNR across Modulation Orders (GD=0ns, numBits={table_numBits})")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            for m_val in sorted(df["M"].unique()):
                subset = df[(df["M"] == m_val) & (df["GD_ns"] == 0) & (df["numBits"] == table_numBits)]
                if len(subset):
                    ax3.semilogy(subset["SNR_dB"], subset["BER"].clip(lower=1e-6), "-o", label=f"M={m_val}")
            ax3.set_xlabel("SNR (dB)")
            ax3.set_ylabel("BER")
            ax3.legend()
            ax3.grid(True, which="both")
            st.pyplot(fig3)

        st.subheader("Regression")
        col_h, col_i = st.columns(2)
        with col_h:
            gd_choice = st.selectbox("BER vs SNR fit at GD =", sorted(df_plot["GD_ns"].unique()), key="gd_choice")
            fit = fit_ber_vs_snr(df, gd_choice, table_numBits, table_M)
            if fit:
                slope, intercept, r2 = fit
                st.write(f"log10(BER) = {slope:.4f} * SNR + {intercept:.4f}")
                st.write(f"R-squared = {r2:.4f}")
            else:
                st.write("Not enough non-zero BER points to fit at this GD.")
        with col_i:
            snr_choice = st.selectbox("EVM vs GD fit at SNR =", sorted(df_plot["SNR_dB"].unique()), key="snr_choice")
            fit2_result = fit_evm_vs_gd(df, snr_choice, table_numBits, table_M)
            if fit2_result:
                slope, intercept, r2 = fit2_result
                st.write(f"EVM = {slope:.6f} * GD_ns + {intercept:.4f}")
                st.write(f"R-squared = {r2:.4f}")
            else:
                st.write("Not enough points to fit at this SNR.")

        st.subheader("Constellation Snapshots")
        st.caption("Pick specific SNR / GD points from the sweep above to visually compare their received constellations.")
        gd_options = sorted(df_plot["GD_ns"].unique())
        snr_options = sorted(df_plot["SNR_dB"].unique())
        try:
            snap_picks = st.multiselect(
                "Pick up to 4 (GD, SNR) points",
                [(gd, snr) for gd in gd_options for snr in snr_options],
                default=[(gd_options[0], snr_options[0]), (gd_options[-1], snr_options[-1])][: min(2, len(gd_options) * len(snr_options))],
                format_func=lambda t: f"GD={t[0]}ns, SNR={t[1]}dB",
                max_selections=4,  # requires Streamlit >= 1.34
            )
        except TypeError:
            # Older Streamlit versions don't support max_selections -- fall back
            # to an unrestricted multiselect; the warning below still catches it.
            snap_picks = st.multiselect(
                "Pick up to 4 (GD, SNR) points",
                [(gd, snr) for gd in gd_options for snr in snr_options],
                default=[(gd_options[0], snr_options[0]), (gd_options[-1], snr_options[-1])][: min(2, len(gd_options) * len(snr_options))],
                format_func=lambda t: f"GD={t[0]}ns, SNR={t[1]}dB",
            )
        if len(snap_picks) > 4:
            st.warning(f"You picked {len(snap_picks)} points -- only the first 4 will be shown: {', '.join(f'GD={g}ns/SNR={s}dB' for g, s in snap_picks[:4])}.")
        if st.button("Show Constellations for Selected Points"):
            with st.spinner("Generating selected constellation snapshots..."):
                t_snap = time.time()
                snap_picks = snap_picks[:4]
                tmpl = st.session_state.get("sweep_impairments_template", {})
                bits_snap = np.random.randint(0, 2, table_numBits)
                snap_stages = []
                for gd, snr in snap_picks:
                    snap_impairments = dict(tmpl)
                    if gd > 0:
                        snap_impairments["group_delay"] = {"gd_max_ns": gd, "gd_type": sweep_gd_type.lower()}
                    elif "group_delay" in snap_impairments:
                        del snap_impairments["group_delay"]
                    snap_result = run_simulation(bits_snap, table_M, snr, snap_impairments)
                    n_active = len([k for k in snap_impairments if k != "group_delay" or gd > 0])
                    label = f"GD={gd}ns, SNR={snr}dB<br>{n_active} impairment(s), BER={snap_result['ber']:.4f}"
                    snap_stages.append((label, snap_result["rx_symbols"]))
                elapsed_snap = time.time() - t_snap
            # Cache in session_state -- otherwise toggling the constellation's own
            # "Interactive (Plotly)" checkbox below triggers a rerun in which this
            # button goes back to False, wiping the snapshots it just generated.
            st.session_state["snap_stages"] = snap_stages
            st.session_state["snap_elapsed"] = elapsed_snap

        if "snap_stages" in st.session_state:
            st.success(f"Snapshots generated in {st.session_state['snap_elapsed']:.2f} seconds.")
            render_constellation_grid(st.session_state["snap_stages"], plot_n=1500, n_cols=len(st.session_state["snap_stages"]), key="sweep_snapshot_constellation")
    else:
        st.info("Set your sweep ranges above, then click **Run Sweep**.")

    st.divider()
    st.subheader("📡 Doppler Sweep (separate axis)")
    st.caption(
        "Doppler / frequency offset is normalized to **cycles per symbol**, not nanoseconds like Group Delay -- "
        "so it needs its own, much smaller value range, or the carrier phase wraps around many full rotations "
        "every symbol and destroys the signal at every SNR (BER pins near 0.5 regardless of noise). "
        "Defaults below are pre-scaled correctly. Phase is built from an explicit symbol-rate time axis "
        "(t = n·Ts, phase = φ₀ + 2π·fd·t) with a random initial phase φ₀ each run, like a real receiver."
    )

    col_dop1, col_dop2, col_dop3 = st.columns(3)
    with col_dop1:
        dop_M = st.selectbox("Modulation order (M-PSK)", [2, 4, 8, 16, 32], index=2, key="dop_sweep_M")
        dop_numBits = st.selectbox("Number of bits", NUM_BITS_OPTIONS, index=1, key="dop_sweep_numBits")
    with col_dop2:
        dop_values = st.multiselect(
            "Doppler values to sweep (cycles/symbol)",
            DEFAULT_DOPPLER_SWEEP_VALUES,
            default=DEFAULT_DOPPLER_SWEEP_VALUES,
            key="dop_sweep_values",
            format_func=lambda v: f"{v:.5f}",
        )
        dop_custom = st.text_input(
            "Add custom values (comma-separated, cycles/symbol)",
            value="",
            key="dop_sweep_custom",
            placeholder="e.g. 0.00003, 0.00012",
        )
    with col_dop3:
        dop_snr_min, dop_snr_max = st.slider("SNR range (dB)", 0, 25, (6, 18), step=1, key="dop_sweep_snr_range")
        dop_snr_step = st.selectbox("SNR step (dB)", [1, 2, 3], index=1, key="dop_sweep_snr_step")

    with st.expander("⚙️ Advanced Doppler options"):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            dop_n_trials = st.slider(
                "Trials averaged per point", 1, 8, 3, key="dop_n_trials",
                help="Each (Doppler, SNR) point runs this many independent trials (fresh random phase + noise) and averages BER/EVM -- smooths the curve and firms up the regression fit.",
            )
        with col_adv2:
            dop_residual_walk = st.slider(
                "Residual random-walk phase (deg/symbol std)", 0.0, 1.0, 0.0, 0.05, key="dop_residual_walk",
                help="Optional: adds a small cumulative random-walk phase on top of the deterministic Doppler ramp, approximating residual Doppler left after PLL tracking + oscillator instability. 0 = pure deterministic ramp (default).",
            )

    seed_text_dop = seed_control("dop_sweep")
    run_dop_sweep_button = st.button("Run Doppler Sweep", type="primary", key="run_dop_sweep")
    if run_dop_sweep_button and not seed_text_is_valid(seed_text_dop):
        st.error("Random seed must be a whole number (e.g. 42) or left blank.")
        run_dop_sweep_button = False

    if run_dop_sweep_button:
        all_dop_values = list(dop_values)
        if dop_custom.strip():
            try:
                extra_vals = [float(v.strip()) for v in dop_custom.split(",") if v.strip()]
                bad_vals = [v for v in extra_vals if abs(v) > 0.01]
                if bad_vals:
                    st.error(
                        f"These custom values look like a units mistake: {bad_vals}. "
                        f"freq_offset_norm is cycles/symbol -- keep values under ~0.01, "
                        f"not in the tens/hundreds (that range belongs to Group Delay's ns values)."
                    )
                    all_dop_values = None
                else:
                    all_dop_values += extra_vals
            except ValueError:
                st.error("Couldn't parse custom values -- use comma-separated numbers, e.g. 0.00003, 0.00012")
                all_dop_values = None

        if not all_dop_values:
            if all_dop_values is not None:
                st.error("Pick at least one Doppler value to sweep.")
        else:
            effective_seed_dop = apply_seed(seed_text_dop)

            with st.spinner("Running Doppler sweep..."):
                t_dop = time.time()
                dop_snr_values = list(range(dop_snr_min, dop_snr_max + 1, dop_snr_step))
                progress_dop = st.progress(0, text="Running Doppler sweep...")

                def update_dop_progress(frac):
                    progress_dop.progress(min(frac, 1.0), text=f"Running Doppler sweep... {int(frac * 100)}%")

                dop_df = run_doppler_sweep(
                    M=dop_M,
                    numBits=dop_numBits,
                    doppler_values_norm=sorted(set(all_dop_values)),
                    snr_values_db=dop_snr_values,
                    residual_walk_std_deg=dop_residual_walk,
                    n_trials=dop_n_trials,
                    progress_callback=update_dop_progress,
                )
                progress_dop.empty()
                dop_elapsed = time.time() - t_dop

            st.success(f"Doppler sweep completed in {dop_elapsed:.2f} seconds ({dop_n_trials} trial(s)/point, averaged).")
            st.caption(f"🎲 Seed used: **{effective_seed_dop}** -- enter this value in the seed field above to reproduce this exact sweep later.")
            st.session_state["dop_sweep_df"] = dop_df
            log_history(
                "Doppler Sweep",
                {"M": dop_M, "numBits": dop_numBits, "Doppler_values": sorted(set(all_dop_values)), "SNR_range": (dop_snr_min, dop_snr_max), "n_trials": dop_n_trials, "seed": effective_seed_dop, "elapsed_s": round(dop_elapsed, 2)},
                ber=dop_df["BER"].mean(),
                evm=dop_df["EVM"].mean(),
            )

    if "dop_sweep_df" in st.session_state:
        dop_df = st.session_state["dop_sweep_df"]
        st.markdown("**Doppler Sweep Results**")
        st.caption("BER/EVM are averaged across trials; BER_std/EVM_std show the spread across those trials (smaller = more stable estimate).")
        st.dataframe(
            dop_df.style.format({"Doppler_norm": "{:.5f}", "BER": "{:.6f}", "EVM": "{:.6f}", "BER_std": "{:.6f}", "EVM_std": "{:.6f}"}),
            use_container_width=True,
        )
        st.download_button("Download Doppler sweep CSV", dop_df.to_csv(index=False).encode("utf-8"), "doppler_sweep_results.csv", "text/csv")

        col_dopplot1, col_dopplot2 = st.columns(2)
        with col_dopplot1:
            fig_dop1, ax_dop1 = plt.subplots(figsize=(7, 5))
            for dop in sorted(dop_df["Doppler_norm"].unique()):
                subset = dop_df[dop_df["Doppler_norm"] == dop]
                ax_dop1.semilogy(subset["SNR_dB"], subset["BER"].clip(lower=1e-6), "-o", label=f"Doppler={dop:.5f}")
            ax_dop1.set_xlabel("SNR (dB)")
            ax_dop1.set_ylabel("BER")
            ax_dop1.set_title("BER vs SNR (by Doppler offset)")
            ax_dop1.legend(fontsize=8)
            ax_dop1.grid(True, which="both")
            st.pyplot(fig_dop1)

        with col_dopplot2:
            fig_dop2, ax_dop2 = plt.subplots(figsize=(7, 5))
            for dop in sorted(dop_df["Doppler_norm"].unique()):
                subset = dop_df[dop_df["Doppler_norm"] == dop]
                ax_dop2.plot(subset["SNR_dB"], subset["EVM"], "-s", label=f"Doppler={dop:.5f}")
            ax_dop2.set_xlabel("SNR (dB)")
            ax_dop2.set_ylabel("EVM")
            ax_dop2.set_title("EVM vs SNR (by Doppler offset)")
            ax_dop2.legend(fontsize=8)
            ax_dop2.grid(True)
            st.pyplot(fig_dop2)

        st.markdown("**Regression: BER vs SNR at a chosen Doppler value**")

        _fit_candidates = []
        for _dv in sorted(dop_df["Doppler_norm"].unique()):
            _f = fit_ber_vs_snr_doppler(dop_df, _dv)
            if _f is not None:
                _slope, _intercept, _r2 = _f
                _fit_candidates.append((_dv, _slope, _r2))
        _meaningful = [c for c in _fit_candidates if abs(c[1]) >= 0.02]
        if _meaningful:
            _best_dv, _best_slope, _best_r2 = max(_meaningful, key=lambda c: c[2])
            st.caption(
                f"Best fit among your swept Doppler values (excluding flat/floored ones): "
                f"**{_best_dv:.5f}** cycles/symbol -- R\u00b2={_best_r2:.4f}, slope={_best_slope:.4f}. "
                f"Pick this below for the cleanest BER-vs-SNR curve."
            )
        elif _fit_candidates:
            st.caption(
                "All of your currently swept Doppler values are flat/floored (BER doesn't meaningfully "
                "respond to SNR at any of them) -- add a smaller Doppler value above (e.g. 1e-6 to 1e-5) "
                "and re-run the sweep to get a value with a real log-linear fit."
            )

        dop_choice = st.selectbox("Doppler value =", sorted(dop_df["Doppler_norm"].unique()), key="dop_fit_choice", format_func=lambda v: f"{v:.5f}")
        dop_fit = fit_ber_vs_snr_doppler(dop_df, dop_choice)
        if dop_fit:
            slope, intercept, r2 = dop_fit
            st.write(f"log10(BER) = {slope:.4f} * SNR + {intercept:.4f}")
            st.write(f"R-squared = {r2:.4f}")
            if r2 < 0.9:
                st.warning(
                    f"R\u00b2 = {r2:.4f} (below 0.9) -- try more trials or a wider SNR range above for a cleaner fit, "
                    f"or pick a different Doppler value."
                )
        else:
            st.write("Not enough non-zero BER points to fit at this Doppler value.")
    else:
        st.info("Set your Doppler sweep values above, then click **Run Doppler Sweep**.")

# ====================================================================
# TAB 3: COMPARE CONFIGURATIONS
# ====================================================================
with tab_compare:
    reset_parameters_button("compare")
    st.subheader("Compare two impairment configurations side by side")
    st.caption("e.g. Configuration A = Group Delay only, Configuration B = Group Delay + AM-AM.")
    col_shared1, col_shared2, col_shared3 = st.columns(3)
    with col_shared1:
        cmp_M = st.selectbox("Modulation order (M-PSK)", [2, 4, 8, 16, 32], index=2, key="cmp_M")
        cmp_numBits = st.selectbox("Number of bits", NUM_BITS_OPTIONS, index=1, key="cmp_numBits")
    with col_shared2:
        cmp_snr_min, cmp_snr_max = st.slider("SNR range for BER curves (dB)", 0, 25, (6, 18), key="cmp_snr_range")
        cmp_snr_step = st.selectbox("SNR step (dB)", [1, 2, 3], index=1, key="cmp_snr_step")
    with col_shared3:
        cmp_snapshot_snr = st.slider("SNR for constellation snapshot (dB)", 0, 25, 13, key="cmp_snap_snr")
    
    col_cfgA, col_cfgB = st.columns(2)
    with col_cfgA:
        st.markdown("### Configuration A")
        cfg_a = impairment_controls("cfgA")
    with col_cfgB:
        st.markdown("### Configuration B")
        cfg_b = impairment_controls("cfgB")
        
    seed_text_cmp = seed_control("compare")
    run_compare_button = st.button("Run Comparison", type="primary")
    if run_compare_button and not seed_text_is_valid(seed_text_cmp):
        st.error("Random seed must be a whole number (e.g. 42) or left blank.")
        run_compare_button = False

    if run_compare_button:
        with st.spinner("Executing side-by-side comparison simulation..."):
            run_start_time = time.time()
            effective_seed_cmp = apply_seed(seed_text_cmp)
            bits_cmp = np.random.randint(0, 2, cmp_numBits)
            snr_values_cmp = list(range(cmp_snr_min, cmp_snr_max + 1, cmp_snr_step))
            total_steps = 2 + 2 * len(snr_values_cmp)
            progress = st.progress(0, text="Running comparison...")
            step = 0
            
            result_a = run_simulation(bits_cmp, cmp_M, cmp_snapshot_snr, cfg_a)
            step += 1
            progress.progress(step / total_steps, text=f"Config A snapshot... {int(step / total_steps * 100)}%")
            
            result_b = run_simulation(bits_cmp, cmp_M, cmp_snapshot_snr, cfg_b)
            step += 1
            progress.progress(step / total_steps, text=f"Config B snapshot... {int(step / total_steps * 100)}%")
            
            ber_a, ber_b = [], []
            for snr in snr_values_cmp:
                ber_a.append(run_simulation(bits_cmp, cmp_M, snr, cfg_a)["ber"])
                step += 1
                progress.progress(step / total_steps, text=f"Config A BER curve... {int(step / total_steps * 100)}%")
            for snr in snr_values_cmp:
                ber_b.append(run_simulation(bits_cmp, cmp_M, snr, cfg_b)["ber"])
                step += 1
                progress.progress(step / total_steps, text=f"Config B BER curve... {int(step / total_steps * 100)}%")
            progress.empty()
            elapsed = time.time() - run_start_time

        st.success(f"Comparison completed in {elapsed:.2f} seconds.")
        st.session_state["cmp_results"] = {
            "result_a": result_a, "result_b": result_b,
            "snr_values": snr_values_cmp, "ber_a": ber_a, "ber_b": ber_b,
            "n_a": len(cfg_a), "n_b": len(cfg_b),
            "names_a": impairment_names_str(cfg_a), "names_b": impairment_names_str(cfg_b),
        }
        log_history(
            "Compare Configs",
            {"M": cmp_M, "numBits": cmp_numBits, "cfg_a": list(cfg_a.keys()), "cfg_b": list(cfg_b.keys()), "seed": effective_seed_cmp, "elapsed_s": round(elapsed, 2)},
            ber=None, evm=None,
        )

    if "cmp_results" in st.session_state:
        cr = st.session_state["cmp_results"]
        st.subheader("Constellations")
        cmp_stages = [
            (f"Config A ({cr['names_a']})<br>BER={cr['result_a']['ber']:.4f}, EVM={cr['result_a']['evm']:.4f}", cr["result_a"]["rx_symbols"]),
            (f"Config B ({cr['names_b']})<br>BER={cr['result_b']['ber']:.4f}, EVM={cr['result_b']['evm']:.4f}", cr["result_b"]["rx_symbols"]),
        ]
        render_constellation_grid(cmp_stages, plot_n=1500, n_cols=2, key="compare_constellation")
        
        st.subheader("BER vs SNR: Config A vs Config B")
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ax6.semilogy(cr["snr_values"], np.clip(cr["ber_a"], 1e-6, None), "-o", label=f"Config A ({cr['names_a']})")
        ax6.semilogy(cr["snr_values"], np.clip(cr["ber_b"], 1e-6, None), "-s", label=f"Config B ({cr['names_b']})")
        ax6.set_xlabel("SNR (dB)")
        ax6.set_ylabel("BER")
        ax6.legend()
        ax6.grid(True, which="both")
        st.pyplot(fig6)
        
        cmp_csv = pd.DataFrame({"SNR_dB": cr["snr_values"], "BER_ConfigA": cr["ber_a"], "BER_ConfigB": cr["ber_b"]})
        st.download_button("Download comparison CSV", cmp_csv.to_csv(index=False).encode("utf-8"), "config_comparison.csv", "text/csv")
    else:
        st.info("Set up Configuration A and B above, then click **Run Comparison**.")

# ====================================================================
# TAB 4: ADVANCED ANALYSIS
# ====================================================================
with tab_advanced:
    reset_parameters_button("advanced")
    st.subheader("Advanced Analysis")
    analysis_choice = st.radio(
        "Choose analysis type",
        ["Spectrum Analyzer", "Eye Diagram", "Equalizer (Group Delay)", "Carrier Recovery (Doppler)"],
        horizontal=True,
        key="advanced_analysis_choice",
    )
    st.divider()

    # ---------------- SPECTRUM ANALYZER ----------------
    if analysis_choice == "Spectrum Analyzer":
        st.markdown("**Spectrum Analyzer** -- shows the power spectral density at every enabled stage.")
        st.caption("Look for: spectral regrowth (AM-AM), a shifted spectrum (Doppler), a raised noise floor (AWGN), and phase-noise sidebands around the carrier.")
        col1, col2 = st.columns(2)
        with col1:
            spec_M = st.selectbox("Modulation order (M-PSK)", [2, 4, 8, 16, 32], index=2, key="spec_M")
            spec_numBits = st.selectbox("Number of bits", NUM_BITS_OPTIONS, index=1, key="spec_numBits")
        with col2:
            spec_snr = st.slider("SNR (dB)", 0, 25, 13, key="spec_snr")
        spec_cols = st.columns(4)
        spec_impairments = impairment_controls("spec", columns=spec_cols)
        spec_normalize = st.checkbox(
            "Normalize PSD",
            value=True,
            key="spec_normalize",
            help=(
                "ON (default): each stage's curve is shifted so its own peak sits at 0 dB -- "
                "easiest for comparing spectral SHAPE across stages (e.g. how wide AM-AM's "
                "shoulders are, or where phase-noise sidebands sit), independent of absolute level.\n\n"
                "OFF: true absolute PSD, exactly as computed -- no rescaling. Any impairment that "
                "actually changes signal or noise power (Rain Fade's higher noise floor, AM-AM's "
                "slight average-power change, etc.) shows up directly as a vertical shift between "
                "curves, for every impairment, not just Rain Fade."
            ),
        )
        
        seed_text_spec = seed_control("spec")
        run_spec_button = st.button("Run Spectrum Analysis", type="primary")
        if run_spec_button and not seed_text_is_valid(seed_text_spec):
            st.error("Random seed must be a whole number (e.g. 42) or left blank.")
            run_spec_button = False

        if run_spec_button:
            progress = st.progress(0, text="Starting spectrum analysis...")

            def update_spec_progress(frac, label):
                progress.progress(frac, text=f"{label}... {int(frac * 100)}%")

            spec_start = time.time()
            effective_seed_spec = apply_seed(seed_text_spec)
            bits = np.random.randint(0, 2, spec_numBits)
            result = run_simulation(bits, spec_M, spec_snr, spec_impairments, progress_callback=update_spec_progress)
            spec_elapsed = time.time() - spec_start
            progress.empty()
                
            log_history(
                "Advanced: Spectrum Analyzer",
                {"M": spec_M, "numBits": spec_numBits, "SNR": spec_snr, "impairments": list(spec_impairments.keys()), "seed": effective_seed_spec, "elapsed_s": round(spec_elapsed, 2)},
                ber=result["ber"], evm=result["evm"],
            )
            st.success(f"Completed in {spec_elapsed:.2f} seconds.")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for label, syms in stages_for_display(result["stages"], spec_impairments):
                freqs, psd_db = compute_psd(syms, fs=1.0)
                if spec_normalize:
                    psd_db = psd_db - np.max(psd_db)
                ax.plot(freqs, psd_db, label=label)
            ax.set_xlabel("Normalized Frequency (cycles/symbol)")
            ax.set_ylabel("PSD (dB, normalized to each stage's peak)" if spec_normalize else "PSD (dB, absolute)")
            ax.legend(fontsize=8)
            ax.grid(True)
            st.pyplot(fig)
            st.download_button("Download spectrum plot (PNG)", fig_to_png_bytes(fig), "spectrum.png", "image/png")

    # ---------------- EYE DIAGRAM ----------------
    elif analysis_choice == "Eye Diagram":
        st.markdown("**Eye Diagram** -- shows inter-symbol interference (ISI) far more clearly than a constellation.")
        st.caption("Group Delay is the impairment to try here: watch the eye opening close as you increase it. SNR adds channel noise on top, so you can see ISI and noise closing the eye together.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            eye_M = st.selectbox("Modulation order (M-PSK)", [2, 4, 8, 16, 32], index=1, key="eye_M")
            eye_numBits = st.selectbox("Number of symbols (approx)", [500, 1000, 2000], index=1, key="eye_numBits")
        with col2:
            eye_gd_type = st.radio("Group Delay model", ["Linear", "Parabolic"], key="eye_gd_type", horizontal=True)
            eye_gd_ns = st.slider("Group Delay (ns)", 0, 300, 150, key="eye_gd")
        with col3:
            eye_snr = st.slider("SNR (dB)", 0, 25, 13, key="eye_snr")
            eye_sps = st.selectbox("Samples per symbol", [4, 8, 16], index=1, key="eye_sps")
        with col4:
            eye_beta = st.slider("RRC roll-off (beta)", 0.1, 1.0, 0.35, 0.05, key="eye_beta")

        seed_text_eye = seed_control("eye")
        run_eye_button = st.button("Generate Eye Diagrams", type="primary")
        if run_eye_button and not seed_text_is_valid(seed_text_eye):
            st.error("Random seed must be a whole number (e.g. 42) or left blank.")
            run_eye_button = False

        if run_eye_button:
            progress = st.progress(0, text="Starting eye diagram generation...")
            eye_start = time.time()
            effective_seed_eye = apply_seed(seed_text_eye)

            k = int(np.log2(eye_M))
            bits = np.random.randint(0, 2, eye_numBits * k)
            progress.progress(15, text="Modulating bits... 15%")
            symbols, _ = bits_to_psk_symbols(bits, eye_M)

            noisy_symbols = awgn_channel(symbols, eye_snr)
            progress.progress(35, text="Adding AWGN (clean path)... 35%")
            wf_clean = pulse_shape(noisy_symbols, sps=eye_sps, beta=eye_beta)
            traces_clean = eye_traces(wf_clean, sps=eye_sps)
            progress.progress(55, text="Pulse-shaping clean path... 55%")

            gd_symbols = group_delay(symbols, eye_gd_ns * 1e-9, gd_type=eye_gd_type.lower())
            noisy_gd_symbols = awgn_channel(gd_symbols, eye_snr)
            progress.progress(75, text="Applying Group Delay + AWGN... 75%")
            wf_gd = pulse_shape(noisy_gd_symbols, sps=eye_sps, beta=eye_beta)
            traces_gd = eye_traces(wf_gd, sps=eye_sps)
            progress.progress(95, text="Pulse-shaping Group Delay path... 95%")

            eye_elapsed = time.time() - eye_start
            progress.progress(100, text="Done... 100%")
            progress.empty()

            log_history(
                "Advanced: Eye Diagram",
                {"M": eye_M, "numSymbols": eye_numBits, "GD_ns": eye_gd_ns, "GD_type": eye_gd_type, "SNR": eye_snr, "seed": effective_seed_eye, "elapsed_s": round(eye_elapsed, 2)},
                ber=None, evm=None,
            )
            st.success(f"Completed in {eye_elapsed:.2f} seconds.")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for trace in traces_clean:
                axes[0].plot(trace, color="steelblue", alpha=0.3, linewidth=0.8)
            axes[0].set_title(f"Eye Diagram: No Group Delay (SNR={eye_snr}dB)")
            axes[0].grid(True)
            for trace in traces_gd:
                axes[1].plot(trace, color="firebrick", alpha=0.3, linewidth=0.8)
            axes[1].set_title(f"Eye Diagram: {eye_gd_type} GD = {eye_gd_ns}ns (SNR={eye_snr}dB)")
            axes[1].grid(True)
            fig.tight_layout()
            st.pyplot(fig)
            st.download_button("Download eye diagrams (PNG)", fig_to_png_bytes(fig), "eye_diagrams.png", "image/png")

    # ---------------- EQUALIZER ----------------
    elif analysis_choice == "Equalizer (Group Delay)":
        st.markdown("**Equalizer** -- Zero-Forcing and MMSE both invert the *known* Group Delay channel response.")
        st.caption("This targets Group Delay specifically (a linear, invertible distortion) -- it will NOT fix AM-AM/AM-PM, IQ imbalance, phase noise, or Doppler, since those aren't static linear filters.")
        col1, col2, col3 = st.columns(3)
        with col1:
            eq_M = st.selectbox("Modulation order (M-PSK)", [2, 4, 8, 16, 32], index=2, key="eq_M")
            eq_numBits = st.selectbox("Number of bits", NUM_BITS_OPTIONS, index=1, key="eq_numBits")
        with col2:
            eq_gd_type = st.radio("Group Delay model", ["Linear", "Parabolic"], key="eq_gd_type", horizontal=True)
            eq_gd_ns = st.slider("Group Delay (ns)", 10, 300, 200, key="eq_gd")
            eq_snr = st.slider("SNR (dB)", 0, 25, 13, key="eq_snr")
        with col3:
            eq_type = st.radio("Equalizer type", ["Zero-Forcing (ZF)", "MMSE"], key="eq_type")
            
        seed_text_eq = seed_control("eq")
        run_eq_button = st.button("Run Equalization Comparison", type="primary")
        if run_eq_button and not seed_text_is_valid(seed_text_eq):
            st.error("Random seed must be a whole number (e.g. 42) or left blank.")
            run_eq_button = False

        if run_eq_button:
            progress = st.progress(0, text="Starting equalizer comparison...")
            eq_start = time.time()
            effective_seed_eq = apply_seed(seed_text_eq)

            bits = np.random.randint(0, 2, eq_numBits)
            symbols, padded_bits = bits_to_psk_symbols(bits, eq_M)
            progress.progress(15, text="Modulating bits... 15%")

            gd_symbols = group_delay(symbols, eq_gd_ns * 1e-9, gd_type=eq_gd_type.lower())
            rx = awgn_channel(gd_symbols, eq_snr)
            progress.progress(40, text="Applying Group Delay + AWGN... 40%")

            rx_before = carrier_phase_recovery(rx, symbols)
            ber_before = np.mean(padded_bits != psk_symbols_to_bits(rx_before, eq_M))
            progress.progress(60, text="Measuring BER before equalization... 60%")

            if eq_type == "Zero-Forcing (ZF)":
                rx_eq = zero_forcing_equalize(rx, eq_gd_ns * 1e-9, gd_type=eq_gd_type.lower())
            else:
                snr_linear = 10 ** (eq_snr / 10)
                rx_eq = mmse_equalize(rx, eq_gd_ns * 1e-9, snr_linear, gd_type=eq_gd_type.lower())
            progress.progress(85, text=f"Applying {eq_type}... 85%")

            rx_after = carrier_phase_recovery(rx_eq, symbols)
            ber_after = np.mean(padded_bits != psk_symbols_to_bits(rx_after, eq_M))
            eq_elapsed = time.time() - eq_start
            progress.progress(100, text="Done... 100%")
            progress.empty()
                
            log_history(
                "Advanced: Equalizer",
                {"M": eq_M, "numBits": eq_numBits, "GD_ns": eq_gd_ns, "GD_type": eq_gd_type, "SNR": eq_snr, "equalizer": eq_type, "seed": effective_seed_eq, "elapsed_s": round(eq_elapsed, 2)},
                ber=ber_after, evm=None,
            )

            # Cache in session_state -- otherwise toggling the constellation's own
            # "Interactive (Plotly)" checkbox below triggers a rerun in which
            # `run_eq_button` goes back to False, wiping this whole comparison.
            st.session_state["eq_output"] = {
                "ber_before": ber_before, "ber_after": ber_after,
                "eq_type": eq_type, "elapsed": eq_elapsed,
                "stages": [
                    (f"Before Equalization<br>BER={ber_before:.4f}", rx_before),
                    (f"After {eq_type}<br>BER={ber_after:.4f}", rx_after),
                ],
            }

        if "eq_output" in st.session_state:
            eo = st.session_state["eq_output"]
            st.success(f"Completed in {eo['elapsed']:.2f} seconds.")

            col_a, col_b = st.columns(2)
            col_a.metric("BER before equalization", f"{eo['ber_before']:.6f}")
            col_b.metric("BER after equalization", f"{eo['ber_after']:.6f}", delta=f"{eo['ber_after'] - eo['ber_before']:.6f}")

            render_constellation_grid(eo["stages"], plot_n=1500, n_cols=2, key="equalizer_constellation")

    # ---------------- CARRIER RECOVERY ----------------
    elif analysis_choice == "Carrier Recovery (Doppler)":
        st.markdown("**Carrier Recovery** -- tracks a continuously drifting phase (like Doppler) instead of one static correction.")
        st.caption("M-th power Costas loop works well for M=4/8 but suffers noise enhancement at M=16+. Decision-directed tracks higher orders better, but needs mostly-correct decisions to lock -- very dense constellations (32-PSK) remain hard without a training preamble.")
        st.caption("Note: blind loops (Costas / decision-directed) can only track phase up to an unknown multiple of 2\u03c0/M -- a real receiver resolves which one with a short training preamble. We emulate that resolution step here so the comparison reflects tracking performance, not which of the M rotations the loop happened to land on.")
        col1, col2, col3 = st.columns(3)
        with col1:
            cr_M = st.selectbox("Modulation order (M-PSK)", [4, 8, 16, 32], index=1, key="cr_M")
            cr_numBits = st.selectbox("Number of bits", [1_000, 10_000, 20_000, 100_000, 1_000_000, 10_000_000], index=1, key="cr_numBits")
        with col2:
            cr_snr = st.slider("SNR (dB)", 0, 25, 15, key="cr_snr")
            cr_doppler = st.slider("Doppler freq offset (cycles/symbol)", 0.0, 0.001, 0.0003, 0.00005, format="%.5f", key="cr_dop")
        with col3:
            cr_method = st.radio("Recovery method", ["Static (one-shot)", "Costas Loop (M-th power)", "Decision-Directed"], key="cr_method")
            cr_loop_bw = st.slider("Loop bandwidth", 0.005, 0.05, 0.02, 0.005, key="cr_lbw")
            
        seed_text_cr = seed_control("cr")
        run_cr_button = st.button("Run Carrier Recovery Comparison", type="primary")
        if run_cr_button and not seed_text_is_valid(seed_text_cr):
            st.error("Random seed must be a whole number (e.g. 42) or left blank.")
            run_cr_button = False

        if run_cr_button:
            progress = st.progress(0, text="Starting carrier recovery comparison...")
            cr_start = time.time()
            effective_seed_cr = apply_seed(seed_text_cr)

            k = int(np.log2(cr_M))
            bits = np.random.randint(0, 2, cr_numBits)
            symbols, padded_bits = bits_to_psk_symbols(bits, cr_M)
            progress.progress(15, text="Modulating bits... 15%")

            dop_symbols = doppler_shift(symbols, cr_doppler)
            rx = awgn_channel(dop_symbols, cr_snr)
            progress.progress(35, text="Applying Doppler shift + AWGN... 35%")

            # Static (one-shot) is NOT blind -- it has a known reference, so no
            # rotation ambiguity to resolve.
            rx_static = carrier_phase_recovery(rx, symbols)
            ber_static = np.mean(padded_bits != psk_symbols_to_bits(rx_static, cr_M))
            progress.progress(55, text="Measuring static (one-shot) BER... 55%")

            if cr_method == "Static (one-shot)":
                rx_chosen, ber_chosen = rx_static, ber_static
            elif cr_method == "Costas Loop (M-th power)":
                rx_blind = costas_loop_recovery(rx, cr_M, loop_bandwidth=cr_loop_bw)
                ber_chosen, rx_chosen = resolve_rotation_ambiguity(rx_blind, padded_bits, cr_M)
            else:
                rx_blind = decision_directed_costas(rx, cr_M, loop_bandwidth=cr_loop_bw)
                ber_chosen, rx_chosen = resolve_rotation_ambiguity(rx_blind, padded_bits, cr_M)
            progress.progress(90, text=f"Tracking with {cr_method}... 90%")

            cr_elapsed = time.time() - cr_start
            progress.progress(100, text="Done... 100%")
            progress.empty()
                
            log_history(
                "Advanced: Carrier Recovery",
                {"M": cr_M, "numBits": cr_numBits, "SNR": cr_snr, "doppler": cr_doppler, "method": cr_method, "seed": effective_seed_cr, "elapsed_s": round(cr_elapsed, 2)},
                ber=ber_chosen, evm=None,
            )

            # Cache in session_state -- otherwise toggling the constellation's own
            # "Interactive (Plotly)" checkbox below triggers a rerun in which
            # `run_cr_button` goes back to False, wiping this whole comparison.
            st.session_state["cr_output"] = {
                "ber_static": ber_static, "ber_chosen": ber_chosen,
                "cr_method": cr_method, "elapsed": cr_elapsed,
                "stages": [
                    (f"Before Recovery (static)<br>BER={ber_static:.4f}", rx_static),
                    (f"After {cr_method}<br>BER={ber_chosen:.4f}", rx_chosen),
                ],
            }

        if "cr_output" in st.session_state:
            co = st.session_state["cr_output"]
            st.success(f"Completed in {co['elapsed']:.2f} seconds.")

            col_a, col_b = st.columns(2)
            col_a.metric("BER: Static (one-shot)", f"{co['ber_static']:.6f}")
            col_b.metric(f"BER: {co['cr_method']}", f"{co['ber_chosen']:.6f}", delta=f"{co['ber_chosen'] - co['ber_static']:.6f}")

            render_constellation_grid(co["stages"], plot_n=1500, n_cols=2, key="carrier_recovery_constellation")

# ====================================================================
# TAB: PERFORMANCE DASHBOARD
# ====================================================================
with tab_dashboard:
    st.markdown(feature_card_html(
        "📈", "Performance Dashboard",
        "A live rollup of every simulation you've run this session -- across Single Run, Sweep, Compare, and Advanced Analysis.",
        PALETTE[0],
    ), unsafe_allow_html=True)
    st.write("")

    if not st.session_state.history:
        st.info("No runs yet this session. Go run something in Single Run, Sweep, Compare, or Advanced Analysis -- your stats will show up here.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["BER_num"] = pd.to_numeric(hist_df["BER"], errors="coerce")
        hist_df["EVM_num"] = pd.to_numeric(hist_df["EVM"], errors="coerce")

        total_runs = len(hist_df)
        avg_ber = hist_df["BER_num"].mean()
        avg_evm = hist_df["EVM_num"].mean()
        mode_counts = hist_df["mode"].value_counts()
        top_mode = mode_counts.index[0] if len(mode_counts) else "--"

        dash_stats = [
            (str(total_runs), "Total Runs"),
            (f"{avg_ber:.4f}" if pd.notna(avg_ber) else "--", "Avg BER"),
            (f"{avg_evm:.4f}" if pd.notna(avg_evm) else "--", "Avg EVM"),
            (top_mode, "Most-Used Tool"),
        ]
        dash_stat_cols = st.columns(4)
        for col, (num, label), color in zip(dash_stat_cols, dash_stats, PALETTE):
            with col:
                st.markdown(stat_chip_html(num, label, color), unsafe_allow_html=True)

        st.write("")
        st.write("")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("🧮 Runs by Tool")
            fig_d1, ax_d1 = plt.subplots(figsize=(6, 4))
            bar_colors = (PALETTE * (len(mode_counts) // len(PALETTE) + 1))[: len(mode_counts)]
            ax_d1.bar(mode_counts.index, mode_counts.values, color=bar_colors)
            ax_d1.set_ylabel("Number of runs")
            ax_d1.tick_params(axis="x", rotation=25)
            ax_d1.grid(True, axis="y", alpha=0.3)
            fig_d1.tight_layout()
            st.pyplot(fig_d1)

        with col_right:
            st.subheader("📉 BER Trend (this session)")
            ber_series = hist_df["BER_num"].dropna()
            if len(ber_series):
                fig_d2, ax_d2 = plt.subplots(figsize=(6, 4))
                ax_d2.plot(range(len(ber_series)), ber_series.values, "-o", color=PALETTE[1])
                ax_d2.set_xlabel("Run # (BER-producing runs, chronological)")
                ax_d2.set_ylabel("BER")
                ax_d2.grid(True, alpha=0.3)
                fig_d2.tight_layout()
                st.pyplot(fig_d2)
            else:
                st.info("No BER-producing runs yet.")

        st.subheader("🏆 Best & Worst BER this session")
        valid_ber = hist_df.dropna(subset=["BER_num"])
        if len(valid_ber):
            best_row = valid_ber.loc[valid_ber["BER_num"].idxmin()]
            worst_row = valid_ber.loc[valid_ber["BER_num"].idxmax()]
            col_best, col_worst = st.columns(2)
            with col_best:
                st.markdown(feature_card_html(
                    "✅", "Best BER",
                    f"<b>{best_row['BER_num']:.6f}</b> -- {best_row['mode']} at {best_row['time']}",
                    "#22C55E",
                ), unsafe_allow_html=True)
            with col_worst:
                st.markdown(feature_card_html(
                    "⚠️", "Worst BER",
                    f"<b>{worst_row['BER_num']:.6f}</b> -- {worst_row['mode']} at {worst_row['time']}",
                    "#EF4444",
                ), unsafe_allow_html=True)
        else:
            st.info("No BER-producing runs yet to rank.")

        st.subheader("📋 Full Run Log")
        st.dataframe(hist_df.drop(columns=["BER_num", "EVM_num"]), use_container_width=True)
        st.download_button(
            "Download dashboard data (CSV)",
            hist_df.drop(columns=["BER_num", "EVM_num"]).to_csv(index=False).encode("utf-8"),
            "performance_dashboard.csv", "text/csv",
        )

        st.divider()
        st.subheader("📄 Final Report")
        st.caption("Generates a single self-contained HTML report covering this session: summary stats, the full run log, and (if you've run one) the theoretical BER reference comparison. Open it in any browser, or print to PDF from there.")

        if st.button("Generate Report", type="primary"):
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hist_table_html = hist_df.drop(columns=["BER_num", "EVM_num"]).to_html(index=False, border=0)

            reference_html = ""
            if "sweep_df" in st.session_state:
                reference_html = """
                <h2>Theoretical Reference</h2>
                <p>A Sweep & Comparison run was performed this session. Clean-link (AWGN-only) simulated BER was
                compared against the theoretical M-PSK BER formula on the Sweep & Comparison tab -- see that tab's
                reference-comparison message for the agreement level and percentage match.</p>
                """

            report_html = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <title>Satcom RF Impairment Simulator -- Session Report</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1a1a2e; }}
                    h1 {{ color: #023e7d; }}
                    h2 {{ color: #023e7d; border-bottom: 2px solid #023e7d33; padding-bottom: 6px; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 0.85rem; }}
                    th {{ background: #023e7d; color: white; }}
                    .stat-row {{ display: flex; gap: 20px; margin: 20px 0; }}
                    .stat-box {{ background: #023e7d11; border: 1px solid #023e7d33; border-radius: 10px;
                                 padding: 14px 20px; text-align: center; }}
                    .stat-box .num {{ font-size: 1.6rem; font-weight: 700; color: #023e7d; }}
                    .stat-box .label {{ font-size: 0.8rem; color: #555; }}
                    .footer {{ margin-top: 40px; font-size: 0.75rem; color: #888; }}
                </style>
            </head>
            <body>
                <h1>🛰️ Satcom RF Impairment Simulator -- Session Report</h1>
                <p>Generated: {report_time} &nbsp;|&nbsp; User: {st.session_state.username}</p>

                <h2>Summary</h2>
                <div class="stat-row">
                    <div class="stat-box"><div class="num">{total_runs}</div><div class="label">Total Runs</div></div>
                    <div class="stat-box"><div class="num">{f"{avg_ber:.4f}" if pd.notna(avg_ber) else "--"}</div><div class="label">Avg BER</div></div>
                    <div class="stat-box"><div class="num">{f"{avg_evm:.4f}" if pd.notna(avg_evm) else "--"}</div><div class="label">Avg EVM</div></div>
                    <div class="stat-box"><div class="num">{top_mode}</div><div class="label">Most-Used Tool</div></div>
                </div>

                {reference_html}

                <h2>Full Run Log</h2>
                {hist_table_html}

                <div class="footer">
                    Generated automatically by the Satcom RF Impairment Simulator.
                </div>
            </body>
            </html>
            """

            st.download_button(
                "Download Final Report (HTML)",
                report_html.encode("utf-8"),
                f"satcom_simulator_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                "text/html",
            )
            st.success("Report generated -- click the download button above. Open the file in any browser; you can print it to PDF from there (Ctrl+P → Save as PDF).")


# TAB 5: HISTORY
# ====================================================================
with tab_history:
    reset_parameters_button("history")
    st.subheader("Run History (this session)")
    st.caption("Every run you've made in Single Run, Sweep, Compare, AND Advanced Analysis (Spectrum, Eye Diagram, Equalizer, Carrier Recovery) during this session, most recent last. This resets when the browser tab/session ends -- it is not saved permanently.")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        st.download_button("Download history (CSV)", hist_df.to_csv(index=False).encode("utf-8"), "run_history.csv", "text/csv")
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No runs yet this session. Go run something in Single Run, Sweep, or Compare Configurations.")