"""Shared data/model loading helpers for the multipage Streamlit dashboard."""
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


@st.cache_data
def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_best_model():
    path = PROJECT_ROOT / "models" / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)  # {"name", "model", "feature_cols"}


@st.cache_data
def load_eval_metrics() -> dict | None:
    path = PROJECT_ROOT / "outputs" / "eval_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_test_data() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "processed" / "test_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_data
def load_test_results() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "outputs" / "test_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def pipeline_ready() -> bool:
    return (
        load_best_model() is not None
        and load_test_data() is not None
        and load_eval_metrics() is not None
    )


def is_synthetic_run() -> bool:
    """True if data_collection.py fell back to synthetic demand (no ENTSOE_API_KEY)."""
    return (PROJECT_ROOT / "data" / "raw" / "SYNTHETIC_DATA_WARNING.txt").exists()


# ------------------------------------------------------------------------ #
# Design system -- palette, chrome, chart theme (see dataviz skill).
# Fixed categorical order; never cycled or reassigned per filter.
# ------------------------------------------------------------------------ #
PALETTE = {
    "categorical": [
        "#2a78d6",  # 1 blue
        "#1baf7a",  # 2 aqua
        "#eda100",  # 3 yellow
        "#008300",  # 4 green
        "#4a3aa7",  # 5 violet
        "#e34948",  # 6 red
        "#e87ba4",  # 7 magenta
        "#eb6834",  # 8 orange
    ],
    "surface": "#fcfcfb",
    "page_plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "status": {
        "good": "#0ca30c",
        "warning": "#fab219",
        "serious": "#ec835a",
        "critical": "#d03b3b",
    },
}

FONT_STACK = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def inject_base_css() -> None:
    """Production chrome: hide framework branding, style KPI cards, brand header,
    sidebar nav, hover/motion polish."""
    st.markdown(
        f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        div[data-testid="stDecoration"] {{display: none;}}

        html, body, [class*="css"] {{
            font-family: {FONT_STACK};
        }}

        .block-container {{
            max-width: 1200px;
            padding-top: 4.5rem;
            padding-bottom: 3rem;
        }}

        /* -------------------------------------------------- KPI cards -- */
        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid rgba(11,11,11,0.08);
            border-radius: 10px;
            padding: 0.9rem 1.1rem 0.75rem;
            box-shadow: 0 1px 2px rgba(11,11,11,0.04);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(11,11,11,0.08);
            border-color: rgba(42,120,214,0.35);
        }}
        div[data-testid="stMetricLabel"] > p {{
            color: {PALETTE["ink_secondary"]};
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        div[data-testid="stMetricValue"] {{
            color: {PALETTE["ink"]};
            font-weight: 650;
        }}

        /* -------------------------------------------------- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: {PALETTE["page_plane"]};
            border-right: 1px solid rgba(11,11,11,0.08);
        }}
        section[data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem;
        }}
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.25rem 0.5rem 1.1rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid rgba(11,11,11,0.08);
        }}
        .sidebar-brand .mark {{
            font-size: 1.35rem;
            line-height: 1;
            filter: drop-shadow(0 1px 1px rgba(0,0,0,0.08));
        }}
        .sidebar-brand .name {{
            font-size: 0.98rem;
            font-weight: 700;
            color: {PALETTE["ink"]};
            line-height: 1.15;
        }}
        .sidebar-brand .tagline {{
            font-size: 0.72rem;
            color: {PALETTE["ink_muted"]};
        }}
        [data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {{
            border-radius: 8px !important;
            transition: background 0.14s ease, padding-left 0.14s ease;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(42,120,214,0.08) !important;
            padding-left: 0.9rem !important;
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(42,120,214,0.12) !important;
            font-weight: 650;
        }}
        [data-testid="stSidebarNav"] a[aria-current="page"] span {{
            color: {PALETTE["categorical"][0]} !important;
        }}

        /* -------------------------------------------------- Header ----- */
        .app-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            padding-bottom: 0.9rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid rgba(11,11,11,0.08);
        }}
        .app-header .brand {{
            display: flex;
            align-items: baseline;
            gap: 0.65rem;
        }}
        .app-header .brand-mark {{
            font-size: 1.5rem;
            line-height: 1;
        }}
        .app-header .brand-title {{
            font-size: 1.3rem;
            font-weight: 700;
            color: {PALETTE["ink"]};
        }}
        .app-header .brand-subtitle {{
            font-size: 0.88rem;
            color: {PALETTE["ink_secondary"]};
            margin-left: 0.15rem;
        }}

        /* -------------------------------------------------- Hero band -- */
        .hero-band {{
            background: linear-gradient(135deg, rgba(42,120,214,0.07), rgba(27,175,122,0.05) 55%, rgba(237,161,0,0.05));
            border: 1px solid rgba(42,120,214,0.12);
            border-radius: 16px;
            padding: 1.75rem 1.9rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-band::before {{
            content: "";
            position: absolute;
            top: -60%; right: -10%;
            width: 260px; height: 260px;
            background: radial-gradient(circle, rgba(42,120,214,0.10), transparent 70%);
            pointer-events: none;
        }}

        /* -------------------------------------------------- Badges ----- */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }}
        .status-badge .dot {{
            width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex: none;
        }}
        .status-good .dot {{
            animation: pulse-dot 2s ease-in-out infinite;
        }}
        @keyframes pulse-dot {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(12,163,12,0.35); }}
            50% {{ opacity: 0.75; box-shadow: 0 0 0 4px rgba(12,163,12,0); }}
        }}
        .status-good {{ background: rgba(12,163,12,0.12); color: #006300; }}
        .status-warning {{ background: rgba(250,178,25,0.18); color: #7a5200; }}
        .status-critical {{ background: rgba(208,59,59,0.12); color: #8f1f1f; }}

        /* -------------------------------------------------- Nav cards --- */
        div[data-testid="stPageLink"] {{
            border: 1px solid rgba(11,11,11,0.08);
            border-radius: 12px;
            background: #ffffff;
            padding: 0.15rem 0.4rem;
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }}
        div[data-testid="stPageLink"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(11,11,11,0.10);
            border-color: rgba(42,120,214,0.4);
        }}

        /* -------------------------------------------------- Buttons ----- */
        .stButton > button {{
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(42,120,214,0.25);
        }}

        /* -------------------------------------------------- Footer ------ */
        .app-footer {{
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(11,11,11,0.08);
            color: {PALETTE["ink_muted"]};
            font-size: 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    """Small brand block pinned above the native page nav in the sidebar."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span class="mark">⚡</span>
            <div>
                <div class="name">Demand Forecasting</div>
                <div class="tagline">French grid &middot; MSc AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str = "", icon: str = "⚡") -> None:
    """Consistent branded header used at the top of every page."""
    badge = (
        '<span class="status-badge status-good"><span class="dot"></span>Live</span>'
        if not is_synthetic_run()
        else '<span class="status-badge status-warning"><span class="dot"></span>Synthetic data</span>'
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div class="brand">
                <span class="brand-mark">{icon}</span>
                <span class="brand-title">{title}</span>
                <span class="brand-subtitle">{subtitle}</span>
            </div>
            {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(extra: str = "") -> None:
    metrics = load_eval_metrics()
    model_bit = f"Model: {metrics['best_model'].replace('_', ' ').title()}" if metrics else "Model: not trained yet"
    st.markdown(
        f'<div class="app-footer">Electricity Demand Forecasting &middot; {model_bit}'
        f'{" &middot; " + extra if extra else ""}</div>',
        unsafe_allow_html=True,
    )


def status_badge(label: str, level: str) -> str:
    """level: 'good' | 'warning' | 'critical'"""
    return f'<span class="status-badge status-{level}"><span class="dot"></span>{label}</span>'


def plotly_layout(**overrides) -> dict:
    """Shared Plotly layout: brand font, chart chrome, hairline gridlines."""
    base = dict(
        font=dict(family=FONT_STACK, color=PALETTE["ink"], size=13),
        plot_bgcolor=PALETTE["surface"],
        paper_bgcolor=PALETTE["surface"],
        margin=dict(t=56, l=8, r=8, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=PALETTE["ink_secondary"]),
        ),
        xaxis=dict(
            gridcolor=PALETTE["gridline"], linecolor=PALETTE["baseline"], zeroline=False,
            tickfont=dict(color=PALETTE["ink_muted"]),
        ),
        yaxis=dict(
            gridcolor=PALETTE["gridline"], linecolor=PALETTE["baseline"], zeroline=False,
            tickfont=dict(color=PALETTE["ink_muted"]),
        ),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT_STACK, color=PALETTE["ink"])),
    )
    base.update(overrides)
    return base


def format_mw(value: float) -> str:
    return f"{value:,.0f} MW"


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%"
