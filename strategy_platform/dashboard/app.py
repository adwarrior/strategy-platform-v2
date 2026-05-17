"""
Unified strategy platform — Streamlit dashboard.

Auto-adapts to any registered strategy:
  - Strategy selector in the sidebar (populated from StrategyRegistry)
  - Parameter filters generated from strategy.param_grid
  - IS / Monte Carlo / OOS tabs with charts and tables

Run:
    streamlit run strategy_platform/dashboard/app.py
"""

from __future__ import annotations

import sys
if "streamlit" not in sys.modules:
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "streamlit", "run", __file__] + sys.argv[1:]))

import datetime
import glob
import json
import os
import signal
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter

# Make the package importable when run via `streamlit run`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import strategy_platform  # noqa: F401 — auto-registers all strategies
from strategy_platform import results_store
from strategy_platform.data.loader import (
    INSTRUMENT_META, ONE_MINUTE_SYMBOLS, TICK_DATA_COVERAGE,
    SYMBOL_COVERAGE, ALL_SYMBOLS,
    load_1m, load_5m, load_tick_bars,
)
from strategy_platform.registry import StrategyRegistry
from strategy_platform.optimize.pipeline import _deduplicated_combinations

REPORTS_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports'))

DF_ROW_HEIGHT_PX = 35   # pixel height per dataframe row
SCATTER_CAP = 500       # max rows shown in scatter plots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strategy Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS theme injection — v2 redesign (terminal aesthetic)
st.html("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
/* ---------- Strategy Platform v2 — terminal aesthetic ---------- */

/* Fonts everywhere — but EXEMPT icon glyphs (Material Symbols, etc) */
*, .stApp, .stMarkdown, .stDataFrame, button, input, select, textarea {
    font-family: 'IBM Plex Mono', monospace !important;
}
h1, h2, .topbar-strat, [data-testid="stHeader"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
/* Streamlit's icon font wraps glyphs in spans like material-symbols-rounded /
   material-icons / .stIconMaterial. The * !important above was overriding
   their font-family and rendering the icon NAME as plain text. */
.material-icons, .material-icons-outlined, .material-icons-round,
.material-symbols-rounded, .material-symbols-outlined, .material-symbols-sharp,
.stIconMaterial, [class*="material-symbols"], [class*="material-icons"],
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    letter-spacing: normal !important;
    text-transform: none !important;
}

/* Base body sizing — bump up from Streamlit's tiny default */
.stApp, .stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
    line-height: 1.5 !important;
}

/* Backgrounds */
.stApp { background: #0a0a0c !important; color: #ffffff !important; }
section[data-testid="stSidebar"] {
    background: #070709 !important;
    border-right: 1px solid #1a1a22 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: #070709 !important;
}
[data-testid="stHeader"] { background: #070709 !important; border-bottom: 1px solid #1a1a22; }

/* Block container — slightly more breathing room */
.block-container { padding: 1.25rem 2rem !important; max-width: 100% !important; }

/* Headings — clear hierarchy */
h1 { color: #ffffff !important; letter-spacing: -0.01em; font-size: 30px !important; font-weight: 600 !important; }
h2 { color: #ffffff !important; font-size: 22px !important; font-weight: 600 !important; }
/* h3 = SECTION header — large, white, with green accent bar on the left */
h3 {
    font-size: 18px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    margin: 22px 0 14px 0 !important;
    padding: 10px 0 10px 14px !important;
    border-left: 3px solid #39ff8a !important;
    border-bottom: 1px solid #1f1f2a !important;
    background: linear-gradient(90deg, rgba(57,255,138,0.06) 0%, rgba(57,255,138,0) 60%) !important;
}
/* h4 = sub-header inside a section */
h4 {
    font-size: 13px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #dde0e8 !important;
    font-weight: 600 !important;
    margin-bottom: 10px !important;
    margin-top: 4px !important;
}

/* Tabs — primary navigation, distinctly bigger */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1f1f2a !important;
    gap: 4px !important;
    background: #0d0d12 !important;
    padding: 4px 4px 0 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #8a8e98 !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 14px 26px !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; background: #15151c !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #39ff8a !important;
    border-bottom: 2px solid #39ff8a !important;
    background: #0a1810 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: #39ff8a !important; }

/* Primary buttons — electric green with glow */
.stButton > button[kind="primary"] {
    background: #39ff8a !important;
    color: #020804 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: 0 0 24px rgba(57,255,138,0.3) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    padding: 8px 18px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #50ffaa !important;
    box-shadow: 0 0 32px rgba(57,255,138,0.5) !important;
    color: #020804 !important;
}
.stButton > button[kind="primary"]:active { transform: scale(0.99); }

/* Secondary buttons — terminal style */
.stButton > button:not([kind="primary"]) {
    background: #0e0e14 !important;
    color: #ffffff !important;
    border: 1px solid #1e1e28 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    padding: 6px 14px !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #39ff8a !important;
    color: #39ff8a !important;
    background: #0a1810 !important;
}

/* Inputs — clear borders so they read as inputs */
.stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input,
textarea {
    background: #14141c !important;
    color: #ffffff !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
    padding: 6px 10px !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus, textarea:focus {
    border-color: #39ff8a !important;
    outline: none !important;
    box-shadow: 0 0 0 1px #39ff8a !important;
}
/* Selectbox — distinct dropdown look */
.stSelectbox div[data-baseweb="select"] > div {
    background: #14141c !important;
    color: #ffffff !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
    min-height: 38px !important;
}
.stSelectbox div[data-baseweb="select"]:hover > div { border-color: #39ff8a !important; }
.stSelectbox div[data-baseweb="select"] svg { color: #39ff8a !important; fill: #39ff8a !important; }
/* Dropdown popover */
[data-baseweb="popover"] [role="listbox"] {
    background: #14141c !important;
    border: 1px solid #2a2a36 !important;
}
[data-baseweb="popover"] [role="option"] {
    color: #e2e6ee !important;
    font-size: 14px !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: #1f1f2a !important;
    color: #39ff8a !important;
}

/* Form labels (above inputs) — readable, not microscopic */
label, .stRadio > label, .stCheckbox > label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #e2e6ee !important;
    font-weight: 500 !important;
}
/* Radio/checkbox option text should NOT be uppercase or tiny */
.stRadio [data-baseweb="radio"] label, .stRadio [data-baseweb="radio"] label p,
.stCheckbox label p, .stCheckbox [data-baseweb="checkbox"] label,
[data-baseweb="radio"] div, [data-baseweb="checkbox"] div {
    font-size: 13px !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    color: #ffffff !important;
    font-weight: 400 !important;
}

/* Captions / help text */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
small, .help {
    font-size: 12px !important;
    color: #e2e6ee !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #0e0e14;
    border: 1px solid #1a1a22;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: #39ff8a !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 26px !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    color: #e2e6ee !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* Checkboxes — green accent */
.stCheckbox input[type="checkbox"] { accent-color: #39ff8a; }

/* Dataframes / tables */
[data-testid="stDataFrame"] {
    background: #0e0e14;
    border: 1px solid #1a1a22;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #e2e6ee !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-size: 13px !important;
}

/* Dividers */
hr { border-color: #1a1a22 !important; margin: 1rem 0 !important; }

/* Expander — readable header, distinct surface */
[data-testid="stExpander"] {
    background: #101018 !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 4px !important;
    margin: 8px 0 !important;
}
[data-testid="stExpander"] summary {
    color: #ffffff !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    background: #14141c !important;
}
[data-testid="stExpander"] summary p { font-size: 14px !important; color: #ffffff !important; }
[data-testid="stExpander"] summary:hover { background: #1a1a24 !important; }

/* Bordered containers (st.container(border=True)) — section card look */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0e0e14 !important;
    border: 1px solid #2a2a36 !important;
    border-radius: 4px !important;
    padding: 14px !important;
}

/* Progress bars — green */
.stProgress > div > div > div { background: #39ff8a !important; }

/* Toasts and alerts retain semantic colors but lose default streamlit chrome */
.stAlert { background: #0e0e14 !important; border: 1px solid #1a1a22 !important; border-radius: 0 !important; }
.stAlert p, .stAlert div { font-size: 13px !important; }

/* Scrollbar — thin, minimal */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a32; }
::-webkit-scrollbar-thumb:hover { background: #39ff8a; }
</style>
""")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_dollar(x) -> str:
    return f"${x:,.0f}" if pd.notna(x) else "—"

def fmt_pct(x) -> str:
    return f"{x*100:.1f}%" if pd.notna(x) else "—"

def fmt_float(x, decimals: int = 2) -> str:
    return f"{x:.{decimals}f}" if pd.notna(x) else "—"

def df_height(n_rows: int, row_px: int = DF_ROW_HEIGHT_PX) -> int:
    """Return a pixel height that shows all rows without an inner scrollbar."""
    return (n_rows + 1) * row_px + 3

def config_label(row, param_keys: List[str], max_len: int = 80) -> str:
    label = "  ".join(f"{k}={row[k]}" for k in param_keys if k in row.index)
    return label[:max_len] + "…" if len(label) > max_len else label


_READABLE_LABELS = {
    "_run_display": "Run",
    "config": "Configuration",
    "entry_mode": "Entry Mode",
    "require_retest": "Require Retest",
    "retest_type": "Retest Type",
    "net_pnl": "Net P&L",
    "profit_factor": "Profit Factor",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "win_rate": "Win Rate",
    "max_drawdown": "Max Drawdown",
    "mc_stability": "MC Stability",
    "mc_sharpe_p5": "MC Sharpe P5",
    "mc_pnl_p50": "MC P&L P50",
    "mc_pnl_p5": "MC P&L P5",
    "oos_net_pnl": "OOS Net P&L",
    "oos_sharpe": "OOS Sharpe",
    "oos_sortino": "OOS Sortino",
    "oos_profit_factor": "OOS Profit Factor",
    "oos_win_rate": "OOS Win Rate",
    "oos_max_drawdown": "OOS Max Drawdown",
    "runs_represented": "Runs Represented",
    "run_wins": "Run Wins",
    "configs_in_shortlist": "Configs In Shortlist",
    "parameter": "Parameter",
    "most_common": "Most Common",
    "count": "Count",
    "share": "Share",
    "top_values": "Top Values",
    "value": "Value",
}


def _human_label(key: str, display_names: Optional[Dict[str, str]] = None) -> str:
    if display_names and key in display_names:
        return display_names[key]
    if key in _READABLE_LABELS:
        return _READABLE_LABELS[key]
    text = key.replace("_", " ").strip()
    return text.title() if text else key


def _humanize_columns(df: pd.DataFrame, display_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    renamed = {}
    seen: Dict[str, int] = {}
    for col in df.columns:
        base = _human_label(col, display_names)
        count = seen.get(base, 0)
        renamed[col] = base if count == 0 else f"{base} ({count + 1})"
        seen[base] = count + 1
    return df.rename(columns=renamed)


def _cmp_section_break() -> None:
    st.markdown("<div style='margin: 18px 0;'></div>", unsafe_allow_html=True)
    st.markdown("---")

def load_latest_csv(prefix: str) -> Optional[pd.DataFrame]:
    """Return the most recently modified CSV matching prefix*.csv, or None."""
    files = glob.glob(os.path.join(REPORTS_DIR, f"{prefix}*.csv"))
    if not files:
        return None
    return pd.read_csv(max(files, key=os.path.getmtime))


@st.cache_data(ttl=10)
def list_run_timestamps(strategy_name: str, sym_safe: str) -> List[str]:
    """Return all run timestamps for this strategy, newest first.

    Timestamps are extracted from IS_<strategy>_<sym>_<YYYYMMDD_HHMM>.csv filenames.
    """
    pattern = os.path.join(REPORTS_DIR, f"IS_{strategy_name}_{sym_safe}_*.csv")
    files   = glob.glob(pattern)
    tss = []
    prefix_len = len(f"IS_{strategy_name}_{sym_safe}_")
    for f in files:
        name = os.path.basename(f)           # IS_goldbot7_GC_F_20260320_1212.csv
        ts   = name[prefix_len:-4]           # 20260320_1212
        if len(ts) == 13 and os.path.getsize(f) > len(name):   # skip empty stubs
            tss.append(ts)
    db_tss = results_store.list_optimizer_run_timestamps(strategy_name, sym_safe)
    return sorted(set(tss).union(db_tss), reverse=True)        # newest first


@st.cache_data(ttl=60)
def load_run_csv(strategy_name: str, sym_safe: str, stage: str, ts: str) -> Optional[pd.DataFrame]:
    """Load a stage DataFrame for a specific run timestamp, preferring the shared DB."""
    db_df = results_store.load_optimizer_stage(strategy_name, sym_safe, ts, stage)
    if db_df is not None:
        return db_df
    path = os.path.join(REPORTS_DIR, f"{stage}_{strategy_name}_{sym_safe}_{ts}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _format_run_display(strategy_name: str, sym_safe: str, ts: str) -> str:
    """Human-friendly run label with optional saved label + timestamp."""
    try:
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}  {ts[9:11]}:{ts[11:]}"
    except Exception:
        date_str = ts
    label = _get_run_label(strategy_name, sym_safe, ts)
    return f"{label}  —  {date_str}" if label else date_str


_COMPARE_STAGE_METRICS = {
    "IS": [
        "sharpe", "sortino", "profit_factor", "net_pnl", "win_rate", "max_drawdown",
    ],
    "MC": [
        "mc_stability", "mc_sharpe_p5", "mc_pnl_p50", "mc_pnl_p5",
        "sharpe", "profit_factor", "net_pnl",
    ],
    "OOS": [
        "oos_net_pnl", "oos_sharpe", "oos_sortino", "oos_profit_factor",
        "oos_win_rate", "oos_max_drawdown",
    ],
}

_COMPARE_METRIC_ASC = {
    "max_drawdown": True,
    "oos_max_drawdown": True,
}


def _compare_metric_ascending(metric: str) -> bool:
    return _COMPARE_METRIC_ASC.get(metric, False)


def _sort_for_compare(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return *df* sorted best-first by metric, dropping NaN metric rows when possible."""
    if metric not in df.columns:
        return df
    out = df[df[metric].notna()].copy()
    if out.empty:
        out = df.copy()
    return out.sort_values(metric, ascending=_compare_metric_ascending(metric))


@st.cache_data
def _load_compare_stage_runs(
    strategy_name: str,
    sym_safe: str,
    run_ts_tuple: tuple,
    param_keys_tuple: tuple,
    stage: str,
) -> Dict[str, pd.DataFrame]:
    """Load a stage CSV for each selected run and attach run/config labels."""
    param_keys = list(param_keys_tuple)
    stage_name = stage
    out: Dict[str, pd.DataFrame] = {}
    for ts in run_ts_tuple:
        df = load_run_csv(strategy_name, sym_safe, stage_name, ts)
        if df is None or df.empty:
            continue
        df = df.copy()
        if "config" not in df.columns:
            df.insert(0, "config", df.apply(lambda r: config_label(r, param_keys), axis=1))
        df["_run_ts"] = ts
        df["_run_label"] = _get_run_label(strategy_name, sym_safe, ts)
        df["_run_display"] = _format_run_display(strategy_name, sym_safe, ts)
        out[ts] = df
    return out


def _sync_results_run_picker(shared_key: str, widget_key: str) -> None:
    """Copy a tab-local results-run picker value into the shared selected-run state."""
    st.session_state[shared_key] = st.session_state[widget_key]


@st.cache_data
def _list_stage_runs_with_rows(
    strategy_name: str,
    sym_safe: str,
    run_ts_tuple: tuple,
    stage: str,
) -> List[str]:
    """Return run timestamps whose stage output exists and has at least one row."""
    out: List[str] = []
    for ts in run_ts_tuple:
        df = load_run_csv(strategy_name, sym_safe, stage, ts)
        if df is not None and not df.empty:
            out.append(ts)
    return out


def _load_params_into_backtester(
    row: pd.Series,
    param_keys: List[str],
    source_label: str,
    stage: Optional[str] = None,
) -> None:
    """Populate Backtest tab controls from a result row."""
    for key in param_keys:
        if key in row.index and not pd.isna(row[key]):
            st.session_state[f"bt_{key}"] = row[key]

    stage = (stage or "").upper()
    date_pairs = {
        "IS": ("_is_start", "_is_end"),
        "OOS": ("_oos_start", "_oos_end"),
        "MC": ("_is_start", "_is_end"),
    }
    start_key, end_key = date_pairs.get(stage, ("_data_start", "_data_end"))
    start_val = row.get(start_key, row.get("_data_start"))
    end_val = row.get(end_key, row.get("_data_end"))

    if pd.notna(start_val):
        try:
            st.session_state["bt_start"] = pd.Timestamp(start_val).date()
        except Exception:
            pass
    if pd.notna(end_val):
        try:
            st.session_state["bt_end"] = pd.Timestamp(end_val).date()
        except Exception:
            pass

    st.session_state["bt_loaded_from"] = source_label


def _reset_optimizer_param_state(
    key: str,
    selected_strategy_name: str,
    param_values: Optional[List[Any]] = None,
) -> None:
    """Clear Streamlit widget state for one Configure & Run parameter."""
    for state_key in [
        f"run_grid_{key}",
        f"run_grid_pending_{key}",
        f"_nr_from_{selected_strategy_name}_{key}",
        f"_nr_to_{selected_strategy_name}_{key}",
        f"_nr_step_{selected_strategy_name}_{key}",
        f"_nr_shadow_from_{selected_strategy_name}_{key}",
        f"_nr_shadow_to_{selected_strategy_name}_{key}",
        f"_nr_shadow_step_{selected_strategy_name}_{key}",
        f"_boolradio_{selected_strategy_name}_{key}",
        f"_boolradio_shadow_{selected_strategy_name}_{key}",
        f"_bool_{selected_strategy_name}_{key}",
        f"_boolshadow_{selected_strategy_name}_{key}",
        f"_t_shadow_{selected_strategy_name}_{key}",
    ]:
        st.session_state.pop(state_key, None)

    current_ver = st.session_state.get(f"_tbox_ver_{selected_strategy_name}_{key}", 0)
    st.session_state.pop(f"_tbox_{selected_strategy_name}_{key}_{current_ver}", None)
    st.session_state[f"_tbox_ver_{selected_strategy_name}_{key}"] = current_ver + 1

    if param_values:
        checkable = [v for v in param_values if v != "off"]
        for idx in range(len(checkable)):
            st.session_state.pop(f"_cat_{selected_strategy_name}_{key}_{idx}", None)


def _load_params_into_optimizer(
    row: pd.Series,
    param_keys: List[str],
    selected_strategy_name: str,
    param_grid: Dict[str, List[Any]],
    defaults: Optional[Dict[str, Any]] = None,
    strat_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Populate Configure & Run with definitive single-value selections from a result row."""
    pending_setup_key = f"_optimizer_pending_setup_{selected_strategy_name}"
    pending_setup: Dict[str, Any] = {
        "ui_selections": {},
    }
    if strat_groups:
        pending_setup["group_includes"] = {group_name: False for group_name in strat_groups}
    else:
        pending_setup["group_includes"] = {}

    for key in param_keys:
        value = None
        if key in row.index and not pd.isna(row[key]):
            value = row[key]
        elif defaults and key in defaults:
            value = defaults[key]
        elif key in param_grid and param_grid[key]:
            _pg = param_grid[key]
            value = _pg[0] if not isinstance(_pg, tuple) else _pg[0]

        _pg_def = param_grid.get(key)
        if isinstance(_pg_def, tuple) or value is None:
            matched = []
        else:
            matched = _matching_allowed_values([value], list(_pg_def or []))
        final_values = matched if matched else ([value] if value is not None else [])
        if final_values:
            pending_setup["ui_selections"][key] = final_values

    st.session_state[pending_setup_key] = pending_setup


def _apply_pending_optimizer_setup(
    selected_strategy_name: str,
    param_grid: Dict[str, List[Any]],
    param_keys: List[str],
    strat_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Apply a queued optimizer setup before Configure & Run widgets render."""
    pending_setup_key = f"_optimizer_pending_setup_{selected_strategy_name}"
    pending_setup = st.session_state.pop(pending_setup_key, None)
    if not pending_setup:
        return
    _apply_run_setup_to_optimizer(
        pending_setup,
        selected_strategy_name,
        param_grid,
        param_keys,
        strat_groups,
    )


def _apply_pending_optimizer_include_state(selected_strategy_name: str) -> None:
    """Apply deferred include-checkbox state before Configure & Run widgets render."""
    pending_include_key = f"_optimizer_pending_include_state_{selected_strategy_name}"
    pending_state = st.session_state.pop(pending_include_key, None)
    if not pending_state:
        return
    for state_key, value in pending_state.items():
        st.session_state[state_key] = value


def _config_details_df(
    row: pd.Series,
    param_keys: List[str],
    display_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Return a human-readable parameter/value table for a result row."""
    detail_rows: List[Dict[str, str]] = []
    for key in param_keys:
        if key not in row.index or pd.isna(row[key]):
            continue
        value = row[key]
        if isinstance(value, (np.floating, float)) and float(value).is_integer():
            value = int(value)
        detail_rows.append({
            "Parameter": _human_label(key, display_names),
            "Value": str(value),
        })
    return pd.DataFrame(detail_rows)


def _render_config_details(
    row: pd.Series,
    param_keys: List[str],
    display_names: Optional[Dict[str, str]] = None,
    title: str = "Selected config details",
) -> None:
    """Render a full parameter listing for a selected result row."""
    with st.expander(title, expanded=False):
        details_df = _config_details_df(row, param_keys, display_names)
        if details_df.empty:
            st.info("No parameter details are available for this result.")
        else:
            st.dataframe(
                details_df,
                use_container_width=True,
                hide_index=True,
                height=min(700, df_height(len(details_df))),
            )


def _matching_allowed_values(raw_values: List[Any], allowed_values: List[Any]) -> List[Any]:
    """Return allowed_values that are present in raw_values, preserving allowed order.

    Handles int/float type coercion so that CSV-read float(4.0) matches int(4) in a grid.
    """
    matched: List[Any] = []
    for allowed in allowed_values:
        for raw in raw_values:
            try:
                if pd.isna(raw) and pd.isna(allowed):
                    matched.append(allowed)
                    break
            except Exception:
                pass
            if raw == allowed:
                matched.append(allowed)
                break
            # Numeric coercion: float(4.0) should match int(4) and vice-versa
            try:
                if float(raw) == float(allowed):
                    matched.append(allowed)
                    break
            except (TypeError, ValueError):
                pass
    return matched


def _infer_run_setup_from_frames(
    stage_frames: List[Optional[pd.DataFrame]],
    param_grid: Dict[str, List[Any]],
    defaults: Dict[str, Any],
    strat_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Approximate a run setup from saved result frames when explicit settings were not stored."""
    source_df = next((df for df in stage_frames if df is not None and not df.empty), None)
    ui_selections: Dict[str, List[Any]] = {}
    group_includes: Dict[str, bool] = {}

    for key, allowed_values in param_grid.items():
        inferred: List[Any] = []
        if source_df is not None and key in source_df.columns:
            raw_values = [v for v in source_df[key].tolist() if not pd.isna(v)]
            if isinstance(allowed_values, tuple):
                # Numeric bounds tuple — use raw CSV values directly
                inferred = sorted(set(raw_values))
            else:
                inferred = _matching_allowed_values(raw_values, list(allowed_values))
        if not inferred:
            default_val = defaults.get(key)
            if isinstance(allowed_values, tuple):
                inferred = [default_val] if default_val is not None else [allowed_values[0]]
            else:
                inferred = [default_val] if default_val in allowed_values else [allowed_values[0]]
        ui_selections[key] = inferred

    if strat_groups:
        for group_name, group_keys in strat_groups.items():
            group_includes[group_name] = any(len(ui_selections.get(k, [])) > 1 for k in group_keys if k in ui_selections)

    return {
        "ui_selections": ui_selections,
        "group_includes": group_includes,
        "selected_group": next((g for g, inc in group_includes.items() if inc), next(iter(strat_groups), None)) if strat_groups else None,
        "_inferred": True,
    }


def _apply_run_setup_to_optimizer(
    setup: Dict[str, Any],
    selected_strategy_name: str,
    param_grid: Dict[str, List[Any]],
    param_keys: List[str],
    strat_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Apply a saved or inferred run setup to Configure & Run widget state."""
    for key, state_key in [
        ("data_start", "run_start"),
        ("data_end", "run_end"),
    ]:
        value = setup.get(key)
        if value:
            try:
                st.session_state[state_key] = pd.Timestamp(value).date()
            except Exception:
                pass

    if setup.get("train_pct") is not None:
        try:
            st.session_state["run_train_pct"] = int(round(float(setup["train_pct"]) * 100))
        except Exception:
            pass
    for src_key, dst_key in [
        ("mc_sims", "run_mc_sims"),
        ("mc_top_n", "run_mc_top_n"),
        ("oos_top_n", "run_oos_top_n"),
        ("min_trades", "run_min_trades"),
        ("refresh", "run_refresh"),
    ]:
        if src_key in setup and setup[src_key] is not None:
            st.session_state[dst_key] = setup[src_key]

    pending_include_key = f"_optimizer_pending_include_state_{selected_strategy_name}"
    if strat_groups:
        group_includes = setup.get("group_includes", {})
        _pending_inc = {}
        for group_name in strat_groups:
            _v = bool(group_includes.get(group_name, False))
            _pending_inc[f"_inc_state_{selected_strategy_name}_{group_name}"] = _v
        st.session_state[pending_include_key] = _pending_inc
        selected_group = setup.get("selected_group")
        if selected_group in strat_groups:
            st.session_state[f"selgrp_{selected_strategy_name}"] = selected_group
    else:
        included = any(len(v) > 1 for v in setup.get("ui_selections", {}).values())
        st.session_state[pending_include_key] = {
            f"inc_{selected_strategy_name}_all": included
        }

    for key in param_keys:
        values = setup.get("ui_selections", {}).get(key)
        if not values:
            continue
        grid_def = param_grid.get(key)
        if isinstance(grid_def, tuple):
            # Numeric bounds — values are already a computed list, no preset matching needed
            valid_values = list(values)
        else:
            valid_values = _matching_allowed_values(list(values), list(grid_def or []))
        if not valid_values:
            continue
        _reset_optimizer_param_state(key, selected_strategy_name, grid_def)
        st.session_state[f"run_grid_{key}"] = valid_values
        st.session_state[f"run_grid_pending_{key}"] = valid_values


def _load_saved_run_setup_into_optimizer(
    strategy_name: str,
    sym_safe: str,
    run_ts: str,
    param_grid: Dict[str, List[Any]],
    param_keys: List[str],
    defaults: Dict[str, Any],
    strat_groups: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Load a saved run's setup into Configure & Run and return the source note."""
    setup = results_store.load_optimizer_run_settings(strategy_name, sym_safe, run_ts)
    if setup.get("ui_selections"):
        source_note = "saved run setup"
    else:
        stage_frames = [
            load_run_csv(strategy_name, sym_safe, "IS", run_ts),
            load_run_csv(strategy_name, sym_safe, "MC", run_ts),
            load_run_csv(strategy_name, sym_safe, "OOS", run_ts),
        ]
        setup = _infer_run_setup_from_frames(stage_frames, param_grid, defaults, strat_groups)
        source_note = "inferred from run results"
    _apply_run_setup_to_optimizer(setup, strategy_name, param_grid, param_keys, strat_groups)
    st.session_state[f"_optimizer_loaded_source_{strategy_name}_{sym_safe}"] = (
        f"{_format_run_display(strategy_name, sym_safe, run_ts)} ({source_note})"
    )
    return source_note


def _finish_optimizer_load(selected_strategy_name: str) -> None:
    """Rerun after loading params so Configure & Run re-renders with new state."""
    st.session_state[f"_optimizer_loaded_toast_{selected_strategy_name}"] = True
    st.rerun()


def _unique_existing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Return ordered, de-duplicated column names that exist in *df*."""
    seen = set()
    out: List[str] = []
    for col in columns:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _bt_save_path(strategy_name: str, sym_safe: str, ts: str) -> str:
    return os.path.join(REPORTS_DIR, f"BT_{strategy_name}_{sym_safe}_{ts}.json")


def _bt_db_token(bt_ts: str) -> str:
    return f"db:{bt_ts}"


def _is_bt_db_token(value: str) -> bool:
    return isinstance(value, str) and value.startswith("db:")


def _bt_token_ts(value: str) -> str:
    return value.split(":", 1)[1] if _is_bt_db_token(value) else value


@st.cache_data(ttl=15)
def _list_saved_backtests(strategy_name: str, sym_safe: str) -> List[str]:
    """Return saved single-backtest JSON files, newest first."""
    db_tokens = [_bt_db_token(ts) for ts in results_store.list_backtests(strategy_name, sym_safe)]
    db_ts = {_bt_token_ts(token) for token in db_tokens}
    pattern = os.path.join(REPORTS_DIR, f"BT_{strategy_name}_{sym_safe}_*.json")
    prefix_len = len(f"BT_{strategy_name}_{sym_safe}_")
    files = []
    for f in glob.glob(pattern):
        ts = os.path.basename(f)[prefix_len:-5]  # strip prefix + .json
        if len(ts) == 15 and ts not in db_ts:  # YYYYMMDD_HHMMSS
            files.append(f)
    return db_tokens + sorted(files, reverse=True)


@st.cache_data(ttl=15)
def _list_wf_runs(strategy_name: str, sym_safe: str) -> List[str]:
    """Return WF JSON files for strategy+symbol, sorted newest first."""
    pattern = os.path.join(REPORTS_DIR, f"WF_{strategy_name}_{sym_safe}_*.json")
    prefix_len = len(f"WF_{strategy_name}_{sym_safe}_")
    files = []
    for f in glob.glob(pattern):
        ts = os.path.basename(f)[prefix_len:-5]
        if len(ts) == 15:
            files.append(f)
    return sorted(files, reverse=True)


def _fmt_bt_file(path: str, strategy_name: str, sym_safe: str) -> str:
    if _is_bt_db_token(path):
        ts = _bt_token_ts(path)
    else:
        prefix_len = len(f"BT_{strategy_name}_{sym_safe}_")
        ts = os.path.basename(path)[prefix_len:-5]
    try:
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}  {ts[9:11]}:{ts[11:13]}:{ts[13:]}"
    except Exception:
        date_str = ts
    label = _get_bt_label(path, strategy_name, sym_safe)
    return f"{label}  —  {date_str}" if label else date_str


def _save_backtest(strategy_name: str, symbol: str, params: dict,
                   meta: dict, result: dict) -> Optional[str]:
    """Persist a completed single backtest to disk."""
    import json as _json
    from datetime import datetime as _dt
    sym_safe = symbol.replace('=', '_')
    ts   = _dt.now().strftime('%Y%m%d_%H%M%S')
    path = _bt_save_path(strategy_name, sym_safe, ts)
    trades = result.get('trades')
    trades_list: list = []
    if trades is not None and not trades.empty:
        for rec in trades.to_dict(orient='records'):
            trades_list.append({
                k: (str(v) if not isinstance(v, (bool, int, float, str, type(None))) else v)
                for k, v in rec.items()
            })
    metrics = {k: (float(v) if hasattr(v, '__float__') and not isinstance(v, bool) else v)
               for k, v in result.items() if k != 'trades'}
    data = {
        'params':  params,
        'meta':    meta,
        'metrics': metrics,
        'trades':  trades_list,
        'ts':      ts,
    }
    os.makedirs(REPORTS_DIR, exist_ok=True)
    db_token = None
    try:
        with open(path, 'w') as f:
            _json.dump(data, f, default=str)
        db_token = _bt_db_token(ts)
    except Exception:
        path = None
    try:
        results_store.save_backtest(
            strategy_name=strategy_name,
            symbol=symbol,
            bt_ts=ts,
            payload=data,
        )
        return db_token or path
    except Exception:
        return db_token or path


def _bt_label_path(bt_json_path: str) -> str:
    return bt_json_path[:-5] + '.label'


def _get_bt_label(bt_json_path: str, strategy_name: str, sym_safe: str) -> str:
    if _is_bt_db_token(bt_json_path):
        return results_store.get_backtest_label(strategy_name, sym_safe, _bt_token_ts(bt_json_path))
    try:
        with open(_bt_label_path(bt_json_path)) as f:
            return f.read().strip()
    except Exception:
        return ""


def _set_bt_label(bt_json_path: str, label: str, strategy_name: str, sym_safe: str) -> None:
    if _is_bt_db_token(bt_json_path):
        results_store.set_backtest_label(strategy_name, sym_safe, _bt_token_ts(bt_json_path), label)
        return
    try:
        with open(_bt_label_path(bt_json_path), 'w') as f:
            f.write(label.strip())
    except Exception:
        pass


def _run_label_path(strategy_name: str, sym_safe: str, ts: str) -> str:
    return os.path.join(REPORTS_DIR, f"RUN_{strategy_name}_{sym_safe}_{ts}.label")


@st.cache_data(ttl=60)
def _get_run_label(strategy_name: str, sym_safe: str, ts: str) -> str:
    label = results_store.get_run_label(strategy_name, sym_safe, ts)
    if label:
        return label
    try:
        with open(_run_label_path(strategy_name, sym_safe, ts)) as f:
            return f.read().strip()
    except Exception:
        return ""


def _set_run_label(strategy_name: str, sym_safe: str, ts: str, label: str) -> None:
    results_store.set_run_label(strategy_name, sym_safe, ts, label)
    try:
        with open(_run_label_path(strategy_name, sym_safe, ts), 'w') as f:
            f.write(label.strip())
    except Exception:
        pass
    _get_run_label.clear()
    list_run_timestamps.clear()


@st.cache_data(ttl=300, show_spinner=False)
def _wf_data_coverage(symbol: str, bar_type: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """Return bars-per-day DataFrame for the given symbol+bar_type+range. Cached."""
    try:
        if bar_type == '1m':
            df = load_1m(symbol, start=start_iso, end=end_iso)
        elif bar_type == 'tick':
            df = load_tick_bars(symbol, bar_size=1300, start=start_iso, end=end_iso)
        else:
            df = load_5m(symbol, start=start_iso, end=end_iso)
    except Exception:
        return pd.DataFrame(columns=["date", "bars"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "bars"])
    daily = df.groupby(df.index.date).size().rename("bars").reset_index()
    daily.columns = ["date", "bars"]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def _wf_label_path(strategy_name: str, sym_safe: str, ts: str) -> str:
    return os.path.join(REPORTS_DIR, f"WF_{strategy_name}_{sym_safe}_{ts}.label")


@st.cache_data(ttl=60)
def _get_wf_label(strategy_name: str, sym_safe: str, ts: str) -> str:
    try:
        with open(_wf_label_path(strategy_name, sym_safe, ts)) as f:
            return f.read().strip()
    except Exception:
        return ""


def _set_wf_label(strategy_name: str, sym_safe: str, ts: str, label: str) -> None:
    try:
        with open(_wf_label_path(strategy_name, sym_safe, ts), 'w') as f:
            f.write(label.strip())
    except Exception:
        pass
    _get_wf_label.clear()


def _sync_text_input_state(state_key: str, selected_token: str, saved_value: str) -> None:
    """Refresh a keyed text_input when its backing saved value changes elsewhere."""
    loaded_key = f"{state_key}__loaded"
    loaded_value = st.session_state.get(loaded_key)
    target = (selected_token, saved_value)
    if loaded_value != target:
        st.session_state[state_key] = saved_value
        st.session_state[loaded_key] = target


def _terminate_proc_tree(proc: Optional[subprocess.Popen], output_lines: Optional[List[str]] = None) -> bool:
    """Terminate a subprocess and its children when possible."""
    if proc is None:
        return False
    try:
        if proc.poll() is not None:
            return True

        pgid = None
        if os.name == "posix":
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                return True
        else:
            proc.terminate()

        try:
            proc.wait(timeout=2)
            return True
        except subprocess.TimeoutExpired:
            if output_lines is not None:
                output_lines.append("\n[Still stopping — sending force kill]\n")
            if os.name == "posix":
                try:
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    return True
            else:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                return False
            return True
    except Exception as exc:
        if output_lines is not None:
            output_lines.append(f"\n[Stop failed: {exc}]\n")
        return False


def _load_backtest(path: str, strategy_name: str, sym_safe: str) -> tuple:
    """Load a saved backtest. Returns (params, meta, result_dict)."""
    import json as _json
    if _is_bt_db_token(path):
        parts = path.split(":", 1)
        data = results_store.load_backtest(strategy_name, sym_safe, parts[1]) or {}
        trades_df = pd.DataFrame(data.get('trades', []))
        result    = dict(data.get('metrics', {}))
        result['trades'] = trades_df
        return data.get('params', {}), data.get('meta', {}), result
    with open(path) as f:
        data = _json.load(f)
    trades_df = pd.DataFrame(data.get('trades', []))
    result    = dict(data.get('metrics', {}))
    result['trades'] = trades_df
    return data.get('params', {}), data.get('meta', {}), result

def infer_col_type(series: pd.Series) -> str:
    """Return 'numeric', 'categorical', or 'bool'."""
    if pd.api.types.is_bool_dtype(series):
        return 'bool'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    return 'categorical'


# ---------------------------------------------------------------------------
# Param-grid helpers (Configure & Run tab)
# ---------------------------------------------------------------------------

import re as _re
_TIME_RE = _re.compile(r'^\d{2}:\d{2}$')

def _is_time_list(values: list) -> bool:
    return bool(values) and all(isinstance(v, str) and bool(_TIME_RE.match(v)) for v in values)

_BAR_PERIOD_KEYS = ("tick_bar_size", "bar_period", "minute_bar_period")

def _is_bar_period_param(key: str) -> bool:
    kl = key.lower()
    return any(kl == k or kl.endswith("_" + k) or k in kl
               for k in _BAR_PERIOD_KEYS)


def _param_type(values) -> str:
    # Tuple (min, max, step) → numeric bounds definition
    if isinstance(values, tuple):
        return 'numeric'
    if not values:
        return 'categorical'
    s = values[0]
    if isinstance(s, bool):
        return 'bool'
    if isinstance(s, (int, float)):
        return 'numeric'
    if _is_time_list(values):
        return 'time'
    return 'categorical'

_MONETARY_KEYWORDS = frozenset({"risk", "profit", "loss", "daily", "pnl", "value", "dollar", "limit"})

def _is_monetary(key: str) -> bool:
    return any(kw in key.lower() for kw in _MONETARY_KEYWORDS)

# Per-key decimal place overrides for float params (key substring → decimals).
# Checked in order; first match wins. Falls back to 2dp for monetary, 4dp otherwise.
_FLOAT_DP_OVERRIDES: List[tuple] = [
    # 0 decimal places
    ("fvg_fill_pct",    0),
    ("fill_pct",        0),
    # 1 decimal place
    ("_rr",             1),   # first_leg_rr, second_leg_rr, be_trigger_rr, trail_trigger_rr, trail_distance_rr
    ("rr_",             1),
    ("multiplier",      1),   # st_multiplier
    ("stop_pct",        1),   # stop_pct_or
    ("first_leg_pct",   1),   # first_leg_pct (0.33 etc.)
    ("leg_pct",         1),
    ("fvg_fill",        0),
    ("gap_factor",      4),   # gap_factor_up/down need 4dp (1.0002)
]

def _float_fmt(key: str) -> str:
    key_lower = key.lower()
    for substr, dp in _FLOAT_DP_OVERRIDES:
        if substr in key_lower:
            return f"%.{dp}f"
    if _is_monetary(key):
        return "%.2f"
    return "%.4f"


def _infer_step(values: list):
    if len(values) < 2:
        return 1
    sv = sorted(values)
    gaps = [round(sv[i + 1] - sv[i], 10) for i in range(len(sv) - 1)]
    step = Counter(gaps).most_common(1)[0][0]
    return step if step != 0 else 1


def _make_range(from_val, to_val, step) -> list:
    if step == 0:
        return [from_val]
    step_str = f"{step:.10f}".rstrip('0')
    decimals = len(step_str.split('.')[-1]) if '.' in step_str else 0
    raw = np.arange(float(from_val), float(to_val) + float(step) * 0.5, float(step))
    rounded = [round(float(v), decimals) for v in raw]
    if isinstance(from_val, int) and isinstance(step, int):
        rounded = [int(v) for v in rounded]
    return rounded


def _render_param_input(
    key: str,
    param_grid: Dict[str, List[Any]],
    defaults: Dict[str, Any],
    display_names: Dict[str, str],
    selected_name: str,
    custom_grid: Dict[str, List[Any]],
    prefs: dict = None,
) -> None:
    """Render a single parameter's input widget and write result to custom_grid[key]."""
    widget_key  = f"run_grid_{key}"
    pending_key = f"run_grid_pending_{key}"
    label       = display_names.get(key, key)
    values      = param_grid[key]
    ptype       = _param_type(values)
    default_val = defaults.get(key)

    # ── Bar-period: free-text comma-separated integers ───────────────────────
    if ptype == 'numeric' and _is_bar_period_param(key):
        _bp_shadow = f"_bp_shadow_{selected_name}_{key}"

        # Derive default list from param_grid definition
        if isinstance(values, tuple):
            _bp_min, _bp_max, _bp_step = values
            _bp_default_list = _make_range(_bp_min, _bp_max, _bp_step)
        else:
            _bp_default_list = list(values)

        # Consume pending key into shadow text
        if pending_key in st.session_state:
            _pv = st.session_state.pop(pending_key)
            if _pv:
                st.session_state[_bp_shadow] = ", ".join(str(int(v)) for v in _pv)

        # Initialise shadow: saved prefs → single default value → empty
        if _bp_shadow not in st.session_state:
            _saved_bp = (prefs or _load_prefs()).get(f"saved_sel_{selected_name}_{key}")
            if _saved_bp:
                st.session_state[_bp_shadow] = ", ".join(str(int(v)) for v in _saved_bp)
            elif default_val is not None:
                st.session_state[_bp_shadow] = str(int(default_val))
            else:
                st.session_state[_bp_shadow] = ""

        st.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:1.0rem; font-weight:700;'>{label}</span>"
            + (f"<span style='font-size:0.82rem; color:#888; margin-left:8px;'>default: {default_val}</span>"
               if default_val is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        _bp_raw = st.text_input(
            label, value=st.session_state[_bp_shadow],
            key=f"_bp_text_{selected_name}_{key}",
            label_visibility="collapsed",
            help="Comma-separated integers, e.g. 144, 233, 377, 512",
        )
        st.session_state[_bp_shadow] = _bp_raw
        try:
            _bp_parsed = [int(v.strip()) for v in _bp_raw.split(",") if v.strip()]
            if not _bp_parsed:
                raise ValueError
        except ValueError:
            st.caption(":orange[Enter comma-separated integers, e.g. 144, 233, 377]")
            _bp_parsed = [int(default_val)] if default_val is not None else [_bp_default_list[0]]
        st.session_state[widget_key] = _bp_parsed
        custom_grid[key] = _bp_parsed
        return

    # ── Numeric: From / To / Step ────────────────────────────────────────────
    if ptype == 'numeric':
        if isinstance(values, tuple):
            _min_v, _max_v, _step_v = values
        else:
            _min_v  = min(values)
            _max_v  = max(values)
            _step_v = _infer_step(values)
        _is_float = isinstance(_min_v, float) or isinstance(_step_v, float)
        _fmt      = _float_fmt(key) if _is_float else "%d"
        _k_from   = f"_nr_from_{selected_name}_{key}"
        _k_to     = f"_nr_to_{selected_name}_{key}"
        _k_step   = f"_nr_step_{selected_name}_{key}"

        # Consume pending key — stash in shadow so widget keys aren't pre-set
        _shadow_from = f"_nr_shadow_from_{selected_name}_{key}"
        _shadow_to   = f"_nr_shadow_to_{selected_name}_{key}"
        _shadow_step = f"_nr_shadow_step_{selected_name}_{key}"
        if pending_key in st.session_state:
            _pending_vals = st.session_state.pop(pending_key)
            if _pending_vals:
                st.session_state[_shadow_from] = min(_pending_vals)
                st.session_state[_shadow_to]   = max(_pending_vals)
                st.session_state[_shadow_step] = _infer_step(_pending_vals) if len(_pending_vals) > 1 else _step_v
                # Clear widget keys so number_input picks up new value= on next render
                st.session_state.pop(_k_from, None)
                st.session_state.pop(_k_to,   None)
                st.session_state.pop(_k_step, None)

        # Determine initial values. Live widget state must win on reruns triggered by
        # user edits; shadow values are only for restores/resets/group switches.
        if all(_k in st.session_state for _k in (_k_from, _k_to, _k_step)):
            _init_from = st.session_state[_k_from]
            _init_to   = st.session_state[_k_to]
            _init_step = st.session_state[_k_step]
        elif _shadow_from in st.session_state:
            _init_from = st.session_state[_shadow_from]
            _init_to   = st.session_state.get(_shadow_to,   _max_v)
            _init_step = st.session_state.get(_shadow_step, _step_v)
        elif widget_key in st.session_state and st.session_state[widget_key]:
            _cur = st.session_state[widget_key]
            _init_from = st.session_state.get(_k_from, min(_cur))
            _init_to   = st.session_state.get(_k_to,   max(_cur))
            _init_step = st.session_state.get(_k_step, _infer_step(_cur) if len(_cur) > 1 else _step_v)
        else:
            _step_pref_key = f"step_{selected_name}_{key}"
            _saved_step = (prefs or {}).get(_step_pref_key, _step_v)
            _init_from = st.session_state.get(_k_from, _min_v)
            _init_to   = st.session_state.get(_k_to,   _max_v)
            _init_step = st.session_state.get(_k_step, _saved_step)

        st.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:1.0rem; font-weight:700;'>{label}</span>"
            + (f"<span style='font-size:0.82rem; color:#888; margin-left:8px;'>default: {default_val}</span>"
               if default_val is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        nc1, nc2, nc3 = st.columns(3)
        _step_min = 0.01 if _is_float else 1
        _from_val = nc1.number_input("From", value=float(_init_from) if _is_float else int(_init_from),
                                     step=float(_step_v) if _is_float else int(_step_v),
                                     format=_fmt, key=_k_from)
        _to_val   = nc2.number_input("To",   value=float(_init_to)   if _is_float else int(_init_to),
                                     step=float(_step_v) if _is_float else int(_step_v),
                                     format=_fmt, key=_k_to)
        _init_step_clamped = max(_init_step, _step_min)
        _step_val = nc3.number_input("Step", value=float(_init_step_clamped) if _is_float else int(_init_step_clamped),
                                     step=float(_step_min) if _is_float else 1,
                                     format=_fmt, key=_k_step,
                                     min_value=float(_step_min) if _is_float else 1)
        _generated = _make_range(_from_val, _to_val, _step_val)
        if not _generated and _from_val != _to_val:
            st.caption(f":orange[From > To — no values in range. Swap From/To to generate a sweep.]")
        _generated_safe = _generated if _generated else [_from_val]
        st.session_state[widget_key] = _generated_safe
        custom_grid[key] = _generated_safe
        # Keep shadow keys populated so values survive group switches
        st.session_state[_shadow_from] = _from_val
        st.session_state[_shadow_to]   = _to_val
        st.session_state[_shadow_step] = _step_val
        # Persist custom step to prefs so it becomes the default next time
        _step_pref_key = f"step_{selected_name}_{key}"
        if prefs is not None and _step_val != _step_v and _step_val > 0:
            if prefs.get(_step_pref_key) != _step_val:
                prefs[_step_pref_key] = _step_val
                _save_prefs(prefs)

    # ── Bool: True / False / Optimize radio ──────────────────────────────────
    elif ptype == 'bool':
        _radio_key  = f"_boolradio_{selected_name}_{key}"
        _shadow_key = f"_boolradio_shadow_{selected_name}_{key}"
        _opts = ["True", "False", "Optimize"]

        def _list_to_opt(lst):
            if lst == [True, False] or lst == [False, True]:
                return "Optimize"
            return "True" if (lst and lst[0] is True) else "False"

        if pending_key in st.session_state:
            st.session_state[_shadow_key] = _list_to_opt(st.session_state.pop(pending_key))
        elif widget_key in st.session_state and _shadow_key not in st.session_state:
            st.session_state[_shadow_key] = _list_to_opt(st.session_state[widget_key])

        if _shadow_key not in st.session_state:
            _saved = (prefs or _load_prefs()).get(f"saved_sel_{selected_name}_{key}")
            st.session_state[_shadow_key] = _list_to_opt(_saved) if _saved is not None else \
                ("True" if default_val is True else "False")

        st.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:1.0rem; font-weight:700;'>{label}</span>"
            + (f"<span style='font-size:0.82rem; color:#888; margin-left:8px;'>default: {default_val}</span>"
               if default_val is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        _chosen = st.radio(label, options=_opts, index=_opts.index(st.session_state[_shadow_key]),
                           horizontal=True, label_visibility="collapsed")
        if _chosen != st.session_state[_shadow_key]:
            st.session_state[_shadow_key] = _chosen
        if _chosen == "Optimize":
            _sel_bool = [True, False]
        elif _chosen == "True":
            _sel_bool = [True]
        else:
            _sel_bool = [False]
        st.session_state[widget_key] = _sel_bool
        custom_grid[key] = _sel_bool

    # ── Time: single dropdown OR sweep (From / To selects) ───────────────────
    elif ptype == 'time':
        _t_shadow = f"_t_shadow_{selected_name}_{key}"
        _t_sweep_key = f"_t_sweep_{selected_name}_{key}"
        _default_t = default_val if default_val in values else values[0]

        # Consume pending key into shadow
        if pending_key in st.session_state:
            _pending_vals = st.session_state.pop(pending_key)
            if _pending_vals:
                _pv_in = [v for v in _pending_vals if v in values]
                if _pv_in:
                    st.session_state[_t_shadow] = _pv_in[0]

        # Init shadow from prefs if not set
        if _t_shadow not in st.session_state:
            _saved_t = (prefs or _load_prefs()).get(f"saved_sel_{selected_name}_{key}")
            st.session_state[_t_shadow] = _saved_t[0] if (_saved_t and _saved_t[0] in values) else _default_t

        _cur_t = st.session_state.get(_t_shadow, _default_t)
        if _cur_t not in values:
            _cur_t = _default_t

        st.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:1.0rem; font-weight:700;'>{label}</span>"
            + (f"<span style='font-size:0.82rem; color:#888; margin-left:8px;'>default: {default_val}</span>"
               if default_val is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        _t_sweep = st.checkbox("Sweep time range", key=_t_sweep_key,
                               help="Enable From/To pickers to sweep this time parameter across multiple values.")
        if _t_sweep:
            _tf_key = f"_t_from_{selected_name}_{key}"
            _tt_key = f"_t_to_{selected_name}_{key}"
            if _tf_key not in st.session_state:
                st.session_state[_tf_key] = _cur_t
            if _tt_key not in st.session_state:
                st.session_state[_tt_key] = _cur_t
            _tf_idx = values.index(st.session_state[_tf_key]) if st.session_state[_tf_key] in values else values.index(_cur_t)
            _tt_idx = values.index(st.session_state[_tt_key]) if st.session_state[_tt_key] in values else values.index(_cur_t)
            _tc1, _tc2 = st.columns(2)
            _t_from = _tc1.selectbox("From", options=values, index=_tf_idx, key=_tf_key, label_visibility="visible")
            _t_to   = _tc2.selectbox("To",   options=values, index=_tt_idx, key=_tt_key, label_visibility="visible")
            _fi = values.index(_t_from)
            _ti = values.index(_t_to)
            if _fi <= _ti:
                _t_range = values[_fi:_ti + 1]
            else:
                _t_range = values[_ti:_fi + 1]
            st.session_state[widget_key] = _t_range
            custom_grid[key] = _t_range
        else:
            _t_idx = values.index(_cur_t)
            _tbox_ver = st.session_state.get(f"_tbox_ver_{selected_name}_{key}", 0)
            _chosen = st.selectbox("Time", options=values, index=_t_idx,
                                   key=f"_tbox_{selected_name}_{key}_{_tbox_ver}", label_visibility="collapsed")
            st.session_state[_t_shadow] = _chosen
            st.session_state[widget_key] = [_chosen]
            custom_grid[key] = [_chosen]

    # ── Categorical: checkbox per value ──────────────────────────────────────
    else:
        # "off" is an implicit null-option — hide it; if nothing else is checked, "off" is used
        _has_off  = "off" in values
        _checkable = [v for v in values if v != "off"]

        if pending_key in st.session_state:
            st.session_state[widget_key] = st.session_state.pop(pending_key)
        if widget_key not in st.session_state:
            _saved_sel = (prefs or _load_prefs()).get(f"saved_sel_{selected_name}_{key}")
            if _saved_sel is not None:
                _valid = [v for v in _saved_sel if v in values]
                st.session_state[widget_key] = _valid if _valid else list(_checkable)
            else:
                st.session_state[widget_key] = list(_checkable)
        _cur_cat = st.session_state[widget_key]

        _lbl_col, _sa_col = st.columns([4, 1])
        _lbl_col.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:1.0rem; font-weight:700;'>{label}</span>"
            + (f"<span style='font-size:0.82rem; color:#888; margin-left:8px;'>default: {default_val}</span>"
               if default_val is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        if _sa_col.button("Optimize all", key=f"_cat_all_{selected_name}_{key}", use_container_width=True,
                          help="Include all options in the sweep"):
            st.session_state[widget_key] = list(_checkable)
            st.rerun()
        _sel_cat = []
        _cat_cols = st.columns(min(len(_checkable), 4))
        for _vi, _v in enumerate(_checkable):
            # No key= — let value= drive the checkbox so state survives group switches
            _checked = _cat_cols[_vi % len(_cat_cols)].checkbox(
                str(_v), value=(_v in _cur_cat)
            )
            if _checked:
                _sel_cat.append(_v)
        if not _sel_cat:
            # Nothing checked — use "off" if it exists, else first option
            _sel_cat = ["off"] if _has_off else [values[0]]
        st.session_state[widget_key] = _sel_cat
        custom_grid[key] = _sel_cat


def _build_export_text(result: dict, params: dict, meta: dict) -> str:
    """Build a NinjaTrader-style tab-separated export string from backtest results."""
    def g(col, default=float("nan")):
        val = result.get(col, default)
        return default if isinstance(val, pd.DataFrame) else val

    lines = []

    # Header
    lines.append(f"Performance\tAll trades")
    lines.append(f"Total net profit\t{fmt_dollar(g('net_pnl'))}")
    lines.append(f"Gross profit\t{fmt_dollar(g('gross_profit'))}")
    lines.append(f"Gross loss\t{fmt_dollar(g('gross_loss'))}")
    lines.append(f"Profit factor\t{fmt_float(g('profit_factor'))}")
    lines.append(f"Max. drawdown\t{fmt_dollar(g('max_drawdown'))}")
    lines.append(f"Sharpe ratio\t{fmt_float(g('sharpe'))}")
    lines.append(f"Sortino ratio\t{fmt_float(g('sortino'))}")
    lines.append(f"Ulcer index\t{fmt_float(g('ulcer_index'))}")
    lines.append(f"R squared\t{fmt_float(g('r_squared'))}")
    lines.append(f"% Months profitable\t{fmt_pct(g('pct_months_profit'))}")
    lines.append("")
    lines.append(f"Start date\t{meta.get('start', '')}")
    lines.append(f"End date\t{meta.get('end', '')}")
    lines.append("")
    lines.append(f"Total # of trades\t{int(g('total_trades', 0) or 0)}")
    lines.append(f"Percent profitable\t{fmt_pct(g('win_rate'))}")
    lines.append(f"# of winning trades\t{int(g('num_wins', 0) or 0)}")
    lines.append(f"# of losing trades\t{int(g('num_losses', 0) or 0)}")
    lines.append(f"# of even trades\t{int(g('num_even', 0) or 0)}")
    lines.append("")
    lines.append(f"Avg. trade\t{fmt_dollar(g('avg_trade'))}")
    lines.append(f"Avg. winning trade\t{fmt_dollar(g('avg_win'))}")
    lines.append(f"Avg. losing trade\t{fmt_dollar(g('avg_loss'))}")
    lines.append(f"Ratio avg. win / avg. loss\t{fmt_float(g('ratio_win_loss'))}")
    lines.append(f"Max. consec. winners\t{int(g('max_consec_winners', 0) or 0)}")
    lines.append(f"Max. consec. losers\t{int(g('max_consec_losers', 0) or 0)}")
    lines.append(f"Largest winning trade\t{fmt_dollar(g('largest_win'))}")
    lines.append(f"Largest losing trade\t{fmt_dollar(g('largest_loss'))}")
    lines.append(f"Avg. trades / day\t{fmt_float(g('avg_trades_per_day'))}")
    lines.append(f"Profit per month\t{fmt_dollar(g('profit_per_month'))}")
    lines.append(f"Max. time to recover\t{fmt_float(g('max_time_to_recover', 0), 0)} days")
    lines.append(f"Longest flat period\t{fmt_float(g('longest_flat_days', 0), 0)} days")
    lines.append("")

    # Parameters used
    lines.append("Parameters")
    for k, v in params.items():
        lines.append(f"{k}\t{v}")
    lines.append("")

    # Trade list
    trades_df = result.get("trades")
    if trades_df is not None and not trades_df.empty:
        lines.append("Trade number\tMarket pos.\tEntry price\tExit price\tEntry time\tExit time\tProfit\tTicks\tExit reason")
        for i, row in enumerate(trades_df.itertuples(), 1):
            ep  = getattr(row, 'entry_price', None) or getattr(row, 'entry', '')
            xp  = getattr(row, 'exit_price',  None) or getattr(row, 'exit',  '')
            et  = getattr(row, 'entry_time',  '')
            xt  = getattr(row, 'exit_time',   '')
            et  = pd.Timestamp(et).strftime("%d/%m/%Y %H:%M") if et else ''
            xt  = pd.Timestamp(xt).strftime("%d/%m/%Y %H:%M") if xt else ''
            pnl = getattr(row, 'pnl',         0)
            tks = getattr(row, 'pnl_ticks',   '')
            dr  = getattr(row, 'direction',   '')
            rsn = getattr(row, 'exit_reason', '')
            ep_s  = f"{ep:.2f}"  if isinstance(ep,  (int, float)) else str(ep)
            xp_s  = f"{xp:.2f}"  if isinstance(xp,  (int, float)) else str(xp)
            tks_s = f"{tks:+.0f}" if isinstance(tks, (int, float)) else str(tks)
            lines.append(
                f"{i}\t{dr.capitalize()}\t{ep_s}\t{xp_s}\t{et}\t{xt}\t{fmt_dollar(pnl)}\t{tks_s}\t{rsn}"
            )

    return "\n".join(lines)


def _df_to_tsv(df: pd.DataFrame, title: str = "") -> str:
    """Convert a DataFrame to tab-separated text for export."""
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    lines.append("\t".join(str(c) for c in df.columns))
    for _, row in df.iterrows():
        lines.append("\t".join(str(v) for v in row.values))
    return "\n".join(lines)


def _build_oos_export_text(row, param_keys: list) -> str:
    """Build NT-style export text for a single OOS config row."""
    lines = []
    lines.append("Parameters")
    for k in param_keys:
        if k in row.index:
            lines.append(f"{k}\t{row[k]}")
    lines.append("")
    lines.append("Performance\tOOS")
    for label, key in [
        ("Total Net Profit",    "oos_net_pnl"),
        ("Gross Profit",        "oos_gross_profit"),
        ("Gross Loss",          "oos_gross_loss"),
        ("Profit Factor",       "oos_profit_factor"),
        ("Max. Drawdown",       "oos_max_drawdown"),
        ("Sharpe Ratio",        "oos_sharpe"),
        ("Sortino Ratio",       "oos_sortino"),
        ("% Months Profitable", "oos_pct_months_profit"),
        ("Start Date",          "oos_start_date"),
        ("End Date",            "oos_end_date"),
        ("Total # of Trades",   "oos_total_trades"),
        ("Percent Profitable",  "oos_win_rate"),
        ("# Winning Trades",    "oos_num_wins"),
        ("# Losing Trades",     "oos_num_losses"),
        ("Avg. Trade",          "oos_avg_trade"),
        ("Avg. Winning Trade",  "oos_avg_win"),
        ("Avg. Losing Trade",   "oos_avg_loss"),
        ("Largest Win",         "oos_largest_win"),
        ("Largest Loss",        "oos_largest_loss"),
        ("Profit Per Month",    "oos_profit_per_month"),
        ("Max Drawdown",        "oos_max_drawdown"),
    ]:
        val = row.get(key, "")
        lines.append(f"{label}\t{val}")
    return "\n".join(lines)


def _render_nt_metrics(data: dict, prefix: str = "") -> None:
    """Render a NinjaTrader-style two-column performance metrics table."""
    def g(col, default=float("nan")):
        val = data.get(f"{prefix}{col}", default)
        return default if isinstance(val, pd.DataFrame) else val

    n_trades = int(data.get(f"{prefix}total_trades", 0) or 0)

    metrics = [
        ("Total Net Profit",      fmt_dollar(g("net_pnl"))),
        ("Gross Profit",          fmt_dollar(g("gross_profit"))),
        ("Gross Loss",            fmt_dollar(g("gross_loss"))),
        ("Commission",            fmt_dollar(g("total_commission"))),
        ("Profit Factor",         fmt_float(g("profit_factor"))),
        ("Max. Drawdown",         fmt_dollar(g("max_drawdown"))),
        ("Sharpe Ratio",          fmt_float(g("sharpe"))),
        ("Sortino Ratio",         fmt_float(g("sortino"))),
        ("Ulcer Index",           fmt_float(g("ulcer_index"))),
        ("R Squared",             fmt_float(g("r_squared"))),
        ("% Months Profitable",   fmt_pct(g("pct_months_profit"))),
        ("MC Stability",          fmt_float(data.get("mc_stability", float("nan")))),
        ("Start Date",            str(g("start_date", ""))),
        ("End Date",              str(g("end_date", ""))),
        ("Total Trades",          str(n_trades)),
        ("Percent Profitable",    fmt_pct(g("win_rate"))),
        ("Winning Trades",        str(int(g("num_wins", 0) or 0))),
        ("Losing Trades",         str(int(g("num_losses", 0) or 0))),
        ("Even Trades",           str(int(g("num_even", 0) or 0))),
        ("Avg. Trade",            fmt_dollar(g("avg_trade"))),
        ("Avg. Winning Trade",    fmt_dollar(g("avg_win"))),
        ("Avg. Losing Trade",     fmt_dollar(g("avg_loss"))),
        ("Ratio Avg Win / Loss",  fmt_float(g("ratio_win_loss"))),
        ("Max Consec. Winners",   str(int(g("max_consec_winners", 0) or 0))),
        ("Max Consec. Losers",    str(int(g("max_consec_losers", 0) or 0))),
        ("Largest Win",           fmt_dollar(g("largest_win"))),
        ("Largest Loss",          fmt_dollar(g("largest_loss"))),
        ("Avg. Trades / Day",     fmt_float(g("avg_trades_per_day"))),
        ("Profit Per Month",      fmt_dollar(g("profit_per_month"))),
        ("Max Time to Recover",   f"{fmt_float(g('max_time_to_recover', 0), 0)} days"),
        ("Longest Flat Period",   f"{fmt_float(g('longest_flat_days', 0), 0)} days"),
    ]

    # Rows where a $ value > 0 is green and < 0 is red
    _green_rows = {"Total Net Profit", "Gross Profit", "Avg. Trade", "Avg. Winning Trade",
                   "Profit Per Month", "Largest Win"}
    _red_rows   = {"Gross Loss", "Max. Drawdown", "Avg. Losing Trade", "Largest Loss"}

    def _is_negative(val: str) -> bool:
        """Check if a formatted value is negative — handles $-294, -1.23, etc."""
        stripped = val.replace("$", "").replace(",", "").strip()
        try:
            return float(stripped) < 0
        except ValueError:
            return False

    def _colour_row(row):
        name = row.name
        val  = row["Value"]
        if val in ("—",):
            return ["" for _ in row]
        if name in _green_rows:
            colour = "#e74c3c" if _is_negative(val) else "#2ecc71"
        elif name in _red_rows:
            colour = "#e74c3c"
        else:
            colour = ""
        return [f"color: {colour}" if colour else "" for _ in row]

    col_a, col_b = st.columns(2)
    half = len(metrics) // 2
    row_height_px = DF_ROW_HEIGHT_PX
    for col, chunk in [(col_a, metrics[:half]), (col_b, metrics[half:])]:
        df_chunk = pd.DataFrame(chunk, columns=["Performance", "Value"]).set_index("Performance")
        try:
            col.dataframe(
                df_chunk.style.apply(_colour_row, axis=1),
                use_container_width=True,
                height=len(chunk) * row_height_px + 38,
            )
        except AttributeError:
            # jinja2 not available in this environment — render without colour styling
            col.dataframe(
                df_chunk,
                use_container_width=True,
                height=len(chunk) * row_height_px + 38,
            )


def _meta_banner(df: pd.DataFrame, is_tab: bool = False) -> None:
    """Show a coloured date-range banner from embedded _run_ts / _*_start/_end columns."""
    if df is None or df.empty or '_run_ts' not in df.columns:
        return
    row = df.iloc[0]
    ts         = row.get('_run_ts', '')
    data_start = row.get('_data_start', '')
    data_end   = row.get('_data_end', '')
    is_start   = row.get('_is_start', data_start)
    is_end     = row.get('_is_end', '')
    oos_start  = row.get('_oos_start', '')
    oos_end    = row.get('_oos_end', data_end)
    if is_tab:
        st.info(
            f"**Run:** {ts}  ·  "
            f"**Data:** {data_start} → {data_end}  ·  "
            f"**IS period:** {is_start} → {is_end}  ·  "
            f"**OOS from:** {oos_start}"
        )
    else:
        st.info(
            f"**Run:** {ts}  ·  "
            f"**Data:** {data_start} → {data_end}  ·  "
            f"**OOS period:** {oos_start} → {oos_end}"
        )


# ---------------------------------------------------------------------------
# Sidebar: strategy selector
# ---------------------------------------------------------------------------

strategies = StrategyRegistry.list_strategies()

_PREFS_FILE = os.path.join(REPORTS_DIR, '..', '.dashboard_prefs.json')

def _load_prefs() -> dict:
    try:
        with open(_PREFS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_prefs(prefs: dict):
    try:
        with open(_PREFS_FILE, 'w') as f:
            json.dump(prefs, f)
    except Exception:
        pass

with st.sidebar:
    st.title("Strategy Platform")
    st.markdown("---")

    if not strategies:
        st.error("No strategies registered. Check that strategy modules are imported.")
        st.stop()

    _prefs       = _load_prefs()
    _last_strat  = _prefs.get("last_strategy", strategies[0])
    _default_idx = strategies.index(_last_strat) if _last_strat in strategies else 0

    selected_name = st.selectbox(
        "Strategy",
        options=strategies,
        index=_default_idx,
        help="Select a registered strategy to view its optimization results.",
    )
    _save_prefs({**_prefs, "last_strategy": selected_name})

    cls      = StrategyRegistry.get(selected_name)
    strategy = cls()
    db_host  = getattr(strategy, 'db_host', None)

    bar_type = getattr(strategy, 'bar_type', 'time')

    # ── Bar type selector (all strategies) ────────────────────────────────────
    _BAR_CATEGORIES   = ['Minute bars', 'Tick bars']
    _prefs_btcat_key  = f"last_bar_cat_{selected_name}"
    _last_btcat       = _prefs.get(_prefs_btcat_key, 'Minute bars')
    _btcat_idx        = _BAR_CATEGORIES.index(_last_btcat) if _last_btcat in _BAR_CATEGORIES else 0
    _bar_category = st.selectbox(
        "Bar Type",
        options=_BAR_CATEGORIES,
        index=_btcat_idx,
        key="sb_bar_category",
        help="Choose Minute bars or Tick bars. Then set the increment below.",
    )

    if _bar_category == 'Minute bars':
        _prefs_min_key = f"last_minute_inc_{selected_name}"
        _last_min_inc  = int(_prefs.get(_prefs_min_key, 5))
        _minute_inc = st.number_input(
            "Minutes per bar",
            min_value=1, max_value=5, step=1,
            value=_last_min_inc,
            key="sb_minute_inc",
            help="1–5 minute bars. 5M uses the deep historical database; 1–4M uses the 1-minute database.",
        )
        _save_prefs({**_prefs, _prefs_btcat_key: _bar_category, _prefs_min_key: int(_minute_inc)})
        bar_type = 'time' if _minute_inc == 5 else '1m'
        bar_minute_inc = int(_minute_inc)
        bar_type_label = f"🕐 Time bars ({_minute_inc}M)"
    else:
        _prefs_tick_key = f"last_tick_inc_{selected_name}"
        _strat_default_tick = getattr(strategy, 'tick_bar_size', 233)
        _last_tick_inc = int(_prefs.get(_prefs_tick_key, _strat_default_tick))
        _tick_inc = st.number_input(
            "Ticks per bar",
            min_value=1, step=1,
            value=_last_tick_inc,
            key="sb_tick_inc",
            help="Number of ticks per bar. Common values: 233, 377, 500, 1000.",
        )
        _save_prefs({**_prefs, _prefs_btcat_key: _bar_category, _prefs_tick_key: int(_tick_inc)})
        bar_type = 'tick'
        bar_minute_inc = None
        bar_type_label = f"📊 Tick bars ({int(_tick_inc)} ticks)"

    # Keep old prefs key in sync for legacy reads
    _prefs_bt_key = f"last_bar_type_{selected_name}"
    _save_prefs({**_prefs, _prefs_bt_key: bar_type})
    strategy.bar_type = bar_type
    _bt_labels = {'time': '5M time bars', '1m': '1M time bars', 'tick': 'Tick bars'}

    # ── Symbol selector — all known instruments ───────────────────────────────
    _default_symbol = getattr(strategy, 'symbol', 'unknown')
    _prefs_sym_key  = f"last_symbol_{selected_name}"
    _last_symbol    = _prefs.get(_prefs_sym_key, _default_symbol)
    _sym_idx = ALL_SYMBOLS.index(_last_symbol) if _last_symbol in ALL_SYMBOLS else \
               (ALL_SYMBOLS.index(_default_symbol) if _default_symbol in ALL_SYMBOLS else 0)
    symbol = st.selectbox(
        "Symbol",
        options=ALL_SYMBOLS,
        index=_sym_idx,
        help="All instruments shown. A warning appears when the chosen bar type has limited data for this symbol.",
        key="sb_symbol",
    )
    _save_prefs({**_prefs, _prefs_sym_key: symbol})

    # ── Coverage warning ──────────────────────────────────────────────────────
    _sym_cov      = SYMBOL_COVERAGE.get(symbol, {})
    _sym_bt_list  = _sym_cov.get('bar_types', [])
    if _sym_bt_list and bar_type not in _sym_bt_list:
        _primary_labels = ', '.join(_bt_labels.get(bt, bt) for bt in _sym_bt_list)
        st.warning(
            f"**{symbol}** has no primary data for **{_bt_labels.get(bar_type, bar_type)}**. "
            f"Primary coverage: {_primary_labels}.  \n"
            f"Results may be empty or very limited."
        )
    elif bar_type == '1m' and symbol not in ONE_MINUTE_SYMBOLS:
        st.warning(
            f"**{symbol}** has only live-feed 1M data (~12 k bars from 2026-03-29). "
            f"Deep 1M history (2020→present) is available for: "
            f"{', '.join(ONE_MINUTE_SYMBOLS.keys())}."
        )

    # Apply symbol-specific economics so PnL and risk-based sizing are correct
    if symbol in INSTRUMENT_META:
        _meta = INSTRUMENT_META[symbol]
        strategy.tick_size     = _meta['tick_size']
        strategy.tick_value    = _meta['tick_value']
        strategy.commission_rt = _meta['commission']

    st.caption(f"**Bar type:** {bar_type_label}")
    st.caption(f"**Description:** {strategy.description}")

    if bar_type == 'tick':
        _calc_mode_options = ["on_bar_close", "on_each_tick", "on_price_change"]
        _calc_mode_labels  = {
            "on_bar_close":    "On Bar Close",
            "on_each_tick":    "On Each Tick",
            "on_price_change": "On Price Change",
        }
        _calc_mode_prefs_key = f"calc_mode_{selected_name}"
        _calc_mode_ss_key    = f"sidebar_calc_mode_{selected_name}"
        _calc_mode_default   = _prefs.get(_calc_mode_prefs_key, "on_bar_close")
        if _calc_mode_default not in _calc_mode_options:
            _calc_mode_default = "on_bar_close"
        if _calc_mode_ss_key not in st.session_state:
            st.session_state[_calc_mode_ss_key] = _calc_mode_default
        _calc_mode_val = st.selectbox(
            "Calculate Mode",
            options=_calc_mode_options,
            format_func=lambda x: _calc_mode_labels.get(x, x),
            key=_calc_mode_ss_key,
            help="Controls when signals are evaluated within each bar.",
        )
        if _prefs.get(_calc_mode_prefs_key) != _calc_mode_val:
            _save_prefs({**_prefs, _calc_mode_prefs_key: _calc_mode_val})
        if _calc_mode_val in ("on_each_tick", "on_price_change"):
            st.caption("Slow — evaluates signal on every tick")

    st.markdown("---")

    _run_tss = list_run_timestamps(selected_name, symbol.replace('=', '_'))

    # Seed backtest tab with strategy defaults (once per session)
    _dp_seed = strategy.default_params
    for _k, _v in _dp_seed.items():
        if f"bt_{_k}" not in st.session_state:
            st.session_state[f"bt_{_k}"] = _v
    if "bt_loaded_from" not in st.session_state:
        st.session_state["bt_loaded_from"] = "default_params"

    st.markdown("---")
    st.caption(f"Reports: `{REPORTS_DIR}`")

param_grid = strategy.param_grid
param_keys = list(param_grid.keys())
sym_safe   = symbol.replace('=', '_')
_results_run_key = f"results_run_view_{selected_name}_{sym_safe}"

if _run_tss:
    if st.session_state.get(_results_run_key) not in _run_tss:
        st.session_state[_results_run_key] = _run_tss[0]
    selected_ts = st.session_state[_results_run_key]
else:
    selected_ts = None

# ---------------------------------------------------------------------------
# Load CSVs for the selected run
# ---------------------------------------------------------------------------

@st.cache_data
def _load_and_label(selected_name: str, sym_safe: str, selected_ts: str, param_keys_tuple: tuple):
    param_keys = list(param_keys_tuple)
    is_df  = load_run_csv(selected_name, sym_safe, "IS",  selected_ts)
    mc_df  = load_run_csv(selected_name, sym_safe, "MC",  selected_ts)
    oos_df = load_run_csv(selected_name, sym_safe, "OOS", selected_ts)
    for df in [df for df in [is_df, mc_df, oos_df] if df is not None]:
        df.insert(0, "config", df.apply(lambda r: config_label(r, param_keys), axis=1))
    return is_df, mc_df, oos_df

if selected_ts:
    is_df, mc_df, oos_df = _load_and_label(selected_name, sym_safe, selected_ts, tuple(param_keys))
else:
    is_df = mc_df = oos_df = None

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title(f"{selected_name.upper()} — Strategy Optimizer Results")
st.caption(f"Symbol: **{symbol}** · Param dimensions: **{len(param_keys)}** · "
           f"Grid size: **{__import__('math').prod(len(v) for v in param_grid.values()):,}** combos")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_labels = [
    "⚙️ Configure & Run",
    "📈 Results",
    "🔄 Walk-Forward",
    "🔬 Backtest",
    "🔁 Autoresearch",
]

tabs        = st.tabs(tab_labels)
tab_run     = tabs[0]
tab_results = tabs[1]
tab_wf      = tabs[2]
tab_bt      = tabs[3]
tab_ar      = tabs[4]


# ============================================================================
# CONFIGURE & RUN TAB
# ============================================================================

PLATFORM_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

with tab_run:
    st.subheader("Configure & Run Optimization")
    st.caption("Set the date range, enable parameter groups, configure ranges, then click Run.")

    # ── Session state for subprocess tracking ────────────────────────────────
    if 'proc' not in st.session_state:
        st.session_state.proc          = None
        st.session_state.output_lines  = []
        st.session_state.run_done      = False
        st.session_state.reader_thread = None

    # Autoresearch subprocess state (initialised here so it's always present)
    if 'ar_proc' not in st.session_state:
        st.session_state.ar_proc          = None
        st.session_state.ar_output_lines  = []
        st.session_state.ar_run_done      = False
        st.session_state.ar_reader_thread = None

    # ── Parameter grid builder ───────────────────────────────────────────────

    _defaults          = strategy.default_params
    _strat_groups      = getattr(strategy, 'param_groups', None)
    _display_names     = getattr(strategy, 'display_names', {})
    _param_conditional  = getattr(strategy, 'param_conditional', {})
    _param_dependencies = getattr(strategy, 'param_dependencies', {})
    _prefs_cache       = _load_prefs()  # load once per render

    _apply_pending_optimizer_setup(selected_name, param_grid, param_keys, _strat_groups)
    _apply_pending_optimizer_include_state(selected_name)

    # ── Run label input ──────────────────────────────────────────────────────
    _run_label_key     = f"_opt_run_label_{selected_name}"       # kept for read compat below
    _run_label_val_key = f"_opt_run_label_val_{selected_name}"
    _run_label_rst_key = f"_opt_run_label_reset_{selected_name}"
    if st.session_state.pop(_run_label_rst_key, False):
        st.session_state[_run_label_val_key] = ""
    _run_label_input = st.text_input(
        "Run label",
        value=st.session_state.get(_run_label_val_key, ""),
        placeholder=f"e.g. {selected_name} first sweep",
        help="Required before launching. Saved as 'OPT — <your label>' in the results dropdown.",
    )
    st.session_state[_run_label_val_key] = _run_label_input

    # ── Saved run loader ─────────────────────────────────────────────────────
    st.markdown("#### Restore Optimizer Setup")
    if _run_tss:
        _run_setup_pick_key = f"run_setup_pick_{selected_name}_{sym_safe}"
        _run_setup_last_key = f"run_setup_last_restored_{selected_name}_{sym_safe}"
        if st.session_state.get(_run_setup_pick_key) not in _run_tss:
            last_restored_ts = st.session_state.get(_run_setup_last_key)
            st.session_state[_run_setup_pick_key] = last_restored_ts if last_restored_ts in _run_tss else _run_tss[0]
        rl1, rl2 = st.columns([5, 1.5])
        _setup_run_ts = rl1.selectbox(
            "Restore setup from run",
            options=_run_tss,
            format_func=lambda ts: _format_run_display(selected_name, sym_safe, ts),
            key=_run_setup_pick_key,
            help="Copy that run's saved sweep setup into Configure & Run.",
        )
        rl2.markdown("<div style='margin-top:27px'></div>", unsafe_allow_html=True)
        if rl2.button("Restore", key=f"run_setup_apply_{selected_name}_{sym_safe}", use_container_width=True):
            _load_saved_run_setup_into_optimizer(
                selected_name,
                sym_safe,
                _setup_run_ts,
                param_grid,
                param_keys,
                _defaults,
                _strat_groups,
            )
            st.session_state[_run_setup_last_key] = _setup_run_ts
            st.session_state[_run_label_rst_key] = True
            _finish_optimizer_load(selected_name)
        _optimizer_source = st.session_state.get(f"_optimizer_loaded_source_{selected_name}_{sym_safe}")
        if _optimizer_source:
            st.caption(f"Most recently loaded into Configure & Run: `{_optimizer_source}`.")
        else:
            st.caption("This restores the actual optimizer sweep setup from a prior run.")
    else:
        st.caption("No saved runs available yet.")

    # ── Load saved config ────────────────────────────────────────────────────
    _cfg_dir = os.path.join(REPORTS_DIR, "configs")
    _cfg_files = sorted(
        [f for f in os.listdir(_cfg_dir) if f.startswith(f"{selected_name}_{sym_safe}_") and f.endswith(".json")]
        if os.path.isdir(_cfg_dir) else []
    )
    if _cfg_files:
        st.markdown("#### Load Saved Config")
        _lsc_col1, _lsc_col2 = st.columns([5, 1.5])
        _lsc_pick = _lsc_col1.selectbox(
            "Saved config",
            options=_cfg_files,
            format_func=lambda f: f.replace(f"{selected_name}_{sym_safe}_", "").replace(".json", ""),
            key=f"_lsc_pick_{selected_name}_{sym_safe}",
        )
        _lsc_col2.markdown("<div style='margin-top:27px'></div>", unsafe_allow_html=True)
        if _lsc_col2.button("Load", key=f"_lsc_load_{selected_name}_{sym_safe}", use_container_width=True):
            try:
                with open(os.path.join(_cfg_dir, _lsc_pick)) as _f:
                    _lsc_payload = json.load(_f)
                # Translate saved format → _apply_run_setup_to_optimizer format
                _lsc_setup = {
                    "group_includes": _lsc_payload.get("include_state", {}),
                    "ui_selections":  _lsc_payload.get("grid_state", {}),
                }
                _apply_run_setup_to_optimizer(_lsc_setup, selected_name, param_grid, param_keys, _strat_groups)
                st.session_state[f"_optimizer_loaded_source_{selected_name}_{sym_safe}"] = (
                    f"saved config: {_lsc_pick.replace('.json', '')}"
                )
                _finish_optimizer_load(selected_name)
            except Exception as _lsc_e:
                st.error(f"Load failed: {_lsc_e}")

    # ── Save current config ───────────────────────────────────────────────────
    st.markdown("#### Save Current Config")
    _save_cfg_name_key = f"_save_config_name_{selected_name}"
    _save_cfg_name = st.text_input(
        "Config name",
        placeholder="e.g. mntarget-sweep-v1",
        key=_save_cfg_name_key,
    )
    if st.button("Save config", key=f"_save_config_btn_{selected_name}"):
        _cfg_name_safe = _save_cfg_name.strip().replace(" ", "_")
        if not _cfg_name_safe:
            st.error("Enter a config name before saving.")
        else:
            _cfg_dir = os.path.join(REPORTS_DIR, "configs")
            os.makedirs(_cfg_dir, exist_ok=True)
            _cfg_path = os.path.join(_cfg_dir, f"{selected_name}_{sym_safe}_{_cfg_name_safe}.json")
            _inc_state = {
                _gn: bool(st.session_state.get(f"_inc_state_{selected_name}_{_gn}", False))
                for _gn in (_strat_groups or {})
            }
            _grid_state = {
                _k: list(st.session_state[f"run_grid_{_k}"])
                for _k in param_keys
                if f"run_grid_{_k}" in st.session_state and st.session_state[f"run_grid_{_k}"]
            }
            from datetime import datetime as _dt
            _cfg_payload = {
                "strategy": selected_name,
                "symbol": symbol,
                "bar_type": bar_type,
                "saved_at": _dt.now().isoformat(),
                "include_state": _inc_state,
                "grid_state": _grid_state,
            }
            try:
                with open(_cfg_path, "w") as _f:
                    json.dump(_cfg_payload, _f, indent=2)
                st.success(f"Config saved to `{os.path.basename(_cfg_path)}`.")
            except Exception as _e:
                st.error(f"Save failed: {_e}")

    # ── Data settings ────────────────────────────────────────────────────────
    st.markdown("#### Data Settings")

    _bar_type   = getattr(strategy, 'bar_type', 'time')
    _notes_str  = SYMBOL_COVERAGE.get(symbol, {}).get('notes', '')
    if _bar_type == 'tick':
        _cov = TICK_DATA_COVERAGE.get(symbol, {})
        _cov_str = (
            f"`{_cov['start']}` → `{_cov['end']}`  ({_cov.get('notes', '')})"
            if _cov else f"No tick data found for {symbol} — check emini.tick_data."
        )
        st.info(
            f"**Tick bars** — data loaded from `emini.tick_data`, aggregated to N-tick OHLCV bars.  \n"
            f"Tick bar sizes swept via **tick_bar_size** parameter above.  \n"
            f"**Available data for {symbol}:** {_cov_str}"
        )
    elif _bar_type == '1m':
        _cov = ONE_MINUTE_SYMBOLS.get(symbol, {})
        _cov_end = _cov.get('end') or 'present'
        _cov_str = (
            f"`{_cov['start']}` → `{_cov_end}`  ({_cov.get('notes', '')})"
            if _cov else f"Live-feed only (~12 k bars from 2026-03-29) — no deep history for {symbol}."
        )
        st.info(
            f"**1-minute bars** — data loaded from `emini.historical_data_1m`.  \n"
            f"**Available data for {symbol}:** {_cov_str}"
        )
    else:
        st.info(
            f"**5-minute time bars** — data loaded from `emini.historical_data`.  \n"
            f"**Coverage for {symbol}:** {_notes_str or 'check emini.historical_data.'}"
        )

    dcol1, dcol2, dcol3 = st.columns(3)
    if "run_start" in st.session_state:
        data_start_input = dcol1.date_input(
            "Data start date",
            help="Required. Select an explicit optimization start date.",
            key="run_start",
        )
    else:
        data_start_input = dcol1.date_input(
            "Data start date",
            value=None,
            help="Required. Select an explicit optimization start date.",
            key="run_start",
        )
    if "run_end" in st.session_state:
        data_end_input = dcol2.date_input(
            "Data end date",
            help="Required. Select an explicit optimization end date.",
            key="run_end",
        )
    else:
        data_end_input = dcol2.date_input(
            "Data end date",
            value=None,
            help="Required. Select an explicit optimization end date.",
            key="run_end",
        )
    if "run_train_pct" in st.session_state:
        train_pct_pct = dcol3.slider(
            "In-Sample split",
            min_value=50,
            max_value=90,
            step=5,
            format="%d%%",
            help="Fraction of data used for in-sample grid search",
            key="run_train_pct",
        )
    else:
        train_pct_pct = dcol3.slider(
            "In-Sample split",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            format="%d%%",
            help="Fraction of data used for in-sample grid search",
            key="run_train_pct",
        )
    train_pct_input = train_pct_pct / 100

    with st.expander("Pipeline Settings", expanded=False):
        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
        if "run_mc_sims" in st.session_state:
            mc_sims_input = pcol1.number_input("MC simulations", min_value=50, max_value=2000, step=50, key="run_mc_sims")
        else:
            mc_sims_input = pcol1.number_input("MC simulations", min_value=50, max_value=2000, value=200, step=50, key="run_mc_sims")
        if "run_mc_top_n" in st.session_state:
            mc_top_n_input = pcol2.number_input("MC top N", min_value=5, max_value=100, step=5, key="run_mc_top_n")
        else:
            mc_top_n_input = pcol2.number_input("MC top N", min_value=5, max_value=100, value=20, step=5, key="run_mc_top_n")
        if "run_oos_top_n" in st.session_state:
            oos_top_n_input = pcol3.number_input("OOS top N", min_value=1, max_value=20, step=1, key="run_oos_top_n")
        else:
            oos_top_n_input = pcol3.number_input("OOS top N", min_value=1, max_value=20, value=5, step=1, key="run_oos_top_n")
        if "run_min_trades" in st.session_state:
            min_trades_input = pcol4.number_input("Min trades", min_value=5, max_value=200, step=5, key="run_min_trades")
        else:
            min_trades_input = pcol4.number_input("Min trades", min_value=5, max_value=200, value=20, step=5, key="run_min_trades")
        rank_by_input = pcol5.selectbox("Rank by", options=["sharpe", "profit_factor", "sortino"], key="run_rank_by")

    if _bar_type == 'time':
        if "run_refresh" in st.session_state:
            refresh_input = st.checkbox("Force reload from MySQL (bypass cache)", key="run_refresh")
        else:
            refresh_input = st.checkbox("Force reload from MySQL (bypass cache)", value=False, key="run_refresh")
    else:
        refresh_input = False  # tick and 1m have no parquet cache

    missing_date_range = not data_start_input or not data_end_input
    invalid_date_order = bool(data_start_input and data_end_input and data_start_input > data_end_input)
    if missing_date_range:
        st.warning("Select both a start date and an end date before running an optimization.")
    elif invalid_date_order:
        st.error("Data start date must be on or before the data end date.")

    st.markdown("---")

    # ── Parameter Grid ────────────────────────────────────────────────────────
    st.markdown("#### Parameter Grid")

    custom_grid: Dict[str, List[Any]] = {}

    def _render_group_params(group_keys: List[str], include_in_sweep: bool) -> None:
        """Render param inputs for a list of keys. Only writes to custom_grid if include_in_sweep."""
        # Filter keys by param_conditional — only show if controlling param includes required value
        def _cond_met(k):
            if k not in _param_conditional:
                return True
            _ctrl, _req = _param_conditional[k]
            _ctrl_sel = st.session_state.get(f"run_grid_{_ctrl}", [])
            return _req in _ctrl_sel

        _visible_keys = [k for k in group_keys if _cond_met(k)]
        # Keys that control param_conditional visibility — changes require a rerun
        _ctrl_keys = {ctrl for ctrl, _ in _param_conditional.values()}

        # Action buttons for this group
        _act_c1, _act_c2, _act_c3, _ = st.columns([2, 2, 2, 4])
        _set_def  = _act_c1.button("Set all to defaults", help="Set each param to its default value")
        _reset    = _act_c2.button("Reset grid",          help="Clear saved selections")
        _save_def = _act_c3.button("Save as default",     help="Save current selections for future runs")

        if _set_def:
            for _k in _visible_keys:
                if _k not in param_grid:
                    continue
                if _param_type(param_grid[_k]) == 'time':
                    _saved = _prefs_cache.get(f"saved_sel_{selected_name}_{_k}")
                    _tv = _saved[0] if _saved else _defaults.get(_k)
                    if _tv:
                        st.session_state[f"_t_shadow_{selected_name}_{_k}"] = _tv
                        _ver_key = f"_tbox_ver_{selected_name}_{_k}"
                        st.session_state[_ver_key] = st.session_state.get(_ver_key, 0) + 1
                else:
                    _saved = _prefs_cache.get(f"saved_sel_{selected_name}_{_k}")
                    _dv = _saved if _saved is not None else ([_defaults[_k]] if _k in _defaults else None)
                    if _dv is not None:
                        st.session_state[f"run_grid_pending_{_k}"] = _dv
            st.rerun()
        if _reset:
            for _k in _visible_keys:
                for _sk in [f"run_grid_{_k}", f"run_grid_pending_{_k}",
                            f"_boolshadow_{selected_name}_{_k}", f"_bool_{selected_name}_{_k}",
                            f"_t_shadow_{selected_name}_{_k}"]:
                    st.session_state.pop(_sk, None)
            st.rerun()
        _pending_save = _save_def

        # Render — time params in pairs, others individually
        _grp_time  = [k for k in _visible_keys if k in param_grid and _param_type(param_grid[k]) == 'time']
        _grp_other = [k for k in _visible_keys if k in param_grid and _param_type(param_grid[k]) != 'time']

        _tmp_grid: Dict[str, List[Any]] = {}
        for _ti in range(0, len(_grp_time), 2):
            _pair  = _grp_time[_ti : _ti + 2]
            _pcols = st.columns(len(_pair))
            for _pci, _pkey in enumerate(_pair):
                with _pcols[_pci]:
                    _render_param_input(_pkey, param_grid, _defaults, _display_names,
                                        selected_name, _tmp_grid, prefs=_prefs_cache)
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)
        for _key in _grp_other:
            if _param_type(param_grid.get(_key, [])) == 'bool':
                st.markdown("<hr style='margin:8px 0 4px 0; border-color:#333;'>", unsafe_allow_html=True)
            _prev_val = st.session_state.get(f"run_grid_{_key}")
            _render_param_input(_key, param_grid, _defaults, _display_names,
                                selected_name, _tmp_grid, prefs=_prefs_cache)
            # calculate_mode: warn when slow modes are selected
            if _key == 'calculate_mode':
                _cm_vals = st.session_state.get(f"run_grid_{_key}", [])
                if any(v in ("on_each_tick", "on_price_change") for v in _cm_vals):
                    st.caption("On Each Tick / On Price Change are slow")
            # BUG-5 UI: show caption for dependent params when controller is being swept
            if _key in _param_dependencies:
                _dep_ctrl, _dep_req = _param_dependencies[_key]
                _ctrl_vals = st.session_state.get(f"run_grid_{_dep_ctrl}", [])
                _ctrl_label = _display_names.get(_dep_ctrl, _dep_ctrl)
                if len(_ctrl_vals) > 1:  # controlling param is being swept
                    st.caption(f"Skipped when {_ctrl_label} = {not _dep_req if isinstance(_dep_req, bool) else f'≠ {_dep_req}'} (no wasted combos).")
            if _key in _ctrl_keys and st.session_state.get(f"run_grid_{_key}") != _prev_val:
                st.rerun()
            st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

        if include_in_sweep:
            custom_grid.update(_tmp_grid)

        if st.session_state.pop(f"_saved_toast_{selected_name}", False):
            st.toast("Selections saved.")
        if st.session_state.pop(f"_optimizer_loaded_toast_{selected_name}", False):
            st.toast("Config loaded into Configure & Run.")
        if _pending_save:
            for _k in _visible_keys:
                if _k not in param_grid:
                    continue
                _pk = param_grid[_k]
                if _param_type(_pk) == 'time':
                    _tv = st.session_state.get(f"_t_shadow_{selected_name}_{_k}")
                    if _tv is not None:
                        _prefs_cache[f"saved_sel_{selected_name}_{_k}"] = [_tv]
                elif f"run_grid_{_k}" in st.session_state:
                    _prefs_cache[f"saved_sel_{selected_name}_{_k}"] = st.session_state[f"run_grid_{_k}"]
                _ver_key = f"_tbox_ver_{selected_name}_{_k}"
                st.session_state[_ver_key] = st.session_state.get(_ver_key, 0) + 1
                st.session_state.pop(f"_t_shadow_{selected_name}_{_k}", None)
                st.session_state.pop(f"_boolradio_shadow_{selected_name}_{_k}", None)
            _save_prefs(_prefs_cache)
            st.session_state[f"_saved_toast_{selected_name}"] = True
            st.rerun()

    if _strat_groups:
        _gnames = [g for g, ks in _strat_groups.items() if any(k in param_grid for k in ks)]

        # Use a plain (non-widget) state key per group so Streamlit cannot reset it.
        # Widget keys get reset to False when their widget isn't rendered — plain keys don't.
        for _gn in _gnames:
            _sk = f"_inc_state_{selected_name}_{_gn}"
            if _sk not in st.session_state:
                st.session_state[_sk] = bool(_prefs_cache.get(f"inc_{selected_name}_{_gn}", False))

        # Group selector dropdown
        _selgrp_key = f"selgrp_{selected_name}"
        if _selgrp_key not in st.session_state or st.session_state[_selgrp_key] not in _gnames:
            st.session_state[_selgrp_key] = _gnames[0]
        _selected_group = st.selectbox("Parameter Group", options=_gnames, key=_selgrp_key)

        # Sweep state summary bar — one chip per group
        def _group_combo_count(gn: str) -> int:
            _count = 1
            for _gk in _strat_groups.get(gn, []):
                if _gk not in param_grid:
                    continue
                if _gk in _param_conditional:
                    _ctrl, _req = _param_conditional[_gk]
                    if _req not in st.session_state.get(f"run_grid_{_ctrl}", []):
                        continue
                _vals = st.session_state.get(f"run_grid_{_gk}")
                _count *= max(1, len(_vals)) if _vals else 1
            return _count

        _chips_md = []
        for _gn in _gnames:
            _g_inc = st.session_state.get(f"_inc_state_{selected_name}_{_gn}", False)
            _g_cnt = _group_combo_count(_gn) if _g_inc else None
            _active = "🟢" if _g_inc else "⬜"
            _cnt_str = f" ({_g_cnt:,} combos)" if _g_cnt is not None else ""
            _bold = "**" if _gn == _selected_group else ""
            _chips_md.append(f"{_active} {_bold}{_gn}{_bold}{_cnt_str}")
        st.caption("  ·  ".join(_chips_md))

        st.markdown("---")

        # "Include in sweep" checkbox — no key= so Streamlit cannot reset it between group switches.
        # State lives in _inc_state_ (plain session_state, never touched by Streamlit).
        _inc_state_key = f"_inc_state_{selected_name}_{_selected_group}"
        _inc_pref_key  = f"inc_{selected_name}_{_selected_group}"
        _include = st.checkbox("Include in sweep", value=st.session_state[_inc_state_key])
        if _include != st.session_state[_inc_state_key]:
            st.session_state[_inc_state_key] = _include
            _prefs_cache[_inc_pref_key] = _include
            _save_prefs(_prefs_cache)
        st.caption("Excluded groups still use their configured values for single runs.")

        _group_keys = [k for k in _strat_groups[_selected_group] if k in param_grid]
        if not _group_keys:
            st.caption("No configurable parameters in this group.")
        else:
            if not _include:
                st.caption("Not included in sweep — configure values below, then check 'Include in sweep' to add to run.")
            # Fix 2: warn when both trail_distance variants present in same group
            _grp_lower = _selected_group.lower()
            if ("be" in _grp_lower or "trail" in _grp_lower) and \
               "trail_distance_ticks" in _group_keys and "trail_distance_rr" in _group_keys:
                st.caption("⚠️ Trail distance can be in ticks OR RR — do not include both in the same sweep.")
            _render_group_params(_group_keys, _include)

        # Populate custom_grid from session state for all *other* included groups
        for _gname in _gnames:
            if _gname == _selected_group:
                continue
            if not st.session_state.get(f"_inc_state_{selected_name}_{_gname}", False):
                continue
            for _k in _strat_groups[_gname]:
                if _k not in param_grid:
                    continue
                # Respect param_conditional — skip if condition not met
                if _k in _param_conditional:
                    _ctrl, _req = _param_conditional[_k]
                    if _req not in st.session_state.get(f"run_grid_{_ctrl}", []):
                        continue
                _v = st.session_state.get(f"run_grid_{_k}")
                if _v is not None:
                    custom_grid[_k] = _v

    else:
        # No groups defined — single "Include in sweep" for all params
        _inc_key = f"inc_{selected_name}_all"
        if _inc_key not in st.session_state:
            st.session_state[_inc_key] = _prefs_cache.get(_inc_key, False)
        _include = st.checkbox("Include in sweep", key=_inc_key)
        if _prefs_cache.get(_inc_key) != _include:
            _prefs_cache[_inc_key] = _include
            _save_prefs(_prefs_cache)
        if not _include:
            st.caption("Not included in sweep.")
        _render_group_params(list(param_keys), _include)

    # Combo count — derive values directly from widget shadow keys so the count is always
    # in sync with the current From/To/Step values, even mid-interaction.
    # Combo count — custom_grid has the freshest values for ALL params in the current group
    # (written by _render_param_input → _tmp_grid → custom_grid.update when include_in_sweep).
    # For params in OTHER included groups, read from run_grid_ (written by _render_param_input
    # unconditionally) or fall back to the full tuple range.
    _all_included_vals: Dict[str, List[Any]] = {}
    if _strat_groups:
        for _gn in _gnames:
            if not st.session_state.get(f"_inc_state_{selected_name}_{_gn}", False):
                continue
            for _k in _strat_groups[_gn]:
                if _k not in param_grid:
                    continue
                if _k in _param_conditional:
                    _ctrl, _req = _param_conditional[_k]
                    if _req not in st.session_state.get(f"run_grid_{_ctrl}", []):
                        continue
                # custom_grid is authoritative for currently-rendered group params
                if _k in custom_grid:
                    _val = custom_grid[_k]
                else:
                    # Other groups: run_grid_ written by last render of that param
                    _val = st.session_state.get(f"run_grid_{_k}")
                    if not _val:
                        # Never rendered — use full tuple range as default sweep
                        _pg = param_grid.get(_k)
                        if isinstance(_pg, tuple):
                            _val = _make_range(_pg[0], _pg[1], _pg[2]) or [_pg[0]]
                if _val:
                    _all_included_vals[_k] = _val
    else:
        _all_included_vals = custom_grid

    # Save for Walk-Forward tab to read (cross-tab access)
    st.session_state[f"_wf_custom_grid_{selected_name}"] = {k: list(v) for k, v in _all_included_vals.items()}

    if not _all_included_vals:
        st.info("No parameter groups included in sweep yet. Check 'Include in sweep' to add groups to the run.")
    else:
        _param_deps = getattr(strategy, 'param_dependencies', {})
        total_combos = sum(1 for _ in _deduplicated_combinations(_all_included_vals, _param_deps))
        if total_combos > 10_000:
            st.warning(f"**{total_combos:,} combinations** — large grid, may take a long time.")
        elif total_combos > 2_000:
            st.info(f"**{total_combos:,} combinations** selected.")
        else:
            st.success(f"**{total_combos:,} combinations** selected")

    st.markdown("---")

    # ── Run / Stop controls ──────────────────────────────────────────────────
    is_running = st.session_state.proc is not None and st.session_state.proc.poll() is None
    run_blocked = missing_date_range or invalid_date_order

    btn_col1, btn_col2 = st.columns([1, 4])
    run_clicked  = btn_col1.button("▶ Run",  disabled=is_running or run_blocked,  type="primary", use_container_width=True)
    stop_clicked = btn_col2.button("⏹ Stop", disabled=not is_running, use_container_width=False)

    if stop_clicked and st.session_state.proc:
        st.session_state.output_lines.append("\n[Stop requested by user]\n")
        stopped = _terminate_proc_tree(st.session_state.proc, st.session_state.output_lines)
        st.session_state.run_done = stopped
        st.rerun()

    if run_clicked:
        if not st.session_state.get(_run_label_val_key, "").strip():
            st.error("Add a run label before launching.")
            st.stop()
        ui_selections = {}
        for _k in param_keys:
            _vals = st.session_state.get(f"run_grid_{_k}")
            if _vals is not None:
                ui_selections[_k] = list(_vals)

        if _strat_groups:
            group_includes = {
                _gname: bool(st.session_state.get(f"_inc_state_{selected_name}_{_gname}", False))
                for _gname in _strat_groups
            }
            selected_group_name = st.session_state.get(f"selgrp_{selected_name}")
        else:
            group_includes = {"all": bool(st.session_state.get(f"inc_{selected_name}_all", False))}
            selected_group_name = None

        run_settings_payload = {
            "data_start": str(data_start_input) if data_start_input else None,
            "data_end": str(data_end_input) if data_end_input else None,
            "refresh": refresh_input,
            "train_pct": train_pct_input,
            "mc_sims": int(mc_sims_input),
            "mc_top_n": int(mc_top_n_input),
            "oos_top_n": int(oos_top_n_input),
            "min_trades": int(min_trades_input),
            "rank_by": rank_by_input,
            "param_grid_override": {k: list(v) for k, v in custom_grid.items()},
            "ui_selections": ui_selections,
            "group_includes": group_includes,
            "selected_group": selected_group_name,
        }

        # Build command
        cmd = [
            sys.executable, "-m", "strategy_platform.optimize.pipeline",
            "--strategy",   selected_name,
            "--symbol",     symbol,
            "--bar-type",   bar_type,
            "--train-pct",  str(train_pct_input),
            "--mc-sims",    str(int(mc_sims_input)),
            "--mc-top-n",   str(int(mc_top_n_input)),
            "--oos-top-n",  str(int(oos_top_n_input)),
            "--min-trades", str(int(min_trades_input)),
            "--rank-by",    rank_by_input,
            "--param-grid", json.dumps({k: list(v) for k, v in custom_grid.items()}),
            "--run-settings", json.dumps(run_settings_payload),
        ]
        if data_start_input:
            cmd += ["--start", str(data_start_input)]
        if data_end_input:
            cmd += ["--end", str(data_end_input)]
        if refresh_input:
            cmd.append("--refresh")

        # Save run label (with OPT prefix) and pre-run timestamps for post-run label application
        _user_lbl = st.session_state.get(_run_label_val_key, "").strip()
        _opt_lbl  = f"OPT — {_user_lbl}" if _user_lbl else "OPT"
        st.session_state[f"_pending_run_label_{selected_name}_{sym_safe}"] = _opt_lbl
        st.session_state[f"_pre_run_tss_{selected_name}_{sym_safe}"] = set(list_run_timestamps(selected_name, sym_safe))

        st.session_state.output_lines  = [f"$ {' '.join(cmd)}\n"]
        st.session_state.run_done      = False
        st.session_state.proc          = subprocess.Popen(
            cmd,
            cwd    = os.path.abspath(PLATFORM_DIR),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text   = True,
            bufsize = 1,
            start_new_session = True,
            env    = {**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        def _reader(proc, lines):
            """Background thread: drain stdout into lines list until EOF."""
            for line in proc.stdout:
                lines.append(line)
            rc = proc.wait()
            lines.append(
                "\n✅ Done — switch to the results tabs to view output.\n"
                if rc == 0 else f"\n❌ Failed (exit code {rc})\n"
            )

        t = threading.Thread(
            target=_reader,
            args=(st.session_state.proc, st.session_state.output_lines),
            daemon=True,
        )
        t.start()
        st.session_state.reader_thread = t
        st.rerun()

    # ── Live output display ──────────────────────────────────────────────────
    # Fragment is only defined (and its run_every timer only registered) while
    # a process is actually running. This prevents stale browser timers from
    # cascading "fragment does not exist" warnings on app restart.
    if st.session_state.proc is not None:

        @st.fragment(run_every="1s")
        def _output_panel():
            proc   = st.session_state.proc
            thread = st.session_state.get('reader_thread')
            if proc is None:
                return

            # Reader thread drains stdout continuously; we just detect completion here.
            if not st.session_state.run_done:
                thread_done = thread is None or not thread.is_alive()
                proc_done   = proc.poll() is not None
                if thread_done and proc_done:
                    st.session_state.run_done = True
                    # BUG-4: clear result caches so OOS/IS dropdowns show latest run
                    _list_stage_runs_with_rows.clear()
                    load_run_csv.clear()
                    _load_compare_stage_runs.clear()
                    st.rerun()

            output_text = "".join(st.session_state.output_lines)
            is_running  = not st.session_state.run_done

            if is_running:
                st.markdown("🟡 **Running…**")
            elif proc.returncode == 0:
                st.success("✅ Pipeline complete! Switch to the In-Sample, Monte Carlo, or OOS Validation tabs to view results.")
            else:
                rc = proc.returncode if proc.returncode is not None else "?"
                st.error(f"❌ Pipeline failed (exit code {rc}) — check the output below for details.")

            st.code(output_text or "Starting…", language=None)

        _output_panel()

    # ── Post-run label application ────────────────────────────────────────────
    _pending_lbl_key = f"_pending_run_label_{selected_name}_{sym_safe}"
    _pre_run_tss_key = f"_pre_run_tss_{selected_name}_{sym_safe}"
    if st.session_state.run_done and _pending_lbl_key in st.session_state:
        _pending_opt_lbl = st.session_state.pop(_pending_lbl_key)
        _pre_tss         = st.session_state.pop(_pre_run_tss_key, set())
        _cur_tss         = set(list_run_timestamps(selected_name, sym_safe))
        _new_tss         = _cur_tss - _pre_tss
        if _new_tss:
            for _new_ts in sorted(_new_tss, reverse=True)[:1]:
                _set_run_label(selected_name, sym_safe, _new_ts, _pending_opt_lbl)


# ============================================================================
# RESULTS TAB — shared run selector + nested IS / MC / OOS / Compare sub-tabs
# ============================================================================

with tab_results:
    if _run_tss:
        _res_col1, _res_col2, _res_col3 = st.columns([4, 3, 1])
        _res_col1.selectbox(
            "View results from run",
            options=_run_tss,
            format_func=lambda ts: _format_run_display(selected_name, sym_safe, ts),
            key=_results_run_key,
            help="Selecting a run updates In-Sample, Monte Carlo, and OOS Validation simultaneously.",
        )
        selected_ts = st.session_state.get(_results_run_key, selected_ts)

        # Inline rename — shows current label, saves on change
        if selected_ts:
            _inline_label_key = f"res_inline_label_{selected_name}_{sym_safe}_{selected_ts}"
            _cur_label = _get_run_label(selected_name, sym_safe, selected_ts)
            _sync_text_input_state(_inline_label_key, selected_ts, _cur_label)
            _new_label = _res_col2.text_input(
                "Run label",
                key=_inline_label_key,
                placeholder="Add a label…",
                help="Edit the run's display label. Saved immediately on change.",
            )
            if _new_label != _cur_label:
                _set_run_label(selected_name, sym_safe, selected_ts, _new_label)

            # Delete run button — at run level, not config level
            _res_del_key = f"_res_del_confirm_{selected_name}_{sym_safe}"
            _res_col3.markdown("<div style='margin-top:27px'></div>", unsafe_allow_html=True)
            if st.session_state.get(_res_del_key) == selected_ts:
                if _res_col3.button("Confirm ✓", key="res_del_confirm", use_container_width=True):
                    results_store.delete_optimizer_run(selected_name, sym_safe, selected_ts)
                    list_run_timestamps.clear()
                    load_run_csv.clear()
                    _list_stage_runs_with_rows.clear()
                    st.session_state.pop(_res_del_key, None)
                    st.rerun()
            else:
                if _res_col3.button("🗑 Delete", key="res_del_run", use_container_width=True,
                                    help="Permanently delete this entire run — IS, MC, and OOS results."):
                    st.session_state[_res_del_key] = selected_ts
                    st.rerun()

        if selected_ts:
            is_df, mc_df, oos_df = _load_and_label(selected_name, sym_safe, selected_ts, tuple(param_keys))

    tab_is, tab_mc, tab_oos, tab_cmp = st.tabs([
        "📊 In-Sample", "🎲 Monte Carlo", "✅ OOS Validation", "🧭 Compare Runs"
    ])

# ============================================================================
# IN-SAMPLE TAB
# ============================================================================

with tab_is:
    st.subheader("In-Sample Grid Search")

    if is_df is None or is_df.empty:
        st.info("No results yet — use the ⚙️ Configure & Run tab to run the optimizer.")
    else:
        _meta_banner(is_df, is_tab=True)
        st.caption(f"{len(is_df):,} parameter combinations.")

        # ── Parameter filters ─────────────────────────────────────────────────
        # Filters narrow results to specific param values (e.g. profit_target=40 only).
        # Each checked value is included; unchecking all values shows zero results.
        with st.expander("Parameter Filters", expanded=True):
            st.caption("Check values to include in the results table. Unchecking all values for a param shows no results for that param.")
            fa, fb, _ = st.columns([1, 1, 6])
            select_all = fa.button("Select All", key="is_filter_all")
            clear_all  = fb.button("Clear All",  key="is_filter_clear")

            # Version key: bumped on Select All / Clear All to force checkbox widget re-render
            _filt_ver_key   = f"is_filter_ver_{selected_name}_{sym_safe}"
            _filt_state_key = f"is_filter_cleared_{selected_name}_{sym_safe}"  # True = all cleared
            if select_all:
                st.session_state[_filt_ver_key]   = st.session_state.get(_filt_ver_key, 0) + 1
                st.session_state[_filt_state_key] = False
                st.rerun()
            if clear_all:
                st.session_state[_filt_ver_key]   = st.session_state.get(_filt_ver_key, 0) + 1
                st.session_state[_filt_state_key] = True
                st.rerun()

            _filt_ver     = st.session_state.get(_filt_ver_key, 0)
            _default_on   = not st.session_state.get(_filt_state_key, False)  # False after Clear All

            filters: Dict[str, Any] = {}

            # Build ordered list of (group_name, [keys]) — use param_groups if available
            if _strat_groups:
                _is_filter_groups = [
                    (gname, [k for k in gkeys if k in is_df.columns and k in param_keys])
                    for gname, gkeys in _strat_groups.items()
                ]
                _is_filter_groups = [(gn, ks) for gn, ks in _is_filter_groups if ks]
            else:
                _is_filter_groups = [("Parameters", [k for k in param_keys if k in is_df.columns])]

            for _gname, _gkeys in _is_filter_groups:
                if not _gkeys:
                    continue
                st.markdown(f"**{_gname}**")
                for _fkey in _gkeys:
                    _vals = sorted(is_df[_fkey].unique(), key=str)
                    _label = _display_names.get(_fkey, _fkey)
                    _n_cb = min(len(_vals), 8)
                    # Inline: label col (fixed width) + one col per checkbox value
                    _row_cols = st.columns([2] + [1] * _n_cb)
                    _row_cols[0].markdown(
                        f"<div style='padding-top:6px; font-size:0.82rem; color:#ccc;'>{_label}</div>",
                        unsafe_allow_html=True,
                    )
                    _selected = []
                    for _vi, _v in enumerate(_vals[:_n_cb]):
                        _cb_key = f"is_filter_cb_{_fkey}_{_vi}_v{_filt_ver}"
                        _checked = _row_cols[_vi + 1].checkbox(
                            str(_v), value=_default_on, key=_cb_key
                        )
                        if _checked:
                            _selected.append(_v)
                    # If param has more values than fit inline, render overflow on next row
                    if len(_vals) > _n_cb:
                        _overflow = _vals[_n_cb:]
                        _ov_cols = st.columns([2] + [1] * len(_overflow))
                        for _vi2, _v2 in enumerate(_overflow):
                            _cb_key2 = f"is_filter_cb_{_fkey}_{_n_cb + _vi2}_v{_filt_ver}"
                            _checked2 = _ov_cols[_vi2 + 1].checkbox(
                                str(_v2), value=_default_on, key=_cb_key2
                            )
                            if _checked2:
                                _selected.append(_v2)
                    filters[_fkey] = _selected  # empty list → no rows pass

        # ── Day-of-week filter ────────────────────────────────────────────────
        dow_days = ['mon', 'tue', 'wed', 'thu', 'fri']
        has_dow  = any(f'{d}_pnl' in is_df.columns for d in dow_days)
        dow_checks: Dict[str, bool] = {}
        if has_dow:
            with st.expander("Day-of-Week Filter", expanded=False):
                st.caption("Show only combos with positive P&L on the selected days.")
                dow_cols = st.columns(5)
                for i, day in enumerate(dow_days):
                    if f'{day}_pnl' in is_df.columns:
                        dow_checks[day] = dow_cols[i].checkbox(day.title(), value=False)

        # ── Quick filters ─────────────────────────────────────────────────────
        qf1, qf2, qf3 = st.columns([1, 1, 4])
        only_profitable = qf1.checkbox("Profitable only (net P&L > 0)", value=False, key="is_profitable_only")
        only_pos_sharpe = qf2.checkbox("Positive Sharpe only", value=False, key="is_pos_sharpe")

        # ── Apply filters ─────────────────────────────────────────────────────
        mask = pd.Series([True] * len(is_df), index=is_df.index)
        for key, selected_vals in filters.items():
            mask &= is_df[key].isin(selected_vals)
        filtered = is_df[mask].copy()
        for day, checked in dow_checks.items():
            if checked and f'{day}_pnl' in filtered.columns:
                filtered = filtered[filtered[f'{day}_pnl'] > 0]
        if only_profitable and "net_pnl" in filtered.columns:
            filtered = filtered[filtered["net_pnl"] > 0]
        if only_pos_sharpe and "sharpe" in filtered.columns:
            filtered = filtered[filtered["sharpe"] > 0]

        st.caption(f"Showing **{len(filtered):,}** combos after filters.")

        if filtered.empty:
            st.warning("No results match the current filters.")
        else:
            # ── Summary metrics ───────────────────────────────────────────────
            top = filtered.sort_values("sharpe", ascending=False).iloc[0]
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Best Sharpe",   fmt_float(top.get("sharpe")))
            m2.metric("Best Sortino",  fmt_float(top.get("sortino", float("nan"))))
            m3.metric("Win Rate",      fmt_pct(top.get("win_rate")))
            m4.metric("Net P&L",       fmt_dollar(top.get("net_pnl")))
            m5.metric("Profit Factor", fmt_float(top.get("profit_factor")))
            m6.metric("Max Drawdown",  fmt_dollar(top.get("max_drawdown")))

            if st.button("Load best result into Configure & Run", key="is_load_best"):
                _load_params_into_optimizer(top, param_keys, selected_name, param_grid, _defaults, _strat_groups)
                _finish_optimizer_load(selected_name)

            st.markdown("---")

            # ── Charts ────────────────────────────────────────────────────────
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("#### Profit Factor vs Win Rate (top 500 by Sharpe)")
                top500 = filtered.nlargest(SCATTER_CAP, "sharpe")
                hover  = [k for k in param_keys if k in top500.columns] + ["net_pnl"]
                fig = px.scatter(
                    top500, x="profit_factor", y="win_rate",
                    size="total_trades" if "total_trades" in top500.columns else None,
                    color="sharpe", color_continuous_scale="Viridis",
                    hover_data=hover,
                    labels={"profit_factor": "Profit Factor", "win_rate": "Win Rate", "sharpe": "Sharpe"},
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            with chart_col2:
                numeric_params = [k for k in param_keys
                                  if k in filtered.columns and pd.api.types.is_numeric_dtype(filtered[k]) and filtered[k].nunique() > 1]
                if len(numeric_params) >= 2:
                    hm_col1, hm_col2 = st.columns(2)
                    x_param = hm_col1.selectbox("Heatmap X axis", numeric_params,
                                                index=0, key="is_hm_x")
                    y_param = hm_col2.selectbox("Heatmap Y axis", numeric_params,
                                                index=min(1, len(numeric_params)-1), key="is_hm_y")
                    st.markdown(f"#### Avg Sharpe — {y_param} × {x_param}")
                    pivot = (
                        filtered.groupby([y_param, x_param])["sharpe"]
                        .mean().reset_index()
                        .pivot(index=y_param, columns=x_param, values="sharpe")
                    )
                    fig = px.imshow(
                        pivot, color_continuous_scale="RdYlGn", aspect="auto",
                        labels={"x": x_param, "y": y_param, "color": "Avg Sharpe"},
                        text_auto=".2f",
                    )
                    fig.update_layout(height=380)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Day-of-week P&L ───────────────────────────────────────────────
            if has_dow:
                st.markdown("#### Day-of-Week P&L — Best Combo")
                dow_data = [{"Day": d.title(), "P&L": top.get(f"{d}_pnl", 0)}
                            for d in dow_days if f"{d}_pnl" in top.index]
                if dow_data:
                    fig = px.bar(pd.DataFrame(dow_data), x="Day", y="P&L",
                                 color="P&L", color_continuous_scale=["red", "green"],
                                 labels={"P&L": "Net P&L ($)"})
                    fig.update_layout(coloraxis_showscale=False, height=280)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Top N table ───────────────────────────────────────────────────
            _show_n_col, _ = st.columns([1, 4])
            _show_n = _show_n_col.selectbox("Show top N", [10, 25, 50], index=0, key="is_show_n")
            st.markdown(f"#### Top {_show_n} by Sharpe")
            priority_cols = param_keys + [
                "total_trades", "trades", "num_wins", "num_losses", "win_rate",
                "net_pnl", "avg_trade", "largest_win", "largest_loss",
                "gross_profit", "gross_loss", "profit_factor",
                "sharpe", "sortino", "max_drawdown",
                "max_consec_winners", "max_consec_losers",
                "avg_trades_per_day", "profit_per_month", "max_time_to_recover",
                "bs_sharpe_p5", "bs_pnl_p5",
            ]
            show_cols = [c for c in priority_cols if c in filtered.columns]
            top50 = filtered.nlargest(_show_n, "sharpe")[show_cols].copy()
            if "win_rate" in top50:
                top50["win_rate"] = top50["win_rate"].apply(fmt_pct)
            for col in ["net_pnl", "avg_trade", "largest_win", "largest_loss",
                        "gross_profit", "gross_loss", "max_drawdown",
                        "profit_per_month", "bs_pnl_p5"]:
                if col in top50:
                    top50[col] = top50[col].apply(fmt_dollar)
            top50 = _humanize_columns(top50, strategy.display_names)
            st.dataframe(top50, use_container_width=True, hide_index=True, height=df_height(10))

            meta_row = is_df.iloc[0] if not is_df.empty else {}
            is_header = (
                f"In-Sample Top {_show_n} by Sharpe\n"
                f"Period\t{meta_row.get('_is_start', meta_row.get('_data_start', '?'))} → "
                f"{meta_row.get('_is_end', meta_row.get('_data_end', '?'))}\n"
                f"Total combos tested\t{len(is_df):,}\n\n"
            )
            st.download_button(
                label=f"⬇ Export Top {_show_n} (txt)",
                data=is_header + _df_to_tsv(top50, title=f"Top {_show_n} configurations"),
                file_name="is_results.txt",
                mime="text/plain",
            )

            # ── Parameter Heatmap ─────────────────────────────────────────────
            st.markdown("#### Parameter Heatmap")
            heatmap_path = os.path.join(REPORTS_DIR, f"heatmap_{selected_name}_{sym_safe}_{selected_ts}.png")
            if not os.path.exists(heatmap_path):
                heatmap_files = glob.glob(os.path.join(REPORTS_DIR, f"heatmap_{selected_name}_{sym_safe}_*.png"))
                if heatmap_files:
                    heatmap_path = max(heatmap_files, key=os.path.getmtime)
                else:
                    heatmap_path = None
            if heatmap_path and os.path.exists(heatmap_path):
                st.image(heatmap_path, use_container_width=True)
            else:
                st.caption("No heatmap available for this run.")


# ============================================================================
# MONTE CARLO TAB
# ============================================================================

with tab_mc:
    st.subheader("Monte Carlo Stability")
    st.caption("Stability = fraction of shuffled-order simulations with positive net P&L. Target > 0.60.")

    if mc_df is None:
        st.info(
            "No Monte Carlo results found. Run the optimizer to generate them.\n\n"
            f"```\npython -m strategy_platform.optimize.pipeline --strategy {selected_name}\n```"
        )
    else:
        _meta_banner(mc_df, is_tab=True)
        mc_has_data = not mc_df["mc_stability"].isna().all()

        if not mc_has_data:
            st.warning(
                f"Strategy **{selected_name}** does not implement `run_monte_carlo()` — "
                "MC results are not available. Override the method in the strategy class to enable this."
            )
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Configs tested",     len(mc_df))
            m2.metric("Avg MC Stability",   fmt_float(mc_df["mc_stability"].mean()))
            m3.metric("Fully stable (1.0)", int((mc_df["mc_stability"] == 1.0).sum()))

            st.markdown("---")

            # ── Table ────────────────────────────────────────────────────────

            priority_cols = param_keys + [
                "total_trades", "trades", "sharpe", "max_drawdown",
                "mc_stability", "mc_sharpe_p5", "mc_pnl_p5", "mc_pnl_p50",
            ]
            show_cols = [c for c in priority_cols if c in mc_df.columns]
            mc_sorted = mc_df.sort_values("mc_stability", ascending=False)
            pretty    = mc_sorted[show_cols].copy()

            for col in ["mc_pnl_p5", "mc_pnl_p50", "max_drawdown"]:
                if col in pretty:
                    pretty[col] = pretty[col].apply(fmt_dollar)
            pretty = _humanize_columns(pretty, strategy.display_names)
            st.dataframe(pretty, use_container_width=True, hide_index=True, height=df_height(len(pretty)))
            st.download_button(
                label="⬇ Export MC Results (txt)",
                data=_df_to_tsv(pretty, title="Monte Carlo Stability Results"),
                file_name="mc_results.txt",
                mime="text/plain",
            )

            # ── Charts ───────────────────────────────────────────────────────

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### MC Stability per config")
                fig = px.bar(
                    mc_sorted, x="config", y="mc_stability",
                    color="mc_stability", color_continuous_scale="RdYlGn",
                    labels={"mc_stability": "Stability", "config": "Config"},
                    range_y=[0, 1.05],
                )
                fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "sharpe" in mc_df.columns:
                    trades_col = "total_trades" if "total_trades" in mc_df.columns else (
                        "trades" if "trades" in mc_df.columns else None
                    )
                    st.markdown("#### IS Sharpe vs MC Stability")
                    fig = px.scatter(
                        mc_sorted, x="sharpe", y="mc_stability",
                        size=trades_col,
                        hover_data=[k for k in param_keys if k in mc_sorted.columns],
                        labels={"sharpe": "IS Sharpe", "mc_stability": "MC Stability"},
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# OOS VALIDATION TAB
# ============================================================================

with tab_oos:
    st.subheader("Out-of-Sample Validation")
    st.caption("Final candidates that survived IS → Monte Carlo → OOS walk-forward.")

    # Phase 4: use shared selected_ts from Results tab header (no per-tab run selector)
    _oos_ts = selected_ts  # alias for clarity
    active_oos_df = oos_df if (oos_df is not None and not oos_df.empty) else (
        load_run_csv(selected_name, sym_safe, "OOS", _oos_ts) if _oos_ts else None
    )

    if _oos_ts is None:
        st.info("No OOS results yet — use the ⚙️ Configure & Run tab to run the optimizer.")
    elif active_oos_df is None or active_oos_df.empty:
        st.info(f"No OOS results for the selected run. Run has IS/MC data but OOS validation may not have completed.")
    else:
        active_oos_df = active_oos_df.copy()
        if "config" not in active_oos_df.columns:
            active_oos_df.insert(0, "config", active_oos_df.apply(lambda r: config_label(r, param_keys), axis=1))
        _meta_banner(active_oos_df, is_tab=False)

        # ── Config selector ───────────────────────────────────────────────────
        oos_sorted = active_oos_df.sort_values("oos_net_pnl", ascending=False).reset_index(drop=True)
        config_options = [
            f"#{i+1}  {config_label(row, param_keys)}"
            for i, row in oos_sorted.iterrows()
        ]

        sel_idx = st.selectbox(
            "Configuration",
            range(len(config_options)),
            format_func=lambda i: config_options[i],
            key=f"oos_config_sel_{selected_name}_{sym_safe}_{_oos_ts}",
        )
        selected = oos_sorted.iloc[sel_idx]

        # ── Param preview (shows exactly what each load button will apply) ────
        with st.expander(f"Preview params for #{sel_idx + 1}", expanded=False):
            _preview_rows = []
            for _pk in param_keys:
                if _pk in selected.index and not pd.isna(selected[_pk]):
                    _preview_rows.append({"param": _pk, "value": selected[_pk]})
            if _preview_rows:
                st.dataframe(pd.DataFrame(_preview_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No swept params recorded for this row.")

        # ── Load actions ──────────────────────────────────────────────────────
        la1, la2 = st.columns([1, 1])
        if la1.button(f"📋 Load THIS config (#{sel_idx + 1}) into Backtester", key="oos_load_bt"):
            for key in param_keys:
                if key in selected.index:
                    st.session_state[f"bt_{key}"] = selected[key]
                    st.session_state.pop(f"_bt_val_{selected_name}_{key}", None)
            for key, val in strategy.default_params.items():
                if key not in param_keys and f"bt_{key}" not in st.session_state:
                    st.session_state[f"bt_{key}"] = val
            st.session_state["bt_loaded_from"] = (
                f"{_format_run_display(selected_name, sym_safe, _oos_ts)}  |  {config_options[sel_idx]}"
            )
            oos_start_val = selected.get("oos_start_date", None)
            oos_end_val   = selected.get("oos_end_date",   None)
            if oos_start_val:
                try:
                    st.session_state["bt_start"] = pd.Timestamp(oos_start_val).date()
                except Exception:
                    pass
            if oos_end_val:
                try:
                    st.session_state["bt_end"] = pd.Timestamp(oos_end_val).date()
                except Exception:
                    pass
            _loaded_sym = selected.get("symbol", None)
            if _loaded_sym and isinstance(_loaded_sym, str):
                st.session_state["sb_symbol"] = _loaded_sym
            _dates_loaded = bool(oos_start_val and oos_end_val)
            if _dates_loaded:
                st.toast("Parameters and OOS date range loaded — switch to the 🔬 Backtest tab.", icon="✅")
            else:
                st.toast("Parameters loaded — switch to the 🔬 Backtest tab.", icon="✅")
            st.rerun()
        if la2.button(f"⚙️ Send THIS config (#{sel_idx + 1}) to Optimizer", key="oos_load_optimizer"):
            _load_params_into_optimizer(selected, param_keys, selected_name, param_grid, _defaults, _strat_groups)
            st.session_state[f"_optimizer_loaded_source_{selected_name}_{sym_safe}"] = (
                f"OOS config from {_format_run_display(selected_name, sym_safe, _oos_ts)}"
            )
            _finish_optimizer_load(selected_name)

        st.markdown("---")
        st.markdown(f"#### Config {sel_idx + 1} — full performance")
        _render_nt_metrics(selected.to_dict(), prefix="oos_")

        # Fix 6: show all params, highlight optimized ones in red
        st.markdown("##### Config parameters")
        # Determine which params were actually swept in THIS run (>1 distinct value in IS results)
        _swept_in_run = set()
        if is_df is not None and not is_df.empty:
            for _sk in param_keys:
                if _sk in is_df.columns and is_df[_sk].nunique() > 1:
                    _swept_in_run.add(_sk)
        else:
            _swept_in_run = set(param_keys)
        _all_param_rows = []
        for _pk in strategy.default_params.keys():
            _pval = selected[_pk] if _pk in selected.index else strategy.default_params[_pk]
            _plabel = strategy.display_names.get(_pk, _pk)
            _is_opt = _pk in _swept_in_run
            _all_param_rows.append({"Parameter": _plabel, "Value": str(_pval), "_optimized": _is_opt})
        _all_params_df = pd.DataFrame(_all_param_rows)

        def _highlight_optimized(row):
            if row["_optimized"]:
                return ["color: #e05252; font-weight: bold"] * len(row)
            return [""] * len(row)

        _styled = (
            _all_params_df.drop(columns=["_optimized"])
            .style.apply(_highlight_optimized, axis=1, subset=pd.IndexSlice[:, :])
        )
        # apply needs the _optimized col — use the full df with hidden col
        _styled = _all_params_df.style.apply(
            lambda row: ["color: #e05252; font-weight: bold" if row["_optimized"] else ""] * len(row),
            axis=1,
        ).hide(axis="columns", subset=["_optimized"])
        st.dataframe(_styled, use_container_width=True, hide_index=True, height=df_height(len(_all_params_df)))
        st.caption("Red = optimized parameter (swept in this run) · Grey = fixed at default")

        # Fix 7: lock params button
        if st.button("🔒 Lock params into Configure & Run", key="oos_lock_params"):
            for _lk in param_keys:
                if _lk in selected.index:
                    st.session_state[f"run_grid_pending_{_lk}"] = [selected[_lk]]
            _finish_optimizer_load(selected_name)

        st.download_button(
            label="⬇ Export this config (txt)",
            data=_build_oos_export_text(selected, param_keys),
            file_name=f"oos_config_{sel_idx + 1}.txt",
            mime="text/plain",
        )

        st.markdown("---")

        # ── All configurations summary table ─────────────────────────────────
        st.markdown("#### All configurations")
        priority_cols = param_keys + [
            "mc_stability", "mc_pnl_p50",
            "oos_total_trades", "oos_win_rate", "oos_net_pnl",
            "oos_profit_factor", "oos_sharpe", "oos_sortino",
            "oos_max_drawdown", "oos_start_date", "oos_end_date",
        ]
        show_cols = [c for c in priority_cols if c in active_oos_df.columns]
        pretty = oos_sorted[show_cols].copy()
        for col in ["oos_win_rate"]:
            if col in pretty:
                pretty[col] = pretty[col].apply(fmt_pct)
        for col in ["oos_net_pnl", "mc_pnl_p50", "oos_max_drawdown"]:
            if col in pretty:
                pretty[col] = pretty[col].apply(fmt_dollar)
        pretty = _humanize_columns(pretty, strategy.display_names)
        st.dataframe(pretty, use_container_width=True, hide_index=True, height=df_height(len(pretty)))
        st.download_button(
            label="⬇ Export all OOS configs (txt)",
            data=_df_to_tsv(pretty, title="All OOS Configurations"),
            file_name="oos_all_configs.txt",
            mime="text/plain",
        )

        # ── OOS P&L bar chart ─────────────────────────────────────────────────
        if len(active_oos_df) > 1:
            st.markdown("#### OOS Net P&L by configuration")
            fig = px.bar(
                oos_sorted.iloc[::-1], x="oos_net_pnl", y="config", orientation="h",
                color="oos_net_pnl", color_continuous_scale="RdYlGn",
                labels={"oos_net_pnl": "OOS Net P&L ($)", "config": "Config"},
            )
            fig.update_layout(coloraxis_showscale=False, height=max(300, len(active_oos_df) * 50))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# COMPARE RUNS TAB
# ============================================================================

with tab_cmp:
    st.subheader("Compare Runs")
    st.caption(
        "Compare multiple saved optimization runs side by side. "
        "Use this to see which parameter values and entry structures keep surviving."
    )

    if not _run_tss:
        st.info("No saved runs yet — run the optimizer first, then come back here to compare them.")
    else:
        st.markdown("#### 1. Choose Runs")
        run_sel_key = f"cmp_run_sel_init_{selected_name}_{sym_safe}"
        if run_sel_key not in st.session_state:
            for ts in _run_tss[:min(4, len(_run_tss))]:
                st.session_state[f"cmp_run_{selected_name}_{sym_safe}_{ts}"] = True
            st.session_state[run_sel_key] = True

        with st.expander("Runs To Compare", expanded=False):
            rs1, rs2, _ = st.columns([1, 1, 4])
            if rs1.button("Recent 4", key=f"cmp_runs_recent4_{selected_name}_{sym_safe}"):
                for ts in _run_tss:
                    st.session_state[f"cmp_run_{selected_name}_{sym_safe}_{ts}"] = ts in _run_tss[:min(4, len(_run_tss))]
                st.rerun()
            if rs2.button("Select All", key=f"cmp_runs_all_{selected_name}_{sym_safe}"):
                for ts in _run_tss:
                    st.session_state[f"cmp_run_{selected_name}_{sym_safe}_{ts}"] = True
                st.rerun()

            run_cols = st.columns(2)
            for i, ts in enumerate(_run_tss):
                run_cols[i % 2].checkbox(
                    _format_run_display(selected_name, sym_safe, ts),
                    key=f"cmp_run_{selected_name}_{sym_safe}_{ts}",
                )

        compare_runs = [
            ts for ts in _run_tss
            if st.session_state.get(f"cmp_run_{selected_name}_{sym_safe}_{ts}", False)
        ]
        if compare_runs:
            st.caption(
                "Selected: " + " | ".join(
                    _format_run_display(selected_name, sym_safe, ts) for ts in compare_runs[:6]
                ) + (" | ..." if len(compare_runs) > 6 else "")
            )
        else:
            st.caption("No runs selected yet.")

        st.markdown("#### 2. Comparison Settings")
        st.caption(
            "`Stage` controls which result set is being compared. If you choose `OOS`, the tables below show "
            "the best out-of-sample row from each selected run. That is usually the best robustness check when available."
        )

        def _has_stage_rows(stage_name: str, run_ts: str) -> bool:
            df = load_run_csv(selected_name, sym_safe, stage_name, run_ts)
            return df is not None and not df.empty

        preferred_stage = "IS"
        if compare_runs:
            if all(_has_stage_rows("OOS", ts) for ts in compare_runs):
                preferred_stage = "OOS"
            elif all(_has_stage_rows("MC", ts) for ts in compare_runs):
                preferred_stage = "MC"

        cmp_stage_key = f"cmp_stage_{selected_name}_{sym_safe}"
        if cmp_stage_key not in st.session_state:
            st.session_state[cmp_stage_key] = preferred_stage

        cmp_stage_col, cmp_top_col = st.columns([1, 1])
        compare_stage = cmp_stage_col.selectbox(
            "Stage",
            options=["IS", "MC", "OOS"],
            index=["IS", "MC", "OOS"].index(st.session_state.get(cmp_stage_key, preferred_stage)),
            key=cmp_stage_key,
        )
        compare_top_n = int(cmp_top_col.number_input(
            "Top N / run",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key=f"cmp_top_n_{selected_name}_{sym_safe}",
        ))

        if not compare_runs:
            st.info("Select at least one run to compare.")
        else:
            stage_runs = _load_compare_stage_runs(
                selected_name,
                sym_safe,
                tuple(compare_runs),
                tuple(param_keys),
                compare_stage,
            )

            if not stage_runs:
                st.warning(f"No {compare_stage} results found for the selected runs.")
            else:
                metric_options = [
                    metric
                    for metric in _COMPARE_STAGE_METRICS.get(compare_stage, [])
                    if any(metric in df.columns and df[metric].notna().any() for df in stage_runs.values())
                ]
                if not metric_options:
                    numeric_candidates = []
                    for df in stage_runs.values():
                        for col in df.columns:
                            if col.startswith("_") or col == "config":
                                continue
                            if pd.api.types.is_numeric_dtype(df[col]) and col not in numeric_candidates:
                                numeric_candidates.append(col)
                    metric_options = numeric_candidates

                if not metric_options:
                    st.warning("No comparable numeric metrics found for these runs at this stage.")
                else:
                    compare_metric = st.selectbox(
                        "Rank Metric",
                        options=metric_options,
                        index=0,
                        format_func=lambda key: _human_label(key, strategy.display_names),
                        key=f"cmp_metric_{compare_stage}",
                    )

                    st.markdown("#### 3. Comparison Results")
                    st.info(
                        f"Comparing **optimization** runs at the **{compare_stage}** stage, "
                        f"ranked by **{_human_label(compare_metric, strategy.display_names)}**."
                    )

                    usable_runs = {
                        ts: df for ts, df in stage_runs.items()
                        if compare_metric in df.columns and df[compare_metric].notna().any()
                    }
                    if not usable_runs:
                        st.warning(f"No selected runs contain usable values for `{compare_metric}`.")
                    else:
                        top_rows = []
                        shortlist_rows = []
                        for ts, df in usable_runs.items():
                            sorted_df = _sort_for_compare(df, compare_metric)
                            if sorted_df.empty:
                                continue
                            top_rows.append(sorted_df.head(1))
                            shortlist_rows.append(sorted_df.head(compare_top_n))

                        if not top_rows:
                            st.warning("No sortable rows found for the selected runs.")
                        else:
                            top_per_run = pd.concat(top_rows, ignore_index=True)
                            shortlist = pd.concat(shortlist_rows, ignore_index=True)

                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Runs Compared", len(usable_runs))
                            m2.metric("Stage", compare_stage)
                            m3.metric("Rank Metric", _human_label(compare_metric, strategy.display_names))
                            m4.metric("Shortlist Rows", len(shortlist))

                            def _pretty_compare(df: pd.DataFrame) -> pd.DataFrame:
                                pretty = df.copy()
                                for col in [c for c in pretty.columns if "win_rate" in c]:
                                    pretty[col] = pretty[col].apply(fmt_pct)
                                for col in [c for c in pretty.columns
                                            if any(tok in c for tok in ["pnl", "drawdown", "profit_limit", "loss_limit"])]:
                                        pretty[col] = pretty[col].apply(fmt_dollar)
                                return _humanize_columns(pretty, strategy.display_names)

                            st.markdown("#### Best Config Per Run")
                            st.caption("This is the main side-by-side table. It shows one winning row per selected run for the current stage.")
                            best_cols = [
                                "_run_display", "config", "entry_mode", "require_retest", "retest_type",
                                compare_metric,
                                "total_trades", "trades",
                                "net_pnl", "profit_factor", "sharpe", "win_rate", "max_drawdown",
                                "mc_stability", "mc_pnl_p50",
                                "oos_net_pnl", "oos_sharpe", "oos_win_rate", "oos_max_drawdown",
                            ]
                            best_show = _unique_existing_columns(top_per_run, best_cols)
                            best_df = _pretty_compare(top_per_run[best_show].copy())
                            best_df = best_df.rename(columns={"_run_display": "run"})
                            st.dataframe(
                                best_df,
                                use_container_width=True,
                                hide_index=True,
                                height=df_height(len(best_df)),
                            )

                            # ── Top N Pool expander ───────────────────────────
                            _cmp_section_break()
                            with st.expander(f"Top {compare_top_n} Pool", expanded=False):
                                st.caption("Pools the top N rows from every selected run. Useful for spotting patterns across runs, not just the single winner from each.")
                                shortlist_cols = [
                                    "_run_display", "config", "entry_mode", "require_retest", "retest_type",
                                    compare_metric,
                                    "total_trades", "trades",
                                    "net_pnl", "profit_factor", "sharpe", "win_rate", "max_drawdown",
                                    "mc_stability", "mc_pnl_p50",
                                    "oos_net_pnl", "oos_sharpe", "oos_win_rate", "oos_max_drawdown",
                                ]
                                shortlist_show = _unique_existing_columns(shortlist, shortlist_cols)
                                shortlist_df = _pretty_compare(shortlist[shortlist_show].copy())
                                shortlist_df = shortlist_df.rename(columns={"_run_display": "run"})
                                st.dataframe(
                                    shortlist_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=min(700, df_height(len(shortlist_df))),
                                )

                            # ── Parameter Stability expander ──────────────────
                            _cmp_section_break()
                            with st.expander("Parameter Stability", expanded=False):
                                st.caption("Which values appear most often across the pooled shortlist — useful for spotting stable settings instead of chasing one lucky row.")
                                freq_rows = []
                                freq_detail_rows = []
                                for key in param_keys:
                                    if key not in shortlist.columns:
                                        continue
                                    counts = shortlist[key].value_counts(dropna=False)
                                    if counts.empty:
                                        continue
                                    top_values = counts.head(3)
                                    freq_rows.append({
                                        "parameter": key,
                                        "most_common": str(top_values.index[0]),
                                        "count": int(top_values.iloc[0]),
                                        "share": top_values.iloc[0] / len(shortlist),
                                        "top_values": " | ".join(f"{idx} ({cnt})" for idx, cnt in top_values.items()),
                                    })
                                    for value, count in counts.items():
                                        freq_detail_rows.append({
                                            "parameter": key,
                                            "value": str(value),
                                            "count": int(count),
                                            "share": count / len(shortlist),
                                        })

                                freq_df = pd.DataFrame(freq_rows)
                                if freq_df.empty:
                                    st.info("No parameter frequency data available for the current shortlist.")
                                else:
                                    freq_df["share"] = freq_df["share"].apply(fmt_pct)
                                    freq_df = _humanize_columns(freq_df, strategy.display_names)
                                    st.dataframe(
                                        freq_df,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=min(700, df_height(len(freq_df))),
                                    )
                                    with st.expander("Full parameter frequency detail", expanded=False):
                                        freq_detail_df = pd.DataFrame(freq_detail_rows)
                                        if not freq_detail_df.empty:
                                            freq_detail_df["share"] = freq_detail_df["share"].apply(fmt_pct)
                                            freq_detail_df = _humanize_columns(freq_detail_df, strategy.display_names)
                                            st.dataframe(
                                                freq_detail_df,
                                                use_container_width=True,
                                                hide_index=True,
                                                height=min(700, df_height(len(freq_detail_df))),
                                            )

                                # Head-to-Head — generic categorical grouping
                                _cat_params = [
                                    k for k in param_keys
                                    if k in shortlist.columns
                                    and not pd.api.types.is_numeric_dtype(shortlist[k])
                                    and shortlist[k].nunique() > 1
                                ]
                                if _cat_params:
                                    st.markdown("##### Head-to-Head by Parameter")
                                    st.caption("Roll up the shortlist by any categorical param to see which values survive most often.")
                                    _h2h_default = "entry_mode" if "entry_mode" in _cat_params else _cat_params[0]
                                    _h2h_param = st.selectbox(
                                        "Group by",
                                        options=_cat_params,
                                        index=_cat_params.index(_h2h_default),
                                        format_func=lambda k: _human_label(k, strategy.display_names),
                                        key=f"cmp_h2h_param_{compare_stage}_{compare_metric}",
                                    )
                                    wins = Counter(top_per_run[_h2h_param].dropna().tolist()) if _h2h_param in top_per_run.columns else {}
                                    agg_spec = {
                                        "configs_in_shortlist": (_h2h_param, "size"),
                                        "runs_represented": ("_run_ts", "nunique"),
                                        "run_wins": (_h2h_param, lambda s: wins.get(s.iloc[0], 0)),
                                        f"median_{compare_metric}": (compare_metric, "median"),
                                        f"best_{compare_metric}": (
                                            compare_metric,
                                            "min" if _compare_metric_ascending(compare_metric) else "max",
                                        ),
                                    }
                                    for extra_col in [
                                        "net_pnl", "profit_factor", "win_rate", "max_drawdown",
                                        "mc_stability", "mc_pnl_p50",
                                        "oos_net_pnl", "oos_sharpe", "oos_win_rate", "oos_max_drawdown",
                                    ]:
                                        if extra_col in shortlist.columns:
                                            agg_spec[f"median_{extra_col}"] = (extra_col, "median")

                                    h2h_summary = (
                                        shortlist.groupby(_h2h_param, dropna=False)
                                        .agg(**agg_spec)
                                        .reset_index()
                                    )
                                    h2h_summary = h2h_summary.sort_values(
                                        f"median_{compare_metric}",
                                        ascending=_compare_metric_ascending(compare_metric),
                                    )
                                    pretty_h2h = _pretty_compare(h2h_summary.copy())
                                    st.dataframe(
                                        pretty_h2h,
                                        use_container_width=True,
                                        hide_index=True,
                                        height=df_height(len(pretty_h2h)),
                                    )
                                    fig = px.bar(
                                        h2h_summary,
                                        x=_h2h_param,
                                        y=f"median_{compare_metric}",
                                        color="run_wins",
                                        text="run_wins",
                                        labels={
                                            _h2h_param: _human_label(_h2h_param, strategy.display_names),
                                            f"median_{compare_metric}": f"Median {_human_label(compare_metric, strategy.display_names)}",
                                            "run_wins": "Run Wins",
                                        },
                                    )
                                    fig.update_layout(height=320)
                                    st.plotly_chart(fig, use_container_width=True)

                            # ── Single promote section ────────────────────────
                            _cmp_section_break()
                            st.markdown("#### Promote Config")
                            st.caption("Load any config from Best Per Run or Top N Pool into Backtest or Configure & Run.")
                            shortlist_sorted = _sort_for_compare(shortlist, compare_metric).reset_index(drop=True)
                            # Build unified options: top_per_run rows first, then remaining shortlist rows
                            _top_ts_set = set(
                                zip(top_per_run.get("_run_ts", pd.Series(dtype=str)),
                                    top_per_run.get("config", pd.Series(dtype=str)))
                            )
                            promote_options = []
                            promote_rows = []
                            for i, row in top_per_run.reset_index(drop=True).iterrows():
                                promote_options.append(
                                    f"[Best] #{i+1}  {row.get('_run_display', '')}  |  {row.get('config', '')}"
                                )
                                promote_rows.append(row)
                            for i, row in shortlist_sorted.iterrows():
                                _rts = row.get("_run_ts", "")
                                _cfg = row.get("config", "")
                                if (_rts, _cfg) in _top_ts_set:
                                    continue  # already listed above
                                metric_val = row.get(compare_metric)
                                if pd.notna(metric_val):
                                    if "win_rate" in compare_metric:
                                        metric_txt = fmt_pct(metric_val)
                                    elif any(tok in compare_metric for tok in ["pnl", "drawdown", "profit_limit", "loss_limit"]):
                                        metric_txt = fmt_dollar(metric_val)
                                    else:
                                        metric_txt = fmt_float(metric_val)
                                else:
                                    metric_txt = "—"
                                promote_options.append(
                                    f"[Pool] #{i+1}  {row.get('_run_display', '')}  |  "
                                    f"{row.get('config', '')}  |  {compare_metric}={metric_txt}"
                                )
                                promote_rows.append(row)

                            if promote_options:
                                promote_idx = st.selectbox(
                                    "Choose config",
                                    options=range(len(promote_options)),
                                    format_func=lambda i: promote_options[i],
                                    key=f"cmp_promote_{compare_stage}_{compare_metric}",
                                )
                                promote_row = promote_rows[promote_idx]
                                pr_col1, pr_col2 = st.columns([1, 1])
                                if pr_col1.button("📋 Send to Backtester", key=f"cmp_load_bt_{compare_stage}_{compare_metric}"):
                                    _load_params_into_backtester(
                                        promote_row,
                                        param_keys,
                                        source_label=promote_options[promote_idx],
                                        stage=compare_stage,
                                    )
                                    st.success("Config loaded into Backtest — switch to the 🔬 tab.")
                                if pr_col2.button("⚙️ Send to Optimizer", key=f"cmp_send_opt_{compare_stage}_{compare_metric}"):
                                    _load_params_into_optimizer(promote_row, param_keys, selected_name, param_grid, _defaults, _strat_groups)
                                    _finish_optimizer_load(selected_name)
                                _render_config_details(
                                    promote_row,
                                    param_keys,
                                    strategy.display_names,
                                    title="Selected config details",
                                )


# ============================================================================
# WALK-FORWARD TAB
# ============================================================================

with tab_wf:
    st.subheader("Walk-Forward Optimization")
    st.caption("Sliding-window IS/OOS validation — finds the best params on each IS slice, then evaluates OOS.")

    # ── Session state ─────────────────────────────────────────────────────────
    if 'wf_proc' not in st.session_state:
        st.session_state.wf_proc          = None
        st.session_state.wf_output_lines  = []
        st.session_state.wf_run_done      = False
        st.session_state.wf_reader_thread = None

    # ── Section A: Configure ──────────────────────────────────────────────────
    st.markdown("#### Configure")

    # Window controls
    _wf_c1, _wf_c2, _wf_c3 = st.columns(3)
    wf_is_window = _wf_c1.number_input(
        "IS window (days)", min_value=30, max_value=730, value=180, step=10,
        key="wf_is_window",
    )
    wf_oos_window = _wf_c2.number_input(
        "OOS window (days)", min_value=7, max_value=180, value=30, step=1,
        key="wf_oos_window",
    )
    wf_step = _wf_c3.number_input(
        "Step (days)", min_value=1, max_value=180, value=30, step=1,
        key="wf_step",
    )

    # Date range
    _wf_d1, _wf_d2 = st.columns(2)
    wf_start = _wf_d1.date_input(
        "Start date",
        value=data_start_input if data_start_input else None,
        key="wf_start",
    )
    wf_end = _wf_d2.date_input(
        "End date",
        value=data_end_input if data_end_input else None,
        key="wf_end",
    )

    # Rank / min-trades
    _wf_r1, _wf_r2, _ = st.columns([1, 1, 2])
    wf_rank_by = _wf_r1.selectbox(
        "Rank by",
        options=["sharpe", "profit_factor", "sortino", "net_pnl", "max_drawdown"],
        index=0,
        key="wf_rank_by",
    )
    wf_min_trades = _wf_r2.number_input(
        "Min trades", min_value=1, max_value=500, value=20, step=1,
        key="wf_min_trades",
    )

    # Param grid — read from session_state (written by tab_run on each render)
    _wf_grid_key = f"_wf_custom_grid_{selected_name}"
    _wf_grid = st.session_state.get(_wf_grid_key)
    _wf_has_grid = bool(_wf_grid and any(len(v) > 0 for v in _wf_grid.values()))
    if not _wf_has_grid:
        st.caption(
            "⚠️ Configure the parameter grid in the ⚙️ Configure & Run tab first "
            "(enable at least one param group's 'Include in sweep')."
        )
    else:
        _swept_count = sum(1 for v in _wf_grid.values() if len(v) > 0)
        st.caption(f"Using param grid from Configure & Run — {_swept_count} swept param(s).")

    # Slice preview table
    _wf_date_ok = bool(wf_start and wf_end and wf_start < wf_end)
    _wf_slices_preview = []
    _wf_invalid_dates_msg = None

    if wf_start and wf_end and wf_start >= wf_end:
        _wf_invalid_dates_msg = "End date must be after start date."
    elif _wf_date_ok:
        import datetime as _dt
        _cur = wf_start
        _idx = 0
        while True:
            _is_s = _cur
            _is_e = _is_s + _dt.timedelta(days=int(wf_is_window))
            _oos_s = _is_e
            _oos_e = _oos_s + _dt.timedelta(days=int(wf_oos_window))
            if _oos_e > wf_end:
                break
            _wf_slices_preview.append({
                "slice_idx": _idx,
                "is_start": str(_is_s),
                "is_end": str(_is_e),
                "oos_start": str(_oos_s),
                "oos_end": str(_oos_e),
            })
            _idx += 1
            _cur = _cur + _dt.timedelta(days=int(wf_step))

    _wf_n_slices = len(_wf_slices_preview)

    if _wf_invalid_dates_msg:
        st.error(_wf_invalid_dates_msg)
    elif _wf_date_ok:
        if _wf_n_slices == 0:
            st.error(
                "No valid slices — the date range is too short for the current IS + OOS window sizes. "
                "Increase the date range or reduce the window sizes."
            )
        else:
            st.caption(f"**{_wf_n_slices} slice(s)** will be evaluated.")
            _preview_df = pd.DataFrame(_wf_slices_preview[:50])
            st.dataframe(_preview_df, use_container_width=True, hide_index=True)
            if _wf_n_slices > 50:
                st.caption(f"Showing first 50 of {_wf_n_slices} slices.")

            # ── Pre-flight: data coverage check ───────────────────────────────
            with st.expander("Data coverage check", expanded=True):
                _wf_cov = _wf_data_coverage(symbol, bar_type, str(wf_start), str(wf_end))
                if _wf_cov.empty:
                    st.warning(f"No {bar_type} bars found for **{symbol}** in this range.")
                else:
                    _full_idx = pd.date_range(start=wf_start, end=wf_end - pd.Timedelta(days=1), freq="D")
                    _cov_full = _wf_cov.set_index("date").reindex(_full_idx, fill_value=0).rename_axis("date").reset_index()
                    _cov_full["bars"] = _cov_full["bars"].astype(int)

                    _zero_days = int((_cov_full["bars"] == 0).sum())
                    _total_days = len(_cov_full)
                    _coverage_pct = 100.0 * (1 - _zero_days / _total_days) if _total_days else 0.0
                    st.caption(
                        f"**{_total_days}** days in range · **{_zero_days}** with 0 bars · "
                        f"**{_coverage_pct:.1f}%** day-coverage"
                    )

                    fig_cov = px.bar(_cov_full, x="date", y="bars", height=180)
                    fig_cov.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title=None, yaxis_title="bars/day",
                    )
                    st.plotly_chart(fig_cov, use_container_width=True)

                    _affected = []
                    for _s in _wf_slices_preview:
                        _is_s = pd.Timestamp(_s["is_start"])
                        _is_e = pd.Timestamp(_s["is_end"])
                        _oos_s = pd.Timestamp(_s["oos_start"])
                        _oos_e = pd.Timestamp(_s["oos_end"])
                        _is_bars = int(_cov_full[(_cov_full["date"] >= _is_s) & (_cov_full["date"] < _is_e)]["bars"].sum())
                        _oos_bars = int(_cov_full[(_cov_full["date"] >= _oos_s) & (_cov_full["date"] < _oos_e)]["bars"].sum())
                        if _is_bars == 0 or _oos_bars == 0:
                            _affected.append({
                                "slice_idx": _s["slice_idx"],
                                "is_bars": _is_bars,
                                "oos_bars": _oos_bars,
                                "issue": "IS empty" if _is_bars == 0 else "OOS empty",
                            })
                    if _affected:
                        st.error(
                            f"⚠ {len(_affected)} of {_wf_n_slices} slices will be skipped due to DB data gaps "
                            "(reimport the affected date range to fix):"
                        )
                        st.dataframe(pd.DataFrame(_affected), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section B: Run / Stop ─────────────────────────────────────────────────
    _wf_is_running = (
        st.session_state.wf_proc is not None
        and st.session_state.wf_proc.poll() is None
    )

    _wf_run_blocked = (
        not _wf_has_grid
        or not _wf_date_ok
        or _wf_n_slices == 0
        or bool(_wf_invalid_dates_msg)
    )

    btn_col1, btn_col2 = st.columns([1, 4])
    wf_run_clicked  = btn_col1.button(
        "▶ Run Walk-Forward",
        disabled=_wf_is_running or _wf_run_blocked,
        type="primary",
        use_container_width=True,
        key="wf_run_btn",
    )
    wf_stop_clicked = btn_col2.button(
        "⏹ Stop",
        disabled=not _wf_is_running,
        use_container_width=False,
        key="wf_stop_btn",
    )

    if not _wf_has_grid and not _wf_is_running:
        st.caption("Run button disabled: no param grid configured in ⚙️ Configure & Run.")
    if not _wf_date_ok and not _wf_invalid_dates_msg:
        st.caption("Run button disabled: set both start and end dates.")
    if _wf_date_ok and _wf_n_slices == 0:
        st.caption("Run button disabled: no valid slices for this date range + window config.")

    if wf_stop_clicked and st.session_state.wf_proc:
        st.session_state.wf_output_lines.append("\n[Stop requested by user]\n")
        stopped = _terminate_proc_tree(st.session_state.wf_proc, st.session_state.wf_output_lines)
        st.session_state.wf_run_done = stopped
        st.rerun()

    if wf_run_clicked:
        _wf_cmd = [
            sys.executable, "-m", "strategy_platform.optimize.walk_forward",
            "--strategy",        selected_name,
            "--symbol",          symbol,
            "--bar-type",        bar_type,
            "--start",           str(wf_start),
            "--end",             str(wf_end),
            "--is-window-days",  str(int(wf_is_window)),
            "--oos-window-days", str(int(wf_oos_window)),
            "--step-days",       str(int(wf_step)),
            "--rank-by",         wf_rank_by,
            "--min-trades",      str(int(wf_min_trades)),
            "--param-grid",      json.dumps(_wf_grid),
        ]

        st.session_state.wf_output_lines  = [f"$ {' '.join(_wf_cmd)}\n"]
        st.session_state.wf_run_done      = False
        st.session_state.wf_proc          = subprocess.Popen(
            _wf_cmd,
            cwd    = os.path.abspath(PLATFORM_DIR),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text   = True,
            bufsize = 1,
            start_new_session = True,
            env    = {**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        def _wf_reader(proc, lines):
            """Background thread: drain stdout into lines list until EOF."""
            for line in proc.stdout:
                lines.append(line)
            rc = proc.wait()
            lines.append(
                "\n✅ Walk-forward complete!\n"
                if rc == 0 else f"\n❌ Failed (exit code {rc})\n"
            )

        wf_t = threading.Thread(
            target=_wf_reader,
            args=(st.session_state.wf_proc, st.session_state.wf_output_lines),
            daemon=True,
        )
        wf_t.start()
        st.session_state.wf_reader_thread = wf_t
        st.rerun()

    # ── Live output fragment ──────────────────────────────────────────────────
    if st.session_state.wf_proc is not None:

        @st.fragment(run_every="1s")
        def _wf_output_panel():
            proc   = st.session_state.wf_proc
            thread = st.session_state.get('wf_reader_thread')
            if proc is None:
                return

            if not st.session_state.wf_run_done:
                thread_done = thread is None or not thread.is_alive()
                proc_done   = proc.poll() is not None
                if thread_done and proc_done:
                    st.session_state.wf_run_done = True
                    _list_wf_runs.clear()
                    st.rerun()

            output_text = "".join(st.session_state.wf_output_lines)
            is_running  = not st.session_state.wf_run_done

            if is_running:
                st.markdown("🟡 **Running…**")
            elif proc.returncode == 0:
                st.success("✅ Walk-forward complete! See results below.")
            else:
                rc = proc.returncode if proc.returncode is not None else "?"
                st.error(f"❌ Walk-forward failed (exit code {rc}) — check the output below.")

            st.code(output_text or "Starting…", language=None)

        _wf_output_panel()

    st.markdown("---")

    # ── Section C: Results ────────────────────────────────────────────────────
    st.markdown("#### Results")

    _wf_files = _list_wf_runs(selected_name, sym_safe)

    if not _wf_files:
        st.info("No walk-forward results yet — configure and run above.")
    else:
        _wf_prefix_len = len(f"WF_{selected_name}_{sym_safe}_")

        def _fmt_wf_file(path: str) -> str:
            ts = os.path.basename(path)[_wf_prefix_len:-5]
            try:
                date_part = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}:{ts[13:15]}"
            except Exception:
                date_part = ts
            label = _get_wf_label(selected_name, sym_safe, ts)
            return f"{label} — {date_part}" if label else date_part

        _wf_pick = st.selectbox(
            f"Walk-forward runs ({len(_wf_files)})",
            options=_wf_files,
            format_func=_fmt_wf_file,
            key="wf_results_pick",
        )

        if _wf_pick:
            _wf_pick_ts = os.path.basename(_wf_pick)[_wf_prefix_len:-5]
            _wf_cur_label = _get_wf_label(selected_name, sym_safe, _wf_pick_ts)
            _lc1, _lc2 = st.columns([4, 1])
            _wf_lbl_input = _lc1.text_input(
                "Run label",
                value=_wf_cur_label,
                key=f"_wf_lbl_input_{_wf_pick_ts}",
                placeholder="Add a descriptive label…",
            )
            if _lc2.button("Save label", key=f"_wf_lbl_save_{_wf_pick_ts}", use_container_width=True):
                _set_wf_label(selected_name, sym_safe, _wf_pick_ts, _wf_lbl_input)
                st.toast("Label saved.", icon="✅")
                st.rerun()

        # Load selected WF JSON
        _wf_data = None
        if _wf_pick:
            try:
                with open(_wf_pick) as _fh:
                    _wf_data = json.load(_fh)
            except Exception as _e:
                st.error(f"Failed to load {_wf_pick}: {_e}")

        if _wf_data:
            _wf_meta   = _wf_data.get("meta", {})
            _wf_slices = _wf_data.get("slices", [])

            # 1. Header banner
            _wfe = _wf_meta.get("wfe")
            _wf_banner_col1, _wf_banner_col2 = st.columns([1, 3])
            with _wf_banner_col1:
                st.metric(
                    "Walk-Forward Efficiency",
                    f"{_wfe:.3f}" if _wfe is not None else "N/A",
                    help="WFE = avg(OOS metric) / avg(IS metric). Higher is better.",
                )
            with _wf_banner_col2:
                st.caption(
                    f"{_wf_meta.get('n_slices', len(_wf_slices))} slices · "
                    f"IS {_wf_meta.get('is_window_days', '?')}d / "
                    f"OOS {_wf_meta.get('oos_window_days', '?')}d / "
                    f"step {_wf_meta.get('step_days', '?')}d · "
                    f"Ranked by **{_wf_meta.get('rank_by', '?')}** · "
                    f"Symbol: **{_wf_meta.get('symbol', '?')}** · "
                    f"{_wf_meta.get('data_start', '')} → {_wf_meta.get('data_end', '')}"
                )

            # 2. Stitched OOS equity curve
            _all_trades = []
            _slice_boundaries = []
            for _sl in _wf_slices:
                _oos_trades = _sl.get("oos_trades") or []
                if _oos_trades:
                    for _tr in _oos_trades:
                        _tr_copy = dict(_tr)
                        _tr_copy["_slice_idx"] = _sl.get("slice_idx", 0)
                        _all_trades.append(_tr_copy)
                    if _sl.get("oos_start"):
                        _slice_boundaries.append(_sl["oos_start"])

            if _all_trades:
                _trades_df = pd.DataFrame(_all_trades)
                if "exit_time" in _trades_df.columns:
                    _trades_df["exit_time"] = pd.to_datetime(_trades_df["exit_time"])
                    _trades_df = _trades_df.sort_values("exit_time")
                    _trades_df["cumulative_pnl"] = _trades_df["pnl"].cumsum()

                    _fig_eq = px.line(
                        _trades_df,
                        x="exit_time",
                        y="cumulative_pnl",
                        title="Stitched OOS Equity (across all slices)",
                        labels={"exit_time": "Exit Time", "cumulative_pnl": "Cumulative P&L ($)"},
                    )
                    # Add vertical lines at slice boundaries (skip the first — it's the chart start)
                    for _bnd in _slice_boundaries[1:]:
                        try:
                            _fig_eq.add_vline(
                                x=str(pd.Timestamp(_bnd)),
                                line_dash="dash",
                                line_color="rgba(150,150,150,0.5)",
                            )
                        except Exception:
                            pass
                    _fig_eq.update_layout(height=380)
                    st.plotly_chart(_fig_eq, use_container_width=True)
            else:
                st.info("No OOS trades to plot — all slices had no valid combos or empty OOS.")

            # 3. Per-slice metrics table
            _slice_rows = []
            for _sl in _wf_slices:
                _ism = _sl.get("is_metrics") or {}
                _oosm = _sl.get("oos_metrics") or {}
                _oos_t = _sl.get("oos_trades") or []
                _slice_rows.append({
                    "slice_idx":    _sl.get("slice_idx"),
                    "is_start":     _sl.get("is_start", ""),
                    "is_end":       _sl.get("is_end", ""),
                    "oos_start":    _sl.get("oos_start", ""),
                    "oos_end":      _sl.get("oos_end", ""),
                    "is_sharpe":    _ism.get("sharpe"),
                    "is_net_pnl":   _ism.get("net_pnl"),
                    "is_trades":    _ism.get("trades"),
                    "oos_sharpe":   _oosm.get("sharpe"),
                    "oos_net_pnl":  _oosm.get("net_pnl"),
                    "oos_trades":   len(_oos_t),
                    "oos_win_rate": _oosm.get("win_rate"),
                })
            _slices_tbl = pd.DataFrame(_slice_rows)
            st.markdown("##### Per-Slice Metrics")
            st.dataframe(
                _slices_tbl,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "is_net_pnl":   st.column_config.NumberColumn("IS Net P&L",   format="$%.2f"),
                    "oos_net_pnl":  st.column_config.NumberColumn("OOS Net P&L",  format="$%.2f"),
                    "is_sharpe":    st.column_config.NumberColumn("IS Sharpe",    format="%.2f"),
                    "oos_sharpe":   st.column_config.NumberColumn("OOS Sharpe",   format="%.2f"),
                    "oos_win_rate": st.column_config.NumberColumn("OOS Win Rate", format="%.1f%%"),
                    "is_trades":    st.column_config.NumberColumn("IS Trades"),
                    "oos_trades":   st.column_config.NumberColumn("OOS Trades"),
                },
            )

            # 4. Param stability table
            _param_keys_wf = _wf_meta.get("param_keys") or []
            if _param_keys_wf:
                _valid_slices = [
                    _sl for _sl in _wf_slices if _sl.get("best_params") is not None
                ]
                if _valid_slices:
                    _stability_rows = {}
                    for _pk in _param_keys_wf:
                        _stability_rows[_pk] = {
                            f"slice_{_sl['slice_idx']}": _sl["best_params"].get(_pk)
                            for _sl in _valid_slices
                        }
                    _stability_df = pd.DataFrame(_stability_rows).T
                    _stability_df.index.name = "param"

                    # Try to color-shade numeric rows by relative deviation from median
                    try:
                        _num_cols = [c for c in _stability_df.columns
                                     if pd.to_numeric(_stability_df[c], errors='coerce').notna().all()]
                        if _num_cols:
                            _num_df = _stability_df[_num_cols].apply(pd.to_numeric, errors='coerce')
                            _medians = _num_df.median(axis=1)
                            _range = _num_df.max(axis=1) - _num_df.min(axis=1)

                            def _shade_row(row):
                                med = _medians[row.name]
                                rng = _range[row.name]
                                styles = []
                                for v in row:
                                    if rng == 0 or pd.isna(v):
                                        styles.append("")
                                    else:
                                        intensity = min(abs(v - med) / rng, 1.0)
                                        r = int(255 * intensity)
                                        g = int(100 * (1 - intensity))
                                        styles.append(f"background-color: rgba({r},{g},60,0.35)")
                                return styles

                            _styled = _stability_df.style.apply(_shade_row, axis=1, subset=_num_cols)
                            st.markdown("##### Param Stability (best param per slice)")
                            st.dataframe(_styled, use_container_width=True)
                        else:
                            raise ValueError("no numeric cols")
                    except Exception:
                        st.markdown("##### Param Stability (best param per slice)")
                        st.dataframe(_stability_df, use_container_width=True)


# ============================================================================
# BACKTEST TAB
# ============================================================================

with tab_bt:
    st.subheader("Single Configuration Backtest")
    st.caption("Test any parameter set on any date range — ideal for validating on unseen data after the OOS period.")

    # ── Saved runs — always visible, auto-loads on selection ─────────────────
    # Consume pending pick BEFORE the selectbox renders
    if "_bt_pick_pending" in st.session_state:
        st.session_state["bt_saved_pick"] = st.session_state.pop("_bt_pick_pending")
    _saved_bts = _list_saved_backtests(selected_name, sym_safe)
    if _saved_bts:
        _sl1, _sl2, _sl3 = st.columns([5, 1, 1])
        _bt_pick = _sl1.selectbox(
            f"Saved backtests ({len(_saved_bts)})",
            options=_saved_bts,
            format_func=lambda p: _fmt_bt_file(p, selected_name, sym_safe),
            key="bt_saved_pick",
        )
        # Auto-load when selection changes (compare to last-loaded token)
        _bt_loaded_key = f"_bt_last_loaded_{selected_name}_{sym_safe}"
        if st.session_state.get(_bt_loaded_key) != _bt_pick:
            try:
                _lp, _lm, _lr = _load_backtest(_bt_pick, selected_name, sym_safe)
                for _k, _v in _lp.items():
                    st.session_state[f"bt_{_k}"] = _v
                if _lm.get('start'):
                    try: st.session_state["bt_start"] = pd.Timestamp(_lm['start']).date()
                    except Exception: pass
                if _lm.get('end'):
                    try: st.session_state["bt_end"] = pd.Timestamp(_lm['end']).date()
                    except Exception: pass
                st.session_state["bt_result"]      = _lr
                st.session_state["bt_result_meta"] = _lm
                st.session_state["bt_loaded_from"] = f"Saved: {_fmt_bt_file(_bt_pick, selected_name, sym_safe)}"
                st.session_state[_bt_loaded_key]   = _bt_pick
            except Exception as _e:
                if _bt_pick:
                    st.toast(f"Could not load saved backtest: {_e}", icon="⚠️")

        _sl2.markdown("<div style='margin-top:27px'></div>", unsafe_allow_html=True)
        if _sl2.button("⚙️ Optimize", key="bt_saved_to_optimizer", use_container_width=True,
                       help="Send this backtest's params directly to Configure & Run"):
            try:
                _lp2, _, _ = _load_backtest(_bt_pick, selected_name, sym_safe)
                _load_params_into_optimizer(pd.Series(_lp2), param_keys, selected_name, param_grid, _defaults, _strat_groups)
                _finish_optimizer_load(selected_name)
            except Exception as _e:
                st.error(f"Failed: {_e}")

        # Delete with confirmation
        _del_key = f"_bt_del_confirm_{selected_name}_{sym_safe}"
        _sl3.markdown("<div style='margin-top:27px'></div>", unsafe_allow_html=True)
        if st.session_state.get(_del_key) == _bt_pick:
            if _sl3.button("Confirm ✓", key="bt_del_confirm_yes", use_container_width=True):
                try:
                    if _is_bt_db_token(_bt_pick):
                        results_store.delete_backtest(selected_name, sym_safe, _bt_token_ts(_bt_pick))
                    elif os.path.exists(_bt_pick):
                        os.remove(_bt_pick)
                        label_path = _bt_label_path(_bt_pick)
                        if os.path.exists(label_path):
                            os.remove(label_path)
                    st.session_state.pop(_del_key, None)
                    st.session_state.pop(_bt_loaded_key, None)
                    st.session_state.pop("bt_result", None)
                    st.rerun()
                except Exception as _de:
                    st.error(f"Delete failed: {_de}")
        else:
            if _sl3.button("🗑 Delete", key="bt_del_btn", use_container_width=True):
                st.session_state[_del_key] = _bt_pick
                st.rerun()

    # ── Load from OOS results ─────────────────────────────────────────────────
    if oos_df is not None:
        oos_sorted_bt = oos_df.sort_values("oos_net_pnl", ascending=False).reset_index(drop=True)
        bt_load_options = {
            f"#{i+1}  {config_label(row, param_keys)}  —  P&L {fmt_dollar(row.get('oos_net_pnl', 0))}  Sharpe {fmt_float(row.get('oos_sharpe', 0))}": i
            for i, row in oos_sorted_bt.iterrows()
        }
        lc1, lc2 = st.columns([4, 1])
        chosen_label = lc1.selectbox(
            "Load from OOS results",
            options=list(bt_load_options.keys()),
            index=0,
            key="bt_oos_picker",
        )
        if lc2.button("Load ▶", key="bt_load_from_oos", use_container_width=True):
            row = oos_sorted_bt.iloc[bt_load_options[chosen_label]]
            for key in param_keys:
                if key in row.index:
                    st.session_state[f"bt_{key}"] = row[key]
                    st.session_state.pop(f"_bt_val_{selected_name}_{key}", None)
            st.session_state["bt_loaded_from"] = chosen_label
            st.rerun()
        st.markdown("---")

    if st.session_state.get("bt_loaded_from"):
        st.info(f"Loaded: **{st.session_state.bt_loaded_from}**")

    # ── Parameter selectors ───────────────────────────────────────────────────
    st.markdown("#### Parameters")
    _bt_defaults = strategy.default_params
    _BT_EXCLUDED_KEYS = {'calculate_mode', 'tick_bar_size'}
    _BT_EXCLUDED_GROUPS = {'Bar Settings', 'Calculation'}

    def _bt_default_val(key: str) -> Any:
        """Return the best default value for a backtester param."""
        pg = param_grid.get(key)
        dv = _bt_defaults.get(key)
        if dv is not None:
            return dv
        if isinstance(pg, tuple):
            return pg[0]
        return pg[0] if pg else None

    # Initialise session state defaults only if not already set (avoids conflict with Load buttons)
    for key in param_keys:
        if f"bt_{key}" not in st.session_state:
            st.session_state[f"bt_{key}"] = _bt_default_val(key)

    def _bt_render_param(key: str, col) -> Any:
        """Render a single param widget in the given column, return value."""
        _bt_shadow = f"_bt_val_{selected_name}_{key}"
        pg = param_grid.get(key)
        label = _display_names.get(key, key)
        dv = _bt_default_val(key)
        cur = st.session_state.get(_bt_shadow, st.session_state.get(f"bt_{key}", dv))
        st.session_state.pop(f"bt_{key}", None)

        if isinstance(pg, tuple):
            # Numeric bounds — use number_input
            _mn, _mx, _stp = pg
            _is_flt = isinstance(_mn, float) or isinstance(_stp, float)
            _fmt = _float_fmt(key) if _is_flt else "%d"
            _cur_v = float(cur) if _is_flt else int(cur) if cur is not None else _mn
            val = col.number_input(label, min_value=_mn, max_value=_mx,
                                   value=_cur_v, step=_stp, format=_fmt)
        elif isinstance(pg, list) and pg and isinstance(pg[0], bool):
            # Bool — checkbox
            val = col.checkbox(label, value=bool(cur) if cur is not None else bool(pg[0]))
        else:
            # Categorical / time list — selectbox
            opts = list(pg) if pg else [cur]
            if cur not in opts:
                opts = [cur] + opts
            val = col.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0)

        st.session_state[_bt_shadow] = val
        return val

    # Groups to render as rows-of-related-params (bool + sub-params on one line)
    # Each entry: (bool_key, [sub_param_keys...])
    _BT_FILTER_ROWS = {
        'Filters': [
            ('enable_divergence_filter', ['divergence_lookback']),
            ('enable_bw_filter',         ['bw_period', 'bw_multiplier']),
            ('enable_time_filter',       ['trade_start_time', 'trade_end_time']),
        ],
    }

    bt_params: Dict[str, Any] = {}
    if _strat_groups:
        # Grouped display: group dropdown → params for that group
        _bt_grp_names = [
            g for g, ks in _strat_groups.items()
            if g not in _BT_EXCLUDED_GROUPS and any(k in param_grid and k not in _BT_EXCLUDED_KEYS for k in ks)
        ]
        _bt_grp_key   = f"bt_selgrp_{selected_name}"
        if _bt_grp_key not in st.session_state or st.session_state[_bt_grp_key] not in _bt_grp_names:
            st.session_state[_bt_grp_key] = _bt_grp_names[0] if _bt_grp_names else None
        if _bt_grp_names:
            _bt_grp = st.selectbox("Parameter group", options=_bt_grp_names, key=_bt_grp_key)
        else:
            _bt_grp = None

        if _bt_grp and _bt_grp in _BT_FILTER_ROWS:
            # Special layout: each filter on its own row
            for _bool_key, _sub_keys in _BT_FILTER_ROWS[_bt_grp]:
                _row_keys = [_bool_key] + [k for k in _sub_keys if k in param_grid]
                _row_cols = st.columns(max(len(_row_keys), 1))
                for i, key in enumerate(_row_keys):
                    if key in param_grid:
                        bt_params[key] = _bt_render_param(key, _row_cols[i])
        elif _bt_grp:
            _bt_grp_keys = [k for k in _strat_groups[_bt_grp] if k in param_grid and k not in _BT_EXCLUDED_KEYS]
            if _bt_grp_keys:
                bt_gcols = st.columns(min(len(_bt_grp_keys), 4))
                for i, key in enumerate(_bt_grp_keys):
                    bt_params[key] = _bt_render_param(key, bt_gcols[i % len(bt_gcols)])
            else:
                st.caption("No configurable parameters in this group.")

        # Collect all other groups' current values from shadow keys (not reset by Streamlit)
        for _gn, _gkeys in _strat_groups.items():
            if _gn == _bt_grp:
                continue
            for key in _gkeys:
                if key in param_grid and key not in bt_params:
                    _bt_shadow = f"_bt_val_{selected_name}_{key}"
                    bt_params[key] = st.session_state.get(_bt_shadow, st.session_state.get(f"bt_{key}", _bt_default_val(key)))
    else:
        # No groups — compact 4-column grid
        _ungrouped_keys = [k for k in param_keys if k not in _BT_EXCLUDED_KEYS]
        if _ungrouped_keys:
            bt_cols = st.columns(min(len(_ungrouped_keys), 4))
            for i, key in enumerate(_ungrouped_keys):
                bt_params[key] = _bt_render_param(key, bt_cols[i % len(bt_cols)])

    # ── Quick actions (above Data Source) ────────────────────────────────────
    _bt_act1, _bt_act2 = st.columns([3, 1])
    if _bt_act1.button("⚙️ Send to Optimizer →", key="bt_send_to_optimizer",
                       help="Load these parameters as fixed values in the Configure & Run tab."):
        _load_params_into_optimizer(pd.Series(bt_params), param_keys, selected_name, param_grid, _defaults, _strat_groups)
        _finish_optimizer_load(selected_name)
    if _bt_act2.button("💾 Save as default params", key="bt_save_defaults",
                       help="Write these parameters to default_params in strategy.py"):
        try:
            import sys as _sys
            _ar_path = os.path.join(PLATFORM_DIR, 'autoresearch')
            if _ar_path not in _sys.path:
                _sys.path.insert(0, _ar_path)
            from autoresearch_loop import _write_default_params
            merged = {**strategy.default_params, **bt_params}
            _write_default_params(selected_name, merged)
            st.success("default_params updated in strategy.py — restart dashboard to see the change in the sidebar.")
        except Exception as _e:
            st.error(f"Save failed: {_e}")

    st.markdown("---")

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown("#### Data Source")
    data_source = st.radio(
        "Load data from",
        ["MySQL (default)", "NinjaTrader CSV export"],
        horizontal=True,
        key="bt_data_source",
    )

    nt_csv_path = None
    if data_source == "NinjaTrader CSV export":
        st.info(
            "Export from NT8: right-click the chart data series → Export Data. "
            "Select 1-Minute bars for best accuracy. The file format is semicolon-delimited "
            "with no header (YYYYMMDD HHMMSS;O;H;L;C;V). Timestamps must be in UTC "
            "(NT exports in UTC regardless of display timezone)."
        )
        nt_csv_path = st.text_input(
            "Path to NT .txt file",
            placeholder="e.g. /mnt/e/Ticktemp/GC 08-25.Last.txt",
            key="bt_nt_csv_path",
        )
        st.caption(
            f"CSV will be resampled to **{bar_minute_inc}-minute** bars to match the sidebar bar-size selection."
        )

    # ── Date range ────────────────────────────────────────────────────────────
    st.markdown("#### Date Range")
    dc1, dc2 = st.columns(2)
    bt_start = dc1.date_input("Start date", value=None, key="bt_start",
                               help="First bar of data to load")
    bt_end   = dc2.date_input("End date",   value=None, key="bt_end",
                               help="Last bar of data to load")
    tc1, tc2 = st.columns(2)
    bt_start_time = tc1.time_input("Start time (ET)", value=datetime.time(0, 0), key="bt_start_time",
                                    help="Start time within the start date (ET). Leave at 00:00 to load from market open.")
    bt_end_time   = tc2.time_input("End time (ET)",   value=datetime.time(23, 59), key="bt_end_time",
                                    help="End time within the end date (ET). Leave at 23:59 to load through market close.")

    if st.session_state.pop("_bt_label_reset", False):
        st.session_state["_bt_label_val"] = ""
    bt_label = st.text_input(
        "Backtest label",
        placeholder="e.g. baseline pre-optimization",
        value=st.session_state.get("_bt_label_val", ""),
        help="Saved as 'BAC — <your label>'. Always written — leave blank for just 'BAC'.",
    )
    st.session_state["_bt_label_val"] = bt_label
    # Inject sidebar calculate_mode into bt_params for tick strategies
    if bar_type == 'tick':
        _sb_calc_mode = st.session_state.get(
            f"sidebar_calc_mode_{selected_name}",
            _load_prefs().get(f"calc_mode_{selected_name}", "on_bar_close"),
        )
        bt_params["calculate_mode"] = _sb_calc_mode

    run_bt = st.button("▶ Run Backtest", type="primary", key="bt_run")

    if run_bt:
        if not bt_start or not bt_end:
            st.error("Please select both a start and end date.")
        elif bt_end < bt_start:
            st.error("End date must not be before start date.")
        elif data_source == "NinjaTrader CSV export" and not nt_csv_path:
            st.error("Please enter the path to your NT CSV file.")
        elif data_source == "NinjaTrader CSV export" and not os.path.exists(nt_csv_path):
            st.error(f"File not found: {nt_csv_path}")
        else:
            with st.spinner("Loading data & running backtest…"):
                try:
                    _bt_start_str = f"{bt_start}T{bt_start_time}"
                    _bt_end_str   = f"{bt_end}T{bt_end_time}"
                    if data_source == "NinjaTrader CSV export":
                        from strategy_platform.data.loader import load_nt_csv
                        _csv_resample = f"{bar_minute_inc}min" if bar_minute_inc else "1min"
                        df_bt = load_nt_csv(
                            nt_csv_path,
                            resample=_csv_resample,
                            start=_bt_start_str,
                            end=_bt_end_str,
                        )
                        # CSV is already at target bar size — skip the strategy's _prepare_df resample
                        strategy.bar_type = 'time'
                        data_source_label = f"NT CSV: {os.path.basename(nt_csv_path)} ({_csv_resample})"
                    elif bar_type == '1m':
                        from strategy_platform.data.loader import load_1m
                        df_bt = load_1m(symbol, start=_bt_start_str, end=_bt_end_str, host=db_host)
                        if bar_minute_inc and bar_minute_inc > 1:
                            df_bt = df_bt.resample(f"{bar_minute_inc}min", label='right', closed='right').agg(
                                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                            ).dropna()
                        data_source_label = f"MySQL {bar_minute_inc or 1}M"
                    elif bar_type == 'tick':
                        from strategy_platform.data.loader import load_tick_bars
                        _tick_sz = bt_params.get('tick_bar_size', st.session_state.get("sb_tick_inc", getattr(strategy, 'tick_bar_size', 500)))
                        df_bt = load_tick_bars(symbol, bar_size=int(_tick_sz), start=_bt_start_str, end=_bt_end_str, host=db_host)
                        bt_params['_symbol'] = symbol  # needed by on_each_tick raw-tick loader
                        data_source_label = f"MySQL tick ({_tick_sz}-tick bars)"
                    else:
                        from strategy_platform.data.loader import load_5m
                        df_bt = load_5m(symbol, start=_bt_start_str, end=_bt_end_str, host=db_host)
                        data_source_label = "MySQL 5M"
                    sess_bt    = strategy.prepare_data(df_bt)
                    result_bt  = strategy.run_backtest_prepared(sess_bt, bt_params)
                    _bt_meta = {
                        "start":       str(bt_start),
                        "end":         str(bt_end),
                        "start_time":  str(bt_start_time),
                        "end_time":    str(bt_end_time),
                        "bars":        len(df_bt),
                        "sessions":    len(sess_bt) if isinstance(sess_bt, list) else "N/A",
                        "data_source": data_source_label,
                    }
                    st.session_state["bt_result"]      = result_bt
                    st.session_state["bt_result_meta"] = _bt_meta
                    _saved_bt_path = _save_backtest(selected_name, symbol, bt_params, _bt_meta, result_bt)
                    if _saved_bt_path:
                        _user_bt_label = st.session_state.get("_bt_label_val", "").strip()
                        _full_bt_label = f"BAC — {_user_bt_label}" if _user_bt_label else "BAC"
                        _set_bt_label(_saved_bt_path, _full_bt_label, selected_name, sym_safe)
                        st.session_state["_bt_label_reset"] = True
                        _list_saved_backtests.clear()
                        st.session_state[f"_bt_last_loaded_{selected_name}_{sym_safe}"] = _saved_bt_path
                        st.session_state["_bt_pick_pending"] = _saved_bt_path
                        st.rerun()
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    st.session_state.pop("bt_result", None)

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("bt_result"):
        result = st.session_state["bt_result"]
        meta   = st.session_state.get("bt_result_meta", {})

        st.markdown("---")
        st.info(
            f"**Period:** {meta.get('start')} → {meta.get('end')}  ·  "
            f"**Bars:** {meta.get('bars', '?'):,}  ·  "
            f"**Sessions:** {meta.get('sessions', '?')}  ·  "
            f"**Data:** {meta.get('data_source', 'MySQL')}"
        )

        n = int(result.get("total_trades", 0) or 0)
        if n == 0:
            st.warning("No trades generated for this parameter set and date range.")
        else:
            st.markdown(f"#### Performance — {n} trades")
            # BUG-1: fallback largest_win/loss from trades DataFrame if missing
            trades_df = result.get("trades")
            if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
                if "largest_win" not in result or pd.isna(result.get("largest_win")):
                    result["largest_win"] = trades_df["pnl"].max()
                if "largest_loss" not in result or pd.isna(result.get("largest_loss")):
                    result["largest_loss"] = trades_df["pnl"].min()
            _render_nt_metrics(result)

            with st.expander("Parameters used", expanded=False):
                _params_ser = pd.Series(bt_params)
                _params_df = _config_details_df(_params_ser, param_keys, _display_names)
                if not _params_df.empty:
                    st.dataframe(_params_df, use_container_width=True, hide_index=True, height=df_height(len(_params_df)))

            # Equity curve + trade list
            trades_df = result.get("trades")
            if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
                st.markdown("#### Equity Curve")
                eq_mode = st.radio(
                    "X axis", ["By date", "By trade"],
                    horizontal=True, key="bt_eq_mode",
                    help="'By trade' gives equal spacing per trade — clearer when trade count is low.",
                )
                _time_col = "entry_time" if "entry_time" in trades_df.columns else "session_date"
                _eq_cols = ["session_date", "pnl"] if _time_col == "session_date" else ["session_date", "entry_time", "pnl"]
                eq = trades_df[_eq_cols].copy().reset_index(drop=True)
                eq["Cumulative P&L"] = eq["pnl"].cumsum()
                # v1: kept for one revision in case of revert
                # if eq_mode == "By trade":
                #     import numpy as _np
                #     eq["Trade #"] = eq.index + 1
                #     eq[_time_col] = eq[_time_col].astype(str)
                #     coef = _np.polyfit(eq["Trade #"].values, eq["Cumulative P&L"].values, 1)
                #     eq["Trend"] = _np.polyval(coef, eq["Trade #"].values)
                #     fig = px.line(
                #         eq, x="Trade #", y="Cumulative P&L",
                #         hover_data={_time_col: True, "pnl": True, "Trade #": False},
                #         labels={"Cumulative P&L": "Cumulative P&L ($)", _time_col: "Date", "pnl": "Trade P&L"},
                #     )
                #     fig.add_scatter(
                #         x=eq["Trade #"], y=eq["Trend"],
                #         mode="lines", name="Trend",
                #         line=dict(color="rgba(100,100,255,0.5)", width=2, dash="dot"),
                #     )
                # else:
                #     fig = px.line(
                #         eq, x="session_date", y="Cumulative P&L",
                #         labels={"session_date": "Date"},
                #     )
                # fig.add_hline(y=0, line_dash="dash", line_color="gray")
                # fig.update_layout(height=380, legend=dict(orientation="h", y=1.05))
                # st.plotly_chart(fig, use_container_width=True)
                if eq_mode == "By trade":
                    _eq_data = [
                        {"time": int(i + 1), "value": float(v)}
                        for i, v in enumerate(eq["Cumulative P&L"].values)
                    ]
                else:
                    _eq_ts = pd.to_datetime(eq[_time_col], format="mixed", utc=True)
                    _eq_series = dict(zip(_eq_ts.map(lambda t: int(t.timestamp())), eq["Cumulative P&L"].values))
                    _eq_data = [{"time": int(k), "value": float(v)} for k, v in sorted(_eq_series.items())]
                _eq_data_json = json.dumps(_eq_data)
                _eq_html = f"""
<div id="eqchart" style="width:100%;height:360px;"></div>
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
  var container = document.getElementById('eqchart');
  var chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth,
    height: 360,
    layout: {{
      background: {{ color: '#0a0a0c' }},
      textColor: '#e2e6ee',
      fontFamily: 'IBM Plex Mono, monospace'
    }},
    grid: {{
      vertLines: {{ color: '#141418' }},
      horzLines: {{ color: '#141418' }}
    }},
    rightPriceScale: {{ borderColor: '#1a1a22' }},
    timeScale: {{ borderColor: '#1a1a22', timeVisible: true, secondsVisible: false }},
    crosshair: {{ mode: 0 }}
  }});
  var areaSeries = chart.addAreaSeries({{
    lineColor: '#39ff8a',
    topColor: 'rgba(57,255,138,0.25)',
    bottomColor: 'rgba(57,255,138,0.0)',
    lineWidth: 2
  }});
  areaSeries.setData({_eq_data_json});
  chart.timeScale().fitContent();
  var ro = new ResizeObserver(function() {{
    chart.applyOptions({{ width: container.clientWidth }});
  }});
  ro.observe(container);
}})();
</script>
"""
                import streamlit.components.v1 as _components
                _components.html(_eq_html, height=360)

                # ── Monthly P&L breakdown ─────────────────────────────────
                monthly_data = trades_df.copy()
                monthly_data["_dt"]   = pd.to_datetime(monthly_data[_time_col], format="mixed")
                monthly_data["_year"] = monthly_data["_dt"].dt.year
                monthly_data["_mon"]  = monthly_data["_dt"].dt.month
                monthly_pnl_grp = monthly_data.groupby(["_year", "_mon"])["pnl"].sum().reset_index()

                # Bar chart
                monthly_pnl_grp["Month"] = monthly_pnl_grp.apply(
                    lambda r: f"{int(r['_year'])}-{int(r['_mon']):02d}", axis=1)
                st.markdown("#### Monthly P&L")
                fig_m = px.bar(
                    monthly_pnl_grp, x="Month", y="pnl",
                    color="pnl", color_continuous_scale=["#e74c3c", "#2ecc71"],
                    text=monthly_pnl_grp["pnl"].apply(lambda x: f"${x:,.0f}"),
                    labels={"pnl": "Net P&L ($)"},
                )
                fig_m.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_m.update_layout(height=300, coloraxis_showscale=False, xaxis_tickangle=-45)
                fig_m.update_traces(textposition="outside")
                st.plotly_chart(fig_m, use_container_width=True)

                # Calendar heatmap (month × year)
                years  = sorted(monthly_pnl_grp["_year"].unique())
                months = list(range(1, 13))
                month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                pivot_cal = monthly_pnl_grp.pivot(index="_year", columns="_mon", values="pnl").reindex(
                    index=years, columns=months
                )
                if len(years) >= 1:
                    st.markdown("#### Monthly P&L Calendar")
                    fig_cal = px.imshow(
                        pivot_cal,
                        color_continuous_scale=["#e74c3c", "#ffffff", "#2ecc71"],
                        color_continuous_midpoint=0,
                        labels={"x": "Month", "y": "Year", "color": "P&L ($)"},
                        x=month_names,
                        y=[str(y) for y in years],
                        text_auto=".0f",
                        aspect="auto",
                    )
                    fig_cal.update_layout(height=max(150, len(years) * 60))
                    st.plotly_chart(fig_cal, use_container_width=True)

                # ── Day-of-week P&L ───────────────────────────────────────
                dow_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                dow_data = [
                    {"Day": d, "Net P&L ($)": result.get(f"{d[:3].lower()}_pnl", 0),
                     "Trades": result.get(f"{d[:3].lower()}_trades", 0)}
                    for d in dow_days
                ]
                dow_df = pd.DataFrame(dow_data)
                st.markdown("#### Day-of-Week P&L")
                fig_d = px.bar(
                    dow_df, x="Day", y="Net P&L ($)",
                    color="Net P&L ($)", color_continuous_scale=["#e74c3c", "#2ecc71"],
                    text=dow_df["Net P&L ($)"].apply(lambda x: f"${x:,.0f}").tolist(),
                    hover_data=["Trades"],
                )
                fig_d.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_d.update_layout(height=280, coloraxis_showscale=False)
                fig_d.update_traces(textposition="outside")
                st.plotly_chart(fig_d, use_container_width=True)

                st.markdown("#### Trade List")
                tl = trades_df.copy()
                display_cols = [c for c in [
                    "entry_time", "exit_time", "direction",
                    "entry_price", "exit_price", "pnl_ticks", "pnl", "exit_reason",
                ] if c in tl.columns]
                tl = tl[display_cols].copy()
                if "entry_time" in tl.columns:
                    tl["entry_time"] = pd.to_datetime(tl["entry_time"], format="mixed").dt.strftime("%d/%m/%Y %H:%M")
                if "exit_time" in tl.columns:
                    tl["exit_time"] = pd.to_datetime(tl["exit_time"], format="mixed").dt.strftime("%d/%m/%Y %H:%M")
                if "pnl" in tl.columns:
                    tl["pnl"] = tl["pnl"].apply(lambda x: f"${x:,.0f}")
                if "pnl_ticks" in tl.columns:
                    tl["pnl_ticks"] = tl["pnl_ticks"].apply(lambda x: f"{x:+.0f}")
                if "entry_price" in tl.columns:
                    tl["entry_price"] = tl["entry_price"].apply(lambda x: f"{x:.2f}")
                if "exit_price" in tl.columns:
                    tl["exit_price"] = tl["exit_price"].apply(lambda x: f"{x:.2f}")
                tl = _humanize_columns(tl, strategy.display_names)
                st.dataframe(tl, use_container_width=True, hide_index=True, height=df_height(10))

                export_text = _build_export_text(result, bt_params, meta)
                st.download_button(
                    label="⬇ Export results (txt)",
                    data=export_text,
                    file_name="backtest_results.txt",
                    mime="text/plain",
                )


# ============================================================================
# AUTORESEARCH TAB
# ============================================================================

AUTORESEARCH_DIR = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'autoresearch')
)
RESULTS_TSV = os.path.join(AUTORESEARCH_DIR, 'results.tsv')


def _list_ar_runs(strategy_name: str, sym_safe: str) -> List[str]:
    """Return timestamps of saved AR runs for this strategy, newest first."""
    pattern = os.path.join(REPORTS_DIR, f"AR_{strategy_name}_{sym_safe}_*.tsv")
    files   = glob.glob(pattern)
    prefix_len = len(f"AR_{strategy_name}_{sym_safe}_")
    tss = []
    for f in files:
        name = os.path.basename(f)
        ts   = name[prefix_len:-4]
        if len(ts) == 13:
            tss.append(ts)
    return sorted(tss, reverse=True)


def _load_ar_run(strategy_name: str, sym_safe: str, ts: str) -> Optional[pd.DataFrame]:
    """Load a specific AR run TSV, skipping # metadata lines."""
    path = os.path.join(REPORTS_DIR, f"AR_{strategy_name}_{sym_safe}_{ts}.tsv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, sep='\t', comment='#')
    except Exception:
        return None


def _load_ar_run_meta(strategy_name: str, sym_safe: str, ts: str) -> dict:
    """Parse # metadata lines from an AR run file."""
    path = os.path.join(REPORTS_DIR, f"AR_{strategy_name}_{sym_safe}_{ts}.tsv")
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                break
            line = line[1:].strip()
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
    return meta


def _load_results_tsv() -> Optional[pd.DataFrame]:
    if not os.path.exists(RESULTS_TSV):
        return None
    try:
        return pd.read_csv(RESULTS_TSV, sep='\t')
    except Exception:
        return None


with tab_ar:
    st.subheader("Autoresearch — Autonomous Parameter Optimisation")
    st.caption(
        "Uses an LLM (Haiku by default) to propose one parameter change per generation, "
        "tests it against the IS backtest, and keeps improvements automatically. "
        "~$0.0005 per generation with Haiku."
    )

    # ── Editable starting params ──────────────────────────────────────────────
    with st.expander("Starting parameters (current default_params)", expanded=True):
        try:
            from autoresearch.autoresearch_loop import _read_default_params, _write_default_params
        except ImportError:
            sys.path.insert(0, AUTORESEARCH_DIR)
            from autoresearch_loop import _read_default_params, _write_default_params

        dp = _read_default_params(selected_name)
        edited_params: Dict[str, Any] = {}

        if _strat_groups:
            # Group dropdown — only groups with at least one key in dp
            _ar_gnames = [g for g, ks in _strat_groups.items() if any(k in dp for k in ks)]
            _ar_selgrp_key = f"ar_selgrp_{selected_name}"
            if _ar_selgrp_key not in st.session_state or st.session_state[_ar_selgrp_key] not in _ar_gnames:
                st.session_state[_ar_selgrp_key] = _ar_gnames[0]
            _ar_grp = st.selectbox("Parameter Group", options=_ar_gnames, key=_ar_selgrp_key,
                                   format_func=lambda g: g)
            _ar_grp_keys = [k for k in _strat_groups[_ar_grp] if k in dp]

            # Render selected group's params (4 per row)
            _ar_cols_n = min(len(_ar_grp_keys), 4)
            _ar_cols = st.columns(_ar_cols_n) if _ar_cols_n > 0 else []
            for i, k in enumerate(_ar_grp_keys):
                v = dp[k]
                lbl = _display_names.get(k, k)
                col = _ar_cols[i % _ar_cols_n] if _ar_cols else st
                if isinstance(v, bool):
                    edited_params[k] = col.checkbox(lbl, value=v, key=f"ar_dp_{k}")
                elif isinstance(v, int):
                    edited_params[k] = col.number_input(lbl, value=v, step=1, key=f"ar_dp_{k}")
                elif isinstance(v, float):
                    edited_params[k] = col.number_input(lbl, value=v, step=0.05, format="%.2f", key=f"ar_dp_{k}")
                else:
                    edited_params[k] = col.text_input(lbl, value=str(v), key=f"ar_dp_{k}")

            # Collect other groups' current values from session state or dp defaults
            for _ag, _agkeys in _strat_groups.items():
                if _ag == _ar_grp:
                    continue
                for k in _agkeys:
                    if k in dp and k not in edited_params:
                        edited_params[k] = st.session_state.get(f"ar_dp_{k}", dp[k])
        else:
            # No groups — flat 4-per-row display
            bool_keys  = [k for k, v in dp.items() if isinstance(v, bool)]
            other_keys = [k for k in dp if k not in bool_keys]
            if bool_keys:
                _bc = st.columns(min(len(bool_keys), 4))
                for i, k in enumerate(bool_keys):
                    lbl = _display_names.get(k, k)
                    edited_params[k] = _bc[i % len(_bc)].checkbox(lbl, value=dp[k], key=f"ar_dp_{k}")
            chunk_size = 4
            for row_start in range(0, len(other_keys), chunk_size):
                chunk = other_keys[row_start:row_start + chunk_size]
                cols  = st.columns(chunk_size)
                for i, k in enumerate(chunk):
                    v = dp[k]
                    lbl = _display_names.get(k, k)
                    if isinstance(v, int):
                        edited_params[k] = cols[i].number_input(lbl, value=v, step=1, key=f"ar_dp_{k}")
                    elif isinstance(v, float):
                        edited_params[k] = cols[i].number_input(lbl, value=v, step=0.05, format="%.2f", key=f"ar_dp_{k}")
                    else:
                        edited_params[k] = cols[i].text_input(lbl, value=str(v), key=f"ar_dp_{k}")

        if st.button("💾 Save params to strategy", key="ar_save_params"):
            _write_default_params(selected_name, edited_params)
            st.success("Saved — these will be used as the baseline for autoresearch.")

    # ── Config ───────────────────────────────────────────────────────────────
    st.markdown("#### Settings")
    ar_col1, ar_col2, ar_col3, ar_col4, ar_col5 = st.columns(5)

    ar_model = ar_col1.selectbox(
        "Model",
        options=["haiku", "sonnet"],
        index=0,
        help="haiku = cheapest (~$0.0005/gen). sonnet = smarter but ~10× more expensive.",
        key="ar_model",
    )
    ar_max_gens = ar_col2.number_input(
        "Max generations (0 = unlimited)",
        min_value=0, max_value=10000, value=100, step=10,
        key="ar_max_gens",
    )
    ar_min_trades = ar_col3.number_input(
        "Min trades",
        min_value=1, max_value=200, value=10, step=1,
        help="Reject runs with fewer trades than this on the IS slice.",
        key="ar_min_trades",
    )
    ar_start = ar_col4.date_input("IS data start", value=None, key="ar_start",
                                   help="Leave blank for 2 years ago")
    ar_end   = ar_col5.date_input("IS data end",   value=None, key="ar_end",
                                   help="Leave blank for 1 year ago (leaves OOS headroom)")

    st.caption(
        f"**Estimated cost:** "
        f"{'~$' + str(round(ar_max_gens * 0.0005, 2)) if ar_max_gens > 0 else 'unlimited'} "
        f"with {ar_model}  ·  "
        f"Strategy: **{selected_name}** ({symbol})  ·  "
        f"Results log: `autoresearch/results.tsv`"
    )
    st.markdown("---")

    # ── Run / Stop ────────────────────────────────────────────────────────────
    ar_is_running = (
        st.session_state.ar_proc is not None
        and st.session_state.ar_proc.poll() is None
    )

    ar_btn1, ar_btn2 = st.columns([1, 4])
    ar_run_clicked  = ar_btn1.button("▶ Start", disabled=ar_is_running,
                                      type="primary", use_container_width=True, key="ar_run")
    ar_stop_clicked = ar_btn2.button("⏹ Stop",  disabled=not ar_is_running,
                                      use_container_width=False, key="ar_stop")

    if ar_stop_clicked and st.session_state.ar_proc:
        st.session_state.ar_output_lines.append("\n[Stop requested by user]\n")
        stopped = _terminate_proc_tree(st.session_state.ar_proc, st.session_state.ar_output_lines)
        st.session_state.ar_run_done = stopped
        st.rerun()

    if ar_run_clicked:
        cmd = [
            sys.executable,
            os.path.join(AUTORESEARCH_DIR, 'autoresearch_loop.py'),
            '--strategy', selected_name,
            '--symbol',   symbol,
            '--model',    ar_model,
        ]
        cmd += ['--bar-type', bar_type]
        if ar_max_gens:
            cmd += ['--max-gens', str(int(ar_max_gens))]
        if ar_min_trades:
            cmd += ['--min-trades', str(int(ar_min_trades))]
        if ar_start:
            cmd += ['--start', str(ar_start)]
        if ar_end:
            cmd += ['--end', str(ar_end)]

        st.session_state.ar_output_lines  = [f"$ {' '.join(cmd)}\n"]
        st.session_state.ar_run_done      = False
        st.session_state.ar_proc          = subprocess.Popen(
            cmd,
            cwd     = os.path.abspath(PLATFORM_DIR),
            stdout  = subprocess.PIPE,
            stderr  = subprocess.STDOUT,
            text    = True,
            bufsize = 1,
            start_new_session = True,
            env     = {**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        def _ar_reader(proc, lines):
            for line in proc.stdout:
                lines.append(line)
            rc = proc.wait()
            lines.append(
                "\n✅ Autoresearch complete.\n"
                if rc == 0 else f"\n❌ Failed (exit code {rc})\n"
            )

        ar_t = threading.Thread(
            target=_ar_reader,
            args=(st.session_state.ar_proc, st.session_state.ar_output_lines),
            daemon=True,
        )
        ar_t.start()
        st.session_state.ar_reader_thread = ar_t
        st.rerun()

    # ── Live output ───────────────────────────────────────────────────────────
    if st.session_state.ar_proc is not None:

        @st.fragment(run_every="2s")
        def _ar_output_panel():
            proc   = st.session_state.ar_proc
            thread = st.session_state.get('ar_reader_thread')
            if proc is None:
                return
            if not st.session_state.ar_run_done:
                if (thread is None or not thread.is_alive()) and proc.poll() is not None:
                    st.session_state.ar_run_done = True

            output_text = "".join(st.session_state.ar_output_lines)
            is_running  = not st.session_state.ar_run_done

            if is_running:
                st.markdown("🟡 **Running…**")
            elif proc.returncode == 0:
                st.success("✅ Autoresearch finished. See results below.")
            else:
                st.error(f"❌ Failed (exit code {proc.returncode})")

            st.code(output_text or "Starting…", language=None)

        _ar_output_panel()

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Results")

    # Run selector
    _sym_safe_ar = symbol.replace('=', '_')
    _ar_run_tss  = _list_ar_runs(selected_name, _sym_safe_ar)
    if _ar_run_tss:
        selected_ar_ts = st.selectbox(
            "Run",
            options=["Live (results.tsv)"] + _ar_run_tss,
            format_func=lambda ts: ts if ts == "Live (results.tsv)" else
                f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}  {ts[9:11]}:{ts[11:]}",
            index=1,
            key="ar_run_selector",
        )
    else:
        selected_ar_ts = "Live (results.tsv)"

    @st.fragment(run_every="5s" if st.session_state.ar_proc is not None else None)
    def _ar_results_panel():
        if selected_ar_ts == "Live (results.tsv)":
            df = _load_results_tsv()
        else:
            df = _load_ar_run(selected_name, _sym_safe_ar, selected_ar_ts)
            meta = _load_ar_run_meta(selected_name, _sym_safe_ar, selected_ar_ts)
            if meta:
                st.caption(
                    f"**Strategy:** {meta.get('strategy','')}  ·  "
                    f"**Model:** {meta.get('model','')}  ·  "
                    f"**IS:** {meta.get('is_start','')} → {meta.get('is_end','')}"
                )
        if df is None or df.empty:
            st.info("No results yet. Click **▶ Start** to begin.")
            return

        # Summary metrics
        kept_df    = df[df['kept'] == 'yes']
        best_row   = kept_df.loc[kept_df['sharpe'].idxmax()] if not kept_df.empty else None
        baseline   = df[df['param_changed'] == 'baseline']
        base_sharpe = float(baseline['sharpe'].iloc[0]) if not baseline.empty else None

        mc1, mc2, mc3, mc4 = st.columns(4)
        total_gens   = int(df['gen'].max()) if 'gen' in df.columns else 0
        n_kept       = len(kept_df)
        best_sharpe  = float(best_row['sharpe']) if best_row is not None else float('nan')
        delta        = best_sharpe - base_sharpe if base_sharpe is not None else float('nan')

        mc1.metric("Generations run",   total_gens)
        mc2.metric("Improvements kept", n_kept)
        mc3.metric("Best Sharpe",       fmt_float(best_sharpe))
        mc4.metric("Δ vs baseline",     fmt_float(delta, 4),
                   delta_color="normal" if delta >= 0 else "inverse")

        # ── Best accumulated param set ────────────────────────────────────────
        if not kept_df.empty:
            st.markdown("#### Best parameters found")
            # Replay all kept changes onto the starting default_params
            try:
                sys.path.insert(0, AUTORESEARCH_DIR)
                from autoresearch_loop import _read_default_params
                best_params = dict(_read_default_params(selected_name))
            except Exception:
                best_params = {}

            kept_ordered = kept_df[
                (kept_df['param_changed'] != 'baseline') &
                (kept_df['gen'] <= best_row['gen'])
            ].sort_values('gen')
            for _, row in kept_ordered.iterrows():
                p = row.get('param_changed')
                v = row.get('new_value')
                if p and p != 'baseline' and p in best_params:
                    orig = best_params[p]
                    try:
                        if isinstance(orig, bool):
                            v = str(v).lower() == 'true'
                        elif isinstance(orig, int):
                            v = int(float(v))
                        elif isinstance(orig, float):
                            v = float(v)
                    except Exception:
                        pass
                    best_params[p] = v

            # Best params — wide 4-column table + metrics side by side
            bp_col, info_col = st.columns([3, 1])
            with bp_col:
                # Render as a compact 4-column grid (label: value pairs)
                _bp_items = [(k, str(v)) for k, v in best_params.items()]
                _bp_chunk = 4
                for _bpi in range(0, len(_bp_items), _bp_chunk):
                    _chunk = _bp_items[_bpi:_bpi + _bp_chunk]
                    _bpcols = st.columns(_bp_chunk)
                    for _ci, (_bpk, _bpv) in enumerate(_chunk):
                        _bpcols[_ci].metric(_display_names.get(_bpk, _bpk), _bpv)
            with info_col:
                info_col.metric("Best Sharpe",   fmt_float(best_sharpe))
                info_col.metric("Trades",        int(best_row.get('trades', 0)))
                info_col.metric("Net P&L",       fmt_dollar(float(best_row.get('net_pnl', 0))))
                info_col.metric("Δ vs baseline", fmt_float(delta, 4))

        # Sharpe over generations chart
        plot_df = df[df['param_changed'] != 'baseline'].copy()
        if not plot_df.empty:
            plot_df['kept_label'] = plot_df['kept'].map({'yes': 'Kept', 'no': 'Reverted'})
            st.markdown("#### Sharpe by generation")
            fig = px.scatter(
                plot_df, x='gen', y='sharpe',
                color='kept_label',
                color_discrete_map={'Kept': '#2ecc71', 'Reverted': '#e74c3c'},
                hover_data=['param_changed', 'old_value', 'new_value', 'trades'],
                labels={'gen': 'Generation', 'sharpe': 'IS Sharpe', 'kept_label': ''},
            )
            # Running best line
            if not kept_df.empty:
                running_best = (
                    kept_df.sort_values('gen')
                    .assign(running_best=lambda d: d['sharpe'].cummax())
                )
                fig.add_scatter(
                    x=running_best['gen'], y=running_best['running_best'],
                    mode='lines', name='Running best',
                    line=dict(color='#3498db', width=2, dash='dot'),
                )
            if base_sharpe is not None:
                fig.add_hline(y=base_sharpe, line_dash='dash', line_color='gray',
                              annotation_text='Baseline')
            fig.update_layout(height=380, legend=dict(orientation='h', y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        # Full log table
        st.markdown("#### Generation log")
        show_cols = [c for c in ['gen', 'timestamp', 'sharpe', 'net_pnl', 'trades',
                                  'win_rate', 'param_changed', 'old_value', 'new_value', 'kept']
                     if c in df.columns]
        pretty = df[show_cols].copy().sort_values('gen', ascending=False)
        if 'net_pnl' in pretty.columns:
            pretty['net_pnl'] = pd.to_numeric(pretty['net_pnl'], errors='coerce').apply(fmt_dollar)
        if 'win_rate' in pretty.columns:
            pretty['win_rate'] = pd.to_numeric(pretty['win_rate'], errors='coerce').apply(fmt_pct)
        if 'sharpe' in pretty.columns:
            pretty['sharpe'] = pd.to_numeric(pretty['sharpe'], errors='coerce').apply(
                lambda x: fmt_float(x, 4))
        pretty = pretty.rename(columns={
            'gen': 'Gen', 'timestamp': 'Time', 'sharpe': 'Sharpe', 'net_pnl': 'Net P&L',
            'trades': 'Trades', 'win_rate': 'Win Rate', 'param_changed': 'Param Changed',
            'old_value': 'Old Value', 'new_value': 'New Value', 'kept': 'Kept',
        })
        st.dataframe(pretty, use_container_width=True, hide_index=True, height=df_height(10))

        # Store export data in session state so download button can live outside the fragment
        try:
            cur_params = strategy.default_params
        except Exception:
            cur_params = {}
        params_lines = "Current default_params (after autoresearch)\n"
        params_lines += "\n".join(f"{k}\t{v}" for k, v in cur_params.items())
        st.session_state["ar_export_text"] = (
            params_lines + "\n\n" + _df_to_tsv(df[show_cols], title="Generation Log")
        )

    _ar_results_panel()

    if st.session_state.get("ar_export_text"):
        st.download_button(
            label="⬇ Export autoresearch log (txt)",
            data=st.session_state["ar_export_text"],
            file_name="autoresearch_results.txt",
            mime="text/plain",
        )
