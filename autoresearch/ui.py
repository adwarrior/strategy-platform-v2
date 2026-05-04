"""
Autoresearch UI — configure and launch the autoresearch loop from a browser.

Run:
    python3 autoresearch/ui.py
"""
import sys, os
if "streamlit" not in sys.modules:
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "streamlit", "run", __file__] + sys.argv[1:]))

import ast
import subprocess
import threading
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import strategy_platform  # noqa: F401
from strategy_platform.registry import StrategyRegistry
from autoresearch_loop import _read_default_params, _write_default_params

RESULTS_TSV = Path(__file__).parent / "results.tsv"

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Autoresearch", layout="wide", initial_sidebar_state="expanded")
st.title("Autoresearch Loop")
st.caption("AI-driven parameter optimiser — uses Claude Haiku to propose one change per generation.")

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    strategies = StrategyRegistry.list_strategies()
    strategy   = st.selectbox("Strategy", strategies)

    cls    = StrategyRegistry.get(strategy)
    symbol = getattr(cls, "symbol", "")
    st.caption(f"Symbol: **{symbol}**")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("IS Start", value=pd.Timestamp("2024-03-20").date())
    with col2:
        end_date   = st.date_input("IS End",   value=pd.Timestamp("2026-03-20").date())

    st.markdown("---")
    max_gens   = st.number_input("Max generations (0 = unlimited)", min_value=0, value=100, step=10)
    min_trades = st.number_input("Min trades (IS slice)", min_value=1, value=10, step=1,
                                 help="Reject runs with fewer trades than this on the IS slice.")
    model    = st.selectbox("Model", ["haiku", "sonnet"], index=0,
                            help="Haiku: cheap (~$0.0005/gen). Sonnet: smarter but 10× costlier.")

    cost_est = max_gens * 0.0005 if model == "haiku" else max_gens * 0.005
    if max_gens > 0:
        st.caption(f"Estimated cost: ~${cost_est:.2f}")

    # ── Starting parameters ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Starting Parameters")
    st.caption("These become the baseline the autoresearch optimises from.")

    current_params = _read_default_params(strategy)
    edited_params  = {}
    for key, default in current_params.items():
        if isinstance(default, bool):
            edited_params[key] = st.checkbox(key, value=default)
        elif isinstance(default, int):
            edited_params[key] = st.number_input(key, value=default, step=1)
        elif isinstance(default, float):
            edited_params[key] = st.number_input(key, value=default, step=0.05, format="%.2f")
        else:
            edited_params[key] = st.text_input(key, value=str(default))

    save_btn = st.button("💾  Save params to strategy", use_container_width=True)
    if save_btn:
        _write_default_params(strategy, edited_params)
        st.success("Saved — these will be used as the baseline.")

    st.markdown("---")
    run_btn  = st.button("▶  Start Autoresearch", type="primary", use_container_width=True)
    stop_btn = st.button("⏹  Stop", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "proc"    not in st.session_state: st.session_state.proc    = None
if "lines"   not in st.session_state: st.session_state.lines   = []
if "running" not in st.session_state: st.session_state.running = False

# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------
if stop_btn and st.session_state.proc:
    st.session_state.proc.terminate()
    st.session_state.running = False
    st.session_state.proc    = None
    st.warning("Stopped.")

if run_btn:
    if st.session_state.running:
        st.warning("Already running — click Stop first.")
    else:
        st.session_state.lines   = []
        st.session_state.running = True

        cmd = [
            sys.executable, "-u",
            str(Path(__file__).parent / "autoresearch_loop.py"),
            "--strategy", strategy,
            "--symbol",   symbol,
            "--start",    str(start_date),
            "--end",      str(end_date),
            "--model",    model,
        ]
        if max_gens > 0:
            cmd += ["--max-gens", str(max_gens)]
        cmd += ["--min-trades", str(int(min_trades))]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Load API key from .env if not already in environment
        env_file = ROOT / ".env"
        if env_file.exists() and "ANTHROPIC_API_KEY" not in env:
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    env["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()

        st.session_state.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(ROOT),
        )

        def _reader(proc, lines):
            for line in iter(proc.stdout.readline, ""):
                lines.append(line.rstrip())
            proc.stdout.close()
            st.session_state.running = False

        t = threading.Thread(target=_reader, args=(st.session_state.proc, st.session_state.lines), daemon=True)
        t.start()

# ---------------------------------------------------------------------------
# Main area — two columns
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])

# ── Live output ──────────────────────────────────────────────────────────────
with left:
    st.subheader("Live Output")
    status = "🟢 Running" if st.session_state.running else ("⚪ Idle" if not st.session_state.lines else "🔴 Stopped")
    st.caption(status)

    output_box = st.empty()
    output_box.code("\n".join(st.session_state.lines[-200:]) or "Click ▶ Start to begin.", language="")

# ── Live chart from results.tsv ──────────────────────────────────────────────
with right:
    st.subheader("Progress")
    chart_box = st.empty()
    stats_box = st.empty()

    def _render_chart():
        if not RESULTS_TSV.exists():
            chart_box.info("No results yet.")
            return
        try:
            df = pd.read_csv(RESULTS_TSV, sep="\t")
        except Exception:
            return
        if df.empty or "sharpe" not in df.columns:
            return

        df = df[pd.to_numeric(df["sharpe"], errors="coerce").notna()].copy()
        df["sharpe"] = df["sharpe"].astype(float)
        df["kept"]   = df["kept"].astype(str)

        fig = px.scatter(
            df, x="gen", y="sharpe",
            color="kept",
            color_discrete_map={"yes": "green", "no": "lightgrey"},
            title="Sharpe per generation",
            labels={"gen": "Generation", "sharpe": "Sharpe"},
            hover_data=["param_changed", "old_value", "new_value", "net_pnl", "trades"],
        )
        # Add running best line
        kept_df = df[df["kept"] == "yes"].copy()
        if not kept_df.empty:
            kept_df = kept_df.sort_values("gen")
            kept_df["best"] = kept_df["sharpe"].cummax()
            fig.add_scatter(x=kept_df["gen"], y=kept_df["best"],
                            mode="lines", name="Best so far",
                            line=dict(color="green", width=2, dash="dot"))

        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        chart_box.plotly_chart(fig, use_container_width=True)

        # Summary stats
        kept = df[df["kept"] == "yes"]
        total = len(df) - 1  # exclude baseline row
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Generations run | {total} |
| Improvements kept | {len(kept)} |
| Best Sharpe | {df['sharpe'].max():.4f} |
| Current Sharpe | {df['sharpe'].iloc[-1]:.4f} |
""")

    _render_chart()

# ---------------------------------------------------------------------------
# Auto-refresh while running
# ---------------------------------------------------------------------------
if st.session_state.running:
    time.sleep(2)
    st.rerun()
