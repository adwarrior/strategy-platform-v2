"""
Strategy Lab — interactive parameter tweaker.

Adjust parameters with sliders, pick a date range, hit Run and see
the equity curve and metrics instantly.

Run:
    python3 lab.py
    python3 -m streamlit run lab.py
"""
import sys
if "streamlit" not in sys.modules:
    import subprocess
    sys.exit(subprocess.call(["python3", "-m", "streamlit", "run", __file__] + sys.argv[1:]))

import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy_platform  # noqa: F401 — registers all strategies
from strategy_platform.registry import StrategyRegistry
from strategy_platform.data.loader import load_5m

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Strategy Lab", layout="wide", initial_sidebar_state="expanded")
st.title("Strategy Lab")
st.caption("Tweak parameters and run a quick backtest instantly.")

# ---------------------------------------------------------------------------
# Sidebar — strategy + date range
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    strategies = StrategyRegistry.list_strategies()
    selected_name = st.selectbox("Strategy", strategies)
    cls = StrategyRegistry.get(selected_name)
    strategy = cls()

    st.markdown("---")
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.Timestamp("2024-01-01").date())
    with col2:
        end_date   = st.date_input("To",   value=pd.Timestamp("2026-03-20").date())

    st.markdown("---")
    st.subheader("Parameters")

    params = {}
    for key, default in strategy.default_params.items():
        if isinstance(default, bool):
            params[key] = st.checkbox(key.capitalize(), value=default)
        elif isinstance(default, int):
            params[key] = st.slider(key, min_value=0, max_value=max(default * 4, 20), value=default, step=1)
        elif isinstance(default, float):
            params[key] = st.slider(key, min_value=0.0, max_value=max(default * 4, 4.0), value=default, step=0.05)
        elif isinstance(default, str) and "-" in default and ":" in default:
            # trade_window — two time pickers
            parts = default.split("-")
            start_t = st.text_input(f"{key} start", value=parts[0])
            end_t   = st.text_input(f"{key} end",   value=parts[1])
            params[key] = f"{start_t}-{end_t}"
        else:
            params[key] = st.text_input(key, value=str(default))

    st.markdown("---")
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main — results
# ---------------------------------------------------------------------------

if not run_btn:
    st.info("Set your parameters in the sidebar and click **▶ Run Backtest**.")
    st.stop()

with st.spinner("Loading data and running backtest..."):
    try:
        df = load_5m(
            strategy.symbol,
            start=str(start_date),
            end=str(end_date),
            host=getattr(strategy, "db_host", None),
        )
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

    try:
        result = strategy.run_backtest(df, params)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

if "error" in result:
    st.warning(f"Backtest returned: {result['error']}")
    st.stop()

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

net_pnl      = result.get("net_pnl", 0)
total_trades = result.get("total_trades", result.get("trades", 0))
win_rate     = result.get("win_rate", 0)
sharpe       = result.get("sharpe", 0)
max_dd       = result.get("max_drawdown", 0)
profit_f     = result.get("profit_factor", 0)

pnl_color = "normal" if net_pnl >= 0 else "inverse"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Net P&L",       f"${net_pnl:,.0f}")
c2.metric("Trades",        f"{total_trades}")
c3.metric("Win Rate",      f"{win_rate*100:.1f}%")
c4.metric("Sharpe",        f"{sharpe:.2f}")
c5.metric("Max Drawdown",  f"${abs(max_dd):,.0f}")
c6.metric("Profit Factor", f"{profit_f:.2f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

trades_df = result.get("trades")
equity    = result.get("equity_curve")

if equity is not None and len(equity) > 0:
    if isinstance(equity, pd.Series):
        eq_df = equity.reset_index()
        eq_df.columns = ["date", "equity"]
    else:
        eq_df = pd.DataFrame({"equity": equity}).reset_index(names="date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["date"], y=eq_df["equity"],
        mode="lines", name="Equity",
        line=dict(color="green" if net_pnl >= 0 else "red", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,200,0,0.1)" if net_pnl >= 0 else "rgba(200,0,0,0.1)",
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
        height=400, margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

elif trades_df is not None and len(trades_df) > 0:
    # Build equity curve from trades if not provided
    pnl_col = next((c for c in ["pnl_dollars", "pnl", "net_pnl"] if c in trades_df.columns), None)
    time_col = next((c for c in ["entry_time", "date", "datetime"] if c in trades_df.columns), None)
    if pnl_col and time_col:
        eq = trades_df.set_index(time_col)[pnl_col].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            mode="lines", name="Equity",
            line=dict(color="green" if net_pnl >= 0 else "red", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,200,0,0.1)" if net_pnl >= 0 else "rgba(200,0,0,0.1)",
        ))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
            height=400, margin=dict(l=0, r=0, t=40, b=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No equity curve data returned by this strategy.")

# ---------------------------------------------------------------------------
# Day-of-week breakdown
# ---------------------------------------------------------------------------

if trades_df is not None and len(trades_df) > 0:
    pnl_col  = next((c for c in ["pnl_dollars", "pnl", "net_pnl"] if c in trades_df.columns), None)
    time_col = next((c for c in ["entry_time", "date", "datetime"] if c in trades_df.columns), None)

    if pnl_col and time_col:
        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            tdf = trades_df.copy()
            tdf["day"] = pd.to_datetime(tdf[time_col]).dt.day_name()
            dow = tdf.groupby("day")[pnl_col].agg(["sum", "count"]).reindex(
                ["Monday","Tuesday","Wednesday","Thursday","Friday"]
            ).fillna(0)
            dow.columns = ["P&L ($)", "Trades"]
            fig2 = px.bar(
                dow.reset_index(), x="day", y="P&L ($)",
                color="P&L ($)", color_continuous_scale=["red","white","green"],
                color_continuous_midpoint=0, title="P&L by Day of Week",
            )
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown("#### Trade List")
            show_cols = [c for c in [time_col, pnl_col, "direction", "win"] if c in trades_df.columns]
            st.dataframe(
                trades_df[show_cols].tail(50),
                use_container_width=True,
                hide_index=True,
            )
