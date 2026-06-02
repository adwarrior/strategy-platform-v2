"""
Results Browser — read/curate the shared `strategy_results` MySQL store.

Reuses strategy_platform.results_store for the engine, loaders, and
label/delete mutators so it can never drift from the real schema.

Run:  streamlit run results_browser/app.py   (from the repo root)
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import text

# --- make the platform package importable when run from anywhere ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from strategy_platform.results_store import (  # noqa: E402
    _results_engine,
    _settings,
    delete_backtest,
    delete_optimizer_run,
    load_backtest,
    load_optimizer_stage,
    set_backtest_label,
    set_run_label,
)

st.set_page_config(page_title="Results Browser", page_icon="📊", layout="wide")

STAGE_ORDER = ["IS", "MC", "OOS"]


# --------------------------------------------------------------------------- #
# Data access (read-only catalog queries; cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def catalog_runs() -> pd.DataFrame:
    sql = text("""
        SELECT r.strategy_name, r.symbol, r.sym_safe, r.run_ts, r.label,
               r.created_at,
               GROUP_CONCAT(s.stage ORDER BY s.stage SEPARATOR ',') AS stages,
               MAX(CASE WHEN s.stage='IS'  THEN s.row_count END) AS is_rows,
               MAX(CASE WHEN s.stage='MC'  THEN s.row_count END) AS mc_rows,
               MAX(CASE WHEN s.stage='OOS' THEN s.row_count END) AS oos_rows
        FROM sp_optimizer_runs r
        LEFT JOIN sp_optimizer_run_stages s ON s.run_id = r.id
        GROUP BY r.id
        ORDER BY r.run_ts DESC
    """)
    with _results_engine().connect() as c:
        return pd.read_sql(sql, c)


@st.cache_data(ttl=30, show_spinner=False)
def catalog_backtests() -> pd.DataFrame:
    # Parse metrics in Python: payloads may contain NaN/Infinity, which MySQL's
    # JSON_EXTRACT rejects but json.loads tolerates.
    sql = text("""
        SELECT strategy_name, symbol, sym_safe, bt_ts, label, created_at, payload_json
        FROM sp_backtests
        ORDER BY bt_ts DESC
    """)
    with _results_engine().connect() as c:
        rows = c.execute(sql).mappings().all()
    wanted = ("net_pnl", "profit_factor", "sharpe", "sortino",
              "max_drawdown", "total_trades", "win_rate")
    out = []
    for r in rows:
        try:
            m = json.loads(r["payload_json"]).get("metrics", {})
        except Exception:  # noqa: BLE001
            m = {}
        rec = {k: r[k] for k in ("strategy_name", "symbol", "sym_safe", "bt_ts",
                                 "label", "created_at")}
        for w in wanted:
            rec[w] = finite(m.get(w))
        out.append(rec)
    return pd.DataFrame(out)


@st.cache_data(ttl=30, show_spinner=False)
def stage_df(strategy: str, sym_safe: str, run_ts: str, stage: str) -> pd.DataFrame | None:
    return load_optimizer_stage(strategy, sym_safe, run_ts, stage)


@st.cache_data(ttl=30, show_spinner=False)
def backtest_payload(strategy: str, sym_safe: str, bt_ts: str) -> dict | None:
    return load_backtest(strategy, sym_safe, bt_ts)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def finite(v):
    """Return a real number or None (NaN/Inf/non-numeric -> None)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fmt_ts(ts: str) -> str:
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(str(ts), fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return str(ts)


def apply_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty:
        return df
    c1, c2 = st.columns(2)
    strats = sorted(df["strategy_name"].dropna().unique())
    syms = sorted(df["symbol"].dropna().unique())
    sel_s = c1.multiselect("Strategy", strats, key=f"{key}_strat")
    sel_y = c2.multiselect("Symbol", syms, key=f"{key}_sym")
    out = df
    if sel_s:
        out = out[out["strategy_name"].isin(sel_s)]
    if sel_y:
        out = out[out["symbol"].isin(sel_y)]
    return out


def clear_caches():
    catalog_runs.clear()
    catalog_backtests.clear()
    stage_df.clear()
    backtest_payload.clear()


# --------------------------------------------------------------------------- #
# Sidebar — connection status
# --------------------------------------------------------------------------- #
cfg = _settings(use_database=True)
with st.sidebar:
    st.subheader("Connection")
    try:
        with _results_engine().connect() as c:
            c.execute(text("SELECT 1"))
        st.success("Connected", icon="✅")
    except Exception as e:  # noqa: BLE001
        st.error(f"DB unreachable\n\n{e}", icon="🚫")
        st.stop()
    st.caption(f"**Host:** {cfg['host']}:{cfg['port']}")
    st.caption(f"**DB:** {cfg['database']}")
    st.caption(f"**User:** {cfg['user']}")
    if st.button("🔄 Refresh data", use_container_width=True):
        clear_caches()
        st.rerun()

st.title("📊 Strategy Results Browser")

tab_runs, tab_bt = st.tabs(["⚙️ Optimizer Runs", "💾 Saved Backtests"])


# --------------------------------------------------------------------------- #
# TAB 1 — Optimizer runs
# --------------------------------------------------------------------------- #
with tab_runs:
    runs = catalog_runs()
    st.caption(f"{len(runs)} runs in store")
    runs = apply_filters(runs, "runs")

    if runs.empty:
        st.info("No runs match the current filter.")
    else:
        disp = runs.copy()
        disp["when"] = disp["run_ts"].map(fmt_ts)
        st.dataframe(
            disp[["strategy_name", "symbol", "when", "label", "stages",
                  "is_rows", "mc_rows", "oos_rows", "run_ts"]],
            use_container_width=True, hide_index=True,
            column_config={
                "strategy_name": "Strategy", "symbol": "Symbol", "when": "When",
                "label": "Label", "stages": "Stages",
                "is_rows": st.column_config.NumberColumn("IS rows"),
                "mc_rows": st.column_config.NumberColumn("MC rows"),
                "oos_rows": st.column_config.NumberColumn("OOS rows"),
                "run_ts": None,
            },
        )

        # --- select a run ---
        labels = [
            f"{r.strategy_name} · {r.symbol} · {fmt_ts(r.run_ts)}"
            + (f" · {r.label}" if r.label else "")
            for r in runs.itertuples()
        ]
        idx = st.selectbox("Inspect run", range(len(labels)),
                           format_func=lambda i: labels[i], key="run_pick")
        row = runs.iloc[idx]

        st.divider()
        # --- label edit + delete ---
        a, b = st.columns([3, 1])
        with a:
            new_label = st.text_input("Label", value=row.label or "", key="run_label")
            if st.button("Save label", key="run_label_save"):
                set_run_label(row.strategy_name, row.sym_safe, row.run_ts, new_label)
                clear_caches()
                st.rerun()
        with b:
            st.write("")
            confirm = st.checkbox("Confirm delete", key="run_del_confirm")
            if st.button("🗑 Delete run", disabled=not confirm, key="run_del"):
                delete_optimizer_run(row.strategy_name, row.sym_safe, row.run_ts)
                clear_caches()
                st.rerun()

        # --- stages ---
        avail = [s for s in STAGE_ORDER if row.stages and s in row.stages.split(",")]
        if not avail:
            st.warning("No stage data stored for this run.")
        else:
            for stage, sub in zip(avail, st.tabs(avail)):
                with sub:
                    df = stage_df(row.strategy_name, row.sym_safe, row.run_ts, stage)
                    if df is None or df.empty:
                        st.info("Empty stage.")
                        continue
                    # quick stats over the param grid
                    metric_cols = [c for c in ("net_pnl", "profit_factor",
                                               "max_drawdown", "win_rate")
                                   if c in df.columns]
                    if metric_cols:
                        cards = st.columns(len(metric_cols) + 1)
                        cards[0].metric("Combos", f"{len(df):,}")
                        for col, mc in zip(cards[1:], metric_cols):
                            best = df[mc].max() if mc != "max_drawdown" else df[mc].min()
                            col.metric(f"best {mc}", f"{best:,.2f}")
                    sort_col = st.selectbox(
                        "Sort by", df.columns.tolist(),
                        index=(df.columns.tolist().index("net_pnl")
                               if "net_pnl" in df.columns else 0),
                        key=f"sort_{stage}",
                    )
                    df = df.sort_values(sort_col, ascending=False)
                    st.dataframe(df, use_container_width=True, hide_index=True, height=360)
                    st.download_button(
                        f"⬇ Download {stage} CSV", df.to_csv(index=False),
                        file_name=f"{stage}_{row.strategy_name}_{row.sym_safe}_{row.run_ts}.csv",
                        mime="text/csv", key=f"dl_{stage}",
                    )


# --------------------------------------------------------------------------- #
# TAB 2 — Saved backtests
# --------------------------------------------------------------------------- #
with tab_bt:
    bts = catalog_backtests()
    st.caption(f"{len(bts)} saved backtests in store")
    bts = apply_filters(bts, "bt")

    if bts.empty:
        st.info("No backtests match the current filter.")
    else:
        disp = bts.copy()
        disp["when"] = disp["bt_ts"].map(fmt_ts)
        st.dataframe(
            disp[["strategy_name", "symbol", "when", "label", "net_pnl",
                  "profit_factor", "sharpe", "sortino", "max_drawdown",
                  "win_rate", "total_trades", "bt_ts"]],
            use_container_width=True, hide_index=True,
            column_config={
                "strategy_name": "Strategy", "symbol": "Symbol", "when": "When",
                "label": "Label",
                "net_pnl": st.column_config.NumberColumn("Net P&L", format="$%.2f"),
                "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn("Max DD", format="$%.2f"),
                "win_rate": st.column_config.NumberColumn("Win %", format="%.1f"),
                "total_trades": st.column_config.NumberColumn("Trades"),
                "bt_ts": None,
            },
        )

        labels = [
            f"{r.strategy_name} · {r.symbol} · {fmt_ts(r.bt_ts)}"
            + (f" · {r.label}" if r.label else "")
            for r in bts.itertuples()
        ]
        idx = st.selectbox("Inspect backtest", range(len(labels)),
                           format_func=lambda i: labels[i], key="bt_pick")
        row = bts.iloc[idx]
        payload = backtest_payload(row.strategy_name, row.sym_safe, row.bt_ts)

        st.divider()
        a, b = st.columns([3, 1])
        with a:
            new_label = st.text_input("Label", value=row.label or "", key="bt_label")
            if st.button("Save label", key="bt_label_save"):
                set_backtest_label(row.strategy_name, row.sym_safe, row.bt_ts, new_label)
                clear_caches()
                st.rerun()
        with b:
            st.write("")
            confirm = st.checkbox("Confirm delete", key="bt_del_confirm")
            if st.button("🗑 Delete backtest", disabled=not confirm, key="bt_del"):
                delete_backtest(row.strategy_name, row.sym_safe, row.bt_ts)
                clear_caches()
                st.rerun()

        if not payload:
            st.warning("Could not load payload.")
            st.stop()

        metrics = payload.get("metrics", {})
        meta = payload.get("meta", {})
        params = payload.get("params", {})
        trades = payload.get("trades", [])

        # --- headline stats ---
        st.subheader("Performance")
        keys = [
            ("net_pnl", "Net P&L", "$%.0f"), ("profit_factor", "Profit Factor", "%.2f"),
            ("sharpe", "Sharpe", "%.2f"), ("sortino", "Sortino", "%.2f"),
            ("max_drawdown", "Max Drawdown", "$%.0f"), ("win_rate", "Win Rate", "%.1f%%"),
            ("total_trades", "Trades", "%.0f"), ("avg_trade", "Avg Trade", "$%.2f"),
            ("ratio_win_loss", "Win/Loss", "%.2f"), ("r_squared", "R²", "%.3f"),
            ("profit_per_month", "P&L / Month", "$%.0f"), ("max_consec_losers", "Max Consec L", "%.0f"),
        ]
        cols = st.columns(6)
        for i, (k, lbl, fmt) in enumerate(keys):
            v = finite(metrics.get(k))
            cols[i % 6].metric(lbl, fmt % v if v is not None else "—")

        # --- equity curve ---
        if trades:
            tdf = pd.DataFrame(trades)
            if "entry_time" in tdf.columns:
                tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], errors="coerce")
                tdf = tdf.sort_values("entry_time")
            if "pnl" in tdf.columns:
                tdf["equity"] = tdf["pnl"].cumsum()
                st.subheader("Equity curve")
                eq = tdf.set_index("entry_time")["equity"] if "entry_time" in tdf.columns \
                    else tdf["equity"]
                st.area_chart(eq, height=260)

        # --- breakdowns ---
        left, right = st.columns(2)
        with left:
            st.subheader("Parameters")
            st.dataframe(
                pd.DataFrame([(k, str(v)) for k, v in sorted(params.items())],
                             columns=["param", "value"]),
                width="stretch", hide_index=True, height=320,
            )
        with right:
            st.subheader("Day of week")
            dow = [(d, metrics.get(f"{d}_pnl"), metrics.get(f"{d}_trades"))
                   for d in ("mon", "tue", "wed", "thu", "fri")]
            dow_df = pd.DataFrame(dow, columns=["day", "pnl", "trades"]).dropna(how="all", subset=["pnl"])
            if not dow_df.empty:
                st.bar_chart(dow_df.set_index("day")["pnl"], height=260)

        st.caption(
            f"Data: {meta.get('data_source','?')} · {meta.get('start','?')} → "
            f"{meta.get('end','?')} · {meta.get('bars','?')} bars · "
            f"{meta.get('sessions','?')} sessions"
        )

        with st.expander(f"Trades ({len(trades)})"):
            if trades:
                st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Download trades CSV", pd.DataFrame(trades).to_csv(index=False),
                    file_name=f"trades_{row.strategy_name}_{row.sym_safe}_{row.bt_ts}.csv",
                    mime="text/csv",
                )

        with st.expander("Raw metrics (all 44)"):
            st.dataframe(
                pd.DataFrame([(k, str(v)) for k, v in sorted(metrics.items())],
                             columns=["metric", "value"]),
                width="stretch", hide_index=True,
            )
