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
import re
import sys
from datetime import datetime

import altair as alt
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

# Walk-forward runs are JSON files on disk (reports/<strategy>/WF_*.json), not in
# the results DB. The browser reads them directly — see catalog_wf_runs().
_WF_REPORTS_GLOB = os.path.join(_REPO_ROOT, "reports", "*", "WF_*.json")
_WF_FNAME_RE = re.compile(r"^WF_(?P<strategy>.+)_(?P<sym>[^_]+)_(?P<ts>\d{8}_\d{4,6})\.json$")


# --------------------------------------------------------------------------- #
# Data access (read-only catalog queries; cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30, show_spinner=False)
def catalog_runs() -> pd.DataFrame:
    sql = text("""
        SELECT r.strategy_name, r.symbol, r.sym_safe, r.run_ts, r.label,
               r.created_at, r.run_meta_json, r.settings_json,
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
        df = pd.read_sql(sql, c)

    def _tf(rec):
        try:
            s = json.loads(rec["settings_json"]) if rec["settings_json"] else {}
        except Exception:  # noqa: BLE001
            s = {}
        return infer_run_timeframe(rec["strategy_name"], s)

    def _window(rec):
        try:
            m = json.loads(rec["run_meta_json"]) if rec["run_meta_json"] else {}
        except Exception:  # noqa: BLE001
            m = {}
        a, b = m.get("_data_start"), m.get("_data_end")
        return f"{a} → {b}" if a and b else ""

    if not df.empty:
        df["timeframe"] = df.apply(_tf, axis=1)
        df["data_window"] = df.apply(_window, axis=1)
    return df


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
            payload = json.loads(r["payload_json"])
        except Exception:  # noqa: BLE001
            payload = {}
        m = payload.get("metrics", {})
        meta = payload.get("meta", {})
        rec = {k: r[k] for k in ("strategy_name", "symbol", "sym_safe", "bt_ts",
                                 "label", "created_at")}
        rec["bar_type"] = meta.get("data_source") or "—"
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


@st.cache_data(ttl=30, show_spinner=False)
def catalog_wf_runs() -> pd.DataFrame:
    """Catalog of walk-forward runs read from reports/<strategy>/WF_*.json.

    WF runs are not persisted to the results DB — they live as JSON on disk.
    We read each file's `meta` block (cheap) to build the listing; the heavy
    `slices` array is only loaded when a run is inspected (wf_payload()).
    """
    rows = []
    for path in glob.glob(_WF_REPORTS_GLOB):
        base = os.path.basename(path)
        m = _WF_FNAME_RE.match(base)
        if not m:
            continue
        try:
            with open(path) as fh:
                meta = json.load(fh).get("meta", {})
        except Exception:  # noqa: BLE001
            meta = {}
        rows.append({
            "strategy_name": meta.get("strategy", m.group("strategy")),
            "symbol":        meta.get("symbol", m.group("sym")),
            "sym_safe":      m.group("sym"),
            "run_ts":        meta.get("ts", m.group("ts")),
            "bar_type":      meta.get("bar_type", "—"),
            "data_window":   (f"{meta.get('data_start')} → {meta.get('data_end')}"
                              if meta.get("data_start") else ""),
            "is_window_days":  meta.get("is_window_days"),
            "oos_window_days": meta.get("oos_window_days"),
            "step_days":       meta.get("step_days"),
            "n_slices":        meta.get("n_slices"),
            "wfe":             finite(meta.get("wfe")),
            "rank_by":         meta.get("rank_by", ""),
            "path":            path,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("run_ts", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=30, show_spinner=False)
def wf_payload(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


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


# --- human-friendly labels -------------------------------------------------- #
# Explicit overrides; anything not listed falls back to a generic prettifier.
PRETTY = {
    # performance metrics
    "net_pnl": "Net P&L", "gross_profit": "Gross Profit", "gross_loss": "Gross Loss",
    "profit_factor": "Profit Factor", "max_drawdown": "Max Drawdown",
    "win_rate": "Win Rate", "avg_trade": "Avg Trade", "avg_win": "Avg Win",
    "avg_loss": "Avg Loss", "ratio_win_loss": "Win / Loss Ratio",
    "largest_win": "Largest Win", "largest_loss": "Largest Loss",
    "num_wins": "Wins", "num_losses": "Losses", "num_even": "Scratch",
    "total_trades": "Total Trades", "total_commission": "Total Commission",
    "max_consec_winners": "Max Consecutive Winners",
    "max_consec_losers": "Max Consecutive Losers", "max_consec": "Max Consecutive",
    "sharpe": "Sharpe Ratio", "sortino": "Sortino Ratio", "r_squared": "R²",
    "profit_per_month": "P&L per Month", "pct_months_profit": "% Months Profitable",
    "avg_trades_per_day": "Avg Trades / Day", "longest_flat_days": "Longest Flat (days)",
    "max_time_to_recover": "Max Recovery Time", "start_date": "Start Date",
    "end_date": "End Date",
    # params
    "stop_loss_ticks": "Stop Loss (ticks)", "profit_target_ticks": "Profit Target (ticks)",
    "trailing_trigger_ticks": "Trailing Trigger (ticks)",
    "trailing_stop_ticks": "Trailing Stop (ticks)",
    "breakout_offset_ticks": "Breakout Offset (ticks)", "sl_ticks": "Stop Loss (ticks)",
    "tp_ticks": "Profit Target (ticks)", "qty": "Quantity", "direction": "Direction",
    "minute_bar_period": "Bar Period (min)", "htf_minutes": "Higher Timeframe (min)",
    "range_bars_minutes": "Range Bars (min)", "range_duration_minutes": "Range Duration (min)",
    "use_pd_levels": "Use Prior-Day Levels", "use_prev_session": "Use Previous Session",
    "start_trading": "Start Trading", "stop_trading": "Stop Trading",
    "ps_window_start": "Prior-Session Window Start", "ps_window_end": "Prior-Session Window End",
    "block_ps_while_forming": "Block While Forming",
    "atr_period": "ATR Period", "atr_multiplier": "ATR Multiplier",
    "fractal_length": "Fractal Length", "eod_exit_time": "End-of-Day Exit Time",
    "bars_between_trades": "Bars Between Trades", "rr_ratio": "Risk:Reward Ratio",
}
_ABBR = {"pnl": "P&L", "atr": "ATR", "tp": "TP", "sl": "SL", "tf": "TF", "pct": "%",
         "ps": "PS", "pd": "PD", "fvg": "FVG", "poi": "POI", "htf": "HTF",
         "rr": "R:R", "eod": "EOD", "ny": "NY", "mc": "MC", "oos": "OOS",
         "is": "IS", "wae": "WAE", "id": "ID", "ema": "EMA", "rsi": "RSI"}
_DAY = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
        "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}


def pretty(key) -> str:
    """snake_case / code key -> human-friendly Title Case label."""
    k = str(key)
    if k in PRETTY:
        return PRETTY[k]
    # day-of-week metrics: mon_pnl, tue_trades, ...
    parts = k.split("_")
    if parts and parts[0] in _DAY:
        tail = " ".join(_ABBR.get(p.lower(), p.capitalize()) for p in parts[1:])
        return f"{_DAY[parts[0]]} {tail}".strip()
    if k.startswith("bs_"):  # bootstrap percentiles
        return "Bootstrap " + " ".join(
            _ABBR.get(p.lower(), p.upper() if p.startswith("p") and p[1:].isdigit() else p.capitalize())
            for p in parts[1:])
    return " ".join(_ABBR.get(p.lower(), p.capitalize()) for p in parts)


_TF_RE = re.compile(r"_(\d+)\s*(m|min|s|h)\b", re.I)


def infer_run_timeframe(strategy_name: str, settings: dict) -> str:
    """Timeframe for an optimizer run.

    Newer runs persist it explicitly (settings.data_source / bar_period); older
    runs fall back to best-effort inference from the strategy name or core.
    """
    ds = settings.get("data_source")
    if ds:
        return str(ds)
    bp = settings.get("bar_period")
    if bp:
        return str(bp)
    m = _TF_RE.search(strategy_name) or re.search(r"(\d+)(m|min)$", strategy_name, re.I)
    if m:
        return f"{m.group(1)}-min (from name)"
    core = settings.get("core")
    if isinstance(core, int):
        return f"{core}-min? (settings.core)"
    return "not recorded"


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


def ordered_bar(df: pd.DataFrame, x_col: str, order: list, height: int = 260):
    """P&L bar chart with an explicit category order and green/red by sign."""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", sort=order, title=x_col),
            y=alt.Y("P&L:Q", title="P&L ($)"),
            color=alt.condition(alt.datum["P&L"] >= 0,
                                alt.value("#2e7d32"), alt.value("#c62828")),
            tooltip=[x_col, alt.Tooltip("P&L:Q", format="$,.0f"), "Trades"],
        )
        .properties(width="container", height=height)
    )
    st.altair_chart(chart)


def clear_caches():
    catalog_runs.clear()
    catalog_backtests.clear()
    stage_df.clear()
    backtest_payload.clear()
    catalog_wf_runs.clear()
    wf_payload.clear()


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
    if st.button("🔄 Refresh data", width="stretch"):
        clear_caches()
        st.rerun()

st.title("📊 Strategy Results Browser")

tab_runs, tab_bt, tab_wf = st.tabs(
    ["⚙️ Optimizer Runs", "💾 Saved Backtests", "🔄 Walk-Forward"])


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
            disp[["strategy_name", "symbol", "timeframe", "data_window", "when",
                  "label", "stages", "is_rows", "mc_rows", "oos_rows", "run_ts"]],
            width="stretch", hide_index=True,
            column_config={
                "strategy_name": "Strategy", "symbol": "Symbol",
                "timeframe": "Timeframe", "data_window": "Data Window",
                "when": "Run At", "label": "Label", "stages": "Stages",
                "is_rows": st.column_config.NumberColumn("IS Combos"),
                "mc_rows": st.column_config.NumberColumn("MC Rows"),
                "oos_rows": st.column_config.NumberColumn("OOS Rows"),
                "run_ts": None,
            },
        )
        st.caption("ℹ️ New optimizer runs record their bar type exactly (e.g. "
                   "*MySQL 5M*). Older runs predating this show an inferred value "
                   "(from the strategy name / settings).")

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
                        cards[0].metric("Param Combos", f"{len(df):,}")
                        for col, mc in zip(cards[1:], metric_cols):
                            best = df[mc].max() if mc != "max_drawdown" else df[mc].min()
                            col.metric(f"Best {pretty(mc)}", f"{best:,.2f}")
                    sort_col = st.selectbox(
                        "Sort by", df.columns.tolist(),
                        index=(df.columns.tolist().index("net_pnl")
                               if "net_pnl" in df.columns else 0),
                        format_func=pretty, key=f"sort_{stage}",
                    )
                    df = df.sort_values(sort_col, ascending=False)
                    st.dataframe(
                        df, width="stretch", hide_index=True, height=360,
                        column_config={c: pretty(c) for c in df.columns},
                    )
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
            disp[["strategy_name", "symbol", "bar_type", "when", "label", "net_pnl",
                  "profit_factor", "sharpe", "sortino", "max_drawdown",
                  "win_rate", "total_trades", "bt_ts"]],
            width="stretch", hide_index=True,
            column_config={
                "strategy_name": "Strategy", "symbol": "Symbol",
                "bar_type": "Bar Type", "when": "Run At", "label": "Label",
                "net_pnl": st.column_config.NumberColumn("Net P&L", format="$%.2f"),
                "profit_factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn("Max Drawdown", format="$%.2f"),
                "win_rate": st.column_config.NumberColumn("Win Rate %", format="%.1f"),
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

        # --- setup / bar type ---
        st.info(
            f"**Bar type:** {meta.get('data_source','—')}  ·  "
            f"**Period:** {meta.get('start','?')} → {meta.get('end','?')}  ·  "
            f"**Bars:** {meta.get('bars','?'):,}" if isinstance(meta.get('bars'), int)
            else f"**Bar type:** {meta.get('data_source','—')}  ·  "
                 f"**Period:** {meta.get('start','?')} → {meta.get('end','?')}",
            icon="🧱",
        )

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

        # --- trades frame (shared by equity curve + time breakdowns) ---
        tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
        if not tdf.empty and "entry_time" in tdf.columns:
            tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], errors="coerce")
            tdf = tdf.sort_values("entry_time")

        # --- equity curve ---
        if not tdf.empty and "pnl" in tdf.columns:
            tdf["equity"] = tdf["pnl"].cumsum()
            st.subheader("Equity curve")
            eq = (tdf.set_index("entry_time")["equity"]
                  if "entry_time" in tdf.columns else tdf["equity"])
            st.area_chart(eq, height=260)

        # --- breakdowns: parameters + day of week ---
        left, right = st.columns(2)
        with left:
            st.subheader("Parameters")
            st.dataframe(
                pd.DataFrame([(pretty(k), str(v)) for k, v in sorted(params.items())],
                             columns=["Parameter", "Value"]),
                width="stretch", hide_index=True, height=320,
            )
        with right:
            st.subheader("Day of Week P&L")
            dow = [(_DAY[d], finite(metrics.get(f"{d}_pnl")), finite(metrics.get(f"{d}_trades")))
                   for d in ("mon", "tue", "wed", "thu", "fri")]
            dow_df = pd.DataFrame(dow, columns=["Day", "P&L", "Trades"]).dropna(
                how="all", subset=["P&L"])
            if not dow_df.empty:
                ordered_bar(dow_df, "Day",
                            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            else:
                st.caption("No day-of-week data stored.")

        # --- time of day ---
        st.subheader("Time of Day P&L")
        if (not tdf.empty and "entry_time" in tdf.columns and "pnl" in tdf.columns
                and tdf["entry_time"].notna().any()):
            gran = st.radio("Bucket size", ["Hour", "Half-hour"],
                            horizontal=True, key="tod_gran")
            et = tdf["entry_time"]
            if gran == "Hour":
                bucket = et.dt.strftime("%H:00")
            else:
                bucket = et.dt.strftime("%H:") + (et.dt.minute // 30 * 30).map(
                    lambda m: f"{int(m):02d}" if pd.notna(m) else "")
            tod = tdf.assign(_bucket=bucket).dropna(subset=["_bucket"])
            tod = tod[tod["_bucket"] != ""]
            g = (tod.groupby("_bucket")
                 .agg(**{"P&L": ("pnl", "sum"), "Trades": ("pnl", "size")})
                 .reset_index().rename(columns={"_bucket": "Time"}))
            ordered_bar(g, "Time", sorted(g["Time"]), height=280)
            st.caption("Bucketed by trade entry time (Eastern, matching the source "
                       "data). Green = net-profitable, red = net-losing. Hover for totals.")
        else:
            st.caption("No entry-time data on these trades for a time-of-day breakdown.")

        st.caption(
            f"Data: {meta.get('data_source','?')} · {meta.get('start','?')} → "
            f"{meta.get('end','?')} · {meta.get('bars','?')} bars · "
            f"{meta.get('sessions','?')} sessions"
        )

        with st.expander(f"Trades ({len(trades)})"):
            if trades:
                tr_df = pd.DataFrame(trades)
                st.dataframe(
                    tr_df, width="stretch", hide_index=True,
                    column_config={c: pretty(c) for c in tr_df.columns},
                )
                st.download_button(
                    "⬇ Download trades CSV", pd.DataFrame(trades).to_csv(index=False),
                    file_name=f"trades_{row.strategy_name}_{row.sym_safe}_{row.bt_ts}.csv",
                    mime="text/csv",
                )

        with st.expander(f"All metrics ({len(metrics)})"):
            st.dataframe(
                pd.DataFrame([(pretty(k), str(v)) for k, v in sorted(metrics.items())],
                             columns=["Metric", "Value"]),
                width="stretch", hide_index=True,
            )


# --------------------------------------------------------------------------- #
# TAB 3 — Walk-forward runs (read from reports/<strategy>/WF_*.json)
# --------------------------------------------------------------------------- #
with tab_wf:
    wf = catalog_wf_runs()
    st.caption(f"{len(wf)} walk-forward runs on disk")
    wf = apply_filters(wf, "wf")

    if wf.empty:
        st.info("No walk-forward runs found under reports/*/WF_*.json.")
    else:
        disp = wf.copy()
        disp["when"] = disp["run_ts"].map(fmt_ts)
        st.dataframe(
            disp[["strategy_name", "symbol", "bar_type", "data_window", "when",
                  "is_window_days", "oos_window_days", "step_days", "n_slices",
                  "wfe", "rank_by"]],
            width="stretch", hide_index=True,
            column_config={
                "strategy_name": "Strategy", "symbol": "Symbol",
                "bar_type": "Bar Type", "data_window": "Data Window", "when": "Run At",
                "is_window_days": st.column_config.NumberColumn("IS Days"),
                "oos_window_days": st.column_config.NumberColumn("OOS Days"),
                "step_days": st.column_config.NumberColumn("Step Days"),
                "n_slices": st.column_config.NumberColumn("Slices"),
                "wfe": st.column_config.NumberColumn("WFE", format="%.3f"),
                "rank_by": "Ranked By",
            },
        )

        labels = [
            f"{r.strategy_name} · {r.symbol} · {fmt_ts(r.run_ts)}"
            for r in wf.itertuples()
        ]
        idx = st.selectbox("Inspect walk-forward run", range(len(labels)),
                           format_func=lambda i: labels[i], key="wf_pick")
        row = wf.iloc[idx]
        data = wf_payload(row.path)

        st.divider()
        if not data:
            st.warning("Could not load this walk-forward file.")
            st.stop()

        meta = data.get("meta", {})
        slices = data.get("slices", [])

        # --- WFE banner: OOS efficiency vs IS (1.0 = OOS matches IS) ---
        wfe = finite(meta.get("wfe"))
        b1, b2 = st.columns([1, 3])
        with b1:
            st.metric("Walk-Forward Efficiency",
                      f"{wfe:.3f}" if wfe is not None else "—")
        with b2:
            st.info(
                f"**{meta.get('strategy','?')} · {meta.get('symbol','?')}**  ·  "
                f"{meta.get('n_slices','?')} slices  ·  "
                f"IS {meta.get('is_window_days','?')}d / OOS {meta.get('oos_window_days','?')}d "
                f"step {meta.get('step_days','?')}d  ·  ranked by {meta.get('rank_by','?')}  ·  "
                f"{meta.get('data_start','?')} → {meta.get('data_end','?')}",
                icon="🔄",
            )

        if not slices:
            st.warning("No slices recorded in this run.")
            st.stop()

        # --- flatten slices: one row per OOS fold ---
        srows = []
        for s in slices:
            om = s.get("oos_metrics", {}) or {}
            im = s.get("is_metrics", {}) or {}
            srows.append({
                "Slice": s.get("slice_idx"),
                "OOS Start": s.get("oos_start"), "OOS End": s.get("oos_end"),
                "OOS P&L": finite(om.get("net_pnl")),
                "OOS Sharpe": finite(om.get("sharpe")),
                "OOS PF": finite(om.get("profit_factor")),
                "OOS Win %": (finite(om.get("win_rate")) or 0) * 100,
                "OOS Trades": finite(om.get("total_trades")),
                "IS Sharpe": finite(im.get("sharpe")),
            })
        sdf = pd.DataFrame(srows)

        # --- aggregate OOS stats across slices ---
        oos_pnls = sdf["OOS P&L"].dropna()
        total_oos_pnl = float(oos_pnls.sum()) if not oos_pnls.empty else 0.0
        pos_slices = int((oos_pnls > 0).sum())
        mean_oos_sharpe = float(sdf["OOS Sharpe"].dropna().mean()) if sdf["OOS Sharpe"].notna().any() else None
        total_oos_trades = int(sdf["OOS Trades"].dropna().sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total OOS P&L", f"${total_oos_pnl:,.0f}")
        m2.metric("Profitable Slices", f"{pos_slices}/{len(sdf)}")
        m3.metric("Mean OOS Sharpe",
                  f"{mean_oos_sharpe:.2f}" if mean_oos_sharpe is not None else "—")
        m4.metric("Total OOS Trades", f"{total_oos_trades:,}")

        # --- stitched OOS equity curve (cumulative OOS P&L across slices) ---
        st.subheader("Stitched OOS equity (cumulative P&L across slices)")
        eq = sdf.dropna(subset=["OOS P&L"]).copy()
        if not eq.empty:
            eq["Cumulative P&L"] = eq["OOS P&L"].cumsum()
            eq_idx = eq.set_index("OOS End")["Cumulative P&L"]
            st.area_chart(eq_idx, height=260)

        # --- per-slice table ---
        st.subheader("Per-slice OOS results")
        st.dataframe(
            sdf, width="stretch", hide_index=True, height=400,
            column_config={
                "OOS P&L": st.column_config.NumberColumn("OOS P&L", format="$%.0f"),
                "OOS Sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.2f"),
                "OOS PF": st.column_config.NumberColumn("OOS PF", format="%.2f"),
                "OOS Win %": st.column_config.NumberColumn("OOS Win %", format="%.1f"),
                "IS Sharpe": st.column_config.NumberColumn("IS Sharpe", format="%.2f"),
            },
        )
        st.download_button(
            "⬇ Download slices CSV", sdf.to_csv(index=False),
            file_name=f"WF_{row.strategy_name}_{row.sym_safe}_{row.run_ts}_slices.csv",
            mime="text/csv", key="wf_dl",
        )

        # --- chosen params per slice (parameter stability across folds) ---
        with st.expander("Best params per slice (parameter stability)"):
            prows = []
            for s in slices:
                bp = s.get("best_params", {}) or {}
                prows.append({"Slice": s.get("slice_idx"),
                              "OOS Start": s.get("oos_start"),
                              **{pretty(k): v for k, v in bp.items()}})
            st.dataframe(pd.DataFrame(prows), width="stretch", hide_index=True)
