"""VWAPDrift prop-firm challenge simulation — the question the PF tables can't answer.

The strategy is break-even as an income strategy (PF 1.05). Its author never
claimed otherwise; he pitched it as a PROP-CHALLENGE vehicle: high win rate +
inverted RR passes evaluations fast. Passing is PATH-DEPENDENT — it depends on
whether the profit target is reached before the drawdown limit, in what order
trades land — so profit factor genuinely cannot answer it. This does.

His claims (transcript 31:48-33:11), on 1 NQ == 10 MNQ:
    single-challenge pass rate 49.8%, avg ~3.4 days to pass,
    74.8% within 2 attempts, 87.3% within 3, 93.6% within 4.

METHOD — two simulations, deliberately different:
  1. SEQUENTIAL (primary): walk the REAL trade sequence in date order, starting a
     challenge on every eligible start date. Preserves autocorrelation, volatility
     clustering, and the real order of wins/losses. This is what would have
     happened had you started on that day.
  2. BOOTSTRAP (secondary): resample trade DAYS with replacement, his "20,000
     simulations" method. Reported for comparability with his number, but it
     destroys sequencing and so flatters any strategy with streaky losses.

The gap between the two is itself the finding.

Rules default to a standard 50K evaluation (Topstep/Apex-like):
  profit target $3,000, trailing max drawdown $2,000 (intraday, from peak),
  daily loss limit $1,000, min 1 trading day, 20-day deadline.
Trailing drawdown trails the account PEAK and does NOT stop trailing at breakeven
(the stricter, more common variant). Peak is tracked on CLOSED-trade equity;
we do NOT model intra-trade excursion, so these results are, if anything,
OPTIMISTIC vs a real evaluation that trails on unrealised equity.

Usage: python scripts/prop_sim_vwapdrift.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.data.loader import load_1m                       # noqa: E402
from strategy_platform.strategies.vwapdrift.strategy import VwapDrift   # noqa: E402

SYMBOL = os.environ.get("SWEEP_SYMBOL", "MNQ")
START, END = "2020-01-01", "2026-08-07"

# 10 MNQ == 1 NQ, his stated sizing.
CONTRACTS = int(os.environ.get("PROP_CONTRACTS", "10"))

ACCOUNTS = {
    # label: (profit_target, trailing_dd, daily_loss_limit)
    "50K  (3k target / 2k trail)":  (3000.0, 2000.0, 1000.0),
    "100K (6k target / 3k trail)":  (6000.0, 3000.0, 2000.0),
}
MAX_DAYS = 20
MIN_DAYS = 1


def load_daily_pnl() -> pd.Series:
    df = load_1m(SYMBOL, start=START, end=END, to_et=True)
    res = VwapDrift().run_backtest(df, {})
    tr = res["trades"]
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    # Per-trade PnL is for 1 contract; scale to the challenge size.
    tr["pnl_scaled"] = tr["pnl"] * CONTRACTS
    return tr, tr.groupby("session_date")["pnl_scaled"].sum().sort_index()


def run_challenge(day_pnls, day_trades, target, trail_dd, daily_limit):
    """Walk days until pass/fail. Returns (passed, days_used, reason).

    Trailing drawdown trails the peak of CLOSED-trade equity. Daily loss limit is
    checked on the day's realised total. Both are evaluated per day, in order."""
    equity = 0.0
    peak = 0.0
    for i, (d, pnl) in enumerate(zip(day_pnls.index, day_pnls.values), start=1):
        # Daily loss limit — a breach ends the challenge that day.
        if pnl <= -daily_limit:
            return False, i, "daily_loss"
        equity += pnl
        peak = max(peak, equity)
        # Trailing max drawdown from peak.
        if equity <= peak - trail_dd:
            return False, i, "trailing_dd"
        if equity >= target and i >= MIN_DAYS:
            return True, i, "target"
        if i >= MAX_DAYS:
            return False, i, "timeout"
    return False, len(day_pnls), "ran_out_of_data"


def sequential_sim(daily, target, trail_dd, daily_limit):
    """Start a challenge on every eligible real start date."""
    days = daily.index.to_numpy()
    out = []
    for s in range(0, len(days) - MAX_DAYS):
        window = daily.iloc[s:s + MAX_DAYS]
        passed, used, reason = run_challenge(window, None, target, trail_dd, daily_limit)
        out.append((passed, used, reason))
    return out


def bootstrap_sim(daily, target, trail_dd, daily_limit, n=20000, seed=42):
    """His method: resample days with replacement. Destroys sequencing."""
    rng = np.random.default_rng(seed)
    vals = daily.to_numpy()
    out = []
    for _ in range(n):
        draw = rng.choice(vals, size=MAX_DAYS, replace=True)
        w = pd.Series(draw, index=pd.RangeIndex(MAX_DAYS))
        passed, used, reason = run_challenge(w, None, target, trail_dd, daily_limit)
        out.append((passed, used, reason))
    return out


def summarise(results, label):
    n = len(results)
    passes = [r for r in results if r[0]]
    p = len(passes) / n if n else 0.0
    avg_days = float(np.mean([r[1] for r in passes])) if passes else float("nan")
    reasons = {}
    for _, _, why in results:
        reasons[why] = reasons.get(why, 0) + 1
    fail_mix = {k: f"{v/n*100:.0f}%" for k, v in sorted(
        reasons.items(), key=lambda kv: -kv[1]) if k != "target"}
    print(f"  {label:14s} pass={p*100:5.1f}%  avg days to pass="
          f"{avg_days:5.1f}  n={n:5d}  fails: {fail_mix}")
    return p


def cumulative(p, k):
    """Probability of passing at least one of k independent attempts."""
    return 1 - (1 - p) ** k


def main():
    tr, daily = load_daily_pnl()
    print(f"{SYMBOL} x{CONTRACTS} contracts ({CONTRACTS//10 or 1} NQ-equivalent)")
    print(f"Trades: {len(tr):,}  trading days: {len(daily):,}  "
          f"total net: ${tr['pnl_scaled'].sum():+,.0f}")
    print(f"Daily PnL: mean ${daily.mean():+.0f}  sd ${daily.std():.0f}  "
          f"best ${daily.max():+,.0f}  worst ${daily.min():+,.0f}")
    print(f"Losing days: {(daily<0).mean()*100:.0f}%")

    for label, (target, trail, dlim) in ACCOUNTS.items():
        print(f"\n=== {label} — target ${target:,.0f}, trail ${trail:,.0f}, "
              f"daily loss ${dlim:,.0f}, {MAX_DAYS}-day limit ===")
        seq = sequential_sim(daily, target, trail, dlim)
        p_seq = summarise(seq, "SEQUENTIAL")
        boot = bootstrap_sim(daily, target, trail, dlim)
        p_boot = summarise(boot, "BOOTSTRAP")

        print(f"  cumulative pass probability (sequential / his-method):")
        for k in (1, 2, 3, 4):
            print(f"    {k} attempt{'s' if k>1 else ' '}: "
                  f"{cumulative(p_seq,k)*100:5.1f}%  /  {cumulative(p_boot,k)*100:5.1f}%")

        # Economics: what 4 attempts cost vs what a funded account is worth.
        fee = 50.0
        exp_cost = fee * (1 / p_seq) if p_seq > 0 else float("inf")
        print(f"  at ${fee:.0f}/attempt: expected ${exp_cost:,.0f} spent per PASS "
              f"(sequential)")


if __name__ == "__main__":
    main()
