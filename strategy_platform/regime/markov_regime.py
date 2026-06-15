"""Markov regime comparison — pure logic, no I/O, no Streamlit.

Lifted from the markov-hedge-fund-method skill's honest core so the platform
and the CLI share one implementation of the math. Used as Step 0 of the
3-period sweep-validation workflow: before an IS/OOS parameter sweep, check
whether the two windows have comparable Bull/Bear/Sideways regime mix. If they
don't, an OOS "winner" is likely regime exposure, not strategy edge.

Key honesty fixes carried over from the skill:
  - stride-sampled transition matrix (non-overlapping windows) for the baseline,
    so the persistence diagonal isn't inflated by overlapping rolling windows.

The public entry point is `compare_windows`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STATES = ["Bear", "Sideways", "Bull"]  # index 0, 1, 2


def label_regimes(close: pd.Series, window: int = 20, threshold: float = 0.05) -> pd.Series:
    """Label each day Bull(2) / Bear(0) / Sideways(1) from the trailing
    `window`-day return. Default ±5% over 20 days (hedge-fund-method convention).
    """
    rolling_return = close.pct_change(window)
    labels = pd.Series(1, index=close.index, dtype=int)  # default Sideways
    labels[rolling_return > threshold] = 2  # Bull
    labels[rolling_return < -threshold] = 0  # Bear
    return labels.dropna()


def build_transition_matrix(labels: pd.Series, stride: int = 1) -> np.ndarray:
    """3x3 MLE transition matrix. stride=window counts non-overlapping windows
    (the statistically honest matrix); stride=1 is the legacy overlapping one.
    """
    counts = np.zeros((3, 3), dtype=float)
    arr = labels.to_numpy()
    for i in range(0, len(arr) - stride, stride):
        counts[arr[i], arr[i + stride]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P at eigenvalue 1, normalised to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def _regime_mix(labels: pd.Series, start, end) -> tuple[np.ndarray, int]:
    """(3-vector of regime fractions, n_obs) over [start, end] inclusive."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub = labels.loc[(labels.index >= s) & (labels.index <= e)]
    n = len(sub)
    if n == 0:
        return np.array([0.0, 0.0, 0.0]), 0
    counts = np.bincount(sub.to_numpy().astype(int), minlength=3).astype(float)
    return counts / n, n


def _window_net_return(close: pd.Series, start, end):
    """Total return over the window, or None if too little data."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    sub = close.loc[(close.index >= s) & (close.index <= e)]
    if len(sub) < 2:
        return None
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JSD base 2 — symmetric, 0=identical, 1=disjoint."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.where(a > 0, a, 1e-12)
        b = np.where(b > 0, b, 1e-12)
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _dominant(p: np.ndarray) -> tuple[str, float]:
    idx = int(np.argmax(p))
    return STATES[idx], float(p[idx])


def _verdict(jsd: float, is_dom: float, oos_dom: float) -> tuple[str, str]:
    """PROCEED / WARN / RED FLAG + one-line advice.

    JSD: <0.15 similar, 0.15-0.30 moderately different, >=0.30 very different.
    Dominance >=0.70 means a window is essentially one regime — a strategy fit
    there will trivially work in similar regimes and fail elsewhere.
    """
    flags = []
    if jsd >= 0.30:
        flags.append("HIGH regime divergence")
    elif jsd >= 0.15:
        flags.append("moderate regime divergence")
    if oos_dom >= 0.70:
        flags.append(f"OOS is {oos_dom*100:.0f}% one regime")
    if is_dom >= 0.70:
        flags.append(f"IS is {is_dom*100:.0f}% one regime")

    if not flags:
        return "PROCEED", ("IS and OOS have comparable regime mix — sweep results "
                           "should test strategy edge, not regime exposure.")
    if jsd < 0.15 and max(is_dom, oos_dom) < 0.70:
        return "PROCEED", "Minor differences but within tolerance."
    if jsd >= 0.30 or max(is_dom, oos_dom) >= 0.85:
        return "RED FLAG", ("Windows differ enough that an OOS 'winner' likely just "
                            "exploited the regime. Re-scope the windows OR add a 3rd "
                            "confirm window in the missing regime BEFORE sweeping.")
    return "WARN", ("Regime mix is meaningfully different. Plan a 3rd-period confirm "
                    "with the strategy winner, ideally in a regime closer to IS.")


def compare_windows(
    daily_close: pd.Series,
    is_start, is_end,
    oos_start, oos_end,
    window: int = 20,
    threshold: float = 0.05,
) -> dict:
    """Compare IS vs OOS regime mix over a daily close series.

    `daily_close` should be the FULL available history (labels are computed over
    all of it, then the windows are sliced out — a longer baseline gives steadier
    Bull/Bear labels). Returns a dict the UI can render directly:

        {
          "ok": bool, "error": str|None,
          "is":  {"mix": [bear,side,bull], "n": int, "dominant": (name,pct),
                  "net_return": float|None, "start": ..., "end": ...},
          "oos": {... same ...},
          "jsd": float,
          "verdict": "PROCEED"|"WARN"|"RED FLAG",
          "advice": str,
          "missing_regime": str|None,   # under-represented in OOS vs IS
          "stationary": [bear,side,bull],  # full-history baseline
        }
    """
    if daily_close is None or len(daily_close) == 0:
        return {"ok": False, "error": "No price data for this symbol/range."}

    daily_close = daily_close.sort_index()
    labels = label_regimes(daily_close, window=window, threshold=threshold)
    if len(labels) < window + 5:
        return {"ok": False,
                "error": f"Only {len(labels)} labelled days — need more history "
                         f"(window={window})."}

    P = build_transition_matrix(labels, stride=window)
    pi = stationary_distribution(P)

    is_mix, is_n = _regime_mix(labels, is_start, is_end)
    oos_mix, oos_n = _regime_mix(labels, oos_start, oos_end)
    if is_n == 0 or oos_n == 0:
        which = "IS" if is_n == 0 else "OOS"
        return {"ok": False,
                "error": f"{which} window has 0 labelled days in range — "
                         f"check the dates / split."}

    is_ret = _window_net_return(daily_close, is_start, is_end)
    oos_ret = _window_net_return(daily_close, oos_start, oos_end)
    jsd = jensen_shannon(is_mix, oos_mix)
    _, is_dom = _dominant(is_mix)
    _, oos_dom = _dominant(oos_mix)
    verdict, advice = _verdict(jsd, is_dom, oos_dom)

    # Regime under-represented in OOS relative to IS — what to confirm against.
    diff = is_mix - oos_mix
    missing = STATES[int(np.argmax(diff))] if verdict != "PROCEED" else None

    return {
        "ok": True, "error": None,
        "is": {"mix": is_mix.tolist(), "n": is_n, "dominant": _dominant(is_mix),
               "net_return": is_ret, "start": str(is_start), "end": str(is_end)},
        "oos": {"mix": oos_mix.tolist(), "n": oos_n, "dominant": _dominant(oos_mix),
                "net_return": oos_ret, "start": str(oos_start), "end": str(oos_end)},
        "jsd": float(jsd),
        "verdict": verdict,
        "advice": advice,
        "missing_regime": missing,
        "stationary": pi.tolist(),
    }
