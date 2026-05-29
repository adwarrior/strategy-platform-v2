"""
autoresearch_loop.py — autonomous parameter-optimisation loop for trading strategies.

Calls claude-haiku-4-5 (cheapest model) to propose ONE parameter change per generation,
runs the IS backtest via run_experiment.py, keeps improvements, reverts failures.

Token budget per generation:
    ~600 input tokens + ~80 output tokens ≈ $0.0005 with Haiku
    100 generations ≈ $0.05  |  1000 generations ≈ $0.50

Usage:
    cd /home/ad/strategy-platform
    python autoresearch/autoresearch_loop.py --strategy wicktest5m --symbol NQ=F

    Optional:
        --max-gens 200          stop after N generations (default: unlimited)
        --model haiku           haiku (default) | sonnet (structural changes)
        --start 2024-01-01
        --end   2025-01-01

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY must be set in environment or .env
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Ensure strategy_platform package is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Anthropic client setup
# ---------------------------------------------------------------------------

try:
    import anthropic
except ImportError:
    sys.exit("anthropic package not found. Run: pip install anthropic")

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

MODELS = {
    'haiku':  'claude-haiku-4-5-20251001',  # cheapest — parameter tweaks
    'sonnet': 'claude-sonnet-4-6',          # smarter — structural changes
}

RESULTS_TSV   = Path(__file__).parent / 'results.tsv'
REPORTS_DIR   = Path(__file__).parent.parent / 'reports'
STRATEGY_BASE = Path(__file__).parent.parent / 'strategy_platform' / 'strategies'


# ---------------------------------------------------------------------------
# Strategy file helpers
# ---------------------------------------------------------------------------

def _strategy_path(strategy_name: str) -> Path:
    p = STRATEGY_BASE / strategy_name / 'strategy.py'
    if p.exists():
        return p
    # Lowercase fallback: registered name may be CamelCase (e.g. WaeJurikPro)
    # while the on-disk directory is lowercase (waejurikpro).
    p = STRATEGY_BASE / strategy_name.lower() / 'strategy.py'
    if p.exists():
        return p
    # Variant fallback: strip trailing _variant suffix (e.g. goldbot6_tick → goldbot6)
    base = strategy_name.rsplit('_', 1)[0]
    return STRATEGY_BASE / base / 'strategy.py'


def _read_default_params(strategy_name: str) -> dict:
    """Extract default_params dict from strategy.py using ast."""
    src = _strategy_path(strategy_name).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                # Plain assignment: default_params = {...}
                if (isinstance(item, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'default_params'
                                for t in item.targets)):
                    return ast.literal_eval(item.value)
                # Annotated assignment: default_params: Dict[str, Any] = {...}
                if (isinstance(item, ast.AnnAssign)
                        and isinstance(item.target, ast.Name)
                        and item.target.id == 'default_params'
                        and item.value is not None):
                    return ast.literal_eval(item.value)
    raise ValueError(f"Could not find default_params in {strategy_name}/strategy.py")


def _read_param_grid(strategy_name: str) -> dict:
    """Return param_grid by importing the strategy at runtime."""
    import strategy_platform  # noqa: F401 — triggers registry
    from strategy_platform.registry import StrategyRegistry
    strat = StrategyRegistry.get(strategy_name)
    if strat is None:
        raise ValueError(f"Strategy '{strategy_name}' not found in registry")
    return dict(strat().param_grid)


def _write_default_params(strategy_name: str, new_params: dict):
    """Update the default_params dict in strategy.py in-place."""
    path  = _strategy_path(strategy_name)
    lines = path.read_text().splitlines()

    # Find the line that starts the default_params block
    start_idx = None
    for i, line in enumerate(lines):
        if re.match(r'\s+default_params\s*[=:].*\{', line):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not locate default_params block to update.")

    # Walk forward counting braces to find the closing line
    depth = 0
    end_idx = None
    for i in range(start_idx, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("Could not find closing brace of default_params block.")

    # Preserve the annotation from the original opening line if present
    orig_open = lines[start_idx]
    if 'Dict[str, Any]' in orig_open:
        open_line = '    default_params: Dict[str, Any] = {'
    else:
        open_line = '    default_params = {'

    # Build the replacement block — cast number_input floats back to int when appropriate
    def _fmt(v):
        if isinstance(v, float) and v == int(v):
            return repr(int(v))
        return repr(v)

    new_block = [open_line]
    for k, v in new_params.items():
        new_block.append(f'        {k!r}: {_fmt(v)},')
    new_block.append('    }')

    updated = lines[:start_idx] + new_block + lines[end_idx + 1:]
    path.write_text('\n'.join(updated) + '\n')


def _git_snapshot(strategy_name: str) -> str:
    """Return the current content of strategy.py (for revert on failure)."""
    return _strategy_path(strategy_name).read_text()


def _git_revert(strategy_name: str, snapshot: str):
    """Restore strategy.py to the saved snapshot."""
    _strategy_path(strategy_name).write_text(snapshot)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _run_experiment(strategy: str, symbol: str, start: str, end: str,
                    min_trades: int = 10, bar_type: str = 'time') -> dict | None:
    """Run run_experiment.py and return parsed JSON result, or None on failure."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'run_experiment.py'),
        '--strategy',   strategy,
        '--symbol',     symbol,
        '--start',      start,
        '--end',        end,
        '--bar-type',   bar_type,
        '--min-trades', str(min_trades),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    [experiment error] stdout: {result.stdout.strip()}")
            print(f"    [experiment error] stderr: {result.stderr.strip()}")
            return None
        last_line = result.stdout.strip().split('\n')[-1]
        return json.loads(last_line)
    except Exception as e:
        print(f"    [experiment exception] {e}")
        return None


# ---------------------------------------------------------------------------
# LLM proposal
# ---------------------------------------------------------------------------

_SYSTEM = textwrap.dedent("""\
    You are a trading strategy parameter optimizer.
    Your job: propose ONE small parameter change that might improve Sharpe ratio.
    Respond with ONLY a valid JSON object: {"param": "<name>", "value": <new_value>}
    No explanation. No markdown. Just the JSON.
""")


def _build_prompt(params: dict, param_grid: dict, history: list[dict]) -> str:
    """Build a compact prompt — minimises tokens sent to the model."""
    history_lines = []
    for h in history[-10:]:  # last 10 generations only
        kept = 'KEPT' if h['kept'] else 'reverted'
        history_lines.append(
            f"  gen {h['gen']:>3}: {h['param']}={h['old']}→{h['new']}  "
            f"sharpe={h['sharpe']:.3f} trades={h['trades']}  [{kept}]"
        )
    history_str = '\n'.join(history_lines) if history_lines else '  (none yet)'

    # Only show params that are in param_grid (changeable ones)
    param_lines = []
    for k in param_grid:
        v = params.get(k)
        allowed = param_grid[k]
        param_lines.append(f"  {k}={v!r}  allowed={allowed}")
    params_str = '\n'.join(param_lines)

    return textwrap.dedent(f"""\
        Changeable parameters (ONLY these may be modified):
        {params_str}

        Recent history (last 10 gens):
        {history_str}

        Propose ONE change to improve Sharpe.
        You MUST only propose a parameter from the list above.
        The new value MUST be in the allowed list for that parameter.
        JSON only: {{"param": "<name>", "value": <new_value>}}
    """)


def _propose_change(
    client: anthropic.Anthropic,
    model: str,
    params: dict,
    param_grid: dict,
    history: list[dict],
    debug: bool = False,
) -> dict | None:
    """Ask the model for a parameter change. Returns {"param": str, "value": Any} or None."""
    prompt = _build_prompt(params, param_grid, history)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=128,
            system=_SYSTEM,
            messages=[{'role': 'user', 'content': prompt}],
        )
        raw = msg.content[0].text.strip()
        if debug:
            print(f"    [model raw] {raw!r}")
        # Strip markdown code fences if present
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)
        raw = raw.strip()
        # Extract first {...} block in case model added surrounding text
        m = re.search(r'\{[^}]+\}', raw)
        if m:
            raw = m.group(0)
        proposal = json.loads(raw)
        if 'param' in proposal and 'value' in proposal:
            return proposal
    except Exception as e:
        print(f"    [propose error] {type(e).__name__}: {e}")
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_proposal(proposal: dict, param_grid: dict, current_params: dict) -> bool:
    """Check the proposed value is in the allowed grid."""
    p, v = proposal['param'], proposal['value']
    if p not in param_grid:
        return False
    allowed = param_grid[p]
    # For bool params not in grid, allow True/False
    if not allowed:
        return isinstance(v, bool)
    return v in allowed


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_RUN_TSV: Path = None  # set at start of main()

TSV_HEADER = 'gen\ttimestamp\tsharpe\tnet_pnl\ttrades\twin_rate\tparam_changed\told_value\tnew_value\tkept\n'


def _ensure_tsv():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)


def _init_run_tsv(strategy: str, symbol: str, model: str, start: str, end: str) -> Path:
    """Create a timestamped run file in reports/ and return its path."""
    global _RUN_TSV
    REPORTS_DIR.mkdir(exist_ok=True)
    sym_safe = symbol.replace('=', '_')
    ts       = datetime.now().strftime('%Y%m%d_%H%M')
    path     = REPORTS_DIR / f"AR_{strategy}_{sym_safe}_{ts}.tsv"
    meta = (
        f"# strategy={strategy}\n"
        f"# symbol={symbol}\n"
        f"# model={model}\n"
        f"# is_start={start}\n"
        f"# is_end={end}\n"
        f"# run_ts={ts}\n"
    )
    path.write_text(meta + TSV_HEADER)
    _RUN_TSV = path
    return path


def _log_result(gen: int, result: dict, param: str, old_val, new_val, kept: bool):
    ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = (f"{gen}\t{ts}\t{result.get('sharpe', 'N/A')}\t{result.get('net_pnl', 'N/A')}\t"
           f"{result.get('trades', 'N/A')}\t{result.get('win_rate', 'N/A')}\t"
           f"{param}\t{old_val}\t{new_val}\t{'yes' if kept else 'no'}\n")
    with open(RESULTS_TSV, 'a') as f:
        f.write(row)
    if _RUN_TSV is not None:
        with open(_RUN_TSV, 'a') as f:
            f.write(row)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', required=True)
    p.add_argument('--symbol',   required=True)
    p.add_argument('--start',    default=None)
    p.add_argument('--end',      default=None)
    p.add_argument('--bar-type',    default='time', choices=['time', '1m', 'tick'],
                   help='Bar type: time=5M (default), 1m=1-minute (MNQ/MES/MGC), tick=tick bars')
    p.add_argument('--max-gens',   type=int, default=0, help='0 = unlimited')
    p.add_argument('--model',      default='haiku', choices=list(MODELS))
    p.add_argument('--min-trades', type=int, default=10, help='Minimum trades to accept a run')
    return p.parse_args()


def main():
    args   = parse_args()
    model  = MODELS[args.model]
    client = anthropic.Anthropic()

    from datetime import timedelta
    today = datetime.today()
    start = args.start or (today - timedelta(days=730)).strftime('%Y-%m-%d')
    end   = args.end   or (today - timedelta(days=365)).strftime('%Y-%m-%d')

    _ensure_tsv()
    _init_run_tsv(args.strategy, args.symbol, args.model, start, end)

    # -- Generation 0: baseline --
    print(f"\n=== autoresearch_loop  strategy={args.strategy}  model={args.model} ===")
    print(f"    IS window: {start} → {end}\n")
    print("Running generation 0 (baseline)...")

    baseline = _run_experiment(args.strategy, args.symbol, start, end, args.min_trades, args.bar_type)
    if baseline is None or 'error' in baseline:
        sys.exit(f"Baseline run failed: {baseline}")

    best_sharpe = baseline['sharpe']
    print(f"  Baseline → sharpe={best_sharpe:.4f}  trades={baseline['trades']}  "
          f"net_pnl=${baseline['net_pnl']:.0f}")
    _log_result(0, baseline, 'baseline', '-', '-', True)

    history: list[dict] = []
    gen = 1
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    while True:
        if args.max_gens and gen > args.max_gens:
            print(f"\nReached max generations ({args.max_gens}). Stopping.")
            break

        params     = _read_default_params(args.strategy)
        param_grid = _read_param_grid(args.strategy)
        snapshot   = _git_snapshot(args.strategy)

        # Ask Haiku for a proposal (show raw output for first 3 gens to aid debugging)
        proposal = _propose_change(client, model, params, param_grid, history, debug=(gen <= 3))
        if proposal is None:
            print(f"  gen {gen:>4}: model returned invalid proposal — skipping")
            gen += 1
            continue

        p_name = proposal['param']
        p_new  = proposal['value']
        p_old  = params.get(p_name)

        # Skip no-op proposals
        if p_new == p_old:
            print(f"  gen {gen:>4}: no-op ({p_name}={p_new}) — skipping")
            gen += 1
            continue

        # Validate against allowed grid
        if not _validate_proposal(proposal, param_grid, params):
            print(f"  gen {gen:>4}: invalid proposal {p_name}={p_new} — skipping")
            gen += 1
            continue

        # Apply change
        new_params = {**params, p_name: p_new}
        try:
            _write_default_params(args.strategy, new_params)
        except Exception as e:
            print(f"  gen {gen:>4}: failed to write params: {e}")
            gen += 1
            continue

        # Run experiment
        result = _run_experiment(args.strategy, args.symbol, start, end, args.min_trades, args.bar_type)

        # Evaluate
        if result is None or 'error' in result:
            _git_revert(args.strategy, snapshot)
            print(f"  gen {gen:>4}: run failed ({result}) — reverted")
            history.append({'gen': gen, 'param': p_name, 'old': p_old, 'new': p_new,
                            'sharpe': 0, 'trades': 0, 'kept': False})
            _log_result(gen, {}, p_name, p_old, p_new, False)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                sys.exit(f"\n{consecutive_failures} consecutive failures — aborting (likely data/connection issue).")
            gen += 1
            continue

        consecutive_failures = 0

        new_sharpe = result['sharpe']
        kept = new_sharpe > best_sharpe

        # Sanity check: reject suspiciously perfect results
        if result.get('win_rate', 0) > 0.90 or new_sharpe > 5.0:
            kept = False
            print(f"  gen {gen:>4}: {p_name}={p_old}→{p_new}  sharpe={new_sharpe:.4f}  "
                  f"⚠ suspicious result — reverted")
        elif kept:
            best_sharpe = new_sharpe
            print(f"  gen {gen:>4}: {p_name}={p_old}→{p_new}  sharpe={new_sharpe:.4f}  ✓ KEPT"
                  f"  (Δ{new_sharpe - baseline['sharpe']:+.4f} vs baseline)")
        else:
            print(f"  gen {gen:>4}: {p_name}={p_old}→{p_new}  sharpe={new_sharpe:.4f}  ✗ reverted")

        if not kept:
            _git_revert(args.strategy, snapshot)

        history.append({'gen': gen, 'param': p_name, 'old': p_old, 'new': p_new,
                        'sharpe': new_sharpe, 'trades': result.get('trades', 0), 'kept': kept})
        _log_result(gen, result, p_name, p_old, p_new, kept)
        gen += 1


if __name__ == '__main__':
    main()
