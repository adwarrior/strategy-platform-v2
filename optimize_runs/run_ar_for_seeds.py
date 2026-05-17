"""
Run autoresearch_loop.py against top-N finalists, seeding strategy.default_params
with each finalist's config in turn.

For each seed (i=1..N_AR):
  1. git stash the strategy file (clean working tree)
  2. patch strategy.default_params with seed[i] via ast rewrite (same mechanism
     AR itself uses)
  3. commit the patched defaults so AR's git-revert points at a sane base
  4. run autoresearch_loop with --max-gens MAX_GENS
  5. capture the produced AR_supertrendfractal_MNQ_*.tsv name
  6. git revert to original strategy file

Output:
  optimize_runs/ar_run_log.txt      — per-seed AR_*.tsv filenames + best gen
  reports/AR_supertrendfractal_MNQ_*.tsv   (one per seed, created by AR itself)
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRAT_FILE = ROOT / "strategy_platform" / "strategies" / "supertrendfractal" / "strategy.py"
AR_SCRIPT = ROOT / "autoresearch" / "autoresearch_loop.py"
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "optimize_runs"

# Tunables
N_AR_SEEDS = 3
MAX_GENS = 500
DATA_START = "2020-01-01"
DATA_END = "2026-05-15"


def _read_default_params(strat_path: Path) -> dict:
    tree = ast.parse(strat_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) \
                        and item.target.id == "default_params":
                    return ast.literal_eval(item.value)
                if isinstance(item, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "default_params" for t in item.targets):
                    return ast.literal_eval(item.value)
    raise ValueError("default_params block not found")


def _write_default_params(strat_path: Path, new_params: dict) -> None:
    """Mirror autoresearch_loop._write_default_params logic (find block by regex,
    rewrite with new dict, preserve indent)."""
    text = strat_path.read_text()
    lines = text.splitlines(keepends=True)
    open_idx = None
    for i, line in enumerate(lines):
        if re.match(r"\s+default_params\s*[=:].*\{", line):
            open_idx = i
            break
    if open_idx is None:
        raise ValueError("default_params block start not found")
    # find closing brace (assume single-line per key)
    depth = lines[open_idx].count("{") - lines[open_idx].count("}")
    close_idx = open_idx
    while depth > 0 and close_idx < len(lines) - 1:
        close_idx += 1
        depth += lines[close_idx].count("{") - lines[close_idx].count("}")
    if depth != 0:
        raise ValueError("default_params block close brace not found")
    is_annotated = ":" in lines[open_idx].split("=")[0] and "Dict" in lines[open_idx]
    open_line = "    default_params: Dict[str, Any] = {\n" if is_annotated else "    default_params = {\n"
    body_lines = ["        " + f"{k!r}: {_fmt(v)},\n" for k, v in new_params.items()]
    close_line = "    }\n"
    new_lines = lines[:open_idx] + [open_line] + body_lines + [close_line] + lines[close_idx + 1:]
    strat_path.write_text("".join(new_lines))


def _fmt(v):
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, float):
        return repr(v)
    return repr(v)


def _git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=check)


def _latest_ar_tsv() -> Path | None:
    files = sorted(REPORTS.glob("AR_supertrendfractal_MNQ_*.tsv"))
    return files[-1] if files else None


def _ensure_clean_git():
    r = _git("status", "--porcelain", str(STRAT_FILE), check=False)
    if r.stdout.strip():
        print(f"  WARN: {STRAT_FILE.name} has uncommitted changes; stashing")
        _git("stash", "push", "-m", "ar_seed_stash", "--", str(STRAT_FILE), check=False)


def main() -> int:
    finalists_file = OUT_DIR / "stage_b_finalists.json"
    if not finalists_file.exists():
        print(f"ERROR: {finalists_file} missing")
        return 2
    finalists = json.loads(finalists_file.read_text())[:N_AR_SEEDS]
    if not finalists:
        print("ERROR: no finalists")
        return 3

    log_lines: list[str] = []
    _ensure_clean_git()
    original = STRAT_FILE.read_text()

    for i, seed in enumerate(finalists, 1):
        print(f"\n=== AR seed #{i}/{len(finalists)} ===")
        # Patch
        current = _read_default_params(STRAT_FILE)
        merged = {**current, **seed}
        _write_default_params(STRAT_FILE, merged)
        _git("add", str(STRAT_FILE), check=False)
        _git("commit", "-m", f"AR seed {i}: lock finalist params", "--allow-empty",
             check=False)

        before_tsv = _latest_ar_tsv()

        # Launch AR
        cmd = [
            "python", str(AR_SCRIPT),
            "--strategy", "supertrendfractal",
            "--symbol", "MNQ",
            "--bar-type", "time",
            "--start", DATA_START,
            "--end", DATA_END,
            "--max-gens", str(MAX_GENS),
            "--model", "haiku",
            "--min-trades", "20",
        ]
        print("  exec: " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        print(f"  AR rc={rc}")

        # Capture the newly produced tsv
        after_tsv = _latest_ar_tsv()
        tsv_name = (after_tsv.name if after_tsv and after_tsv != before_tsv
                    else "no_new_tsv")
        log_lines.append(f"seed_{i}\t{tsv_name}\trc={rc}")

        # Restore original strategy
        STRAT_FILE.write_text(original)
        _git("add", str(STRAT_FILE), check=False)
        _git("commit", "-m", f"AR seed {i}: restore original defaults",
             "--allow-empty", check=False)

    (OUT_DIR / "ar_run_log.txt").write_text("\n".join(log_lines) + "\n")
    print("\nAR sweep complete; per-seed TSV files in reports/")
    for ln in log_lines:
        print("  " + ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
