#!/usr/bin/env bash
# Master orchestrator for SuperTrendFractal MNQ optimization.
#
# Runs phases sequentially with per-phase RUN_STATE.json checkpoints. Each
# pipeline/walk-forward sub-command writes its own checkpoint files in reports/,
# so the underlying combos are resumable on relaunch via this script too.
#
# Phases:
#   1. stage_a            - 24,960-combo core sweep (5M, full data)
#   2. filter_a           - DD<=$2000 + top-5 cores
#   3. stage_b_core_<i>   - 144 session combos per core (5 sub-phases)
#   4. aggregate_b        - top-5 finalists by OOS Sortino
#   5. wfo                - 8-slice walk-forward on union grid
#   6. autoresearch       - 3 seeds * 500 gens each
#   7. synthesis          - hands off to claude (no-op in shell; marker only)
#
# Logs:   reports/logs/run_all.log  (tee'd from stdout)
# State:  optimize_runs/RUN_STATE.json
#
# Usage:
#   ./optimize_runs/run_all.sh                # start or resume from RUN_STATE
#   START_PHASE=stage_b_core_1 ./run_all.sh   # force start phase

set -u
set -o pipefail

ROOT="/home/ad/strategy-platform-v2"
LOG="$ROOT/reports/logs/run_all.log"
STATE="$ROOT/optimize_runs/RUN_STATE.json"
PY="python"
SYM="MNQ"
STR="supertrendfractal"
DATA_START="2020-01-01"
DATA_END="2026-05-15"
BAR_TYPE="time"

mkdir -p "$ROOT/reports/logs" "$ROOT/optimize_runs"
cd "$ROOT"

# ---------------------------------------------------------------------------
# State helpers (jq-free, plain bash + python one-liner)
# ---------------------------------------------------------------------------
state_get() {
    "$PY" -c "import json,sys; d=json.load(open('$STATE')) if __import__('os').path.exists('$STATE') else {}; print(d.get(sys.argv[1],''))" "$1"
}
state_set() {
    "$PY" -c "
import json, os, sys
p = '$STATE'
d = json.load(open(p)) if os.path.exists(p) else {}
d[sys.argv[1]] = sys.argv[2]
d['_updated'] = __import__('datetime').datetime.now().isoformat(timespec='seconds')
open(p, 'w').write(json.dumps(d, indent=2))
" "$1" "$2"
}

phase_done() { [ "$(state_get "$1")" = "done" ]; }
phase_mark() { state_set "$1" "$2"; }

banner() {
    echo
    echo "======================================================================"
    echo "  $1"
    echo "  ts: $(date -Iseconds)"
    echo "======================================================================"
}

# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
run_stage_a() {
    banner "PHASE 1 — Stage A (core sweep, ~24,960 combos)"
    phase_mark stage_a running
    "$PY" -m strategy_platform.optimize.pipeline \
        --strategy "$STR" --symbol "$SYM" --bar-type "$BAR_TYPE" \
        --start "$DATA_START" --end "$DATA_END" \
        --train-pct 0.70 \
        --mc-sims 0 --mc-top-n 300 --oos-top-n 300 --min-trades 20 \
        --rank-by sortino \
        --param-grid "$(cat optimize_runs/stage_a_grid.json)" \
        --run-settings '{"stage":"A","note":"core sweep, all 3 exits, 24h, dir=Both"}'
    rc=$?
    if [ $rc -ne 0 ]; then phase_mark stage_a "failed_rc${rc}"; return $rc; fi
    phase_mark stage_a done
}

run_filter_a() {
    banner "PHASE 2 — Filter Stage A (DD<=\$2000, top-5 cores)"
    phase_mark filter_a running
    "$PY" optimize_runs/filter_stage_a.py 5 || { phase_mark filter_a failed; return 1; }
    "$PY" optimize_runs/build_stage_b_grids.py || { phase_mark filter_a failed; return 1; }
    phase_mark filter_a done
}

run_stage_b() {
    banner "PHASE 3 — Stage B (session sweep, 5 cores × 144 = 720 combos)"
    phase_mark stage_b running
    # Always (re)write the log fresh
    : > optimize_runs/stage_b_run_log.txt
    for i in 1 2 3 4 5; do
        grid_file="optimize_runs/stage_b_core_${i}.json"
        if [ ! -f "$grid_file" ]; then
            echo "  skipping core_$i: $grid_file missing"
            continue
        fi
        sub_phase="stage_b_core_${i}"
        if phase_done "$sub_phase"; then
            echo "  core_$i: already done, skipping"
            # still try to recover its OOS csv name from earlier run if needed
            continue
        fi
        phase_mark "$sub_phase" running
        echo "  --- core $i ---"
        "$PY" -m strategy_platform.optimize.pipeline \
            --strategy "$STR" --symbol "$SYM" --bar-type "$BAR_TYPE" \
            --start "$DATA_START" --end "$DATA_END" \
            --train-pct 0.70 \
            --mc-sims 0 --mc-top-n 60 --oos-top-n 60 --min-trades 20 \
            --rank-by sortino \
            --param-grid "$(cat "$grid_file")" \
            --run-settings "{\"stage\":\"B\",\"core\":$i}"
        rc=$?
        if [ $rc -ne 0 ]; then phase_mark "$sub_phase" "failed_rc${rc}"; phase_mark stage_b failed; return $rc; fi
        # record latest OOS csv
        latest=$(ls -t reports/OOS_${STR}_${SYM}_*.csv 2>/dev/null | head -1 | xargs -n1 basename)
        echo "core_${i}    ${latest}" >> optimize_runs/stage_b_run_log.txt
        phase_mark "$sub_phase" done
    done
    phase_mark stage_b done
}

run_aggregate_b() {
    banner "PHASE 4 — Aggregate Stage B → top-5 finalists"
    phase_mark aggregate_b running
    "$PY" optimize_runs/aggregate_stage_b.py 5 || { phase_mark aggregate_b failed; return 1; }
    phase_mark aggregate_b done
}

run_wfo() {
    banner "PHASE 5 — Walk-forward on union grid (8 slices)"
    phase_mark wfo running
    "$PY" -m strategy_platform.optimize.walk_forward \
        --strategy "$STR" --symbol "$SYM" --bar-type "$BAR_TYPE" \
        --start "$DATA_START" --end "$DATA_END" \
        --is-window-days 720 --oos-window-days 180 --step-days 180 \
        --rank-by sortino --min-trades 20 \
        --param-grid "$(cat optimize_runs/wfo_grid.json)" \
        --run-settings '{"stage":"WFO","note":"validate finalists"}'
    rc=$?
    if [ $rc -ne 0 ]; then phase_mark wfo "failed_rc${rc}"; return $rc; fi
    phase_mark wfo done
}

run_ar() {
    banner "PHASE 6 — Autoresearch on top 3 finalists (haiku, 500 gens each)"
    phase_mark autoresearch running
    "$PY" optimize_runs/run_ar_for_seeds.py || { phase_mark autoresearch failed; return 1; }
    phase_mark autoresearch done
}

run_synthesis_marker() {
    banner "PHASE 7 — Ready for synthesis"
    phase_mark synthesis ready_for_claude
}

notify_done() {
    local msg="$1"
    # Windows toast via PowerShell (WSL2 -> Windows). Silently no-op if not available.
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;
            \$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);
            \$t.GetElementsByTagName('text')[0].AppendChild(\$t.CreateTextNode('STF Optimization')) | Out-Null;
            \$t.GetElementsByTagName('text')[1].AppendChild(\$t.CreateTextNode('$msg')) | Out-Null;
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('WSL').Show([Windows.UI.Notifications.ToastNotification]::new(\$t));
        " 2>/dev/null || true
    fi
    # Terminal bell (×5 to be hard to miss) + banner
    for _ in 1 2 3 4 5; do printf '\a'; sleep 0.2; done
    echo
    echo "######################################################################"
    echo "#  ${msg}"
    echo "#  $(date -Iseconds)"
    echo "######################################################################"
    # Mark with a flag file so Claude/user can detect at a glance
    touch /home/ad/strategy-platform-v2/optimize_runs/COMPLETE.flag
}

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
{
    echo "================ run_all.sh started at $(date -Iseconds) ================"
    echo "WSL host: $(hostname)  pid: $$"
    echo "phase pointer: $(state_get current_phase || echo unknown)"

    # Mark session
    state_set session_start "$(date -Iseconds)"
    state_set pid "$$"

    phase_done stage_a || run_stage_a || exit $?
    phase_done filter_a || run_filter_a || exit $?
    phase_done stage_b || run_stage_b || exit $?
    phase_done aggregate_b || run_aggregate_b || exit $?
    phase_done wfo || run_wfo || exit $?
    phase_done autoresearch || run_ar || exit $?
    run_synthesis_marker

    state_set session_end "$(date -Iseconds)"
    state_set status "complete"
    notify_done "ALL PHASES COMPLETE — ready for synthesis. Reply 'synthesize' to Claude."
    echo
    echo "================ run_all.sh DONE at $(date -Iseconds) ================"
} 2>&1 | tee -a "$LOG"
