#!/usr/bin/env bash
# Resume the SuperTrendFractal optimization after a reboot or kill.
#
# Reads RUN_STATE.json to determine where we left off, then re-launches
# run_all.sh inside a fresh tmux session. The pipeline / walk-forward
# checkpoints in reports/ ensure mid-stage combos are not redone.

set -u
ROOT="/home/ad/strategy-platform-v2"
STATE="$ROOT/optimize_runs/RUN_STATE.json"
LOG="$ROOT/reports/logs/run_all.log"
SESSION="stf_overnight"

cd "$ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is already alive. Attach with:"
    echo "    tmux attach -t $SESSION"
    exit 0
fi

if [ -f "$STATE" ]; then
    echo "Last RUN_STATE:"
    python -c "import json; print(open('$STATE').read())"
fi

tmux new-session -d -s "$SESSION" "cd $ROOT && bash optimize_runs/run_all.sh; echo; echo 'session ended. attach to view: tmux attach -t $SESSION'; sleep infinity"

echo
echo "tmux session '$SESSION' launched."
echo "  attach :  tmux attach -t $SESSION"
echo "  log    :  tail -f $LOG"
echo "  state  :  cat $STATE"
