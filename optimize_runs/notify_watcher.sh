#!/usr/bin/env bash
# Watches RUN_STATE.json for completion, then fires Windows toast + bell.
# Launched in its own tmux session to avoid interfering with the main run.
set -u
STATE="/home/ad/strategy-platform-v2/optimize_runs/RUN_STATE.json"
FLAG="/home/ad/strategy-platform-v2/optimize_runs/COMPLETE.flag"

echo "[notify_watcher] started $(date -Iseconds) — polling $STATE every 60s"

while true; do
    status=$(python -c "import json,sys; d=json.load(open('$STATE')); print(d.get('synthesis',''))" 2>/dev/null)
    if [ "$status" = "ready_for_claude" ]; then
        echo "[notify_watcher] synthesis=ready_for_claude detected at $(date -Iseconds)"
        # Windows MessageBox (most reliable cross-version)
        if command -v powershell.exe >/dev/null 2>&1; then
            powershell.exe -NoProfile -WindowStyle Hidden -Command "
                Add-Type -AssemblyName System.Windows.Forms;
                [System.Windows.Forms.MessageBox]::Show(
                    'All phases complete. Reply ''synthesize'' to Claude.',
                    'STF Optimization DONE',
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Information
                ) | Out-Null
            " 2>/dev/null &
        fi
        # Terminal bells
        for _ in 1 2 3 4 5; do printf '\a'; sleep 0.2; done
        touch "$FLAG"
        echo "[notify_watcher] notification fired. flag: $FLAG"
        exit 0
    fi
    sleep 60
done
