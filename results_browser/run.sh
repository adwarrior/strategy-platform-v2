#!/bin/bash
# Launch the Results Browser. Run from anywhere.
cd "$(dirname "$0")/.." || exit 1
exec streamlit run results_browser/app.py --server.port "${PORT:-8505}"
