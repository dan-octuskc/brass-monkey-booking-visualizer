#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run on a non-default port if 8501 is busy
PORT="${1:-8501}"
exec streamlit run visualizer4.py --server.runOnSave true --server.port "$PORT"
