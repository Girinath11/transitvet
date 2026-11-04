#!/usr/bin/env bash
# TransitVet install for Linux / macOS:  bash install.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m transitvet test
echo "Done. Start the dashboard with:  . .venv/bin/activate && streamlit run app.py"
