#!/bin/bash

echo "🚀 Starting Hand Gesture Recognition Dashboard..."
echo ""
echo "Opening Streamlit dashboard in your browser..."
echo ""
echo "Dashboard will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"

# Prefer the workspace virtualenv (../../.venv) if present.
if [ -x "../../.venv/bin/python" ]; then
	PYTHON_BIN="../../.venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
	PYTHON_BIN="$VIRTUAL_ENV/bin/python"
else
	PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m streamlit run enhanced_dashboard.py
