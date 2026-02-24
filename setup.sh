#!/usr/bin/env bash
set -euo pipefail

echo ""
echo "============================================"
echo "  BioVision Setup"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer python3, fall back to python
PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[ERROR] Python not found."
    echo "        Install Python 3.10+ via your package manager."
    echo "          macOS:  brew install python@3.11"
    echo "          Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo "Using: $($PYTHON --version)"
echo ""

echo "[1/2] Setting up Python backend..."
"$PYTHON" "$SCRIPT_DIR/setup_backend.py"

if [[ "${SKIP_NPM_INSTALL:-0}" == "1" ]]; then
    echo ""
    echo "[2/2] Skipping npm install (SKIP_NPM_INSTALL=1)."
elif command -v npm &>/dev/null; then
    echo ""
    echo "[2/2] Installing frontend dependencies (npm install)..."
    npm install
else
    echo ""
    echo "[WARN] npm not found. Frontend dependencies were not installed."
    echo "       Install Node.js LTS, then run: npm install"
fi

echo ""
echo "Setup complete."
echo "Start app with: npm run dev"
