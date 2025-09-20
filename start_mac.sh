#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# PromGen macOS launcher (Apple Silicon / Intel)
# Usage:
#   ./start_mac.sh                            # defaults: Python 3.10, port 8080
#   ./start_mac.sh --python 3.11
#   ./start_mac.sh --port 7860 --host 127.0.0.1
#
# Notes:
# - Uses Astral's 'uv' for venv + installs.
# - Pins Torch trio to macOS (MPS) wheels you validated: 2.8.0 / 0.23.0 / 2.8.0.
# - Sets MPS envs that behaved well in testing.
# ------------------------------------------------------------

PY_VER="3.10"
PORT="8080"
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)  PY_VER="${2:-3.10}"; shift 2 ;;
    --port)    PORT="${2:-8080}";  shift 2 ;;
    --host)    HOST="${2:-0.0.0.0}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--python 3.10] [--port 8080] [--host 0.0.0.0]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "🍎 Setup PromGen for macOS..."
if [[ "${OSTYPE:-}" != darwin* && "$(uname -s 2>/dev/null || echo)" != "Darwin" ]]; then
  echo "❌ This script is designed for macOS only"
  exit 1
fi

# 1) Require uv
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ 'uv' command not found. Install UV: https://docs.astral.sh/uv/getting-started/installation"
  exit 1
fi

# 2) Ensure a venv exists & activate it
if [[ ! -d .venv ]]; then
  echo "📦 Creating virtual environment with Python ${PY_VER}..."
  uv venv --python="${PY_VER}"
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source ./.venv/bin/activate
  echo "✅ Virtual environment activated"
fi

# 3) Install deps if missing (quick import probe)
if uv run python - <<'PY' >/dev/null 2>&1
import importlib
for m in ("torch","torchvision","torchaudio","transformers","gradio"):
    importlib.import_module(m)
PY
then
  echo "✅ Required packages already installed"
else
  echo "📦 Installing required packages for macOS (MPS)..."
  # Pin to your known-good trio for macOS
  uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
  # Then everything else
  uv pip install -r requirements.txt
  echo "✅ Required packages installed"
fi

# 4) Graceful shutdown (safety net; 'exec' should make this unnecessary)
_term_pid=""
cleanup() {
  echo "🛑 Stopping PromGen..."
  if [[ -n "${_term_pid}" ]] && kill -0 "${_term_pid}" 2>/dev/null; then
    kill -INT "${_term_pid}" || true
    sleep 2
    kill -TERM "${_term_pid}" || true
    sleep 1
    kill -KILL "${_term_pid}" || true
  fi
}
trap cleanup INT TERM

# 5) Apple Silicon perf/env knobs
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# 6) Run the app
echo "🚀 Setup complete - Starting PromGen..."
exec uv run gradio_app.py --enable_flashvdm --mc_algo=mc --host "${HOST}" --port "${PORT}"