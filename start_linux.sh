#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# PromGen Linux launcher (CPU / CUDA / ROCm)
# Usage:
#   ./start_linux.sh                # defaults: CUDA 12.4, Python 3.10
#   ./start_linux.sh --cpu
#   ./start_linux.sh --cuda cu121
#   ./start_linux.sh --cuda cu122
#   ./start_linux.sh --cuda cu124
#   ./start_linux.sh --cuda cu126
#   ./start_linux.sh --rocm rocm6.1
#   ./start_linux.sh --python 3.11 --port 7860 --host 127.0.0.1
# ------------------------------------------------------------

CUDA_FLAVOR="cu126"       # cpu | cu121 | cu122 | cu124 | cu126
ROCM_FLAVOR=""            # e.g. rocm6.1 (leave empty unless using AMD ROCm)
PY_VER="3.10"
PORT="8080"
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)         CUDA_FLAVOR="cpu"; ROCM_FLAVOR=""; shift ;;
    --cuda)        CUDA_FLAVOR="${2:-cu124}"; ROCM_FLAVOR=""; shift 2 ;;
    --rocm)        ROCM_FLAVOR="${2:-rocm6.1}"; CUDA_FLAVOR=""; shift 2 ;;
    --python)      PY_VER="${2:-3.10}"; shift 2 ;;
    --port)        PORT="${2:-8080}"; shift 2 ;;
    --host)        HOST="${2:-0.0.0.0}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--cpu] [--cuda cu121|cu122|cu124|cu126] [--rocm rocm6.1] [--python 3.10] [--port 8080] [--host 0.0.0.0]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "🐧 Setup PromGen for Linux..."
if [[ "${OSTYPE:-}" != "linux-gnu"* && "$(uname -s 2>/dev/null || echo)" != "Linux" ]]; then
  echo "❌ This script is designed for Linux only"
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
  echo "📦 Installing required packages..."

  # Choose the right PyTorch index URL
  INDEX_URL=""
  if [[ -n "${ROCM_FLAVOR}" ]]; then
    INDEX_URL="https://download.pytorch.org/whl/${ROCM_FLAVOR}"
    echo "🔧 Using ROCm wheels: ${ROCM_FLAVOR}"
  else
    if [[ "${CUDA_FLAVOR}" == "cpu" ]]; then
      INDEX_URL="https://download.pytorch.org/whl/cpu"
      echo "🔧 Using CPU wheels"
    else
      INDEX_URL="https://download.pytorch.org/whl/${CUDA_FLAVOR}"
      echo "🔧 Using CUDA wheels: ${CUDA_FLAVOR}"
    fi
  fi

  uv pip install --index-url "${INDEX_URL}" \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

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

# 5) Start the app
echo "🚀 Setup complete - Starting PromGen..."
exec uv run gradio_app.py --enable_flashvdm --mc_algo=mc --host "${HOST}" --port "${PORT}"