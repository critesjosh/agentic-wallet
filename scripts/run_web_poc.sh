#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# The browser is thin; exact target-model inference runs in the host's Ollama.
export AGENTIC_WALLET_INFERENCE_PROVIDER="${AGENTIC_WALLET_INFERENCE_PROVIDER:-ollama}"
export AGENTIC_WALLET_MODEL_ID="${AGENTIC_WALLET_MODEL_ID:-gemma4:e2b}"
export AGENTIC_WALLET_DEBUG_TRANSCRIPTS="${AGENTIC_WALLET_DEBUG_TRANSCRIPTS:-1}"

exec "${ROOT}/.venv/bin/uvicorn" agentic_wallet.web.app:app \
  --host "${AGENTIC_WALLET_WEB_HOST:-127.0.0.1}" \
  --port "${AGENTIC_WALLET_WEB_PORT:-8000}"
