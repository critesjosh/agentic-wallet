#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cleanup() {
  if [[ "${KEEP_ANDROID_RUNNING:-0}" != "1" ]]; then
    "${ROOT}/scripts/android/stop_emulator.sh" || true
  fi
}
trap cleanup EXIT INT TERM

"${ROOT}/scripts/android/start_emulator.sh"
"${ROOT}/scripts/android/start_llama_server.sh"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_android_benchmark.py" "$@"
