#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
ADB="${SDK_ROOT}/platform-tools/adb"
REMOTE_DIR="${AGENTIC_WALLET_ANDROID_DIR:-/data/local/tmp/agentic-wallet}"
HOST_PORT="${AGENTIC_WALLET_LLAMA_HOST_PORT:-18080}"
DEVICE_PORT="${AGENTIC_WALLET_LLAMA_DEVICE_PORT:-8080}"

"${ADB}" get-state >/dev/null
"${ADB}" shell test -r "${REMOTE_DIR}/model.gguf"
"${ADB}" shell test -x "${REMOTE_DIR}/llama-server"
"${ADB}" forward "tcp:${HOST_PORT}" "tcp:${DEVICE_PORT}" >/dev/null

start_server() {
  # This server is a localhost-forwarded development harness, not production
  # app architecture. Production local inference will run in-process.
  # ADB keeps its transport open while the remote process exists, so detach
  # the host-side ADB client as well as redirecting the device process.
  setsid "${ADB}" shell "cd '${REMOTE_DIR}' && \
    export LD_LIBRARY_PATH='${REMOTE_DIR}/lib' && \
    exec ./llama-server -m ./model.gguf --host 127.0.0.1 \
      --port '${DEVICE_PORT}' -c 1024 -t 4 --no-warmup \
      >./llama-server.log 2>&1" \
    >/tmp/agentic-wallet-llama-adb.log 2>&1 </dev/null &
}

started_here=false
if ! "${ADB}" shell pidof llama-server >/dev/null 2>&1; then
  start_server
  started_here=true
fi
deadline=$((SECONDS + 90))
until curl -fsS --max-time 5 "http://127.0.0.1:${HOST_PORT}/health"; do
  # Handle the short window where pidof still reports a process that has been
  # killed but has not finished exiting yet.
  if ! "${ADB}" shell pidof llama-server >/dev/null 2>&1; then
    if [[ "${started_here}" == "true" ]]; then
      "${ADB}" shell tail -80 "${REMOTE_DIR}/llama-server.log" >&2 || true
      echo "llama-server exited during startup" >&2
      exit 1
    fi
    start_server
    started_here=true
  fi
  if (( SECONDS >= deadline )); then
    "${ADB}" shell tail -80 "${REMOTE_DIR}/llama-server.log" >&2 || true
    exit 1
  fi
  sleep 3
done
echo
echo "llama.cpp is healthy at http://127.0.0.1:${HOST_PORT}"
