#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
ADB="${SDK_ROOT}/platform-tools/adb"

pid="$("${ADB}" shell pidof llama-server 2>/dev/null | tr -d '\r' || true)"
if [[ -z "${pid}" ]]; then
  echo "llama-server is already stopped."
  exit 0
fi

"${ADB}" shell kill "${pid}"
deadline=$((SECONDS + 30))
while "${ADB}" shell pidof llama-server >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for llama-server to stop." >&2
    exit 1
  fi
  sleep 1
done
echo "llama-server stopped."
