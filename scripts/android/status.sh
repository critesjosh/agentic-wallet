#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
ADB="${SDK_ROOT}/platform-tools/adb"
HOST_PORT="${AGENTIC_WALLET_LLAMA_HOST_PORT:-18080}"

"${ADB}" devices -l
pid="$("${ADB}" shell pidof llama-server 2>/dev/null | tr -d '\r' || true)"
echo "llama-server pid: ${pid:-stopped}"
"${ADB}" forward --list
curl -fsS --max-time 5 "http://127.0.0.1:${HOST_PORT}/health" || true
echo
