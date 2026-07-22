#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
ADB="${SDK_ROOT}/platform-tools/adb"
EMULATOR="${SDK_ROOT}/emulator/emulator"
AVD="${AGENTIC_WALLET_AVD:-Agentic_Wallet_API_36_1}"
LOG="${AGENTIC_WALLET_EMULATOR_LOG:-/tmp/agentic-wallet-emulator.log}"

if "${ADB}" devices | awk 'NR > 1 && $2 == "device" { found=1 } END { exit !found }'; then
  echo "An Android device is already online."
  "${ADB}" devices -l
  exit 0
fi

if ! "${EMULATOR}" -list-avds | grep -Fxq "${AVD}"; then
  echo "AVD ${AVD} does not exist." >&2
  exit 1
fi

# Detach from the invoking agent/terminal. Without setsid, an interrupted tool
# session can take the emulator down with it and look like a model crash.
setsid "${EMULATOR}" "@${AVD}" \
  -no-window -no-audio -no-boot-anim \
  -gpu swiftshader_indirect -memory 8192 -cores 4 -partition-size 12288 \
  >"${LOG}" 2>&1 </dev/null &

"${ADB}" wait-for-device
deadline=$((SECONDS + 180))
while [[ "$("${ADB}" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" != "1" ]]; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for Android boot; see ${LOG}" >&2
    exit 1
  fi
  sleep 2
done

"${ADB}" devices -l
echo "Android boot completed. Log: ${LOG}"
