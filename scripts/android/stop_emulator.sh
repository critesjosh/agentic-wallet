#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
ADB="${SDK_ROOT}/platform-tools/adb"
AVD="${AGENTIC_WALLET_AVD:-Agentic_Wallet_API_36_1}"
stopped_serials=()

while read -r serial state; do
  [[ "${state:-}" == "device" && "${serial}" == emulator-* ]] || continue
  running_avd="$("${ADB}" -s "${serial}" emu avd name 2>/dev/null | head -1 | tr -d '\r')"
  [[ "${running_avd}" == "${AVD}" ]] || continue
  "${ADB}" -s "${serial}" emu kill >/dev/null
  stopped_serials+=("${serial}")
done < <("${ADB}" devices | tail -n +2)

if (( ${#stopped_serials[@]} == 0 )); then
  echo "Dedicated AVD ${AVD} is already stopped."
  exit 0
fi

deadline=$((SECONDS + 30))
for serial in "${stopped_serials[@]}"; do
  while "${ADB}" devices | awk -v target="${serial}" '
    $1 == target && $2 == "device" { found=1 }
    END { exit !found }
  '; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${AVD} to stop." >&2
      exit 1
    fi
    sleep 1
  done
done
echo "Dedicated AVD ${AVD} stopped."
