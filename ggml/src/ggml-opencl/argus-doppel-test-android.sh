#!/usr/bin/env bash
# Device run of the Doppeladler-arm logic tests, NEON path (argus-engine tickets/032 criterion 1).
#   bash argus-doppel-test-android.sh [serial]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ndk=${ANDROID_NDK:-/opt/android-ndk}
cxx=$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror -static-libstdc++ -o "$out/argus-doppel-test" \
    "$here/argus-doppel.cpp" "$here/argus-doppel-test.cpp"
serial=${1:-${ANDROID_SERIAL:-}}
adb=(adb)
[ -n "$serial" ] && adb=(adb -s "$serial")
"${adb[@]}" push "$out/argus-doppel-test" /data/local/tmp/argus-doppel-test >/dev/null
"${adb[@]}" shell /data/local/tmp/argus-doppel-test
