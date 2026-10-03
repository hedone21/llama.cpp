#!/usr/bin/env bash
# Host tests of the Doppeladler-arm logic (argus-engine tickets/032 criterion 1, scalar path).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -o "$out/argus-doppel-test" \
    "$here/argus-doppel.cpp" "$here/argus-doppel-test.cpp"
"$out/argus-doppel-test"
