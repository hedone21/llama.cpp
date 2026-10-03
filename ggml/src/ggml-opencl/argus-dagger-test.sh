#!/usr/bin/env bash
# Host tests of the ARGUS comparison-arm logic (argus-engine tickets/031 criterion 1).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
g++ -std=c++17 -O1 -Wall -Wextra -Werror -o "$out/argus-dagger-test" \
    "$here/argus-dagger.cpp" "$here/argus-dagger-test.cpp"
"$out/argus-dagger-test"
