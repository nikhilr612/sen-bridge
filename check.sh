#!/bin/sh
# Run every project check. Exits non-zero on the first failure.
#   formatting is strict (ruff format --check), lint is ruff with every rule,
#   types are ty with every rule at error, then the tests.
set -eu
cd "$(dirname "$0")"
BIN=".venv/bin"
[ -x "$BIN/ruff" ] || BIN="$(dirname "$(command -v ruff)")"
"$BIN/ruff" format --check .
"$BIN/ruff" check .
"$BIN/ty" check
"$BIN/python" -m pytest -q
