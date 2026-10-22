#!/bin/sh
# Finish the release once the full build has written all 477 parts:
#   freeze thresholds -> package -> FID stats -> card -> acceptance -> determinism
#   -> project checks. Stops at the first failure; progress goes to stdout.
# Usage: build/finish.sh [BUILD_DIR] [RELEASE_DIR]
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BUILD="${1:-work/build}"
REL="${2:-work/release}"
CAND=work/candidates.parquet
# load_dataset writes a full Arrow cache; keep it on this disk and delete it after.
export HF_DATASETS_CACHE="$PWD/work/hf_cache"
step() { printf '\n=== %s  [%s]\n' "$1" "$(date +%H:%M:%S)"; }

step "wait for all 477 parts"
while [ "$(ls "$BUILD/parts" | grep -c 'parquet$')" -lt 477 ]; do
  if ! pgrep -f "build/build_full.py" >/dev/null; then
    echo "build processes are gone with $(ls "$BUILD/parts" | grep -c 'parquet$')/477 parts"; exit 2
  fi
  sleep 30
done

step "build outcome: rows by status"
$PY - <<PYEOF
import glob, pandas as pd
d = pd.concat([pd.read_parquet(f, columns=["status"]) for f in glob.glob("$BUILD/parts/*.parquet")])
print(d.status.value_counts().to_string())
err = d.status.str.startswith("error").sum()
print("error rows:", err, "of", len(d))
raise SystemExit(0 if err / len(d) < 0.005 else 3)
PYEOF

step "commit state, then freeze thresholds"
git add -A && git diff --cached --quiet || git commit -q -m "Pre-release state"
$PY build/freeze.py
git add docs/thresholds.json && git commit -q -m "Freeze thresholds for the release" || true

step "package"
rm -rf "$REL"
$PY build/package.py "$BUILD" "$CAND" "$REL" --target 150000 --test 20000 --val 5000

step "FID statistics"
$PY scripts/make_fid_stats.py "$REL" "$REL/stats"

step "dataset card"
$PY build/card.py "$REL"

step "acceptance tests"
$PY build/accept.py "$REL"
rm -rf "$HF_DATASETS_CACHE"

step "determinism: package again, compare checksums"
rm -rf "${REL}_b"
$PY build/package.py "$BUILD" "$CAND" "${REL}_b" --target 150000 --test 20000 --val 5000 >/dev/null
if diff -q "$REL/MANIFEST.sha256" "${REL}_b/MANIFEST.sha256" >/dev/null; then
  echo "DETERMINISTIC: identical checksums for $(wc -l < "$REL/MANIFEST.sha256") files"
else
  echo "NOT DETERMINISTIC"; diff "$REL/MANIFEST.sha256" "${REL}_b/MANIFEST.sha256" | head; exit 4
fi
rm -rf "${REL}_b"

step "project checks (format, lint, types, tests)"
./check.sh

step "DONE  release at $REL ($(du -sh "$REL" | cut -f1))"
