#!/usr/bin/env bash
# Build a shareable zip: the committed code (git archive: no history, no licensed data, no real-data models) plus the
# synthetic sandbox/ (synthetic market + models trained on it), so the recipient's `uv run stormcast demo` opens at once.
#   scripts/package_demo.sh [output.zip] [extra files to place at the top level of the zip...]
#   default output: ~/Downloads/stormcast-demo.zip
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-$HOME/Downloads/stormcast-demo.zip}"
shift || true
extras=("$@")
[ -f sandbox/data/SYNTHETIC ] || { echo "sandbox/ is missing or not synthetic; run: uv run stormcast demo" >&2; exit 1; }
[ -f sandbox/results/v2/predictions.parquet ] || { echo "sandbox/ has no trained models yet" >&2; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "commit your changes first" >&2; exit 1; fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir "$tmp/stormcast"
git archive HEAD | tar -x -C "$tmp/stormcast"
rsync -a --exclude '*.log' sandbox/ "$tmp/stormcast/sandbox/"
for f in ${extras[@]+"${extras[@]}"}; do cp "$f" "$tmp/stormcast/"; done
rm -f "$out"
(cd "$tmp" && zip -qr -9 "$out" stormcast -x '*.DS_Store' '*__pycache__*')
echo "wrote $out ($(du -h "$out" | cut -f1))"
