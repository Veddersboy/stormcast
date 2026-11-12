#!/usr/bin/env bash
# Reproduce everything.
#   scripts/run_all.sh            pre-registered study, development years
#   scripts/run_all.sh --final    pre-registered study, final years (once, from the frozen tag)
#   scripts/run_all.sh --v2       Stormcast v2 models (post hoc): StormNet, tuned GBM, stacking, conformal
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f data/dataset.parquet ] || uv run python scripts/00_export_data.py
if [ "${1:-}" = "--v2" ]; then
  [ -f data/dataset_dense.parquet ] || uv run python scripts/09_dense_dataset.py
  for s in 10_v2_stormnet 11_v2_gbm 13_v2_regimes 12_v2_combine; do
    echo "== $s"; uv run python -W ignore scripts/$s.py
  done
  exit 0
fi
for s in 01_direction 02_volatility 03_intervals 04_regimes 06_risk_sizing; do
  echo "== $s $*"; uv run python -W ignore scripts/$s.py "$@"
done
[ "${1:-}" = "--final" ] || uv run python -W ignore scripts/05_leak_lab.py
uv run python -W ignore scripts/07_figures.py "$@"
