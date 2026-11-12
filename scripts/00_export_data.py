"""Export AlphaBench's exploration-window ES bars (2010-06-07 to 2019-05-21) and build the decision-point dataset.

Usage: uv run python scripts/00_export_data.py
Writes data/es_1m.parquet, data/dataset.parquet, data/sequences.npy (all gitignored) and results/data_manifest.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.data import BARS, DATA, ROOT, export_bars, load_bars, load_vix, sha256  # noqa: E402
from stormcast.features import build, sequences  # noqa: E402

if __name__ == "__main__":
    h_bars = BARS.exists() and sha256(BARS) or export_bars()
    bars = load_bars()
    df = build(bars, load_vix())
    df.to_parquet(DATA / "dataset.parquet")
    v = df[df.valid].reset_index(drop=True)
    seq = sequences(bars, v.decision_utc)
    np.save(DATA / "sequences.npy", seq)
    m = {"bars_sha256": h_bars, "bars_rows": len(bars), "first_session": str(bars.session.min().date()),
         "last_session": str(bars.session.max().date()), "decision_points": len(df), "valid": int(df.valid.sum()),
         "days": int(df.date.nunique()), "dataset_sha256": sha256(DATA / "dataset.parquet")}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))
