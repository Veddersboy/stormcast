"""v2: the dense decision dataset (every 5 minutes, 10:00 to 15:30 ET; 67 per day) used to train the v2 models.

On the 12 pre-registered slots it is identical to data/dataset.parquet (checked here), so v2 forecasts can be scored
on exactly the same rows as the pre-registered study.
Usage: uv run python scripts/09_dense_dataset.py
Writes data/dataset_dense.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.data import DATA, load_bars, load_vix  # noqa: E402
from stormcast.features import ALL_FEATURES, DENSE, NOISE, build  # noqa: E402

if __name__ == "__main__":
    dense = build(load_bars(), load_vix(), decisions=DENSE)
    if (DATA / "dataset.parquet").exists():
        std = pd.read_parquet(DATA / "dataset.parquet")
        m = std.merge(dense, on=["date", "k"], suffixes=("", "_d"))
        cols = [c for c in ALL_FEATURES if c not in NOISE] + ["y_lrv", "y_lrange", "y_dir"]
        bad = {c: int((~np.isclose(m[c], m[f"{c}_d"], equal_nan=True)).sum()) for c in cols}
        assert len(m) == len(std) and not any(bad.values()), bad
    dense.to_parquet(DATA / "dataset_dense.parquet")
    print(f"{len(dense):,} rows ({int(dense.valid.sum()):,} valid)")
