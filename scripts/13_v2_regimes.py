"""v2: walk-forward market regimes (3-state Gaussian HMM, filtered probabilities known before each day opens) for the
v2 test years, in the active workspace. Same procedure as the pre-registered scripts/04_regimes.py.
Usage: uv run python scripts/13_v2_regimes.py
Writes results/v2/regimes_daily.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.data import load_bars  # noqa: E402
from stormcast.regimes import NAMES, Regimes, daily_observations  # noqa: E402
from stormcast.v2 import RESULTS, V2_YEARS  # noqa: E402

if __name__ == "__main__":
    obs = daily_observations(load_bars()).dropna()
    X = obs.to_numpy()
    parts = []
    for y in V2_YEARS:
        train = obs.index < pd.Timestamp(f"{y}-01-01")
        p = Regimes().fit(X[train]).filtered(X)
        prior = pd.DataFrame(p, index=obs.index, columns=[f"p_{n}" for n in NAMES]).shift(1)
        test = obs.index.year == y
        part = obs[test].join(prior[test])
        part["state"] = part[[f"p_{n}" for n in NAMES]].to_numpy().argmax(axis=1)
        part["test_year"] = y
        parts.append(part)
        print(f"  year {y}: fit on {int(train.sum())} days", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_parquet(RESULTS / "regimes_daily.parquet")
