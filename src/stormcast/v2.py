"""Stormcast v2: shared data access for the dense (every-5-minute) models.

v2 is post hoc relative to the pre-registered study (docs/plan.md): its models were designed after the final results
were known. It trains on dense, overlapping decision rows but is scored on the original 12 non-overlapping slots, and
any number it reports for 2016-2019 is labelled post hoc. The pre-registered results in results/{dev,final} are not
touched.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from stormcast.data import DATA, HOME
from stormcast.evaluation import EMBARGO_DAYS
from stormcast.features import DECISIONS

DENSE_PATH = DATA / "dataset_dense.parquet"
V2_YEARS = tuple(int(y) for y in os.environ["STORMCAST_YEARS"].split(",")) if os.environ.get("STORMCAST_YEARS") \
    else tuple(range(2012, 2020))
RESULTS = HOME / "results" / "v2"
MODELS = HOME / "models" / "v2"
WORKSPACE = HOME / "workspace.json"  # settings a workspace was built with (written by `stormcast demo`)


def _stride() -> int:
    """Minutes between training rows: STORMCAST_STRIDE, else what this workspace was built with, else 5."""
    if os.environ.get("STORMCAST_STRIDE"):
        return int(os.environ["STORMCAST_STRIDE"])
    if WORKSPACE.exists():
        import json
        return int(json.loads(WORKSPACE.read_text()).get("stride", 5))
    return 5


STRIDE = _stride()


def load_dense() -> pd.DataFrame:
    df = pd.read_parquet(DENSE_PATH)
    df = df[df.valid & (df.k % STRIDE == 0)].reset_index(drop=True)
    df["standard"] = df.k.isin(DECISIONS)
    return df


@lru_cache(maxsize=1)
def bars_and_channels():
    from stormcast.data import load_bars
    from stormcast.deep import bar_channels
    bars = load_bars()
    return bars, bar_channels(bars)


def split(df: pd.DataFrame, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Train on every row dated before `year` minus a one-trading-day embargo; test on every row of `year`."""
    days = np.sort(df.date.unique())
    before = days[days < pd.Timestamp(f"{year}-01-01")]
    last_train = before[-1 - EMBARGO_DAYS]
    tr = np.flatnonzero(df.date.to_numpy() <= last_train)
    te = np.flatnonzero(df.date.dt.year.to_numpy() == year)
    return tr, te


def inner_split(train: pd.DataFrame, frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered split of a training window for tuning: first `frac` of days to fit, the rest to validate, with a
    one-day embargo between them (horizons of dense rows overlap within a day, never across days)."""
    days = np.sort(train.date.unique())
    cut = days[int(frac * len(days))]
    fit = np.flatnonzero(train.date.to_numpy() < cut)
    val = np.flatnonzero(train.date.to_numpy() > cut)
    return fit, val
