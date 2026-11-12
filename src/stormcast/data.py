"""Data access. Only AlphaBench's exploration window (2010-06-07 to 2019-05-21) ever enters this project.

Bar timestamps are UTC bar-open times. Session dates follow CME/AlphaBench: a bar at or after 18:00 ET belongs to the next
day's session, and no session is dated on a weekend.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Workspace: where data, v2 results and models live. Defaults to the repository; the synthetic demo uses ROOT/sandbox so
# it can never overwrite real data or results.
HOME = Path(os.environ.get("STORMCAST_HOME", ROOT))
DATA = HOME / "data"
SOURCES = ROOT / "docs" / "sources"
ALPHABENCH = Path(os.environ.get("ALPHABENCH_DIR", Path.home() / "alphabench"))
ET = "America/New_York"
LAST_DAY = pd.Timestamp("2019-05-21")  # AlphaBench exploration end; later data is reserved there
BARS = DATA / "es_1m.parquet"
SYNTHETIC = (DATA / "SYNTHETIC").exists()


class ReservedDataError(RuntimeError):
    """Raised when a bar from AlphaBench's confirmation window or holdout would be used."""


def session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    et = index.tz_convert(ET)
    day = et.tz_localize(None).normalize() + pd.to_timedelta((et.hour >= 18).astype(np.int64), unit="D")
    dow = day.dayofweek
    return day + pd.to_timedelta(np.where(dow == 5, 2, np.where(dow == 6, 1, 0)), unit="D")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_bars(src: Path = ALPHABENCH / "data" / "clean" / "ES_dev.parquet", dst: Path = BARS) -> str:
    """Copy the exploration window of AlphaBench's cleaned ES series. Never reads AlphaBench's holdout folder."""
    if "holdout" in Path(src).parts:
        raise ReservedDataError("never read AlphaBench's holdout")
    bars = pd.read_parquet(src)
    bars = bars[session_dates(bars.index) <= LAST_DAY]
    dst.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(dst)
    return sha256(dst)


def load_bars(path: Path = BARS) -> pd.DataFrame:
    """Exported bars with roll-adjusted prices, session date, ET minute of day; degraded days dropped."""
    bars = pd.read_parquet(path)
    sd = session_dates(bars.index)
    if sd.max() > LAST_DAY:
        raise ReservedDataError(f"bar dated {sd.max().date()} is after {LAST_DAY.date()}")
    for c in ("open", "high", "low", "close"):
        bars[c] = bars[c] - bars["adj_offset"]
    et = bars.index.tz_convert(ET)
    bars["session"] = sd
    bars["minute"] = et.hour * 60 + et.minute
    degraded = set(pd.to_datetime(pd.read_csv(SOURCES / "databento_degraded_days.csv").date))
    bad = bars["degraded"].to_numpy() | sd.isin(degraded)
    bars = bars[~bad]
    return bars[["open", "high", "low", "close", "volume", "session", "minute"]]


def load_vix(path: Path | None = None) -> pd.Series:
    """Cboe VIX daily closes up to LAST_DAY, indexed by date (data/vix.csv if present, e.g. the synthetic market)."""
    if path is None:
        path = DATA / "vix.csv" if (DATA / "vix.csv").exists() else ALPHABENCH / "data" / "raw" / "vix" / "VIX_History.csv"
    v = pd.read_csv(path)
    s = pd.Series(v["CLOSE"].to_numpy(float), index=pd.to_datetime(v["DATE"], format="%m/%d/%Y"))
    return s[s.index <= LAST_DAY]


def prior_close(series: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """For each day D, the last value dated strictly before D (point-in-time)."""
    s = series.sort_index()
    pos = s.index.searchsorted(days, side="left") - 1
    out = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=days)


def fomc_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.read_csv(SOURCES / "fomc_statements.csv").date))
