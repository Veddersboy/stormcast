"""Point-in-time tests on the real exported bars (skipped if the data have not been exported).

Truncation test: every feature at decision (day, k) must be identical when the dataset is rebuilt from bars that end
just before the decision time. A planted look-ahead feature must fail the same test.
"""
import numpy as np
import pandas as pd
import pytest

from stormcast.data import BARS, LAST_DAY, ReservedDataError, load_bars, load_vix, prior_close
from stormcast.features import ALL_FEATURES, DECISIONS, DENSE, NOISE, build

pytestmark = pytest.mark.skipif(not BARS.exists(), reason="run scripts/00_export_data.py first")
CHECKED = [f for f in ALL_FEATURES if f not in NOISE]


@pytest.fixture(scope="module")
def sample():
    bars = load_bars()
    days = np.unique(bars.session)
    rng = np.random.default_rng(1)
    picks = sorted(rng.choice(np.arange(40, len(days) - 1), size=6, replace=False))
    return bars, days, picks, load_vix()


def _window(bars, days, i):
    return bars[(bars.session >= days[i - 35]) & (bars.session <= days[i])]


def _rebuild_row(bars, vix, day, k, extra=None):
    df = build(bars, vix, decisions=(k,))
    if extra is not None:
        df = extra(df, bars)
    return df[(df.date == day) & (df.k == k)].iloc[0]


@pytest.mark.parametrize("k", [DECISIONS[0], DECISIONS[5], DECISIONS[-1], DENSE[1], DENSE[37], DENSE[-2]])
def test_truncation(sample, k):
    bars, days, picks, vix = sample
    for i in picks:
        w = _window(bars, days, i)
        day = pd.Timestamp(days[i])
        full = _rebuild_row(w, vix, day, k)
        cut = full.decision_utc
        trunc = _rebuild_row(w[w.index < cut], vix, day, k)
        for f in CHECKED:
            assert np.isclose(full[f], trunc[f], equal_nan=True), (f, day, k, full[f], trunc[f])


def test_truncation_catches_planted_leak(sample):
    """A 'centered' 30-minute return (15 minutes into the future) must fail the truncation test."""
    bars, days, picks, vix = sample

    def centered(df, b):
        close = b.close
        t = pd.DatetimeIndex(df.decision_utc)
        pos = close.index.searchsorted(t + pd.Timedelta(minutes=14), side="right") - 1
        pos0 = close.index.searchsorted(t - pd.Timedelta(minutes=16), side="right") - 1
        df["leak"] = np.log(close.to_numpy()[pos]) - np.log(close.to_numpy()[pos0])
        return df

    caught = []
    for i in picks:
        k = DECISIONS[3]
        w = _window(bars, days, i)
        day = pd.Timestamp(days[i])
        full = _rebuild_row(w, vix, day, k, centered)
        trunc = _rebuild_row(w[w.index < full.decision_utc], vix, day, k, centered)
        caught.append(not np.isclose(full["leak"], trunc["leak"]))
    assert sum(caught) >= len(picks) // 2  # equal prices 30 minutes apart can hide it on a single day


def test_no_reserved_data():
    bars = load_bars()
    assert bars.session.max() <= LAST_DAY
    assert load_vix().index.max() <= LAST_DAY


def test_loader_refuses_reserved_dates(tmp_path):
    idx = pd.DatetimeIndex(["2019-05-29 14:00"], tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1, "instrument_id": 1,
                       "adj_offset": 0.0, "degraded": False}, index=idx)
    df.to_parquet(tmp_path / "x.parquet")
    with pytest.raises(ReservedDataError):
        load_bars(tmp_path / "x.parquet")


def test_vix_prior_close_only():
    s = pd.Series([10.0, 20.0, 30.0], index=pd.to_datetime(["2015-01-05", "2015-01-06", "2015-01-07"]))
    out = prior_close(s, pd.DatetimeIndex(["2015-01-06", "2015-01-08"]))
    assert list(out) == [10.0, 30.0]
