"""The synthetic market must be a drop-in replacement for the real export."""
import numpy as np
import pandas as pd
import pytest

from stormcast.data import load_bars, load_vix
from stormcast.features import build
from stormcast.synthetic import GLOBEX, generate, sessions


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    generate(d, start="2015-01-05", end="2015-04-30", verbose=False)
    return d


def test_schema_ticks_and_ohlc(market):
    raw = pd.read_parquet(market / "es_1m.parquet")
    assert list(raw.columns) == ["open", "high", "low", "close", "volume", "instrument_id", "adj_offset", "degraded"]
    assert str(raw.index.tz) == "UTC" and raw.index.is_monotonic_increasing
    for c in ("open", "high", "low", "close"):
        assert np.allclose(raw[c] % 0.25, 0)
    assert (raw.high >= raw[["open", "close"]].max(axis=1)).all() and (raw.low <= raw[["open", "close"]].min(axis=1)).all()
    assert (raw.volume > 0).all() and (raw.close > 0).all()


def test_sessions_and_features_build(market):
    bars = load_bars(market / "es_1m.parquet")
    days = sessions("2015-01-05", "2015-04-30")
    counts = bars.groupby("session").size()
    assert set(counts.index) <= set(days) and counts.median() == GLOBEX
    assert (bars[bars.minute == 9 * 60 + 30].groupby("session").size() == 1).all()  # one 09:30 bar per session
    vix = load_vix(market / "vix.csv")
    assert vix.between(8, 90).all() and len(vix) == len(days)
    df = build(bars, vix)
    assert df.valid.mean() > 0.8


def test_direction_is_a_coin_flip(market):
    bars = load_bars(market / "es_1m.parquet")
    r = np.diff(np.log(bars.close.to_numpy()))
    assert abs(np.corrcoef(r[:-1], r[1:])[0, 1]) < 0.1  # no exploitable autocorrelation at one minute
