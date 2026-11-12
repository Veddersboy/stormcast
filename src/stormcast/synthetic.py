"""A synthetic ES-like futures market, so Stormcast can be run end to end without the licensed data.

The simulator reproduces the stylized facts the real study relies on, and nothing more:
  - volatility clustering across days (persistent latent log-volatility) and within the day (30-minute AR component)
  - a hidden 3-state regime (calm / normal / storm) with sticky transitions, and rare shock days with an afternoon crash
  - the intraday U-shape (busy open and close, quiet lunch), a quieter overnight session, 08:30 ET news spikes,
    and volatility bursts at 14:00 ET on real FOMC statement days
  - fat-tailed one-minute returns, rare jumps, a leverage effect (falls raise later volatility), 0.25-point ticks
  - volume that rises with volatility, and a VIX-like index derived from the latent volatility
Direction is a martingale by construction: no model should beat AUC 0.5 on it, which is the point.

Output has the same schema as the real export (data/es_1m.parquet: UTC bar-open index; open, high, low, close, volume,
instrument_id, adj_offset, degraded) plus data/vix.csv (DATE mm/dd/yyyy, CLOSE), and a SYNTHETIC marker file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from stormcast.data import ET, fomc_days

START, END = "2010-06-07", "2019-05-21"
GLOBEX = 23 * 60  # 18:00 ET (previous evening) to 16:59 ET
RTH0 = 15 * 60 + 30  # index of 09:30 within the Globex day (18:00 + 15h30)
TICK = 0.25


def sessions(start: str = START, end: str = END) -> pd.DatetimeIndex:
    days = pd.bdate_range(start, end)
    hol = USFederalHolidayCalendar().holidays(start, end)
    return days[~days.isin(hol)]


def _profile() -> np.ndarray:
    """Relative per-minute volatility across the Globex day (1 = an average cash-session minute)."""
    m = np.arange(GLOBEX)
    prof = np.full(GLOBEX, 0.32)  # overnight
    prof[(m >= 9 * 60) & (m < 11 * 60)] = 0.45  # Europe opens (03:00-05:00 ET)
    news = 14 * 60 + 30  # 08:30 ET economic releases
    prof[news:news + 15] = 0.45 + 1.3 * np.exp(-np.arange(15) / 4)
    k = m[RTH0:RTH0 + 390] - RTH0
    prof[RTH0:RTH0 + 390] = 0.72 + 1.5 * np.exp(-k / 22) + 0.55 * np.exp(-(389 - k) / 28)
    prof[RTH0 + 390:] = 0.55  # 16:00-16:59
    return prof / prof[RTH0:RTH0 + 390].mean()


def generate(out_dir: Path, seed: int = 7, start: str = START, end: str = END, verbose: bool = True) -> Path:
    rng = np.random.default_rng(seed)
    days = sessions(start, end)
    nd = len(days)
    # ---- daily latent volatility: sticky regimes + persistent AR(1) + rare shocks
    P = np.array([[0.985, 0.014, 0.001], [0.018, 0.970, 0.012], [0.004, 0.056, 0.940]])
    level = np.log(np.array([0.0055, 0.0090, 0.0175]))  # daily return sd by regime
    s = np.zeros(nd, int)
    s[0] = 1
    u = rng.random(nd)
    for d in range(1, nd):
        s[d] = np.searchsorted(np.cumsum(P[s[d - 1]]), u[d])
    h = np.empty(nd)
    h[0] = level[1]
    eps = rng.standard_normal(nd)
    shock = rng.random(nd) < 0.004
    for d in range(1, nd):
        h[d] = 0.85 * h[d - 1] + 0.15 * level[s[d]] + 0.24 * eps[d] + (0.9 if shock[d] else 0.0)
    daily_sd = np.exp(h)
    # ---- minute grid (days x Globex minutes)
    prof = _profile()
    sig = daily_sd[:, None] / np.sqrt(390) * prof[None, :]
    # within-day 30-minute AR(1) volatility component with a leverage effect, built block by block
    nb = GLOBEX // 30
    g = np.zeros((nd, nb))
    z = rng.standard_normal((nd, nb))
    r = np.zeros((nd, GLOBEX))
    noise = rng.standard_t(7, size=(nd, GLOBEX)) / np.sqrt(7 / 5)
    jumps = (rng.random((nd, GLOBEX)) < 2e-5) * rng.standard_normal((nd, GLOBEX)) * 12
    for b in range(nb):
        if b:
            prev = r[:, (b - 1) * 30:b * 30].sum(1) / (sig[:, (b - 1) * 30:b * 30] ** 2).sum(1).clip(1e-18) ** 0.5
            g[:, b] = 0.80 * g[:, b - 1] + 0.26 * z[:, b] - 0.06 * np.clip(prev, -4, 4)
        sl = slice(b * 30, (b + 1) * 30)
        r[:, sl] = sig[:, sl] * np.exp(g[:, b:b + 1]) * (noise[:, sl] + jumps[:, sl])
    # FOMC: a burst at 14:00 ET (minute 20*60 of the Globex day) decaying over 45 minutes
    fomc = days.isin(fomc_days())
    t = np.arange(GLOBEX) - 20 * 60
    burst = np.where(t >= 0, 1 + 2.2 * np.exp(-t.clip(0) / 15), 1.0)
    r[fomc] *= burst[None, :]
    # shock days: an afternoon crash (13:30-15:30 ET) of several daily sds with heavy volatility
    for d in np.flatnonzero(shock):
        a, b2 = RTH0 + 240, RTH0 + 360
        r[d, a:b2] = r[d, a:b2] * 2.5 - daily_sd[d] * 3.5 / (b2 - a)
    # drift so the index trends up like ES in 2010-2019, and the overnight gap is just the overnight returns
    r += np.log(2850 / 1080) / (nd * GLOBEX)
    logp = np.log(1080) + np.cumsum(r.ravel())
    logp -= np.linspace(0, logp[-1] - np.log(2850), logp.size)  # pin the path's end near ES's May 2019 level
    close = np.round(np.exp(logp) / TICK) * TICK
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.standard_normal(close.size)) * np.exp(logp) * np.abs(r.ravel()).clip(1e-5) * 0.8
    high = np.ceil((np.maximum(open_, close) + spread) / TICK) * TICK
    low = np.floor((np.minimum(open_, close) - spread) / TICK) * TICK
    base = np.where((np.arange(GLOBEX) >= RTH0) & (np.arange(GLOBEX) < RTH0 + 390), 1500.0, 140.0)
    rel = (np.abs(r) / sig.clip(1e-12)).clip(0, 8)
    vol = (base[None, :] * (sig / sig.mean()) ** 0.6 * (0.6 + 0.4 * rel) * rng.lognormal(0, 0.35, r.shape)).ravel()
    # timestamps: 18:00 ET on the previous weekday evening (Sunday for Monday) through 16:59 ET
    prev_evening = days - pd.Timedelta(days=1)  # Sunday evening for Monday
    local = (prev_evening.values[:, None] + np.timedelta64(18, "h")
             + np.arange(GLOBEX).astype("timedelta64[m]")[None, :]).ravel()
    idx = pd.DatetimeIndex(local).tz_localize(ET, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.maximum(1, vol).round().astype(np.int64), "instrument_id": 1,
                         "adj_offset": 0.0, "degraded": False}, index=idx)
    bars = bars[~bars.index.duplicated()]
    # VIX-like index: annualized forward-looking vol with a premium and noise, close of each day
    raw = 100 * np.sqrt(252) * np.exp(0.7 * h + 0.3 * np.r_[h[1:], h[-1]]) * 1.18 * rng.lognormal(0, 0.06, nd)
    vix = 7 + 0.65 * raw  # the real VIX never trades near zero: 2010-2019 spanned roughly 9 to 48
    out_dir.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out_dir / "es_1m.parquet")
    pd.DataFrame({"DATE": days.strftime("%m/%d/%Y"), "CLOSE": vix.round(2)}).to_csv(out_dir / "vix.csv", index=False)
    (out_dir / "SYNTHETIC").write_text("Synthetic market generated by stormcast.synthetic; not real ES data.\n")
    pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "regime": np.array(["calm", "normal", "storm"])[s],
                  "shock": shock}).to_csv(out_dir / "synthetic_truth.csv", index=False)
    if verbose:
        print(f"synthetic market: {len(bars):,} bars, {nd} sessions, {int(shock.sum())} shock days, "
              f"regime shares {np.bincount(s, minlength=3) / nd}")
    return out_dir / "es_1m.parquet"
