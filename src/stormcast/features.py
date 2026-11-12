"""Decision-point dataset: point-in-time features and next-30-minute targets.

Cash session 09:30-16:00 ET is laid out as a (day x 390 minute) grid. Decision k (minutes after 09:30) uses grid
minutes 0..k-1 plus earlier days and the overnight session; its horizon is minutes k..k+29.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stormcast.data import ET, fomc_days, prior_close

OPEN = 9 * 60 + 30
N_RTH = 390
H = 30
DECISIONS = tuple(range(30, 361, 30))  # 10:00, 10:30, ..., 15:30 (12)
DENSE = tuple(range(30, 361, 5))  # every 5 minutes, 10:00 to 15:30 (67); v2 training rows, horizons overlap
MIN_BARS = 25
EPS = 1e-10  # floor inside logs of realized variance
SEQ_LEN = 120

VOL_FEATURES = ["lrv_30", "lrv_today", "lrv_on", "lrv_d1", "lrv_w", "lrv_m", "lrv_slot", "lrange_today",
                "lvix", "dvix", "abs_gap", "lvol_rel"]
DIR_FEATURES = ["ret_5", "ret_30", "ret_60", "ret_today", "gap", "ret_d1", "dist_vwap", "pos_range"]
CAL_FEATURES = ["tod", "dow", "fomc"]
NOISE = ["noise1", "noise2"]
ALL_FEATURES = VOL_FEATURES + DIR_FEATURES + CAL_FEATURES + NOISE
HAR = ["lrv_30", "lrv_today", "lrv_on", "lrv_d1", "lrv_w", "lrv_m", "lrv_slot"]


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 1."""
    idx = np.where(np.isnan(a), 0, np.arange(a.shape[1]))
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(a.shape[0])[:, None], idx]


def grid(bars: pd.DataFrame) -> dict:
    rth = bars[(bars.minute >= OPEN) & (bars.minute < OPEN + N_RTH)]
    days = pd.DatetimeIndex(np.unique(rth.session))
    di = days.get_indexer(rth.session)
    mi = (rth.minute - OPEN).to_numpy()
    g = {}
    for c in ("open", "high", "low", "close", "volume"):
        a = np.full((len(days), N_RTH), np.nan)
        a[di, mi] = rth[c].to_numpy(float)
        g[c] = a
    present = ~np.isnan(g["close"])
    C = _ffill(g["close"])
    first_open = np.array([row[~np.isnan(row)][0] for row in g["open"]])
    C = np.where(np.isnan(C), first_open[:, None], C)  # minutes before the first bar of the day
    O = np.where(np.isnan(g["open"]), C, g["open"])
    Hh = np.where(np.isnan(g["high"]), C, g["high"])
    L = np.where(np.isnan(g["low"]), C, g["low"])
    V = np.nan_to_num(g["volume"])
    lc = np.log(C)
    R = np.empty_like(lc)
    R[:, 0] = lc[:, 0] - np.log(O[:, 0])
    R[:, 1:] = np.diff(lc, axis=1)
    return {"days": days, "O": O, "H": Hh, "L": L, "C": C, "V": V, "R": R, "present": present}


def overnight(bars: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Realized variance and bar count of the Globex session before each day's 09:30 open (18:00 ET onward)."""
    on = bars[(bars.minute >= 18 * 60) | (bars.minute < OPEN)]
    lc = np.log(on.close.to_numpy())
    r = np.diff(lc, prepend=np.nan)
    dt = np.diff(on.index.asi8, prepend=on.index.asi8[0]) / 6e10
    same = np.r_[False, on.session.to_numpy()[1:] == on.session.to_numpy()[:-1]]
    r = np.where(same & (dt <= 30), r, 0.0)
    out = pd.DataFrame({"session": on.session.to_numpy(), "r2": r ** 2}).groupby("session").r2.agg(["sum", "size"])
    return out.reindex(days)


def build(bars: pd.DataFrame, vix: pd.Series, seed: int = 0, decisions=DECISIONS) -> pd.DataFrame:
    """One row per (day, decision minute k). The default 12 non-overlapping decisions are the pre-registered design."""
    g = grid(bars)
    days, C, Hh, L, V, R, present = g["days"], g["C"], g["H"], g["L"], g["V"], g["R"], g["present"]
    nd = len(days)
    R2 = R ** 2
    cumR2 = np.concatenate([np.zeros((nd, 1)), np.cumsum(R2, axis=1)], axis=1)
    cumV = np.concatenate([np.zeros((nd, 1)), np.cumsum(V, axis=1)], axis=1)
    cumP = np.concatenate([np.zeros((nd, 1)), np.cumsum(present, axis=1)], axis=1)
    tp = (Hh + L + C) / 3
    cumTPV = np.concatenate([np.zeros((nd, 1)), np.cumsum(tp * V, axis=1)], axis=1)
    run_hi = np.maximum.accumulate(Hh, axis=1)
    run_lo = np.minimum.accumulate(L, axis=1)

    day_rv = cumR2[:, -1]
    day_close = C[:, -1]
    day_open = g["O"][:, 0]
    prev = lambda a: np.r_[np.nan, a[:-1]]  # noqa: E731
    roll_prev = lambda a, n: pd.Series(a).shift(1).rolling(n, min_periods=max(1, n // 2)).mean().to_numpy()  # noqa: E731
    prev_close = prev(day_close)
    on = overnight(bars, days)
    vixp = prior_close(vix, days).to_numpy()
    vix_dates = vix.sort_index().index  # vixp2: the close before the prior close
    pos = vix_dates.searchsorted(days, side="left") - 1
    vixp2 = np.where(pos >= 1, vix.sort_index().to_numpy()[np.clip(pos - 1, 0, None)], np.nan)
    fomc = days.isin(fomc_days()).astype(float)
    ks = np.array(decisions)
    slot_rv = np.stack([cumR2[:, k + H] - cumR2[:, k] for k in ks], axis=1)
    vol_win = np.stack([cumV[:, k] - cumV[:, k - 30] for k in ks], axis=1)
    slot_prof = pd.DataFrame(slot_rv).shift(1).rolling(20, min_periods=10).mean().to_numpy()
    vol_prof = pd.DataFrame(vol_win).shift(1).rolling(20, min_periods=10).mean().to_numpy()

    rows = []
    for j, k in enumerate(ks):
        P = C[:, k - 1]
        lp = np.log(P)
        lo_k, hi_k = run_lo[:, k - 1], run_hi[:, k - 1]
        f = {
            "date": days, "k": k,
            "lrv_30": np.log(cumR2[:, k] - cumR2[:, k - 30] + EPS),
            "lrv_today": np.log(cumR2[:, k] / k * H + EPS),
            "lrv_on": np.log(on["sum"].to_numpy() + EPS),
            "lrv_d1": np.log(prev(day_rv) / 13 + EPS),
            "lrv_w": np.log(roll_prev(day_rv, 5) / 13 + EPS),
            "lrv_m": np.log(roll_prev(day_rv, 22) / 13 + EPS),
            "lrv_slot": np.log(slot_prof[:, j] + EPS),
            "lrange_today": np.log((hi_k - lo_k) / P + EPS),
            "lvix": np.log(vixp), "dvix": np.log(vixp) - np.log(vixp2),
            "abs_gap": np.abs(np.log(day_open) - np.log(prev_close)),
            "lvol_rel": np.log((vol_win[:, j] + 1) / (vol_prof[:, j] + 1)),
            "ret_5": lp - np.log(C[:, k - 6]) if k >= 6 else np.nan,
            "ret_30": lp - np.log(C[:, k - 31]) if k > 30 else lp - np.log(day_open),
            "ret_60": lp - np.log(C[:, k - 61]) if k > 60 else lp - np.log(day_open),
            "ret_today": lp - np.log(day_open),
            "gap": np.log(day_open) - np.log(prev_close),
            "ret_d1": prev(np.log(day_close) - np.log(day_open)),
            "dist_vwap": lp - np.log(cumTPV[:, k] / np.maximum(cumV[:, k], 1)),
            "pos_range": (P - lo_k) / np.maximum(hi_k - lo_k, 1e-9),
            "tod": float(k), "dow": days.dayofweek.to_numpy(float), "fomc": fomc,
            # targets
            "y_lrv": np.log(slot_rv[:, j] + EPS),
            "y_ret": np.log(C[:, k + H - 1]) - lp,
            "move_pts": C[:, k + H - 1] - P, "price": P,
            "y_lrange": np.log(Hh[:, k:k + H].max(axis=1) - L[:, k:k + H].min(axis=1) + 0.25),
            "range_pts": Hh[:, k:k + H].max(axis=1) - L[:, k:k + H].min(axis=1),
            "n_back": cumP[:, k] - cumP[:, k - 30], "n_fwd": cumP[:, k + H] - cumP[:, k],
        }
        rows.append(pd.DataFrame(f))
    df = pd.concat(rows, ignore_index=True).sort_values(["date", "k"], ignore_index=True)
    df["y_dir"] = (df.y_ret > 0).astype(int)
    rng = np.random.default_rng(seed)
    df["noise1"], df["noise2"] = rng.standard_normal(len(df)), rng.standard_normal(len(df))
    df["valid"] = (df.n_back >= MIN_BARS) & (df.n_fwd >= MIN_BARS) & df[ALL_FEATURES].notna().all(axis=1)
    local = pd.Series(pd.DatetimeIndex(df.date).tz_localize(ET), index=df.index)
    df["decision_utc"] = (local + pd.to_timedelta(OPEN + df.k, unit="min")).dt.tz_convert("UTC")
    return df


def sequences(bars: pd.DataFrame, decision_utc: pd.Series, length: int = SEQ_LEN) -> np.ndarray:
    """(n, 2, length) float32: the last `length` 1-minute log returns before each decision, divided by their own
    standard deviation, and log volume relative to the window's mean. Returns across gaps > 30 minutes are zeroed."""
    t = bars.index.asi8
    lc = np.log(bars.close.to_numpy())
    r = np.diff(lc, prepend=lc[0])
    gap = np.diff(t, prepend=t[0]) / 6e10 > 30
    r[gap] = 0.0
    lv = np.log1p(bars.volume.to_numpy(float))
    end = np.searchsorted(t, pd.DatetimeIndex(decision_utc).asi8, side="left")  # first bar at/after decision
    idx = end[:, None] - length + np.arange(length)[None, :]
    idx = np.clip(idx, 0, len(t) - 1)
    rw = r[idx]
    rw = rw / (rw.std(axis=1, keepdims=True) + 1e-12)
    vw = lv[idx]
    vw = vw - vw.mean(axis=1, keepdims=True)
    return np.stack([rw, vw], axis=1).astype(np.float32)
