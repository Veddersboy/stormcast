"""Forecaster: one call that answers "what is the market's weather for the next 30 minutes?" at any decision time.

Time-machine rule: a forecast for a moment in year Y uses only the models trained in the walk-forward run for Y (data
before Y), so the engine never answers with a model that has seen the moment it is forecasting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.linear_model import LinearRegression

from stormcast import stack
from stormcast.conformal import conformal_quantile
from stormcast.data import load_vix
from stormcast.deep import Ensemble, window_ends
from stormcast.features import ALL_FEATURES, HAR, OPEN, build
from stormcast.v2 import MODELS, RESULTS, bars_and_channels, load_dense, split

FEATURE_LABELS = {
    "lrv_30": "volatility, last 30 min", "lrv_today": "volatility, today so far", "lrv_on": "overnight volatility",
    "lrv_d1": "yesterday's volatility", "lrv_w": "last week's volatility", "lrv_m": "last month's volatility",
    "lrv_slot": "usual volatility at this time", "lrange_today": "today's range so far", "lvix": "VIX (prior close)",
    "dvix": "VIX change", "abs_gap": "size of opening gap", "lvol_rel": "volume vs usual", "ret_5": "return, 5 min",
    "ret_30": "return, 30 min", "ret_60": "return, 60 min", "ret_today": "return since open", "gap": "opening gap",
    "ret_d1": "yesterday's return", "dist_vwap": "distance from VWAP", "pos_range": "position in today's range",
    "tod": "time of day", "dow": "day of week", "fomc": "FOMC day", "noise1": "noise (sanity check)",
    "noise2": "noise (sanity check)"}


def annualized_vol(lrv: np.ndarray | float) -> np.ndarray | float:
    """Log 30-minute realized variance -> annualized volatility in % (13 half-hours a day, 252 days): comparable
    to the VIX."""
    return np.sqrt(np.exp(lrv) * 13 * 252) * 100


@dataclass
class Forecast:
    time: pd.Timestamp
    price: float
    vol_lrv: float
    vol_annualized: float
    move_1sd_pts: float
    range_median_pts: float
    range_hi_pts: float
    range_lo_pts: float
    p_up: float
    epistemic_sd: float
    aleatoric_sd: float
    members: dict = field(default_factory=dict)
    drivers: list = field(default_factory=list)
    model_year: int = 0

    def report(self) -> str:
        lines = [
            f"Stormcast forecast for {self.time:%Y-%m-%d %H:%M} ET (models trained on data before {self.model_year})",
            f"  ES price                {self.price:,.2f}",
            f"  next-30-min volatility  {self.vol_annualized:5.1f}% annualized   (1 sd move ~ {self.move_1sd_pts:.2f} pts)",
            f"  90% high-low range      {self.range_lo_pts:.2f} to {self.range_hi_pts:.2f} pts (median {self.range_median_pts:.2f})",
            f"  90% price envelope      {self.price - self.range_hi_pts:,.2f} to {self.price + self.range_hi_pts:,.2f}",
            f"  P(up in 30 min)         {self.p_up:.3f}   (direction is close to a coin flip; do not trade it)",
            f"  model uncertainty       aleatoric sd {self.aleatoric_sd:.2f}, epistemic sd {self.epistemic_sd:.3f} (log RV)",
            "  members                 " + ", ".join(f"{k} {annualized_vol(v):.1f}%" for k, v in self.members.items()),
            "  top drivers (SHAP, gradient boosting, log-variance units):"]
        lines += [f"    {'+' if v >= 0 else '-'}{abs(v):.3f}  {FEATURE_LABELS.get(f, f)}" for f, v in self.drivers[:6]]
        return "\n".join(lines)


class Forecaster:
    def __init__(self):
        self.bars, self.chan = bars_and_channels()
        self.vix = load_vix()
        self.weights = json.loads((RESULTS / "stack_weights.json").read_text())
        self._cache = {}

    def _models(self, year: int):
        if year not in self._cache:
            d = MODELS / str(year)
            if not d.exists():
                raise FileNotFoundError(f"no walk-forward models for {year}; run scripts/10_v2_stormnet.py and 11")
            ens = Ensemble.from_state(torch.load(d / "stormnet.pt", weights_only=False))
            self._cache[year] = (ens, joblib.load(d / "gbm.joblib"), np.load(d / "stormnet_calib_scores.npy"))
        return self._cache[year]

    def features_at(self, when: pd.Timestamp) -> pd.DataFrame:
        """Feature row for a decision time (ET, 10:00 to 15:30, any minute), built only from bars before it."""
        when = pd.Timestamp(when)
        day = when.normalize()
        k = when.hour * 60 + when.minute - OPEN
        if not 30 <= k <= 360:
            raise ValueError("decision time must be between 10:00 and 15:30 ET")
        sessions = np.unique(self.bars.session)
        i = np.searchsorted(sessions, day.to_datetime64())
        if i >= len(sessions) or sessions[i] != day.to_datetime64():
            raise ValueError(f"{day.date()} is not a trading day in the data")
        w = self.bars[(self.bars.session >= sessions[max(0, i - 40)]) & (self.bars.session <= day)]
        row = build(w, self.vix, decisions=(k,))
        return row[row.date == day].reset_index(drop=True)

    def forecast(self, when) -> Forecast:
        X = self.features_at(when)
        year = int(X.date.dt.year.iloc[0])
        ens, gbm, cal = self._models(year)
        ends = window_ends(self.bars, X.decision_utc)
        sn = ens.predict(self.chan, X, ends).iloc[0]
        g = float(gbm.predict(X[ALL_FEATURES])[0])
        har = self._har(year)
        h = float(har.predict(X[har.feature_names_in_])[0])
        members = {"har": h, "gbm_tuned": g, "stormnet": float(sn.stormnet)}
        lrv = float(stack.apply(pd.DataFrame([members]), self.weights[str(year)])[0])
        q = conformal_quantile(cal, 0.10)
        to_pts = lambda x: float(np.exp(x) - 0.25)  # noqa: E731
        vals = shap.TreeExplainer(gbm).shap_values(X[ALL_FEATURES].to_numpy(), check_additivity=False)[0]
        drivers = sorted(zip(ALL_FEATURES, vals.tolist()), key=lambda t: -abs(t[1]))
        price = float(X.price.iloc[0])
        return Forecast(time=pd.Timestamp(when), price=price, vol_lrv=lrv, vol_annualized=float(annualized_vol(lrv)),
                        move_1sd_pts=price * float(np.exp(lrv / 2)), range_median_pts=to_pts(sn.stormnet_q50),
                        range_lo_pts=max(0.0, to_pts(sn.stormnet_q05 - q)), range_hi_pts=to_pts(sn.stormnet_q95 + q),
                        p_up=float(sn.stormnet_p_up), epistemic_sd=float(sn.stormnet_epistemic_sd),
                        aleatoric_sd=float(sn.stormnet_aleatoric_sd), members=members, drivers=drivers,
                        model_year=year)

    def _har(self, year: int):
        """HAR baseline for the year: cheap, so refit from the dense rows before the year instead of stored."""
        key = ("har", year)
        if key not in self._cache:
            df = load_dense()
            tr, _ = split(df, year)
            self._cache[key] = LinearRegression().fit(df.iloc[tr][HAR], df.y_lrv.iloc[tr])
        return self._cache[key]
