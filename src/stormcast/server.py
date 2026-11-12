"""Stormcast web server: a JSON API over the walk-forward results and models, plus the single-page frontend in web/.

Run: uv run stormcast serve   (http://localhost:8000)

Every forecast the API returns for a day in year Y comes from models trained only on data before Y.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stormcast.data import ROOT, SYNTHETIC
from stormcast.engine import FEATURE_LABELS, annualized_vol
from stormcast.features import ALL_FEATURES, OPEN
from stormcast.v2 import MODELS, RESULTS

WEB = ROOT / "web"
DEV, POST = (2012, 2013, 2014, 2015), (2016, 2017, 2018, 2019)
PRESETS = [
    {"date": "2018-02-05", "name": "Volmageddon", "note": "VIX doubles; the Dow's biggest point drop to date"},
    {"date": "2015-08-24", "name": "China devaluation crash", "note": "limit-down open, flash-crash ETFs"},
    {"date": "2016-06-24", "name": "Brexit vote", "note": "the morning after the referendum"},
    {"date": "2018-12-24", "name": "Christmas Eve sell-off", "note": "worst Christmas Eve on record"},
    {"date": "2016-11-09", "name": "Election night reversal", "note": "limit-down overnight, rally by the open"},
    {"date": "2017-07-14", "name": "A calm summer day", "note": "2017: the calmest year on record"},
]
MODEL_ORDER = ["har", "gbm_prereg", "gbm_tuned", "stormnet", "ensemble"]
MODEL_NAMES = {"har": "HAR econometrics", "gbm_prereg": "Gradient boosting (pre-registered)",
               "gbm_tuned": "Gradient boosting + Optuna", "stormnet": "StormNet deep ensemble",
               "ensemble": "Stacked ensemble"}


def _f(x, nd=4):
    """JSON-safe float."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _arr(a, nd=4):
    return [_f(v, nd) for v in np.asarray(a, float)]


class Store:
    """Everything the API serves, loaded once."""

    def __init__(self):
        p = pd.read_parquet(RESULTS / "predictions.parquet")
        p["time"] = pd.to_datetime(p.date) + pd.to_timedelta(OPEN + p.k, unit="min")
        sn_rows = pd.read_parquet(RESULTS / "stormnet_predictions.parquet", columns=["row"]).row.to_numpy()
        p["attn_idx"] = p.row.map(pd.Series(np.arange(len(sn_rows)), index=sn_rows))
        self.P = p
        self.S = p[p.standard]
        self.attn = np.load(RESULTS / "stormnet_attention.npy")
        if (RESULTS / "regimes_daily.parquet").exists():
            reg = pd.read_parquet(RESULTS / "regimes_daily.parquet")
        else:
            reg = pd.concat([pd.read_parquet(ROOT / "results" / s / "regimes_daily.parquet") for s in ("dev", "final")])
        self.regimes = reg[~reg.index.duplicated()]
        self.metrics = pd.read_csv(RESULTS / "metrics.csv")
        self.dm = pd.read_csv(RESULTS / "dm.csv")
        self.summary = json.loads((RESULTS / "summary.json").read_text())
        self._bars = None
        self._lock = threading.Lock()

    @property
    def bars(self):
        with self._lock:
            if self._bars is None:
                from stormcast.data import load_bars
                b = load_bars()
                b["et"] = b.index.tz_convert("America/New_York").tz_localize(None)
                self._bars = b
        return self._bars


@lru_cache(maxsize=1)
def store() -> Store:
    return Store()


app = FastAPI(title="Stormcast", docs_url="/api/docs")


# ------------------------------------------------------------------ summary
@app.get("/api/summary")
def summary():
    s = store()
    M, S = s.metrics, s.S
    vm = M.dropna(subset=["r2"])
    iv = M[(M.model == "intervals") & (M.scope == "all")].iloc[0]
    r2 = vm[vm.scope == "all"].set_index("model").r2
    board = []
    for m in MODEL_ORDER:
        row = {"model": m, "name": MODEL_NAMES[m]}
        for scope, key in (("all", "r2_all"), ("dev 2012-15", "r2_dev"), ("post hoc 2016-19", "r2_post")):
            v = vm[(vm.scope == scope) & (vm.model == m)]
            row[key] = _f(v.r2.iloc[0]) if len(v) else None
        q = vm[(vm.scope == "all") & (vm.model == m)]
        row["qlike"] = _f(q.qlike.iloc[0]) if len(q) else None
        d = s.dm[(s.dm.scope == "all") & (s.dm.model == m)]
        row["dm_t"] = _f(d.t.iloc[0], 2) if len(d) else None
        board.append(row)
    years = sorted(int(y) for y in S.test_year.unique())
    by_year = {m: [_f(vm[(vm.scope == str(y)) & (vm.model == m)].r2.iloc[0])
                   if len(vm[(vm.scope == str(y)) & (vm.model == m)]) else None for y in years] for m in MODEL_ORDER}
    weights = json.loads((RESULTS / "stack_weights.json").read_text())
    tuning = json.loads((RESULTS / "gbm_tuning.json").read_text())
    training = json.loads((RESULTS / "stormnet_training.json").read_text())
    # rolling coverage (standard slots, every 5th point to keep the payload small)
    St = S.sort_values("time")
    roll = St[["covered", "raw_covered"]].rolling(250).mean()
    keep = np.arange(249, len(St), 5)
    # uncertainty calibration: deciles of predicted sd
    bins = pd.qcut(S.stormnet_sd, 10, labels=False)
    cal = pd.DataFrame({"pred": S.stormnet_sd, "e2": (S.y_lrv - S.stormnet) ** 2}).groupby(bins).agg(
        pred=("pred", "mean"), rmse=("e2", lambda e: float(np.sqrt(e.mean()))))
    vt = pd.qcut(S.lvix, 3, labels=["low VIX", "mid VIX", "high VIX"])
    cov_vix = S.groupby(vt, observed=True)[["raw_covered", "covered"]].mean()
    cov_year = S.groupby("test_year")[["raw_covered", "covered"]].mean()
    z = ((S.y_lrv - S.stormnet).abs() / S.stormnet_sd <= 1.645).mean()
    # risk: per-slot P&L of a long position under each sizing scheme, summed by day (client applies limits)
    risk = {}
    for name, col in (("constant", None), ("ensemble", "ensemble"), ("stormnet", "stormnet"), ("har", "har"),
                      ("oracle", "y_lrv")):
        u = np.ones(len(St)) if col is None else 1 / (np.exp(St[col].to_numpy() / 2) * St.price.to_numpy())
        u = u / u.mean()
        risk[name] = _arr(pd.Series(u * St.move_pts.to_numpy()).groupby(St.date.to_numpy()).sum().to_numpy(), 3)
    days = pd.DatetimeIndex(np.unique(St.date.to_numpy()))
    # direction trading P&L, daily
    side = np.where(St.stormnet_p_up > 0.5, 1, -1)
    g = pd.Series(side * St.move_pts.to_numpy()).groupby(St.date.to_numpy()).sum().cumsum()
    n_trades = pd.Series(1, index=St.date.to_numpy()).groupby(level=0).sum().cumsum()
    leak = pd.read_csv(ROOT / "results" / "leak_lab" / "leak_lab.csv")
    available = set(days.strftime("%Y-%m-%d"))
    presets = [dict(p, available=p["date"] in available) for p in PRESETS] if not SYNTHETIC else synthetic_presets(St)
    return {
        "kpi": {"r2": _f(r2.get("ensemble")), "r2_har": _f(r2.get("har")), "dir_auc": _f(iv.dir_auc),
                "dir_auc_ci": [_f(v) for v in s.summary.get("stormnet_dir_auc_ci", [None, None])],
                "coverage": _f(iv.aci_coverage), "n": int(len(S)), "days": int(S.date.nunique()),
                "seeds": len(next(iter(training.values()))["members_best_val"]), "years": years,
                "bars": 3105722, "self_cal": _f(z)},
        "leaderboard": board, "years": years, "r2_by_year": by_year, "model_names": MODEL_NAMES,
        "weights": {y: v["weights"] for y, v in weights.items()},
        "tuning": {y: {"best": _arr(np.minimum.accumulate(v["trial_mse"]) / v["trial_mse"][0], 5),
                       "params": v["best_params"]} for y, v in tuning.items()},
        "training": {y: v["members_history"] for y, v in training.items()},
        "coverage": {"t": St.time.iloc[keep].dt.strftime("%Y-%m-%d %H:%M").tolist(),
                     "aci": _arr(roll.covered.iloc[keep], 3), "raw": _arr(roll.raw_covered.iloc[keep], 3)},
        "calibration": {"pred": _arr(cal.pred), "rmse": _arr(cal.rmse)},
        "cov_vix": {"labels": list(cov_vix.index.astype(str)), "raw": _arr(cov_vix.raw_covered, 3),
                    "aci": _arr(cov_vix.covered, 3)},
        "cov_year": {"labels": [str(y) for y in cov_year.index], "raw": _arr(cov_year.raw_covered, 3),
                     "aci": _arr(cov_year.covered, 3)},
        "risk": {"days": days.strftime("%Y-%m-%d").tolist(), **risk},
        "direction": {"days": [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(g.index)], "gross": _arr(g, 2),
                      "net": _arr(g.to_numpy() - 0.70 * n_trades.to_numpy(), 2)},
        "leak": leak.to_dict(orient="records"),
        "presets": presets, "synthetic": SYNTHETIC,
        "dates": sorted(available),
        "features": [{"key": f, "label": FEATURE_LABELS[f]} for f in ALL_FEATURES],
    }


def synthetic_presets(St: pd.DataFrame) -> list[dict]:
    """On the synthetic market the famous dates mean nothing; pick days by what actually happened in it."""
    daily = St.groupby("date").agg(vol=("y_lrv", "mean"), move=("move_pts", "sum"), fomc=("fomc", "max"))
    picks = [("Stormiest day", daily.vol.idxmax(), "highest realized volatility in the test years"),
             ("Biggest sell-off", daily.move.idxmin(), "largest fall from 10:00 to 16:00"),
             ("Biggest rally", daily.move.idxmax(), "largest rise from 10:00 to 16:00"),
             ("Busiest FOMC day", daily[daily.fomc > 0].vol.idxmax() if (daily.fomc > 0).any() else daily.vol.idxmax(),
              "Fed announcement at 14:00"),
             ("Calmest day", daily.vol.idxmin(), "lowest realized volatility")]
    return [{"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "name": n, "note": note, "available": True}
            for n, d, note in picks]


# ------------------------------------------------------------------ one day
@app.get("/api/day/{day}")
def day(day: str):
    s = store()
    d = pd.Timestamp(day)
    D = s.P[s.P.date == d].sort_values("k")
    if D.empty:
        raise HTTPException(404, f"no walk-forward forecasts for {day}")
    b = s.bars
    sess = b[b.session == d]
    # bars from the first slot's attention window (240 bars before 10:00, often overnight) through the 16:00 close
    i0 = max(0, int(b.index.searchsorted(pd.Timestamp(D.decision_utc.iloc[0]))) - 240)
    rth = sess[(sess.minute >= OPEN) & (sess.minute < OPEN + 390)]
    i1 = int(b.index.get_loc(rth.index[-1])) + 1
    W = b.iloc[i0:i1]
    year = int(D.test_year.iloc[0])
    Y = s.S[s.S.test_year == year]
    typical_sd = float(np.median(np.exp(Y.ensemble / 2) * Y.price))
    slots = []
    for _, r in D.iterrows():
        sh = {f: _f(r[f"shap_{f}"]) for f in ALL_FEATURES}
        attn = s.attn[int(r.attn_idx)].astype(float) if pd.notna(r.attn_idx) else np.zeros(60)
        sd_pts = r.price * np.exp(r.ensemble / 2)
        slots.append({
            "k": int(r.k), "t": r.time.strftime("%H:%M"), "standard": bool(r.standard), "price": _f(r.price, 2),
            "vol": _f(annualized_vol(r.ensemble), 2), "vol_usual": _f(annualized_vol(r.lrv_slot), 2),
            "vol_actual": _f(annualized_vol(r.y_lrv), 2),
            "members": {m: _f(annualized_vol(r[m]), 2) for m in ("har", "gbm_tuned", "stormnet")},
            "sd_pts": _f(sd_pts, 2), "range_lo": _f(r.range_lo_pts, 2), "range_med": _f(r.range_med_pts, 2),
            "range_hi": _f(r.range_hi_pts, 2), "range_actual": _f(r.range_pts, 2), "covered": bool(r.covered),
            "p_up": _f(r.stormnet_p_up, 3), "went_up": bool(r.y_dir), "move_pts": _f(r.move_pts, 2),
            "epistemic": _f(r.stormnet_epistemic_sd), "aleatoric": _f(r.stormnet_aleatoric_sd),
            "size": _f(min(3.0, typical_sd / sd_pts), 2), "vix": _f(np.exp(r.lvix), 2), "fomc": bool(r.fomc),
            "shap": sh, "shap_base": _f(r.shap_base), "attn": _arr(attn, 5),
            "decision_utc": pd.Timestamp(r.decision_utc).isoformat()})
    reg = s.regimes.loc[d] if d in s.regimes.index else None
    return {
        "date": day, "weekday": d.strftime("%A"), "year": year, "post_hoc": year in POST,
        "bars": {"t": W.et.dt.strftime("%Y-%m-%dT%H:%M").tolist(), "c": _arr(W.close, 2), "h": _arr(W.high, 2),
                 "l": _arr(W.low, 2),
                 "rth": ((W.session == d) & (W.minute >= OPEN)).tolist()},
        "slots": slots,
        "regime": None if reg is None else {"calm": _f(reg.p_calm, 3), "normal": _f(reg.p_normal, 3),
                                            "storm": _f(reg.p_storm, 3)},
        "day_stats": {"coverage": _f(D[D.standard].covered.mean(), 3),
                      "corr": _f(np.corrcoef(D.ensemble, D.y_lrv)[0, 1], 3),
                      "open": _f(D.price.iloc[0], 2), "close": _f(sess.close.iloc[-1], 2) if len(sess) else None},
    }


# ------------------------------------------------------------------ live inference
class WhatIf(BaseModel):
    date: str
    k: int
    vix: float | None = None
    vol_mult: float = 1.0
    fomc: bool | None = None
    k_new: int | None = None


@lru_cache(maxsize=8)
def _gbm(year: int):
    import joblib
    return joblib.load(MODELS / str(year) / "gbm.joblib")


@lru_cache(maxsize=8)
def _explainer(year: int):
    import shap
    return shap.TreeExplainer(_gbm(year))


@app.post("/api/whatif")
def whatif(q: WhatIf):
    s = store()
    r = s.P[(s.P.date == pd.Timestamp(q.date)) & (s.P.k == q.k)]
    if r.empty:
        raise HTTPException(404, "no such decision")
    r = r.iloc[0]
    year = int(r.test_year)
    x = r[ALL_FEATURES].astype(float).copy()
    base = x.copy()
    if q.vix is not None:
        x["lvix"] = np.log(q.vix)
    x["lrv_30"] = x["lrv_30"] + 2 * np.log(max(q.vol_mult, 1e-3))
    if q.fomc is not None:
        x["fomc"] = float(q.fomc)
    if q.k_new is not None:
        x["tod"] = float(q.k_new)
    X = pd.DataFrame([base, x])
    pred = _gbm(year).predict(X)
    vals = _explainer(year).shap_values(X.to_numpy()[1:], check_additivity=False)[0]
    top = sorted(zip(ALL_FEATURES, vals.tolist()), key=lambda t: -abs(t[1]))[:6]
    return {"base": _f(annualized_vol(pred[0]), 2), "new": _f(annualized_vol(pred[1]), 2),
            "drivers": [{"key": f, "label": FEATURE_LABELS[f], "value": _f(v)} for f, v in top]}


@lru_cache(maxsize=1)
def _forecaster():
    from stormcast.engine import Forecaster
    return Forecaster()


@app.get("/api/forecast")
def forecast(when: str):
    try:
        f = _forecaster().forecast(pd.Timestamp(when))
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(400, str(e))
    return {"time": f.time.strftime("%Y-%m-%d %H:%M"), "price": _f(f.price, 2), "vol": _f(f.vol_annualized, 2),
            "sd_pts": _f(f.move_1sd_pts, 2), "range_lo": _f(f.range_lo_pts, 2), "range_med": _f(f.range_median_pts, 2),
            "range_hi": _f(f.range_hi_pts, 2), "p_up": _f(f.p_up, 3), "epistemic": _f(f.epistemic_sd),
            "aleatoric": _f(f.aleatoric_sd), "model_year": f.model_year,
            "members": {k: _f(annualized_vol(v), 2) for k, v in f.members.items()},
            "drivers": [{"key": k, "label": FEATURE_LABELS[k], "value": _f(v)} for k, v in f.drivers[:6]]}


LEAK_CACHE = RESULTS / "live_leak.json"


@app.post("/api/leak")
def leak(refresh: bool = False):
    """Same random forest, same rows: shuffled 5-fold CV vs walk-forward (2014 -> 2015) on dense rows whose
    30-minute labels overlap. Cached to disk after the first run."""
    if LEAK_CACHE.exists() and not refresh:
        return json.loads(LEAK_CACHE.read_text()) | {"cached": True}
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold, cross_val_predict
    from stormcast.v2 import load_dense
    d = load_dense()
    d = d[d.date.dt.year.isin([2014, 2015])].sort_values("decision_utc")
    X, y = d[ALL_FEATURES].to_numpy(), d.y_dir.to_numpy()
    rf = RandomForestClassifier(200, min_samples_leaf=1, max_features=0.5, n_jobs=-1, random_state=0)
    shuf = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    tr = d.date.dt.year.to_numpy() == 2014
    wf = rf.fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    out = {"shuffled_auc": _f(roc_auc_score(y, shuf)), "walk_forward_auc": _f(roc_auc_score(y[~tr], wf)),
           "rows": int(len(d))}
    LEAK_CACHE.write_text(json.dumps(out) + "\n")
    return out | {"cached": False}


# ------------------------------------------------------------------ frontend
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
