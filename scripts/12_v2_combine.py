"""v2 (post hoc): combine the walk-forward forecasts into the stacked model, conformal intervals and metrics.

Needs scripts/10_v2_stormnet.py and scripts/11_v2_gbm.py output. Steps, per test year Y:
  - HAR baseline refit on dense rows before Y
  - the pre-registered gradient boosting forecast (results/{dev,final}) joined on the standard slots, for reference
  - ensemble = intercept + non-negative weights on (HAR, tuned GBM, StormNet), weights fit only on earlier years'
    out-of-sample forecasts
  - 90% range intervals: StormNet's 5%/95% quantiles + adaptive conformal inference with delayed feedback (each
    30-minute outcome is used only once it is known), reset each year
Metrics are computed on the standard (non-overlapping) slots, split into development years (2012-2015) and post hoc
years (2016-2019).
Usage: uv run python scripts/12_v2_combine.py
Writes results/v2/{predictions.parquet, metrics.csv, dm.csv, stack_weights.json, summary.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast import stack  # noqa: E402
from stormcast.conformal import aci_delayed  # noqa: E402
from stormcast.data import HOME  # noqa: E402
from stormcast.evaluation import block_bootstrap_ci, diebold_mariano, qlike, r2  # noqa: E402
from stormcast.features import HAR  # noqa: E402
from stormcast.v2 import RESULTS, load_dense, split  # noqa: E402

ALPHA, GAMMA = 0.10, 0.002
BASE = ["har", "gbm_tuned", "stormnet"]
DEV, POST = (2012, 2013, 2014, 2015), (2016, 2017, 2018, 2019)

if __name__ == "__main__":
    df = load_dense()
    sn = pd.read_parquet(RESULTS / "stormnet_predictions.parquet").set_index("row")
    gb = pd.read_parquet(RESULTS / "gbm_predictions.parquet").set_index("row")
    years = sorted(set(df.date.dt.year.iloc[sn.index]) & set(df.date.dt.year.iloc[gb.index]))
    rows = np.intersect1d(sn.index, gb.index)
    pred = df.iloc[rows].copy()
    pred["test_year"] = pred.date.dt.year
    pred = pred.join(sn).join(gb)
    # HAR, refit per year on dense rows
    pred["har"] = np.nan
    for y in years:
        tr, te = split(df, y)
        m = LinearRegression().fit(df.iloc[tr][HAR], df.y_lrv.iloc[tr])
        idx = pred.index[pred.test_year == y]
        pred.loc[idx, "har"] = m.predict(pred.loc[idx, HAR])
    # pre-registered GBM on standard slots
    pre_files = [HOME / "results" / s / "volatility_predictions.parquet" for s in ("dev", "final")]
    pred = pred.reset_index(names="row")
    if all(f.exists() for f in pre_files):  # real-data workspace: compare with the pre-registered model
        pre = pd.concat([pd.read_parquet(f, columns=["date", "k", "gbm"]) for f in pre_files]).rename(
            columns={"gbm": "gbm_prereg"})
        pred = pred.merge(pre, on=["date", "k"], how="left")
    else:
        pred["gbm_prereg"] = np.nan
    pred = pred.sort_values("decision_utc", ignore_index=True)
    # walk-forward stacking
    weights = {}
    pred["ensemble"] = np.nan
    for y in years:
        oos = pred[(pred.test_year < y) & pred.standard]
        blend = stack.fit_weights(oos, BASE)
        weights[y] = blend
        idx = pred.test_year == y
        pred.loc[idx, "ensemble"] = stack.apply(pred[idx], blend)
    # conformal intervals with delayed feedback
    from stormcast.v2 import MODELS
    for col in ("aci_lo", "aci_hi", "aci_alpha"):
        pred[col] = np.nan
    for y in years:
        idx = np.flatnonzero(pred.test_year.to_numpy() == y)
        g = pred.iloc[idx]
        t = pd.DatetimeIndex(g.decision_utc).asi8
        cal = np.load(MODELS / str(y) / "stormnet_calib_scores.npy")
        L, U, A = aci_delayed(t, t + 30 * 60 * 10**9, g.y_lrange.to_numpy(), g.stormnet_q05.to_numpy(),
                              g.stormnet_q95.to_numpy(), cal, ALPHA, GAMMA)
        pred.iloc[idx, pred.columns.get_loc("aci_lo")] = L
        pred.iloc[idx, pred.columns.get_loc("aci_hi")] = U
        pred.iloc[idx, pred.columns.get_loc("aci_alpha")] = A
    to_pts = lambda x: np.exp(x) - 0.25  # noqa: E731
    pred["range_lo_pts"], pred["range_hi_pts"] = to_pts(pred.aci_lo), to_pts(pred.aci_hi)
    pred["range_med_pts"] = to_pts(pred.stormnet_q50)
    pred["covered"] = ((pred.y_lrange >= pred.aci_lo) & (pred.y_lrange <= pred.aci_hi)).astype(float)
    pred["raw_covered"] = ((pred.y_lrange >= pred.stormnet_q05) & (pred.y_lrange <= pred.stormnet_q95)).astype(float)
    pred["stormnet_sd"] = np.sqrt(pred.stormnet_aleatoric_sd ** 2 + pred.stormnet_epistemic_sd ** 2)
    pred.to_parquet(RESULTS / "predictions.parquet")

    # ---- metrics on standard slots
    S = pred[pred.standard].copy()
    vol_models = ["har", "gbm_prereg", "gbm_tuned", "stormnet", "ensemble"]
    scopes = [("all", S), ("dev 2012-15", S[S.test_year.isin(DEV)]), ("post hoc 2016-19", S[S.test_year.isin(POST)])]
    scopes += [(str(y), g) for y, g in S.groupby("test_year")]
    M, DM = [], []
    for name, g in scopes:
        if not len(g):
            continue
        for m in vol_models:
            ok = g[m].notna()
            if not ok.any():
                continue
            gg = g[ok]
            M.append({"scope": name, "model": m, "n": int(ok.sum()), "r2": r2(gg.y_lrv, gg[m]),
                      "qlike": qlike(gg.y_lrv, gg[m], float(np.var(gg.y_lrv - gg[m])))})
            if m != "har":
                d, t, p = diebold_mariano(gg.date, (gg.y_lrv - gg.har) ** 2, (gg.y_lrv - gg[m]) ** 2)
                DM.append({"scope": name, "model": m, "mean_loss_gain_vs_har": d, "t": t, "p": p})
        z = (g.y_lrv - g.stormnet).abs() / g.stormnet_sd
        M.append({"scope": name, "model": "intervals", "n": len(g), "aci_coverage": g.covered.mean(),
                  "raw_quantile_coverage": g.raw_covered.mean(),
                  "median_width_pts": float((g.range_hi_pts - g.range_lo_pts).median()),
                  "vol_sd_coverage_90": float((z <= 1.645).mean()),
                  "dir_auc": roc_auc_score(g.y_dir, g.stormnet_p_up)})
    M, DM = pd.DataFrame(M), pd.DataFrame(DM)
    M.to_csv(RESULTS / "metrics.csv", index=False)
    DM.to_csv(RESULTS / "dm.csv", index=False)
    (RESULTS / "stack_weights.json").write_text(json.dumps({str(k): v for k, v in weights.items()}, indent=1) + "\n")
    allv = M[(M.scope == "all") & M.r2.notna()].set_index("model")
    post = S[S.test_year.isin(POST)]
    summary = {
        "label": "v2 is post hoc: designed after the pre-registered final results; 2016-2019 numbers are not "
                 "pre-registered",
        "years": years, "r2_all": allv.r2.round(4).to_dict(),
        "best_model_all": allv.r2.idxmax(),
        "aci_coverage_all": float(S.covered.mean()),
        "aci_worst_rolling_250": float(S.covered.rolling(250).mean().min()),
        "stormnet_dir_auc_all": float(roc_auc_score(S.y_dir, S.stormnet_p_up)),
        "stormnet_dir_auc_ci": block_bootstrap_ci(S.date, S.y_dir, S.stormnet_p_up, reps=300),
    }
    if len(post) and post.gbm_prereg.notna().all():
        summary["post_hoc_r2_ensemble_vs_prereg_gbm"] = [r2(post.y_lrv, post.ensemble), r2(post.y_lrv, post.gbm_prereg)]
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(M.round(4).to_string(index=False))
        print(DM.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))
