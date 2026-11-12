"""H2/H3: can the same features predict the SIZE of the next 30 minutes' moves (log realized variance)?

Usage: uv run python scripts/02_volatility.py [--final]
Writes results/{dev,final}/volatility_{predictions.parquet,metrics.csv,dm.csv,importance.csv,summary.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.evaluation import diebold_mariano, load_dataset, out_dir, qlike, r2, walk_forward, years  # noqa: E402
from stormcast.features import ALL_FEATURES, NOISE  # noqa: E402
from stormcast.ml import vol_models  # noqa: E402

MODELS = ["persistence", "tod_profile", "har_linear", "ridge_all", "gbm", "mlp"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    yrs = years(a.final, "02_volatility")
    df = load_dataset()
    resid_var, imps = {}, []

    def fit_predict(train, test):
        out = {}
        y = train.y_lrv.to_numpy()
        yr = int(test.date.dt.year.iloc[0])
        for name, m in vol_models().items():
            m.fit(train, y)
            out[name] = m.predict(test)
            resid_var[(yr, name)] = float(np.var(y - m.predict(train)))
            if name == "gbm":
                pi = permutation_importance(m.model, test[ALL_FEATURES], test.y_lrv, scoring="r2", n_repeats=5,
                                            random_state=0)
                imps.append(pd.DataFrame({"year": yr, "feature": ALL_FEATURES, "importance": pi.importances_mean}))
        print(f"  year {yr}: trained on {len(train)} rows, tested on {len(test)}", flush=True)
        return out

    pred = walk_forward(df, yrs, fit_predict)
    rows, dms = [], []
    for scope, g in [("pooled", pred)] + [(str(y), g) for y, g in pred.groupby("test_year")]:
        for m in MODELS:
            rv = np.mean([resid_var[(y, m)] for y in g.test_year.unique()])
            rows.append({"scope": scope, "model": m, "n": len(g), "r2": r2(g.y_lrv, g[m]),
                         "mse": float(np.mean((g.y_lrv - g[m]) ** 2)), "qlike": qlike(g.y_lrv, g[m], rv),
                         "corr": float(np.corrcoef(g.y_lrv, g[m])[0, 1])})
            if m != "har_linear":
                d, t, p = diebold_mariano(g.date, (g.y_lrv - g.har_linear) ** 2, (g.y_lrv - g[m]) ** 2)
                dms.append({"scope": scope, "model": m, "mean_loss_gain_vs_har": d, "t": t, "p": p})
    M, DM = pd.DataFrame(rows), pd.DataFrame(dms)
    imp = pd.concat(imps)
    imp["rank"] = imp.groupby("year").importance.rank(ascending=False)
    rank = imp.groupby("feature")[["importance", "rank"]].mean().sort_values("importance", ascending=False)
    # Magnitude is predictable even though sign is not: AUC of the GBM forecast for "above-median absolute move".
    big = (pred.move_pts.abs() > pred.groupby("test_year").move_pts.transform(lambda x: x.abs().median())).astype(int)
    od = out_dir(a.final)
    pred.to_parquet(od / "volatility_predictions.parquet")
    M.to_csv(od / "volatility_metrics.csv", index=False)
    DM.to_csv(od / "volatility_dm.csv", index=False)
    rank.to_csv(od / "volatility_importance.csv")
    pooled = M[M.scope == "pooled"].set_index("model")
    gbm_dm = DM[(DM.model == "gbm")].set_index("scope")
    yearly_sign = (gbm_dm.drop(index="pooled").mean_loss_gain_vs_har > 0).sum()
    n_feat = len(ALL_FEATURES)
    summary = {"years": list(yrs), "best_model": pooled.r2.idxmax(), "best_r2": float(pooled.r2.max()),
               "H2_best_r2_ge_0.40": bool(pooled.r2.max() >= 0.40),
               "H3_gbm_vs_har_p": float(gbm_dm.loc["pooled", "p"]),
               "H3_gbm_better_years": int(yearly_sign),
               "H3_ml_helps": bool(gbm_dm.loc["pooled", "p"] < 0.05 and gbm_dm.loc["pooled", "mean_loss_gain_vs_har"] > 0
                                   and yearly_sign >= 3),
               "noise_ranks": {f: float(rank.loc[f, "rank"]) for f in NOISE}, "n_features": n_feat,
               "noise_in_bottom_quarter": bool(all(rank.loc[f, "rank"] > 0.75 * n_feat for f in NOISE)),
               "auc_big_move_from_gbm_vol": float(roc_auc_score(big, pred.gbm))}
    (od / "volatility_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 200):
        print(M[M.scope == "pooled"].round(4).to_string(index=False))
        print(M.pivot(index="model", columns="scope", values="r2").round(3))
        print(DM.round(4).to_string(index=False))
        print(rank.round(4))
    print(json.dumps(summary, indent=2))
