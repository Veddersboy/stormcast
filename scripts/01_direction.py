"""H1: can the same features predict the DIRECTION of the next 30 minutes?

Usage: uv run python scripts/01_direction.py [--final]
Without --final: development years 2012-2015. With --final: 2016-2019, once, from the frozen tag.
Writes results/{dev,final}/direction_{predictions.parquet,metrics.csv,summary.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.cnn import CNN  # noqa: E402
from stormcast.data import DATA  # noqa: E402
from stormcast.evaluation import (auc, block_bootstrap_ci, brier, load_dataset, log_loss, newey_west_t,  # noqa: E402
                                  out_dir, walk_forward, years)
from stormcast.ml import dir_models  # noqa: E402

COST_POINTS = 0.70  # AlphaBench round-trip cost model for one contract (slippage + commission), index points
TOO_GOOD_AUC = 0.55

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    yrs = years(a.final, "01_direction")
    df = load_dataset()
    seq = np.load(DATA / "sequences.npy")
    df["row"] = np.arange(len(df))

    def fit_predict(train, test):
        out = {}
        y = train.y_dir.to_numpy()
        for name, m in dir_models().items():
            out[name] = m.fit(train, y).predict_proba(test)[:, 1]
        rng = np.random.default_rng(1)
        for name in ("logistic", "gbm"):
            out[f"{name}_shuffled"] = dir_models()[name].fit(train, rng.permutation(y)).predict_proba(test)[:, 1]
        cnn = CNN("direction").fit(seq[train.row], train.tod.to_numpy(), y)
        out["cnn"] = cnn.predict(seq[test.row], test.tod.to_numpy())
        print(f"  year {test.date.dt.year.iloc[0]}: trained on {len(train)} rows, tested on {len(test)}", flush=True)
        return out

    pred = walk_forward(df, yrs, fit_predict)
    models = ["base_rate", "logistic", "gbm", "mlp", "cnn", "logistic_shuffled", "gbm_shuffled"]
    rows = []
    for scope, g in [("pooled", pred)] + [(str(y), g) for y, g in pred.groupby("test_year")]:
        ll_base = log_loss(g.y_dir, g.base_rate)
        for m in models:
            p = g[m].to_numpy()
            pts = np.where(p > 0.5, 1.0, -1.0) * g.move_pts.to_numpy()
            t, pval = newey_west_t(pd.Series(pts).groupby(g.date.to_numpy()).sum().to_numpy())
            lo, hi = block_bootstrap_ci(g.date, g.y_dir, p) if scope == "pooled" and m != "base_rate" else (np.nan, np.nan)
            rows.append({"scope": scope, "model": m, "n": len(g),
                         "auc": auc(g.y_dir, p) if m != "base_rate" else 0.5, "auc_lo": lo, "auc_hi": hi,
                         "logloss_skill": 1 - log_loss(g.y_dir, p) / ll_base, "brier": brier(g.y_dir, p),
                         "accuracy": float(np.mean((p > 0.5) == g.y_dir)), "share_long": float(np.mean(p > 0.5)),
                         "gross_pts_per_trade": float(pts.mean()), "net_pts_per_trade": float(pts.mean() - COST_POINTS),
                         "t_daily_gross": t})
    M = pd.DataFrame(rows)
    od = out_dir(a.final)
    pred.drop(columns=["row"]).to_parquet(od / "direction_predictions.parquet")
    M.to_csv(od / "direction_metrics.csv", index=False)
    pooled = M[M.scope == "pooled"].set_index("model")
    honest = pooled.loc[["logistic", "gbm", "mlp", "cnn"]]
    summary = {"years": list(yrs), "up_share": float(pred.y_dir.mean()),
               "max_honest_auc": float(honest.auc.max()), "too_good": bool(honest.auc.max() > TOO_GOOD_AUC),
               "H1_all_auc_in_0.48_0.52": bool(honest.auc.between(0.48, 0.52).all()),
               "H1_all_ci_contain_0.5": bool(((honest.auc_lo <= 0.5) & (honest.auc_hi >= 0.5)).all()),
               "H1_logloss_skill_le_0": bool((honest.logloss_skill <= 0).all()),
               "H1_gross_below_cost": bool((honest.gross_pts_per_trade < COST_POINTS).all())}
    (od / "direction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(M[M.scope == "pooled"].round(4).to_string(index=False))
        print(M[M.scope != "pooled"].pivot(index="model", columns="scope", values="auc").round(4))
    print(json.dumps(summary, indent=2))
    if summary["too_good"]:
        print("STOP: honest direction AUC above 0.55 -> leak hunt before anything else (docs/plan.md)")
