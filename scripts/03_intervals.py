"""H4: 90% prediction intervals for the next 30 minutes' high-low range, and whether they hold up when volatility jumps.

Each walk-forward year: quantile gradient boosting (5% and 95%) is fit on the first 80% of the training window; the
last 20% (by time) is the calibration set. Intervals compared on the test year, in time order:
  constant   - the training window's unconditional 5%/95% quantiles of the range
  quantile   - raw quantile gradient boosting
  cqr        - quantile GB widened by the split-conformal correction from the calibration set
  aci        - CQR updated online by adaptive conformal inference (gamma 0.005), reset each year
Usage: uv run python scripts/03_intervals.py [--final]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.conformal import aci, conformal_quantile, cqr_scores  # noqa: E402
from stormcast.evaluation import load_dataset, out_dir, walk_forward, years  # noqa: E402
from stormcast.ml import quantile_gbm  # noqa: E402

ALPHA, GAMMA, ROLL = 0.10, 0.005, 250
METHODS = ["constant", "quantile", "cqr", "aci"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    yrs = years(a.final, "03_intervals")
    df = load_dataset()

    def fit_predict(train, test):
        cut = int(0.8 * len(train))
        fit, cal = train.iloc[:cut], train.iloc[cut:]
        qlo = quantile_gbm(ALPHA / 2).fit(fit, fit.y_lrange.to_numpy())
        qhi = quantile_gbm(1 - ALPHA / 2).fit(fit, fit.y_lrange.to_numpy())
        y = test.y_lrange.to_numpy()
        lo, hi = qlo.predict(test), qhi.predict(test)
        s_cal = cqr_scores(cal.y_lrange.to_numpy(), qlo.predict(cal), qhi.predict(cal))
        q = conformal_quantile(s_cal, ALPHA)
        L, U, A = aci(y, lo, hi, s_cal, ALPHA, GAMMA)
        c_lo, c_hi = np.quantile(train.y_lrange, [ALPHA / 2, 1 - ALPHA / 2])
        print(f"  year {test.date.dt.year.iloc[0]}: CQR correction {q:+.3f} (log points)", flush=True)
        return {"constant_lo": np.full(len(y), c_lo), "constant_hi": np.full(len(y), c_hi),
                "quantile_lo": lo, "quantile_hi": hi, "cqr_lo": lo - q, "cqr_hi": hi + q,
                "aci_lo": L, "aci_hi": U, "aci_alpha": A}

    pred = walk_forward(df, yrs, fit_predict)
    to_pts = lambda x: np.exp(x) - 0.25  # noqa: E731  (y_lrange = log(range + 1 tick))
    rows = []
    for m in METHODS:
        cov = ((pred.y_lrange >= pred[f"{m}_lo"]) & (pred.y_lrange <= pred[f"{m}_hi"])).astype(float)
        pred[f"{m}_cover"] = cov
        width = to_pts(pred[f"{m}_hi"]) - to_pts(pred[f"{m}_lo"])
        roll = cov.rolling(ROLL).mean()
        vix_t = pd.qcut(pred.lvix, 3, labels=["low VIX", "mid VIX", "high VIX"])
        r = {"method": m, "coverage": cov.mean(), "mean_width_pts": float(width[np.isfinite(width)].mean()),
             "median_width_pts": float(np.median(width[np.isfinite(width)])), "infinite_share": float(np.mean(~np.isfinite(width))),
             "worst_rolling_coverage": float(roll.min()), "share_rolling_below_80": float((roll < 0.8).mean())}
        r.update({f"cov_{k}": float(v) for k, v in cov.groupby(vix_t, observed=True).mean().items()})
        r.update({f"cov_{y}": float(v) for y, v in cov.groupby(pred.test_year).mean().items()})
        rows.append(r)
    M = pd.DataFrame(rows)
    od = out_dir(a.final)
    pred.to_parquet(od / "intervals_predictions.parquet")
    M.to_csv(od / "intervals_metrics.csv", index=False)
    m = M.set_index("method")
    summary = {"years": list(yrs), "aci_coverage": float(m.loc["aci", "coverage"]),
               "H4_aci_within_1.5pp": bool(abs(m.loc["aci", "coverage"] - 0.9) <= 0.015),
               "H4_aci_worst_rolling_above_cqr": bool(m.loc["aci", "worst_rolling_coverage"]
                                                       > m.loc["cqr", "worst_rolling_coverage"])}
    (od / "intervals_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(M.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
