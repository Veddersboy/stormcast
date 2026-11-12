"""H7: are volatility forecasts useful for risk? Size a constant long position in each 30-minute slot by the forecast.

units = 1 / forecast standard deviation in points, rescaled so the average size is 1 contract for every scheme
(equal average exposure). Schemes: constant (1 contract), persistence, HAR-intraday, gradient boosting, and an
oracle that knows the realized volatility (upper bound). Needs scripts/02_volatility.py output.
Usage: uv run python scripts/06_risk_sizing.py [--final]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.evaluation import out_dir, years  # noqa: E402

LIMITS = (2, 3, 4)  # daily loss limit = m x mean absolute daily P&L of the constant scheme

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    yrs = years(a.final, "06_risk_sizing")
    od = out_dir(a.final)
    p = pd.read_parquet(od / "volatility_predictions.parquet")
    assert set(p.test_year.unique()) == set(yrs)
    sig = {"constant": np.ones(len(p)), "persistence": np.exp(p.persistence / 2), "har_linear": np.exp(p.har_linear / 2),
           "gbm": np.exp(p.gbm / 2), "oracle": np.exp(p.y_lrv / 2)}
    pnl = {}
    for name, s in sig.items():
        u = 1 / (np.asarray(s) * p.price.to_numpy()) if name != "constant" else np.ones(len(p))
        u = u / u.mean()
        pnl[name] = u * p.move_pts.to_numpy()
    P = pd.DataFrame(pnl)
    daily = P.groupby(p.date.to_numpy()).sum()
    base_abs = daily.constant.abs().mean()
    rows = []
    for name in P:
        z = P[name] / P[name].std()
        yearly_sd = P[name].groupby(p.test_year.to_numpy()).std()
        r = {"scheme": name, "kurtosis_slot": float(stats.kurtosis(P[name])), "share_abs_z_gt_4": float((z.abs() > 4).mean()),
             "yearly_sd_max_over_min": float(yearly_sd.max() / yearly_sd.min()),
             "worst_day_in_const_sd": float(daily[name].min() / daily.constant.std()),
             "daily_sd_pts": float(daily[name].std()), "mean_daily_pts": float(daily[name].mean())}
        for m in LIMITS:
            r[f"breach_rate_{m}x"] = float((daily[name] < -m * base_abs).mean())
        rows.append(r)
    M = pd.DataFrame(rows)
    M.to_csv(od / "risk_sizing_metrics.csv", index=False)
    daily.to_parquet(od / "risk_sizing_daily.parquet")
    m = M.set_index("scheme")
    summary = {"years": list(yrs),
               "H7_lower_kurtosis": bool(m.loc["gbm", "kurtosis_slot"] < m.loc["constant", "kurtosis_slot"]),
               "H7_fewer_breaches_3x": bool(m.loc["gbm", "breach_rate_3x"] < m.loc["constant", "breach_rate_3x"])}
    summary["H7_supported"] = summary["H7_lower_kurtosis"] and summary["H7_fewer_breaches_3x"]
    (od / "risk_sizing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(M.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))
