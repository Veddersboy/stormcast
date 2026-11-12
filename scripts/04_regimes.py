"""H6: unsupervised market regimes (3-state Gaussian HMM on daily log volatility, return and VIX).

Each test year: fit on every earlier day; the regime for day d is the FILTERED probability at the end of day d-1
(point-in-time). Question: does the regime tell us about the next day's volatility? Its direction?
Usage: uv run python scripts/04_regimes.py [--final]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.data import load_bars  # noqa: E402
from stormcast.evaluation import out_dir, years  # noqa: E402
from stormcast.regimes import NAMES, Regimes, daily_observations  # noqa: E402


def eta_squared(y, groups) -> float:
    y = pd.Series(np.asarray(y))
    g = pd.Series(np.asarray(groups))
    between = sum(len(v) * (v.mean() - y.mean()) ** 2 for _, v in y.groupby(g))
    return float(between / ((y - y.mean()) ** 2).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    yrs = years(a.final, "04_regimes")
    obs = daily_observations(load_bars()).dropna()
    X = obs.to_numpy()
    parts, trans = [], {}
    for y in yrs:
        train = obs.index < pd.Timestamp(f"{y}-01-01")
        m = Regimes().fit(X[train])
        p = m.filtered(X)  # forward pass over all days; row d uses days <= d only
        prior = pd.DataFrame(p, index=obs.index, columns=[f"p_{n}" for n in NAMES]).shift(1)  # known before day d opens
        test = obs.index.year == y
        part = obs[test].join(prior[test])
        part["state"] = part[[f"p_{n}" for n in NAMES]].to_numpy().argmax(axis=1)
        part["test_year"] = y
        parts.append(part)
        trans[y] = m.hmm_.transmat_.round(3).tolist()
        print(f"  year {y}: fit on {train.sum()} days; state means (lrv, ret, lvix) {m.state_means().round(3).tolist()}")
    R = pd.concat(parts)
    R["up"] = (R.ret_day > 0).astype(int)
    by_state = R.groupby("state").agg(days=("up", "size"), mean_lrv_next=("lrv_day", "mean"),
                                      mean_ret_next_bp=("ret_day", lambda x: 1e4 * x.mean()),
                                      sd_ret_next_bp=("ret_day", lambda x: 1e4 * x.std()), share_up=("up", "mean"))
    by_state.index = [NAMES[i] for i in by_state.index]
    groups = [g.lrv_day.to_numpy() for _, g in R.groupby("state")]
    summary = {"years": list(yrs), "eta2_next_day_volatility": eta_squared(R.lrv_day, R.state),
               "kruskal_p_volatility": float(stats.kruskal(*groups).pvalue),
               "eta2_next_day_return": eta_squared(R.ret_day, R.state),
               "kruskal_p_return": float(stats.kruskal(*[g.ret_day.to_numpy() for _, g in R.groupby("state")]).pvalue),
               "auc_up_day_from_p_storm": float(roc_auc_score(R.up, R.p_storm)),
               "auc_up_day_from_p_calm": float(roc_auc_score(R.up, R.p_calm)),
               "transition_matrices": {str(k): v for k, v in trans.items()}}
    summary["H6_separates_vol_not_direction"] = bool(summary["kruskal_p_volatility"] < 0.001
                                                     and summary["kruskal_p_return"] > 0.05)  # plan amendment A2
    od = out_dir(a.final)
    R.to_parquet(od / "regimes_daily.parquet")
    by_state.to_csv(od / "regimes_by_state.csv")
    (od / "regimes_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(by_state.round(3))
    print(json.dumps({k: v for k, v in summary.items() if k != "transition_matrices"}, indent=2))
