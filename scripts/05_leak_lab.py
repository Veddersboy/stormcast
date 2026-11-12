"""H5 "leak lab": six common mistakes, each planted on purpose, and how good they make a useless model look.

Development data only (2010-06 to 2015-12; test years 2012-2015), whatever the mode: this is a demonstration of
method, not evidence about the final years.
  L1 centered window     a 30-minute return centered on the decision (15 minutes of future)
  L2 off-by-one bar      momentum features computed through the first bar of the horizon
  L3 overlapping labels  rest-of-day labels (they overlap within a day) + shuffled 5-fold CV, random forest
  L4 smoothed regimes    forward-backward HMM probabilities of the same day, as features (volatility target)
  L5 full-sample scaling standardizing with test-period statistics (logistic regression)
  L6 best of many        200 random logistic models, best one picked on 2012-13, then run on 2014-15
  L6b (supplementary, added after L6 was seen; disclosed in docs/plan.md amendments): the same 200 models as trading
      rules (long if p > 0.5 else short, 30 minutes, gross points), best picked inside each calendar quarter by Sharpe,
      then run in the next quarter
Usage: uv run python scripts/05_leak_lab.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.data import load_bars  # noqa: E402
from stormcast.evaluation import DEV_YEARS, ROOT, auc, load_dataset, r2, walk_forward  # noqa: E402
from stormcast.features import ALL_FEATURES, NOISE, grid  # noqa: E402
from stormcast.ml import GBM  # noqa: E402
from stormcast.regimes import NAMES, Regimes, daily_observations  # noqa: E402

SEED = 0


def wf_auc(df, cols, label="y_dir", model=lambda: HistGradientBoostingClassifier(**GBM), years=DEV_YEARS):
    def fp(tr, te):
        return {"p": model().fit(tr[cols], tr[label]).predict_proba(te[cols])[:, 1]}
    p = walk_forward(df, years, fp)
    return auc(p[label], p.p)


def kfold_auc(df, cols, label, model):
    oof = np.empty(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(df):
        oof[te] = model().fit(df.iloc[tr][cols], df.iloc[tr][label]).predict_proba(df.iloc[te][cols])[:, 1]
    return auc(df[label], oof)


if __name__ == "__main__":
    df = load_dataset()
    df = df[df.date < pd.Timestamp("2016-01-01")].reset_index(drop=True)
    bars = load_bars()
    bars = bars[bars.session < pd.Timestamp("2016-01-01")]
    g = grid(bars)
    di = g["days"].get_indexer(df.date)
    k = df.k.to_numpy()
    lc = np.log(g["C"])
    df["leak_centered"] = lc[di, k + 14] - lc[di, k - 16]
    df["leak_ret_30_ob1"] = lc[di, k] - lc[di, k - 30]
    df["leak_ret_5_ob1"] = lc[di, k] - lc[di, k - 5]
    df["y_rest_of_day"] = (lc[di, -1] > lc[di, k - 1]).astype(int)
    feats = [c for c in ALL_FEATURES]
    rows = []

    def add(leak, what, honest, leaky, metric="AUC"):
        rows.append({"leak": leak, "what": what, "metric": metric, "honest": honest, "leaky": leaky})
        print(f"{leak:<24} {metric} honest {honest:.3f} -> leaky {leaky:.3f}", flush=True)

    honest = wf_auc(df, feats)
    add("L1 centered window", "30-min return centered on the decision", honest, wf_auc(df, feats + ["leak_centered"]))
    ob1 = [c for c in feats if c not in ("ret_5", "ret_30")] + ["leak_ret_30_ob1", "leak_ret_5_ob1"]
    add("L2 off-by-one bar", "momentum through the first horizon bar", honest, wf_auc(df, ob1))

    rf = lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)  # noqa: E731
    dev = df[df.date.dt.year.isin(DEV_YEARS)].reset_index(drop=True)  # same rows as the walk-forward test years
    rows_l3 = {"rf_30min_shuffled_kfold": kfold_auc(dev, feats, "y_dir", rf),
               "rf_rest_of_day_walk_forward": wf_auc(df, feats, "y_rest_of_day", rf),
               "rf_rest_of_day_shuffled_kfold": kfold_auc(dev, feats, "y_rest_of_day", rf)}
    add("L3 overlapping labels", "rest-of-day label, shuffled 5-fold CV (random forest)",
        rows_l3["rf_rest_of_day_walk_forward"], rows_l3["rf_rest_of_day_shuffled_kfold"])

    obs = daily_observations(bars).dropna()
    reg = Regimes().fit(obs[obs.index < pd.Timestamp("2016-01-01")].to_numpy())
    cols_p = [f"p_{n}" for n in NAMES]
    smooth = pd.DataFrame(reg.smoothed(obs.to_numpy()), index=obs.index, columns=cols_p)
    filt = pd.DataFrame(reg.filtered(obs.to_numpy()), index=obs.index, columns=cols_p).shift(1)
    vol_feats = feats

    def wf_r2(frame, cols):
        def fp(tr, te):
            return {"p": HistGradientBoostingRegressor(**GBM).fit(tr[cols], tr.y_lrv).predict(te[cols])}
        p = walk_forward(frame, DEV_YEARS, fp)
        return r2(p.y_lrv, p.p)

    d_s = df.join(smooth, on="date").dropna(subset=cols_p).reset_index(drop=True)
    d_f = df.join(filt, on="date").dropna(subset=cols_p).reset_index(drop=True)
    r2_base = wf_r2(d_s, vol_feats)
    r2_filtered = wf_r2(d_f, vol_feats + cols_p)
    add("L4 smoothed regimes", "same-day forward-backward HMM probabilities (volatility)", r2_base,
        wf_r2(d_s, vol_feats + cols_p), metric="R2")

    other = [c for c in feats if c not in ("tod", "dow")]

    def logit_pipeline():
        ct = ColumnTransformer([("cal", OneHotEncoder(handle_unknown="ignore"), ["tod", "dow"]),
                                ("num", StandardScaler(), other)])
        return ct, LogisticRegression(C=0.1, max_iter=2000)

    def fp_honest(tr, te):
        ct, m = logit_pipeline()
        m.fit(ct.fit_transform(tr[feats]), tr.y_dir)
        return {"p": m.predict_proba(ct.transform(te[feats]))[:, 1]}

    def fp_full(tr, te):
        ct, m = logit_pipeline()
        ct.fit(pd.concat([tr[feats], te[feats]]))  # the mistake: statistics include the test period
        m.fit(ct.transform(tr[feats]), tr.y_dir)
        return {"p": m.predict_proba(ct.transform(te[feats]))[:, 1]}

    a_h = walk_forward(df, DEV_YEARS, fp_honest)
    a_f = walk_forward(df, DEV_YEARS, fp_full)
    add("L5 full-sample scaling", "scaler fit on train + test (logistic)", auc(a_h.y_dir, a_h.p), auc(a_f.y_dir, a_f.p))

    rng = np.random.default_rng(SEED)
    pool = [c for c in feats if c not in NOISE + ["tod", "dow"]]
    trials, trial_pnl = [], {}
    for i in range(200):
        cols = list(rng.choice(pool, size=rng.integers(2, 6), replace=False))
        C = float(rng.choice([0.01, 0.1, 1.0]))
        mk = lambda C=C: LogisticRegression(C=C, max_iter=2000)  # noqa: E731

        def fp(tr, te, cols=cols, mk=mk):
            sc = StandardScaler().fit(tr[cols])
            return {"p": mk().fit(sc.transform(tr[cols]), tr.y_dir).predict_proba(sc.transform(te[cols]))[:, 1]}
        p = walk_forward(df, DEV_YEARS, fp)
        trial_pnl[i] = np.where(p.p > 0.5, 1.0, -1.0) * p.move_pts.to_numpy()
        sel = p.test_year.isin([2012, 2013])
        trials.append({"trial": i, "cols": "|".join(cols), "C": C, "auc_2012_13": auc(p.y_dir[sel], p.p[sel]),
                       "auc_2014_15": auc(p.y_dir[~sel], p.p[~sel])})
    T = pd.DataFrame(trials)
    best = T.loc[T.auc_2012_13.idxmax()]
    add("L6 best of many", "best of 200 random models on 2012-13, then 2014-15", float(best.auc_2014_15),
        float(best.auc_2012_13))
    # L6b: daily gross P&L per trial, Sharpe (annualized, 252 days) inside each quarter; best in quarter q -> q + 1.
    days_q = p.date.dt.to_period("Q").to_numpy()
    daily = pd.DataFrame(trial_pnl).groupby([p.date.to_numpy(), days_q]).sum()
    quarters = sorted(set(daily.index.get_level_values(1)))
    sharpe = lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0  # noqa: E731
    l6b = []
    for q, q_next in zip(quarters[:-1], quarters[1:]):
        inq = daily.xs(q, level=1)
        nxt = daily.xs(q_next, level=1)
        s_in = inq.apply(sharpe)
        b = int(s_in.idxmax())
        l6b.append({"quarter": str(q), "best_trial": b, "sharpe_in_quarter": s_in.max(),
                    "sharpe_next_quarter": sharpe(nxt[b]), "median_trial_sharpe_in_quarter": float(s_in.median())})
    Q = pd.DataFrame(l6b)
    add("L6b best of many (quarters)", "best of 200 by in-quarter Sharpe, then the next quarter",
        float(Q.sharpe_next_quarter.mean()), float(Q.sharpe_in_quarter.mean()), metric="Sharpe")
    out = ROOT / "results" / "leak_lab"
    out.mkdir(parents=True, exist_ok=True)
    L = pd.DataFrame(rows)
    L.to_csv(out / "leak_lab.csv", index=False)
    T.to_csv(out / "best_of_many_trials.csv", index=False)
    Q.to_csv(out / "best_of_many_quarters.csv", index=False)
    summary = {"honest_gbm_auc": honest, "L3_detail": rows_l3, "L4_r2_filtered_prior_day_regimes": r2_filtered,
               "L6_best": {"cols": best.cols, "C": best.C, "auc_selection": best.auc_2012_13,
                           "auc_fresh": best.auc_2014_15},
               "L6_share_trials_auc_above_0.52_on_2012_13": float((T.auc_2012_13 > 0.52).mean()),
               "L6_mean_auc_2014_15": float(T.auc_2014_15.mean()),
               "L6_corr_selection_vs_fresh": float(T[["auc_2012_13", "auc_2014_15"]].corr().iloc[0, 1])}
    (out / "leak_lab_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(L.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))
