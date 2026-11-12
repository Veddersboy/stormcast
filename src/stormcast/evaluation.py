"""Walk-forward protocol, the one-shot guard for the final years, and metrics."""
from __future__ import annotations

import datetime as dt
import subprocess
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from stormcast.data import DATA, ROOT

DEV_YEARS = (2012, 2013, 2014, 2015)
FINAL_YEARS = (2016, 2017, 2018, 2019)
FROZEN_TAG = "final-frozen"
EMBARGO_DAYS = 1


class ProtocolError(RuntimeError):
    pass


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def years(final: bool, script: str) -> tuple[int, ...]:
    """Development years, or the final years once: only from a clean checkout of the `final-frozen` tag."""
    if not final:
        return DEV_YEARS
    head, tag = _git("rev-parse", "HEAD"), _git("rev-list", "-n", "1", FROZEN_TAG)
    if not tag or head != tag:
        raise ProtocolError(f"final years need HEAD at tag {FROZEN_TAG} (HEAD {head[:8]}, tag {tag[:8] or 'missing'})")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ProtocolError("final years need a clean working tree")
    log = ROOT / "results" / "final_runs.log"
    with open(log, "a") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {script} {head[:12]}\n")
    return FINAL_YEARS


def out_dir(final: bool):
    d = ROOT / "results" / ("final" if final else "dev")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "dataset.parquet")
    return df[df.valid].reset_index(drop=True)


def splits(df: pd.DataFrame, test_years) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(year, train rows, test rows): train on every day before the year minus a 1-trading-day embargo."""
    days = np.sort(df.date.unique())
    out = []
    for y in test_years:
        start = pd.Timestamp(f"{y}-01-01")
        before = days[days < start]
        last_train = before[-1 - EMBARGO_DAYS]
        tr = np.flatnonzero(df.date.to_numpy() <= last_train)
        te = np.flatnonzero(df.date.dt.year.to_numpy() == y)
        out.append((y, tr, te))
    return out


def walk_forward(df: pd.DataFrame, test_years, fit_predict: Callable) -> pd.DataFrame:
    """fit_predict(train_df, test_df) -> dict of name -> prediction array. Returns test rows with predictions."""
    parts = []
    for y, tr, te in splits(df, test_years):
        test = df.iloc[te].copy()
        for name, pred in fit_predict(df.iloc[tr], test).items():
            test[name] = pred
        test["test_year"] = y
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


# ---------- metrics ----------
def r2(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def qlike(y_log, p_log, resid_var: float = 0.0) -> float:
    """QLIKE loss on the variance scale; the log forecast is mapped to a variance with a lognormal correction."""
    s2 = np.exp(np.asarray(y_log))
    h = np.exp(np.asarray(p_log) + resid_var / 2)
    ratio = s2 / h
    return float(np.mean(ratio - np.log(ratio) - 1))


def newey_west_t(x: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """t-statistic and two-sided p-value of mean(x) with Newey-West standard errors."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    t = x.mean() / np.sqrt(v / n)
    return float(t), float(2 * stats.norm.sf(abs(t)))


def diebold_mariano(dates, loss_a, loss_b, lags: int = 5) -> tuple[float, float, float]:
    """DM test on daily means of loss_a - loss_b (positive mean = b is better). Returns (mean diff, t, p)."""
    d = pd.Series(np.asarray(loss_a) - np.asarray(loss_b)).groupby(np.asarray(dates)).mean().to_numpy()
    t, p = newey_west_t(d, lags)
    return float(d.mean()), t, p


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def block_bootstrap_ci(dates, y, p, stat=auc, reps: int = 1000, seed: int = 0, level: float = 0.95):
    """Percentile interval of `stat` from resampling whole days with replacement."""
    dates = np.asarray(dates)
    uniq, inv = np.unique(dates, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(stat(y[idx], p[idx]))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))
