"""Walk-forward stacking: the blend used for year Y is fit only on out-of-sample forecasts of years before Y.

forecast = intercept + sum_i w_i * model_i, with w_i >= 0 (non-negative least squares on demeaned forecasts), so the
blend can only reweight models, never short one against another. The first year, with no earlier forecasts, uses equal
weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def fit_weights(oos: pd.DataFrame, models: list[str], target: str = "y_lrv") -> dict:
    if oos is None or len(oos) == 0:
        w = np.full(len(models), 1 / len(models))
        return {"weights": dict(zip(models, w.tolist())), "intercept": 0.0}
    X = oos[models].to_numpy()
    y = oos[target].to_numpy()
    xm, ym = X.mean(0), y.mean()
    w, _ = nnls(X - xm, y - ym)
    return {"weights": dict(zip(models, w.tolist())), "intercept": float(ym - xm @ w)}


def apply(df: pd.DataFrame, blend: dict) -> np.ndarray:
    return blend["intercept"] + sum(w * df[m].to_numpy() for m, w in blend["weights"].items())
