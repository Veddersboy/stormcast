"""Gradient boosting for log realized variance, tuned with Optuna inside each training window, explained with exact
TreeSHAP.

Tuning never sees the test year: each walk-forward training window is split by time (first 80% of days to fit, last 20%
to validate, one-day embargo), Optuna's TPE sampler searches the hyperparameters on that split, and the best setting
is refit on the whole window.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

from stormcast.features import ALL_FEATURES
from stormcast.v2 import inner_split

SEED = 0


def _model(p: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=p["max_iter"], learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        max_features=p["max_features"], early_stopping=False, random_state=SEED)


def tune(train: pd.DataFrame, n_trials: int = 20, timeout: float | None = None) -> tuple[dict, optuna.Study]:
    fit, val = inner_split(train)
    Xf, yf = train[ALL_FEATURES].iloc[fit], train.y_lrv.iloc[fit]
    Xv, yv = train[ALL_FEATURES].iloc[val], train.y_lrv.iloc[val]

    def objective(t: optuna.Trial) -> float:
        p = {"max_iter": t.suggest_int("max_iter", 150, 700, step=50),
             "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
             "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 7, 63),
             "min_samples_leaf": t.suggest_int("min_samples_leaf", 50, 1500, log=True),
             "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
             "max_features": t.suggest_float("max_features", 0.4, 1.0)}
        m = _model(p).fit(Xf, yf)
        return float(np.mean((m.predict(Xv) - yv) ** 2))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 200,
                         "l2_regularization": 1.0, "max_features": 1.0})  # the pre-registered setting, as trial 0
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study.best_params, study


class TunedGBM:
    def __init__(self, n_trials: int = 20):
        self.n_trials = n_trials

    def fit(self, train: pd.DataFrame):
        self.params_, self.study_ = tune(train, self.n_trials)
        self.model_ = _model(self.params_).fit(train[ALL_FEATURES], train.y_lrv)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(df[ALL_FEATURES])

    def explain(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Exact TreeSHAP values (n, features) on the log-variance scale, and the base value: for each row, base +
        sum of the row's values equals the prediction."""
        ex = shap.TreeExplainer(self.model_)
        vals = ex.shap_values(df[ALL_FEATURES].to_numpy(), check_additivity=False)
        return np.asarray(vals, np.float32), float(np.ravel(ex.expected_value)[0])
