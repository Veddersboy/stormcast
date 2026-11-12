"""Models with the pre-registered hyperparameters (docs/plan.md). Every model exposes fit(X, y) / predict(X) or
predict_proba(X) on a DataFrame of features, and is fit on one training window only."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stormcast.features import ALL_FEATURES, HAR

SEED = 0
GBM = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
           early_stopping=False, random_state=SEED)


class Columns(BaseEstimator):
    """Fit `model` on a fixed list of columns."""

    def __init__(self, cols, model):
        self.cols, self.model = cols, model

    def fit(self, X, y):
        self.model.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.cols])

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.cols])


class Shifted(BaseEstimator):
    """Forecast = one feature + the mean training gap (persistence and time-of-day-profile baselines)."""

    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        self.bias_ = float(np.mean(np.asarray(y) - X[self.col].to_numpy()))
        return self

    def predict(self, X):
        return X[self.col].to_numpy() + self.bias_


class BaseRate(BaseEstimator):
    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p_), np.full(len(X), self.p_)])


class TimeEarlyStoppedMLP(BaseEstimator):
    """sklearn MLP trained epoch by epoch; keeps the epoch with the best loss on the last 10% of the training window
    (by time; rows arrive in time order)."""

    def __init__(self, classifier=False, hidden=(64, 32), alpha=1e-3, max_epochs=200, patience=10):
        self.classifier, self.hidden, self.alpha = classifier, hidden, alpha
        self.max_epochs, self.patience = max_epochs, patience

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        cut = int(0.9 * len(X))
        self.scaler_ = StandardScaler().fit(X[:cut])
        Xt, Xv = self.scaler_.transform(X[:cut]), self.scaler_.transform(X[cut:])
        cls = MLPClassifier if self.classifier else MLPRegressor
        net = cls(hidden_layer_sizes=self.hidden, alpha=self.alpha, learning_rate_init=1e-3, batch_size=256,
                  random_state=SEED)
        rng = np.random.default_rng(SEED)
        best, best_loss, bad = None, np.inf, 0
        warnings.filterwarnings("ignore", message="Got `batch_size`")
        for _ in range(self.max_epochs):
            order = rng.permutation(cut)
            for s in range(0, cut, 256):
                b = order[s:s + 256]
                if self.classifier:
                    net.partial_fit(Xt[b], y[:cut][b], classes=np.array([0, 1]))
                else:
                    net.partial_fit(Xt[b], y[:cut][b])
            if self.classifier:
                p = np.clip(net.predict_proba(Xv)[:, 1], 1e-6, 1 - 1e-6)
                loss = -np.mean(y[cut:] * np.log(p) + (1 - y[cut:]) * np.log(1 - p))
            else:
                loss = np.mean((net.predict(Xv) - y[cut:]) ** 2)
            if loss < best_loss - 1e-6:
                best, best_loss, bad = copy.deepcopy(net), loss, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_ = best
        return self

    def predict(self, X):
        return self.net_.predict(self.scaler_.transform(np.asarray(X, float)))

    def predict_proba(self, X):
        return self.net_.predict_proba(self.scaler_.transform(np.asarray(X, float)))


def _linear_all(model):
    other = [c for c in ALL_FEATURES if c not in ("tod", "dow")]
    ct = ColumnTransformer([("cal", OneHotEncoder(handle_unknown="ignore"), ["tod", "dow"]),
                            ("num", StandardScaler(), other)])
    return Pipeline([("prep", ct), ("model", model)])


def vol_models() -> dict:
    return {
        "persistence": Shifted("lrv_30"),
        "tod_profile": Shifted("lrv_slot"),
        "har_linear": Columns(HAR, LinearRegression()),
        "ridge_all": Columns(ALL_FEATURES, _linear_all(Ridge(alpha=1.0))),
        "gbm": Columns(ALL_FEATURES, HistGradientBoostingRegressor(**GBM)),
        "mlp": Columns(ALL_FEATURES, TimeEarlyStoppedMLP()),
    }


def dir_models() -> dict:
    return {
        "base_rate": BaseRate(),
        "logistic": Columns(ALL_FEATURES, _linear_all(LogisticRegression(C=0.1, max_iter=2000))),
        "gbm": Columns(ALL_FEATURES, HistGradientBoostingClassifier(**GBM)),
        "mlp": Columns(ALL_FEATURES, TimeEarlyStoppedMLP(classifier=True)),
    }


def quantile_gbm(alpha: float):
    return Columns(ALL_FEATURES, HistGradientBoostingRegressor(loss="quantile", quantile=alpha, **GBM))
