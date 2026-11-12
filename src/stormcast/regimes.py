"""Market regimes: a Gaussian hidden Markov model on daily observations, with point-in-time (filtered) probabilities.

hmmlearn's predict_proba is forward-backward *smoothing*: the probability for day d uses every later day too, which is
a look-ahead leak if used as a feature. `filtered` runs the forward pass only, so day d's probability uses days <= d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from stormcast.data import load_vix
from stormcast.features import grid

N_STATES = 3
NAMES = ["calm", "normal", "storm"]


def daily_observations(bars: pd.DataFrame) -> pd.DataFrame:
    """Per cash session: log realized variance, open-to-close log return, and that day's VIX close (known at the
    16:15 ET VIX close, so usable from the next day on)."""
    g = grid(bars)
    rv = (g["R"] ** 2).sum(axis=1)
    ret = np.log(g["C"][:, -1]) - np.log(g["O"][:, 0])
    vix = load_vix().reindex(g["days"]).ffill()
    return pd.DataFrame({"lrv_day": np.log(rv), "ret_day": ret, "lvix": np.log(vix.to_numpy())}, index=g["days"])


class Regimes:
    """Standardize with training-window statistics, fit the HMM with several EM restarts (best training
    log-likelihood), and relabel states by mean log volatility: 0 calm ... 2 storm."""

    def __init__(self, restarts: int = 5):
        self.restarts = restarts

    def fit(self, X: np.ndarray):
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0)
        Z = (X - self.mu_) / self.sd_
        best, best_ll = None, -np.inf
        for seed in range(self.restarts):
            m = GaussianHMM(N_STATES, covariance_type="full", n_iter=500, random_state=seed, tol=1e-4)
            m.fit(Z)
            ll = m.score(Z)
            if ll > best_ll:
                best, best_ll = m, ll
        order = np.argsort(best.means_[:, 0])
        m2 = GaussianHMM(N_STATES, covariance_type="full")
        m2.startprob_ = best.startprob_[order]
        m2.transmat_ = best.transmat_[np.ix_(order, order)]
        m2.means_ = best.means_[order]
        m2.covars_ = best.covars_[order]
        m2.n_features = X.shape[1]
        self.hmm_ = m2
        return self

    def _z(self, X):
        return (X - self.mu_) / self.sd_

    def filtered(self, X):
        return filtered(self.hmm_, self._z(X))

    def smoothed(self, X):
        return smoothed(self.hmm_, self._z(X))

    def state_means(self):
        """Per-state means in the original units."""
        return self.hmm_.means_ * self.sd_ + self.mu_


def filtered(m: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """P(state on day d | observations on days <= d): forward algorithm in log space."""
    ll = np.column_stack([multivariate_normal(m.means_[s], m.covars_[s]).logpdf(X) for s in range(N_STATES)])
    with np.errstate(divide="ignore"):  # zero probabilities -> -inf, which logsumexp handles
        log_a, log_pi = np.log(m.transmat_), np.log(m.startprob_)
    out = np.empty_like(ll)
    prev = log_pi + ll[0]
    out[0] = prev - logsumexp(prev)
    for t in range(1, len(X)):
        prev = logsumexp(out[t - 1][:, None] + log_a, axis=0) + ll[t]
        out[t] = prev - logsumexp(prev)
    return np.exp(out)


def smoothed(m: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Forward-backward posteriors (uses the future; for the leak demonstration only)."""
    return m.predict_proba(X)
