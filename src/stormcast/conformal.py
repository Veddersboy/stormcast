"""Prediction intervals: split conformalized quantile regression (CQR; Romano, Patterson & Candes 2019) and adaptive
conformal inference (ACI; Gibbs & Candes 2021)."""
from __future__ import annotations

import numpy as np


def cqr_scores(y, lo, hi) -> np.ndarray:
    """Conformity score: how far y falls outside [lo, hi] (negative when inside)."""
    return np.maximum(np.asarray(lo) - y, np.asarray(y) - hi)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of the scores; +inf if alpha is too small for the sample."""
    n = len(scores)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    if level > 1 or alpha <= 0:
        return np.inf
    if level <= 0:
        return -np.inf
    return float(np.quantile(scores, level, method="higher"))


def aci(y, lo, hi, calib_scores, alpha: float = 0.1, gamma: float = 0.005, window: int | None = None):
    """Online ACI over a time-ordered test sequence. At step t the interval uses the conformal quantile at the current
    level alpha_t of the most recent `window` scores (calibration scores, then each revealed test score); after y_t is
    revealed, alpha_{t+1} = alpha_t + gamma * (alpha - miss_t). Returns (lower, upper, alpha_t path)."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    window = window or len(calib_scores)
    buf = list(calib_scores)[-window:]
    a_t = alpha
    L, U, A = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    for t in range(len(y)):
        q = conformal_quantile(np.asarray(buf), a_t)
        L[t], U[t], A[t] = lo[t] - q, hi[t] + q, a_t
        miss = float(not (L[t] <= y[t] <= U[t]))
        a_t = a_t + gamma * (alpha - miss)
        buf.append(max(lo[t] - y[t], y[t] - hi[t]))
        if len(buf) > window:
            buf.pop(0)
    return L, U, A


def aci_delayed(decision_ns, reveal_ns, y, lo, hi, calib_scores, alpha: float = 0.1, gamma: float = 0.002,
                window: int | None = None):
    """ACI when outcomes arrive late: forecasts are made every few minutes but each 30-minute outcome is only known
    30 minutes later. Rows must be in time order. Before the interval at time t is formed, every earlier row whose
    outcome has been revealed (reveal time <= t) updates the score window and alpha_t, in reveal order. Returns
    (lower, upper, alpha_t path)."""
    decision_ns, reveal_ns = np.asarray(decision_ns, np.int64), np.asarray(reveal_ns, np.int64)
    y, lo, hi = map(np.asarray, (y, lo, hi))
    window = window or len(calib_scores)
    buf = list(calib_scores)[-window:]
    order = np.argsort(reveal_ns, kind="stable")
    j, a_t = 0, alpha
    L, U, A = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    for t in range(len(y)):
        while j < len(order) and reveal_ns[order[j]] <= decision_ns[t] and order[j] < t:
            i = order[j]
            miss = float(not (L[i] <= y[i] <= U[i]))
            a_t = a_t + gamma * (alpha - miss)
            buf.append(max(lo[i] - y[i], y[i] - hi[i]))
            if len(buf) > window:
                buf.pop(0)
            j += 1
        q = conformal_quantile(np.asarray(buf), a_t)
        L[t], U[t], A[t] = lo[t] - q, hi[t] + q, a_t
    return L, U, A
