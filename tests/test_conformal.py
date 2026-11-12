import numpy as np

from stormcast.conformal import aci, conformal_quantile, cqr_scores


def test_split_conformal_coverage_exchangeable():
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(200):
        cal, test = rng.standard_normal(500), rng.standard_normal(500)
        q = conformal_quantile(cqr_scores(cal, -1.0, 1.0), 0.1)
        cover.append(np.mean((test >= -1 - q) & (test <= 1 + q)))
    assert abs(np.mean(cover) - 0.9) < 0.01


def test_aci_recovers_after_variance_shift():
    """Static intervals under-cover after the noise doubles; ACI adapts back toward 90%."""
    rng = np.random.default_rng(1)
    cal = rng.standard_normal(1000)
    y = np.r_[rng.standard_normal(1000), 2 * rng.standard_normal(4000)]
    zeros = np.zeros_like(y)
    q = conformal_quantile(cqr_scores(cal, 0.0, 0.0), 0.1)
    static = np.mean(np.abs(y[2000:]) <= q)
    L, U, _ = aci(y, zeros, zeros, cqr_scores(cal, 0.0, 0.0), alpha=0.1, gamma=0.01)
    adaptive = np.mean((y[2000:] >= L[2000:]) & (y[2000:] <= U[2000:]))
    assert static < 0.65 and abs(adaptive - 0.9) < 0.03


def test_aci_uses_only_past_scores():
    """The interval at t must not change when later observations change."""
    rng = np.random.default_rng(2)
    cal = rng.standard_normal(200)
    y = rng.standard_normal(300)
    z = np.zeros_like(y)
    L1, U1, _ = aci(y, z, z, cal)
    y2 = y.copy()
    y2[150:] *= 10
    L2, U2, _ = aci(y2, z, z, cal)
    assert np.allclose(L1[:151], L2[:151]) and np.allclose(U1[:151], U2[:151])


def test_hmm_filtered_is_point_in_time_and_smoothed_is_not():
    from stormcast.regimes import Regimes
    rng = np.random.default_rng(3)
    states = np.repeat([0, 1, 2, 1, 0], 120)
    X = np.column_stack([np.array([-1.0, 0.0, 1.5])[states] + 0.3 * rng.standard_normal(len(states)),
                         0.2 * rng.standard_normal(len(states))])
    m = Regimes(restarts=2).fit(X)
    X2 = X.copy()
    X2[300:] += 3.0  # change the future only
    assert np.allclose(m.filtered(X)[:300], m.filtered(X2)[:300])
    assert not np.allclose(m.smoothed(X)[:300], m.smoothed(X2)[:300])
    assert np.allclose(m.filtered(X).sum(axis=1), 1.0)
    assert np.all(np.diff(m.state_means()[:, 0]) > 0)
