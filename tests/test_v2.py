"""Tests for the v2 components: dense point-in-time features, StormNet causality and output shape, delayed ACI and
walk-forward stacking."""
import numpy as np
import pandas as pd
import pytest
import torch

from stormcast import stack
from stormcast.conformal import aci, aci_delayed
from stormcast.deep import SEQ, Member, StormNet
from stormcast.features import ALL_FEATURES

# ---------------------------------------------------------------- StormNet


def test_stormnet_shapes_and_noncrossing_quantiles():
    torch.manual_seed(0)
    net = StormNet(len(ALL_FEATURES)).eval()
    o = net(torch.randn(16, SEQ, 3), torch.randn(16, len(ALL_FEATURES)))
    assert o["mu"].shape == (16,) and o["q"].shape == (16, 3) and o["attn"].shape == (16, SEQ // 4)
    assert torch.all(o["q"][:, 1:] >= o["q"][:, :-1])
    assert torch.allclose(o["attn"].sum(1), torch.ones(16), atol=1e-5)


def _tiny_frame(n, rng):
    df = pd.DataFrame(rng.standard_normal((n, len(ALL_FEATURES))), columns=ALL_FEATURES)
    df["y_lrv"] = df.lrv_30 + 0.1 * rng.standard_normal(n)
    df["y_lrange"] = df.lrv_30 / 2 + 0.1 * rng.standard_normal(n)
    df["y_dir"] = (rng.random(n) > 0.5).astype(int)
    return df


def test_stormnet_reads_only_bars_before_the_decision():
    """Changing any bar at or after a sample's window end must not change its forecast."""
    rng = np.random.default_rng(0)
    chan = rng.standard_normal((3000, 3)).astype(np.float32)
    df = _tiny_frame(400, rng)
    ends = rng.integers(SEQ, 2500, size=400)
    m = Member(epochs=1, batch=128).fit(chan, df, ends)
    before = m.predict(chan, df.iloc[:20], ends[:20])
    chan2 = chan.copy()
    cut = int(ends[:20].min())
    chan2[cut:] += 100.0
    after = m.predict(chan2, df.iloc[:20], ends[:20])
    unaffected = ends[:20] == cut  # samples whose window ends exactly at the perturbation
    assert np.allclose(before["mu"][unaffected], after["mu"][unaffected], atol=1e-4)


# ---------------------------------------------------------------- conformal


def test_aci_delayed_without_delay_equals_aci():
    rng = np.random.default_rng(1)
    n = 800
    y = rng.standard_normal(n)
    lo, hi = np.full(n, -1.5), np.full(n, 1.5)
    cal = rng.standard_normal(200) - 1.5
    t = np.arange(n) * 5
    L, U, _ = aci_delayed(t, t, y, lo, hi, cal)
    L0, U0, _ = aci(y, lo, hi, cal, gamma=0.002)
    assert np.allclose(L[1:], L0[1:]) and np.allclose(U[1:], U0[1:])


def test_aci_delayed_recovers_after_variance_shift_and_never_peeks():
    rng = np.random.default_rng(2)
    n = 6000
    y = rng.standard_normal(n) * np.r_[np.ones(n // 2), np.full(n // 2, 2.5)]
    lo, hi = np.full(n, -1.645), np.full(n, 1.645)
    cal = np.maximum(-1.645 - rng.standard_normal(500), rng.standard_normal(500) - 1.645)
    t = np.arange(n) * 5
    L, U, _ = aci_delayed(t, t + 30, y, lo, hi, cal, gamma=0.005)
    cov = (y >= L) & (y <= U)
    assert abs(cov[n // 2 + 1000:].mean() - 0.9) < 0.03
    # rows 0-5 are formed before any outcome is revealed; row 6 (t = 30) may use row 0 (revealed at t = 30)
    y2 = y.copy()
    y2[:6] += 50
    L2, U2, _ = aci_delayed(t, t + 30, y2, lo, hi, cal, gamma=0.005)
    assert np.allclose(L[:6], L2[:6]) and not np.allclose(L[6:20], L2[6:20])


# ---------------------------------------------------------------- stacking


def test_stack_recovers_nonnegative_weights():
    rng = np.random.default_rng(3)
    a, b, c = rng.standard_normal((3, 5000))
    oos = pd.DataFrame({"a": a, "b": b, "c": c, "y_lrv": 0.7 * a + 0.3 * b + 2.0 + 0.01 * rng.standard_normal(5000)})
    w = stack.fit_weights(oos, ["a", "b", "c"])
    assert w["weights"]["a"] == pytest.approx(0.7, abs=0.01) and w["weights"]["c"] == pytest.approx(0, abs=0.01)
    assert w["intercept"] == pytest.approx(2.0, abs=0.01)
    assert min(stack.fit_weights(oos.assign(c=-a), ["a", "c"])["weights"].values()) >= 0


def test_stack_first_year_uses_equal_weights():
    w = stack.fit_weights(pd.DataFrame(), ["a", "b"])
    assert w["weights"] == {"a": 0.5, "b": 0.5}
