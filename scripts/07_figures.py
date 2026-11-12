"""Report figures from saved results (no model fitting).

Usage: uv run python scripts/07_figures.py [--final]
Dev figures go to results/dev/figures/; final figures to report/figures/.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormcast.data import load_bars  # noqa: E402

# Validated categorical palette (fixed order), surface and ink from the dataviz reference instance.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
                     "legend.frameon": False, "figure.dpi": 150})
LABELS = {"base_rate": "Base rate", "logistic": "Logistic", "gbm": "Gradient boosting", "mlp": "MLP", "cnn": "1D-CNN",
          "persistence": "Persistence", "tod_profile": "Time-of-day profile", "har_linear": "HAR (linear)",
          "ridge_all": "Ridge (all)", "constant": "Constant", "quantile": "Quantile GB", "cqr": "Conformal (CQR)",
          "aci": "Adaptive conformal (ACI)", "oracle": "Oracle"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def headline(res):
    dm = pd.read_csv(res / "direction_metrics.csv")
    vm = pd.read_csv(res / "volatility_metrics.csv")
    d = dm[(dm.scope == "pooled") & dm.model.isin(["logistic", "gbm", "mlp", "cnn"])]
    v = vm[(vm.scope == "pooled")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    y = np.arange(len(d))
    a1.barh(y, d.auc - 0.5, left=0.5, height=0.55, color=BLUE)
    a1.errorbar(d.auc, y, xerr=[d.auc - d.auc_lo, d.auc_hi - d.auc], fmt="none", ecolor=INK2, capsize=3, lw=1)
    a1.axvline(0.5, color=INK, lw=1)
    a1.set_yticks(y, [LABELS[m] for m in d.model])
    a1.set_xlim(0.40, 1.0)
    a1.set_xlabel("AUC (0.5 = coin flip, 1.0 = perfect)")
    a1.set_title("Direction of the next 30 minutes")
    for yi, val in zip(y, d.auc):
        a1.text(max(val, d.auc_hi.max()) + 0.015, yi, f"{val:.3f}", va="center", color=INK2, fontsize=9)
    order = ["persistence", "tod_profile", "har_linear", "ridge_all", "gbm", "mlp"]
    v = v.set_index("model").loc[order].reset_index()
    y2 = np.arange(len(v))
    a2.barh(y2, v.r2, height=0.55, color=[MUTED if m in ("persistence", "tod_profile") else ORANGE for m in v.model])
    a2.set_yticks(y2, [LABELS[m] for m in v.model])
    a2.set_xlim(0, 1.0)
    a2.set_xlabel("Out-of-sample R² on log realized variance")
    a2.set_title("Volatility of the next 30 minutes")
    for yi, val in zip(y2, v.r2):
        a2.text(val + 0.015, yi, f"{val:.2f}", va="center", color=INK2, fontsize=9)
    for a in (a1, a2):
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
    save(fig, "fig1_headline")


def calibration(res):
    p = pd.read_parquet(res / "direction_predictions.parquet")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m, c in [("logistic", BLUE), ("gbm", ORANGE), ("cnn", AQUA)]:
        bins = pd.qcut(p[m], 10, duplicates="drop")
        g = p.groupby(bins, observed=True).agg(pred=(m, "mean"), obs=("y_dir", "mean"), n=("y_dir", "size"))
        ax.plot(g.pred, g.obs, marker="o", ms=5, color=c, label=LABELS[m])
    ax.plot([0.3, 0.7], [0.3, 0.7], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    ax.axhline(p.y_dir.mean(), color=INK2, lw=1, ls=":", label="Share of up moves")
    ax.set_xlim(0.38, 0.62)
    ax.set_ylim(0.38, 0.62)
    ax.set_xlabel("Predicted probability of an up move (decile mean)")
    ax.set_ylabel("Observed share of up moves")
    ax.set_title("Direction: reliability diagram")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig3_direction_calibration")


def importance(res):
    r = pd.read_csv(res / "volatility_importance.csv").sort_values("importance")
    fig, ax = plt.subplots(figsize=(5.6, 6))
    colors = [RED if f.startswith("noise") else (ORANGE if f.startswith("lrv") or f in ("tod", "lrange_today", "lvix")
                                                 else BLUE) for f in r.feature]
    ax.barh(r.feature, r.importance, color=colors, height=0.6)
    ax.set_xlabel("Permutation importance (drop in test R² when shuffled)")
    ax.set_title("What drives the volatility forecast")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="Volatility / time / VIX"), Patch(color=BLUE, label="Direction & other"),
                       Patch(color=RED, label="Pure noise (sanity check)")], fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_vol_importance")


def storm_week(res):
    p = pd.read_parquet(res / "intervals_predictions.parquet")
    v = pd.read_parquet(res / "volatility_predictions.parquet")
    daily = p.groupby("date").range_pts.mean()
    peak = daily.idxmax()
    days = sorted(p.date.unique())
    i = days.index(peak)
    sel = days[max(0, i - 3): i + 3]
    w = p[p.date.isin(sel)].reset_index(drop=True)
    to_pts = lambda x: np.exp(x) - 0.25  # noqa: E731
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(w))
    ax.fill_between(x, to_pts(w.constant_lo), to_pts(w.constant_hi), color=GRID, step="mid", label="Constant 90% interval")
    ax.fill_between(x, to_pts(w.aci_lo), to_pts(w.aci_hi), color=BLUE, alpha=0.25, step="mid",
                    label="Adaptive conformal 90% interval")
    ax.plot(x, w.range_pts, color=INK, lw=1.5, marker="o", ms=3, label="Actual 30-minute range")
    miss = (w.aci_cover == 0)
    ax.scatter(x[miss], w.range_pts[miss], s=40, facecolor="none", edgecolor=RED, lw=1.5, zorder=5,
               label="Outside adaptive interval")
    for j, dte in enumerate(sel):
        first = w.index[w.date == dte][0]
        ax.axvline(first - 0.5, color=MUTED, lw=0.6)
        ax.text(first, ax.get_ylim()[1] if j else 0, "", fontsize=8)
    ax.set_xticks([w.index[w.date == d][0] + 5.5 for d in sel], [pd.Timestamp(d).strftime("%a %d %b %Y") for d in sel],
                  fontsize=8)
    ax.set_ylabel("High-low range, index points")
    ax.set_title(f"The stormiest week in the test years: forecast intervals vs what happened")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_storm_week")


def rolling_coverage(res):
    p = pd.read_parquet(res / "intervals_predictions.parquet")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for m, c in [("constant", MUTED), ("quantile", YELLOW), ("cqr", ORANGE), ("aci", BLUE)]:
        ax.plot(p.date, p[f"{m}_cover"].rolling(250).mean(), color=c, lw=1.5, label=LABELS[m])
    ax.axhline(0.9, color=INK, lw=1, ls="--", label="Target 90%")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Coverage, last 250 forecasts")
    ax.set_title("Do the 90% intervals really cover 90%? (rolling, out of sample)")
    ax.legend(fontsize=8, ncol=5, loc="lower left")
    save(fig, "fig5_rolling_coverage")


def regimes(res):
    R = pd.read_parquet(res / "regimes_daily.parquet")
    bars = load_bars()
    close = bars[bars.minute == 15 * 60 + 59].groupby("session").close.last()
    close = close[close.index.isin(R.index)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.3, 1]})
    a1.plot(close.index, close.to_numpy(), color=INK, lw=1)
    lo, hi = close.min() * 0.97, close.max() * 1.02
    for s, c, n in [(1, YELLOW, "normal"), (2, RED, "storm")]:
        a1.fill_between(R.index, lo, hi, where=(R.state == s).to_numpy(), color=c, alpha=0.25, lw=0, label=n.capitalize())
    a1.set_ylim(lo, hi)
    a1.set_ylabel("ES close (roll-adjusted)")
    a1.set_title("Regime known at the start of each day (filtered HMM)")
    a1.legend(fontsize=8, loc="upper left", title="Shaded: regime", title_fontsize=8)
    data = [R.lrv_day[R.state == s].to_numpy() for s in range(3)]
    data2 = [1e4 * R.ret_day[R.state == s].to_numpy() for s in range(3)]
    b = a2.boxplot(data, positions=[0, 1, 2], widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(b["boxes"], [AQUA, YELLOW, RED]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in b["medians"]:
        med.set_color(INK)
    a2.set_xticks([0, 1, 2], ["Calm", "Normal", "Storm"])
    a2.set_ylabel("That day's log realized variance")
    a2.set_title("Regime vs the day's volatility")
    a2.grid(axis="x", visible=False)
    save(fig, "fig6_regimes")
    fig, ax = plt.subplots(figsize=(4, 3.6))
    b = ax.boxplot(data2, positions=[0, 1, 2], widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(b["boxes"], [AQUA, YELLOW, RED]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in b["medians"]:
        med.set_color(INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks([0, 1, 2], ["Calm", "Normal", "Storm"])
    ax.set_ylabel("That day's open-to-close return (bp)")
    ax.set_title("Regime vs the day's direction")
    ax.grid(axis="x", visible=False)
    save(fig, "fig6b_regime_direction")


def leak_lab():
    L = pd.read_csv(ROOT / "results" / "leak_lab" / "leak_lab.csv")
    L = L[L.metric == "AUC"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(L))
    ax.barh(y - 0.17, L.honest, height=0.32, color=MUTED, label="Done correctly")
    ax.barh(y + 0.17, L.leaky, height=0.32, color=RED, label="With the mistake")
    for yi, h, l in zip(y, L.honest, L.leaky):
        ax.text(h + 0.005, yi - 0.17, f"{h:.3f}", va="center", fontsize=8, color=INK2)
        ax.text(l + 0.005, yi + 0.17, f"{l:.3f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0.5, color=INK, lw=1)
    ax.set_yticks(y, L.leak)
    ax.set_xlim(0.45, 1.02)
    ax.invert_yaxis()
    ax.set_xlabel("Direction AUC")
    ax.set_title("Leak lab: how common mistakes manufacture 'skill' from nothing")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_leak_lab")
    Q = pd.read_csv(ROOT / "results" / "leak_lab" / "best_of_many_quarters.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    x = np.arange(len(Q))
    ax.bar(x - 0.2, Q.sharpe_in_quarter, width=0.38, color=RED, label="Best of 200, in the quarter it was picked")
    ax.bar(x + 0.2, Q.sharpe_next_quarter, width=0.38, color=BLUE, label="Same rule, next quarter")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, Q.quarter, rotation=45, fontsize=8)
    ax.set_ylabel("Annualized Sharpe (gross)")
    ax.set_title("Best-of-many selection: the winner of each quarter, one quarter later")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig8_best_of_many")


def risk(res):
    d = pd.read_parquet(res / "risk_sizing_daily.parquet")
    M = pd.read_csv(res / "risk_sizing_metrics.csv").set_index("scheme")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    s = d.constant.std()
    bins = np.linspace(-7, 7, 57)
    a1.hist(d.constant / s, bins=bins, color=MUTED, alpha=0.7, label="Constant size")
    a1.hist(d.gbm / s, bins=bins, histtype="step", color=ORANGE, lw=2, label="Sized by GB forecast")
    a1.set_yscale("log")
    a1.set_xlabel("Daily P&L (in standard deviations of the constant-size P&L)")
    a1.set_ylabel("Days (log scale)")
    a1.set_title("Same average size, thinner tails")
    a1.legend(fontsize=8)
    order = ["constant", "persistence", "har_linear", "gbm", "oracle"]
    x = np.arange(3)
    for i, (m, c) in enumerate(zip(order, [MUTED, YELLOW, AQUA, ORANGE, VIOLET])):
        a2.bar(x + (i - 2) * 0.16, [100 * M.loc[m, f"breach_rate_{k}x"] for k in (2, 3, 4)], width=0.15, color=c,
               label=LABELS[m])
    a2.set_xticks(x, ["2×", "3×", "4×"])
    a2.set_xlabel("Daily loss limit (× mean absolute daily P&L at constant size)")
    a2.set_ylabel("Days breaching the limit (%)")
    a2.set_title("Daily loss-limit breaches")
    a2.legend(fontsize=7)
    a2.grid(axis="x", visible=False)
    save(fig, "fig9_risk_sizing")


def intraday_profile(res):
    v = pd.read_parquet(res / "volatility_predictions.parquet")
    g = v.groupby("k")[["y_lrv", "gbm", "har_linear"]].mean()
    t = [f"{(570 + k) // 60}:{(570 + k) % 60:02d}" for k in g.index]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, g.y_lrv, color=INK, marker="o", ms=4, label="Actual (mean)")
    ax.plot(t, g.gbm, color=ORANGE, marker="o", ms=4, label="Gradient boosting")
    ax.plot(t, g.har_linear, color=AQUA, marker="o", ms=4, label="HAR (linear)")
    ax.set_ylabel("Mean log realized variance")
    ax.set_xlabel("Start of 30-minute window (ET)")
    ax.set_title("The intraday volatility smile")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8)
    save(fig, "fig10_intraday_profile")


def direction_pnl(res):
    p = pd.read_parquet(res / "direction_predictions.parquet").sort_values(["date", "k"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for m, c in [("logistic", BLUE), ("gbm", ORANGE), ("mlp", MAGENTA), ("cnn", AQUA)]:
        pts = np.where(p[m] > 0.5, 1, -1) * p.move_pts.to_numpy()
        ax.plot(p.date, np.cumsum(pts), color=c, lw=1.5, label=f"{LABELS[m]} (gross)")
    ax.plot(p.date, -0.70 * np.arange(1, len(p) + 1), color=RED, lw=1.5, ls="--", label="Cumulative cost (0.70 pt/trade)")
    ax.axhline(0, color=INK, lw=1)
    ax.set_ylabel("Cumulative index points")
    ax.set_title("Trading the direction forecasts: gross P&L vs the cost of trading")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig11_direction_pnl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    res = ROOT / "results" / ("final" if a.final else "dev")
    OUT = ROOT / "report" / "figures" if a.final else res / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (headline, calibration, importance, storm_week, rolling_coverage, regimes, risk, intraday_profile,
              direction_pnl):
        f(res)
    leak_lab()
    print(f"figures written to {OUT}")
