"""Stormcast control room: an interactive tour of what machine learning can and cannot forecast in S&P 500 futures.

Run: uv run stormcast app   (or: uv run streamlit run app/control_room.py)
Every forecast shown is walk-forward: the models for a year were trained only on data before that year.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stormcast.engine import FEATURE_LABELS, annualized_vol  # noqa: E402
from stormcast.features import ALL_FEATURES, OPEN  # noqa: E402

from stormcast.v2 import MODELS as V2_MODELS  # noqa: E402
from stormcast.v2 import RESULTS as V2  # noqa: E402
st.set_page_config(page_title="Stormcast", page_icon="⛈️", layout="wide")

# ------------------------------------------------------------------ style
STORM, CALM, ACCENT, MUTED, UP, DOWN = "#ff6b6b", "#4dabf7", "#ffd43b", "#868e96", "#51cf66", "#ff8787"
MODEL_COLORS = {"har": "#868e96", "gbm_prereg": "#748ffc", "gbm_tuned": "#4dabf7", "stormnet": "#da77f2",
                "ensemble": "#ffd43b"}
MODEL_NAMES = {"har": "HAR (econometrics baseline)", "gbm_prereg": "Gradient boosting (pre-registered)",
               "gbm_tuned": "Gradient boosting + Optuna", "stormnet": "StormNet deep ensemble",
               "ensemble": "Stacked ensemble"}
st.markdown("""<style>
.block-container {padding-top: 1.6rem; max-width: 1500px}
div[data-testid="stMetricValue"] {font-size: 1.55rem}
.kicker {color:#adb5bd; font-size:0.95rem; margin-top:-0.6rem}
.kpi {padding:10px 14px; border-radius:10px; background:#161a20; border:1px solid #262b33; height:100%}
.kpi .l {color:#adb5bd; font-size:0.82rem}
.kpi .v {font-size:1.6rem; font-weight:650; line-height:1.3}
.kpi .s {font-size:0.8rem; color:#868e96}
.kpi .s.good {color:#51cf66} .kpi .s.bad {color:#ff6b6b} .kpi .s.warn {color:#ffd43b}
.tag {display:inline-block; padding:2px 8px; border-radius:6px; background:#2b3035; color:#ced4da; font-size:0.78rem;
      margin-right:6px}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s {tone}">{sub}</div></div>', unsafe_allow_html=True)


def fig_layout(fig, h=360, **kw):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08, x=0), **kw)
    return fig


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading walk-forward forecasts…")
def load_predictions() -> pd.DataFrame:
    p = pd.read_parquet(V2 / "predictions.parquet")
    p["time"] = pd.to_datetime(p.date) + pd.to_timedelta(OPEN + p.k, unit="min")
    sn = pd.read_parquet(V2 / "stormnet_predictions.parquet", columns=["row"])
    p["attn_idx"] = p.row.map(pd.Series(np.arange(len(sn)), index=sn.row.to_numpy()))
    return p


@st.cache_data
def load_attention() -> np.ndarray:
    return np.load(V2 / "stormnet_attention.npy")


@st.cache_resource(show_spinner="Loading one-minute bars…")
def load_bars():
    from stormcast.data import load_bars as lb
    b = lb()
    b["et"] = b.index.tz_convert("America/New_York").tz_localize(None)
    return b


@st.cache_data
def load_regimes() -> pd.DataFrame:
    if (V2 / "regimes_daily.parquet").exists():
        return pd.read_parquet(V2 / "regimes_daily.parquet")
    r = pd.concat([pd.read_parquet(ROOT / "results" / s / "regimes_daily.parquet") for s in ("dev", "final")])
    return r[~r.index.duplicated()]


@st.cache_data
def load_json(name: str) -> dict:
    return json.loads((V2 / name).read_text())


@st.cache_resource
def gbm_for(year: int):
    import joblib
    return joblib.load(V2_MODELS / str(year) / "gbm.joblib")


P = load_predictions()
S = P[P.standard]
METRICS = pd.read_csv(V2 / "metrics.csv")
SUMMARY = load_json("summary.json")

# ------------------------------------------------------------------ header
st.markdown("# ⛈️ Stormcast")
st.markdown('<p class="kicker">A weather forecast for the stock market. Machine learning on 3.1 million one-minute '
            'E-mini S&P 500 bars (2010–2019): it forecasts <b>how much</b> the market will move in the next 30 '
            'minutes, and shows why it cannot forecast <b>which way</b>.</p>', unsafe_allow_html=True)
allm = METRICS[(METRICS.scope == "all")]
r2 = allm.dropna(subset=["r2"]).set_index("model").r2
iv = allm[allm.model == "intervals"].iloc[0]
c = st.columns(5)
kpi(c[0], "Size of the next move: R²", f"{r2['ensemble']:.2f}", f"{r2['ensemble'] - r2['har']:+.3f} vs HAR econometrics",
    "good")
kpi(c[1], "Direction of the next move: AUC", f"{iv.dir_auc:.3f}", "a coin flip scores 0.500", "warn")
kpi(c[2], "90% interval coverage", f"{iv.aci_coverage:.1%}", "adaptive conformal, target 90%")
kpi(c[3], "Forecasts scored", f"{len(S):,}", f"{S.date.nunique():,} trading days, walk-forward")
n_seeds = len(next(iter(load_json("stormnet_training.json").values()))["members_best_val"])
kpi(c[4], "Stacked models", f"3 <span style='font-size:1rem;color:#adb5bd'>+ {n_seeds}-seed deep ensemble</span>",
    "HAR · GBM + Optuna · StormNet")

tabs = st.tabs(["🌩️ Storm replay", "🏟️ Model arena", "🎯 Can we trust it?", "🛡️ Risk desk", "🧪 Leak lab",
                "🧠 How it works"])

# ================================================================== 1. replay
with tabs[0]:
    PRESETS = {"Volmageddon (5 Feb 2018)": "2018-02-05", "China devaluation crash (24 Aug 2015)": "2015-08-24",
               "Brexit vote (24 Jun 2016)": "2016-06-24", "Christmas Eve sell-off (24 Dec 2018)": "2018-12-24",
               "Trump election (9 Nov 2016)": "2016-11-09", "Calm summer day (14 Jul 2017)": "2017-07-14"}
    days = np.sort(P.date.unique())
    top = st.columns([2.2, 1.3, 3])
    preset = top[0].selectbox("Pick a day", ["Custom…"] + list(PRESETS), index=1)
    if preset == "Custom…":
        lo_d, hi_d = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
        day = pd.Timestamp(top[1].date_input("Date", min(max(pd.Timestamp("2018-02-06"), lo_d), hi_d),
                                             min_value=lo_d, max_value=hi_d))
        if day.to_datetime64() not in set(days):
            day = pd.Timestamp(days[np.searchsorted(days, day.to_datetime64()).clip(0, len(days) - 1)])
            top[2].caption(f"Not a trading day in the data; showing {day:%d %b %Y}.")
    else:
        day = pd.Timestamp(PRESETS[preset])
        if day.to_datetime64() not in set(days):
            top[2].caption(f"{day:%d %b %Y} has no forecasts yet (run `stormcast train`); showing the last day.")
            day = pd.Timestamp(days[-1])
    D = P[P.date == day].sort_values("k").reset_index(drop=True)
    times = [f"{(OPEN + k) // 60:02d}:{(OPEN + k) % 60:02d}" for k in D.k]

    if "slot" not in st.session_state or st.session_state.get("slot_day") != day:
        st.session_state.slot, st.session_state.slot_day = times[min(len(times) - 1, 18)], day
    playing = st.session_state.get("playing", False)
    if playing:
        i = times.index(st.session_state.slot) if st.session_state.slot in times else 0
        if i + 1 >= len(times):
            st.session_state.playing = playing = False
        else:
            st.session_state.slot = times[i + 1]
    b = top[2].columns([5, 1, 1])
    slot = b[0].select_slider("Decision time (ET)", options=times, key="slot")
    if b[1].button("▶" if not playing else "⏸", help="Play the day forward, 5 minutes per step"):
        st.session_state.playing = not playing
        st.rerun()
    if b[2].button("⏮", help="Back to 10:00"):
        st.session_state.slot, st.session_state.playing = times[0], False
        st.rerun()
    r = D.iloc[times.index(slot)]
    year = int(r.test_year)
    st.markdown(f'<span class="tag">walk-forward: models trained on data before {year}</span>'
                f'<span class="tag">{"post hoc year" if year >= 2016 else "development year"}</span>',
                unsafe_allow_html=True)

    bars = load_bars()
    B = bars[(bars.session == day) & (bars.minute >= OPEN) & (bars.minute < OPEN + 390)]
    t0 = pd.Timestamp(day) + pd.Timedelta(minutes=OPEN + int(r.k))
    past, fut = B[B.et < t0], B[(B.et >= t0) & (B.et < t0 + pd.Timedelta(minutes=30))]
    later = B[B.et >= t0 + pd.Timedelta(minutes=30)]
    regimes = load_regimes()
    reg = regimes.loc[day] if day in regimes.index else None
    sd_pts = r.price * np.exp(r.ensemble / 2)
    typical = np.exp(D.lrv_slot / 2).median() * r.price
    size_mult = float(np.exp(S[S.test_year == year].ensemble / 2).median() * r.price / sd_pts)

    m = st.columns(6)
    excess = annualized_vol(r.ensemble) - annualized_vol(r.lrv_slot)
    kpi(m[0], "Forecast volatility, next 30 min", f"{annualized_vol(r.ensemble):.0f}%",
        f"{excess:+.0f} pts vs usual at {slot}", "bad" if excess > 3 else "good" if excess < -3 else "")
    kpi(m[1], "Actual (revealed 30 min later)", f"{annualized_vol(r.y_lrv):.0f}%", "annualized, like the VIX")
    kpi(m[2], "VIX, yesterday's close", f"{np.exp(r.lvix):.1f}", "what the options market expected")
    kpi(m[3], "90% interval for the range", f"{r.range_lo_pts:.1f}–{r.range_hi_pts:.1f} pts",
        f"actual {r.range_pts:.1f} pts {'✓ inside' if r.covered else '✗ outside'}", "good" if r.covered else "bad")
    kpi(m[4], "P(up in 30 min)", f"{r.stormnet_p_up:.2f}", "a coin flip, honestly reported", "warn")
    kpi(m[5], "Suggested position size", f"{min(size_mult, 3):.2f}×", "of a normal day's size (equal risk)")

    left, right = st.columns([2.1, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=past.et, y=past.close, line=dict(color="#e9ecef", width=1.6), name="price"))
        fig.add_trace(go.Scatter(x=pd.concat([fut.et, later.et]), y=pd.concat([fut.close, later.close]),
                                 line=dict(color="#495057", width=1.2), name="what happened next"))
        tc = [t0, t0 + pd.Timedelta(minutes=30)]
        for half, op, nm in ((r.range_hi_pts, 0.18, "90% envelope"), (sd_pts, 0.32, "±1 sd move")):
            fig.add_trace(go.Scatter(x=tc + tc[::-1], y=[r.price + half] * 2 + [r.price - half] * 2, fill="toself",
                                     fillcolor=f"rgba(255,107,107,{op})", line=dict(width=0), name=nm, mode="lines",
                                     hoverinfo="skip"))
        if len(fut):
            fig.add_trace(go.Scatter(x=fut.et, y=fut.close, line=dict(color=ACCENT, width=2), name="next 30 min"))
        fig.add_vline(x=t0, line=dict(color=ACCENT, dash="dot", width=1))
        fig_layout(fig, 430, title=f"ES futures {day:%A %d %B %Y}: forecast cone at {slot}",
                   yaxis_title="index points")
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D.time, y=annualized_vol(D.lrv_slot), name="usual for this time of day",
                                 line=dict(color=MUTED, dash="dot")))
        known = D.time <= t0
        revealed = D.time <= t0 - pd.Timedelta(minutes=30)
        fig.add_trace(go.Scatter(x=D.time[known], y=annualized_vol(D.ensemble[known]), name="Stormcast forecast",
                                 line=dict(color=ACCENT, width=2.5)))
        fig.add_trace(go.Scatter(x=D.time[revealed], y=annualized_vol(D.y_lrv[revealed]),
                                 name="actual (once revealed)", mode="markers", marker=dict(color=STORM, size=6)))
        fig.add_vline(x=t0, line=dict(color=ACCENT, dash="dot", width=1))
        fig_layout(fig, 280, title="Volatility through the day (annualized %, like the VIX): only what was known "
                                   f"at {slot}", xaxis=dict(range=[D.time.min(), D.time.max()]))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if reg is not None:
            probs = {"calm": reg.p_calm, "normal": reg.p_normal, "storm": reg.p_storm}
            fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs), orientation="h",
                                   marker_color=[CALM, MUTED, STORM], text=[f"{v:.0%}" for v in probs.values()],
                                   textposition="auto"))
            fig_layout(fig, 170, title="Market regime (hidden Markov model, known at the open)",
                       xaxis=dict(range=[0, 1], visible=False))
            st.plotly_chart(fig, use_container_width=True)
        sh = pd.Series({f: r[f"shap_{f}"] for f in ALL_FEATURES})
        top8 = sh.reindex(sh.abs().sort_values(ascending=False).index[:8])[::-1]
        fig = go.Figure(go.Bar(x=top8.values, y=[FEATURE_LABELS[f] for f in top8.index], orientation="h",
                               marker_color=[STORM if v > 0 else CALM for v in top8.values]))
        fig_layout(fig, 330, title="Why? Top drivers of this forecast (SHAP)",
                   xaxis_title="← calmer   ·   stormier →")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Model disagreement (epistemic sd): {r.stormnet_epistemic_sd:.3f} · "
                   f"HAR {annualized_vol(r.har):.0f}% · GBM {annualized_vol(r.gbm_tuned):.0f}% · "
                   f"StormNet {annualized_vol(r.stormnet):.0f}%")

    with st.expander("🔍 What StormNet looked at: attention over the last 4 hours of bars", expanded=False):
        if pd.notna(r.attn_idx):
            att = load_attention()[int(r.attn_idx)].astype(float)
            win = bars[bars.index < pd.Timestamp(r.decision_utc)].iloc[-240:]
            ax = win.et.iloc[::4].to_numpy()[: len(att)]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=ax, y=att, name="attention weight", marker_color="rgba(218,119,242,0.55)"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=win.et, y=win.close, name="price", line=dict(color="#e9ecef", width=1.2)),
                          secondary_y=True)
            fig_layout(fig, 300, title="Attention pooling weights (4-minute tokens), with price")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The Transformer's pooling layer decides which part of the window to summarize. Bursts of "
                       "activity, the open and the overnight session tend to attract weight.")

    with st.expander("🎛️ What if…? Change the inputs and re-run the gradient-boosting model live"):
        g = gbm_for(year)
        x = r[ALL_FEATURES].astype(float).copy()
        w = st.columns(4)
        vix = w[0].slider("VIX", 9.0, 50.0, float(np.exp(r.lvix)), 0.5)
        mult = w[1].slider("Last 30 min volatility ×", 0.25, 4.0, 1.0, 0.25)
        fomc = w[2].toggle("FOMC day", bool(r.fomc))
        hh = w[3].select_slider("Time of day", options=times, value=slot)
        x["lvix"], x["lrv_30"], x["fomc"] = np.log(vix), x["lrv_30"] + 2 * np.log(mult), float(fomc)
        x["tod"] = float(D.k.iloc[times.index(hh)])
        base, new = g.predict(r[ALL_FEATURES].astype(float).to_frame().T)[0], g.predict(x.to_frame().T)[0]
        kpi(st, "Gradient-boosting forecast with your inputs", f"{annualized_vol(new):.0f}% annualized",
            f"{annualized_vol(new) - annualized_vol(base):+.1f} pts vs the real inputs ({annualized_vol(base):.0f}%)")

    if playing:
        time.sleep(0.7)
        st.rerun()

# ================================================================== 2. arena
with tabs[1]:
    st.markdown("### Five forecasters, one honest walk-forward test")
    st.caption("Each year is forecast by models trained only on earlier years. 2012–2015 are development years; "
               "2016–2019 were the pre-registered one-shot test, so every v2 number there is labelled post hoc.")
    vm = METRICS.dropna(subset=["r2"])
    order = ["har", "gbm_prereg", "gbm_tuned", "stormnet", "ensemble"]
    tbl = vm[vm.scope.isin(["all", "dev 2012-15", "post hoc 2016-19"])].pivot(index="model", columns="scope",
                                                                               values="r2").reindex(index=order, columns=["all", "dev 2012-15", "post hoc 2016-19"])
    q = vm[vm.scope == "all"].set_index("model").qlike.reindex(order)
    dm = pd.read_csv(V2 / "dm.csv")
    dmt = dm[dm.scope == "all"].set_index("model").t.reindex(order)
    board = pd.DataFrame({"Model": [MODEL_NAMES[m] for m in order],
                          "R² all years": tbl["all"].round(3).values,
                          "R² dev 2012–15": tbl["dev 2012-15"].round(3).values,
                          "R² 2016–19 (post hoc)": tbl["post hoc 2016-19"].round(3).values,
                          "QLIKE ↓": q.round(3).values,
                          "Diebold–Mariano t vs HAR": dmt.round(1).values})
    st.dataframe(board, hide_index=True, use_container_width=True)
    a1, a2 = st.columns(2)
    with a1:
        yr = vm[vm.scope.str.fullmatch(r"\d{4}")]
        fig = go.Figure()
        for mname in order:
            g = yr[yr.model == mname]
            fig.add_trace(go.Bar(x=g.scope, y=g.r2, name=MODEL_NAMES[mname], marker_color=MODEL_COLORS[mname]))
        fig_layout(fig, 380, title="R² by year (log realized variance, next 30 minutes)", barmode="group",
                   yaxis=dict(range=[0.3, 0.9]))
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        wts = load_json("stack_weights.json")
        W = pd.DataFrame({y: v["weights"] for y, v in wts.items()}).T
        fig = go.Figure()
        for mname in W.columns:
            fig.add_trace(go.Bar(x=W.index, y=W[mname], name=MODEL_NAMES[mname], marker_color=MODEL_COLORS[mname]))
        fig_layout(fig, 380, title="Stacking weights, refit each year on earlier out-of-sample forecasts",
                   barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    b1, b2 = st.columns(2)
    with b1:
        tune = load_json("gbm_tuning.json")
        fig = go.Figure()
        for y, v in tune.items():
            best = np.minimum.accumulate(v["trial_mse"])
            fig.add_trace(go.Scatter(y=best / best[0], name=y, mode="lines"))
        fig_layout(fig, 330, title="Optuna search inside each training window (best MSE ÷ pre-registered setting)",
                   xaxis_title="trial", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with b2:
        tr = load_json("stormnet_training.json")
        fig = go.Figure()
        for y, v in tr.items():
            for j, hist in enumerate(v["members_history"]):
                fig.add_trace(go.Scatter(y=hist, mode="lines", name=y if j == 0 else None, showlegend=j == 0,
                                         legendgroup=y, line=dict(width=1)))
        fig_layout(fig, 330, title="StormNet validation loss per member (early stopping on the last 10% by time)",
                   xaxis_title="epoch")
        st.plotly_chart(fig, use_container_width=True)

# ================================================================== 3. trust
with tabs[2]:
    st.markdown("### A forecast is only useful if you know when to trust it")
    t1, t2 = st.columns(2)
    with t1:
        roll = S.set_index("time")[["covered", "raw_covered"]].rolling(250).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.raw_covered, name="raw network quantiles",
                                 line=dict(color=MUTED, width=1)))
        fig.add_trace(go.Scatter(x=roll.index, y=roll.covered, name="+ adaptive conformal (ACI)",
                                 line=dict(color=ACCENT, width=1.6)))
        fig.add_hline(y=0.9, line=dict(color=STORM, dash="dot"))
        fig_layout(fig, 360, title="Rolling coverage of 90% range intervals (250 forecasts)",
                   yaxis=dict(range=[0.4, 1.0], tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ACI widens the interval after every miss and narrows it after every hit, so it stays near 90% "
                   "through Aug 2015, Feb 2018 and Dec 2018. Outcomes are fed back only once they are known.")
    with t2:
        z = (S.y_lrv - S.stormnet) / S.stormnet_sd
        bins = pd.qcut(S.stormnet_sd, 10)
        cal = pd.DataFrame({"pred": S.stormnet_sd, "err": (S.y_lrv - S.stormnet).abs()}).groupby(bins, observed=True).agg(
            pred=("pred", "mean"), rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal.pred, y=cal.rmse, mode="markers+lines", name="StormNet",
                                 marker=dict(color="#da77f2", size=9)))
        lim = [cal.min().min() * 0.9, cal.max().max() * 1.1]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="perfect", line=dict(color=MUTED, dash="dot")))
        fig_layout(fig, 360, title="Does the model know when it doesn't know? (deciles of predicted sd)",
                   xaxis_title="predicted sd (aleatoric + epistemic)", yaxis_title="actual error (RMSE)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{(z.abs() <= 1.645).mean():.1%} of outcomes fall inside the model's own 90% band "
                   "(target 90%).")
    vt = pd.qcut(S.lvix, 3, labels=["low VIX", "mid VIX", "high VIX"])
    cov = S.groupby(vt, observed=True)[["raw_covered", "covered"]].mean()
    yrc = S.groupby("test_year")[["raw_covered", "covered"]].mean()
    t3, t4 = st.columns(2)
    for col, data, ttl in ((t3, cov, "Coverage by VIX tercile"), (t4, yrc, "Coverage by year")):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data.index.astype(str), y=data.raw_covered, name="raw quantiles", marker_color=MUTED))
        fig.add_trace(go.Bar(x=data.index.astype(str), y=data.covered, name="+ ACI", marker_color=ACCENT))
        fig.add_hline(y=0.9, line=dict(color=STORM, dash="dot"))
        fig_layout(fig, 300, title=ttl, barmode="group", yaxis=dict(range=[0.5, 1], tickformat=".0%"))
        col.plotly_chart(fig, use_container_width=True)

# ================================================================== 4. risk
with tabs[3]:
    st.markdown("### What a volatility forecast is worth: surviving a daily loss limit")
    st.caption("Hold one long ES position every 30 minutes, 2012–2019. Constant size vs size ∝ 1 / forecast "
               "volatility, both scaled to the same average size. This measures risk control, not an edge.")
    k1, k2 = st.columns([1, 3])
    lim_mult = k1.slider("Daily loss limit (× average daily swing)", 1.5, 5.0, 3.0, 0.25)
    sizer = k1.radio("Size by", ["ensemble", "stormnet", "har"], format_func=lambda m: MODEL_NAMES[m])
    u = 1 / (np.exp(S[sizer] / 2) * S.price)
    u = u / u.mean()
    oracle = 1 / (np.exp(S.y_lrv / 2) * S.price)
    oracle = oracle / oracle.mean()
    pnl = pd.DataFrame({"constant size": S.move_pts.to_numpy(), "forecast-sized": (u * S.move_pts).to_numpy(),
                        "oracle (knows the future)": (oracle * S.move_pts).to_numpy()}, index=S.date.to_numpy())
    daily = pnl.groupby(level=0).sum()
    limit = lim_mult * daily["constant size"].abs().mean()
    br = (daily < -limit).sum()
    kpi(k1, "Limit breaches, constant size", f"{br['constant size']} days",
        f"{br['constant size'] / len(daily):.1%} of {len(daily):,} days", "bad")
    k1.write("")
    kpi(k1, "Limit breaches, forecast-sized", f"{br['forecast-sized']} days",
        f"{(br['forecast-sized'] - br['constant size']) / max(br['constant size'], 1):+.0%} vs constant size", "good")
    with k2:
        roll = daily.rolling(60).std().dropna()
        fig = go.Figure()
        for name, color in (("constant size", MUTED), ("forecast-sized", ACCENT), ("oracle (knows the future)", CALM)):
            fig.add_trace(go.Scatter(x=roll.index, y=roll[name], name=name,
                                     line=dict(color=color, width=2 if name != "oracle (knows the future)" else 1,
                                               dash="dot" if name.startswith("oracle") else None)))
        fig_layout(fig, 300, title="Risk you are actually running: 60-day standard deviation of daily P&L (points)")
        st.plotly_chart(fig, use_container_width=True)
        worst = daily.nsmallest(40, "constant size").sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=worst.index.strftime("%Y-%m-%d"), y=worst["constant size"], name="constant size",
                             marker_color=MUTED))
        fig.add_trace(go.Bar(x=worst.index.strftime("%Y-%m-%d"), y=worst["forecast-sized"], name="forecast-sized",
                             marker_color=ACCENT))
        fig.add_hline(y=-limit, line=dict(color=STORM, dash="dash"), annotation_text="daily loss limit")
        fig_layout(fig, 300, title="The 40 worst days for constant sizing, and the same days forecast-sized",
                   barmode="group", xaxis=dict(type="category", tickangle=-60))
        st.plotly_chart(fig, use_container_width=True)
    stats = pd.DataFrame({"daily sd (pts)": daily.std(), "worst day (pts)": daily.min(),
                          "breach days": br, "kurtosis (slot P&L)": pnl.kurt()}).round(2)
    st.dataframe(stats, use_container_width=True)

# ================================================================== 5. leak lab
with tabs[4]:
    st.markdown("### How easy is it to fool yourself? Six planted mistakes")
    lk = pd.read_csv(ROOT / "results" / "leak_lab" / "leak_lab.csv")
    l1, l2 = st.columns([1.3, 1])
    with l1:
        auc = lk[lk.metric == "AUC"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=auc.leak, x=auc.honest, orientation="h", name="honest pipeline", marker_color=CALM))
        fig.add_trace(go.Bar(y=auc.leak, x=auc.leaky, orientation="h", name="with the mistake", marker_color=STORM))
        fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dot"))
        fig_layout(fig, 360, title="Direction AUC: honest vs leaky (development years)", barmode="group",
                   xaxis=dict(range=[0.45, 1.0]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with l2:
        st.dataframe(lk[["leak", "what", "honest", "leaky"]].round(3), hide_index=True, use_container_width=True,
                     height=300)
    st.markdown("#### Live experiment: the same model, two ways to validate it")
    st.caption("Dense 5-minute rows have overlapping 30-minute labels. A random forest is scored on 2014–2015 "
               "direction twice: shuffled 5-fold cross-validation (neighbours leak into each other) vs walk-forward "
               "by time.")

    @st.cache_data(show_spinner="Training two random forests (about 20 seconds)…", persist="disk")
    def live_leak():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold, cross_val_predict
        from stormcast.v2 import load_dense
        d = load_dense()
        d = d[d.date.dt.year.isin([2014, 2015])].sort_values("decision_utc")
        X, y = d[ALL_FEATURES].to_numpy(), d.y_dir.to_numpy()
        rf = RandomForestClassifier(200, min_samples_leaf=1, max_features=0.5, n_jobs=-1, random_state=0)
        shuf = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        tr = d.date.dt.year.to_numpy() == 2014
        wf = rf.fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
        return roc_auc_score(y, shuf), roc_auc_score(y[~tr], wf), len(d)

    if st.button("Run the experiment", type="primary"):
        a_shuf, a_wf, n = live_leak()
        e = st.columns(3)
        kpi(e[0], "Shuffled 5-fold cross-validation", f"AUC {a_shuf:.3f}", "looks like a money machine", "bad")
        kpi(e[1], "Walk-forward (train 2014 → test 2015)", f"AUC {a_wf:.3f}", "the truth: a coin flip", "good")
        kpi(e[2], "Rows", f"{n:,}", "same model, same features, same data")
    st.markdown("#### And the honest direction model?")
    dd = S.sort_values("time")
    side = np.where(dd.stormnet_p_up > 0.5, 1, -1)
    gross = (side * dd.move_pts).cumsum().to_numpy()
    cost = 0.70 * np.arange(1, len(dd) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.time, y=gross, name="gross points from trading StormNet's direction head",
                             line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=dd.time, y=gross - cost, name="after 0.70 pts round-trip cost",
                             line=dict(color=STORM)))
    lo, hi = SUMMARY["stormnet_dir_auc_ci"]
    fig_layout(fig, 330, title=f"Direction AUC {SUMMARY['stormnet_dir_auc_all']:.3f} (95% CI {lo:.3f}–{hi:.3f}): "
                               "statistically faint, economically zero")
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== 6. how it works
with tabs[5]:
    st.markdown("""
### Pipeline
1. **Data**: 3.1M one-minute ES futures bars (Databento, 2010–2019), prior-day VIX, FOMC calendar. Every feature is
   computed only from bars that closed before the decision, and a unit test rebuilds features from truncated data
   to prove it.
2. **Features**: 25 point-in-time features (HAR volatility components, time-of-day profile, VIX, ranges, returns,
   volume, calendar) plus two pure-noise columns as a sanity check.
3. **Models** (walk-forward: each year trained only on earlier years, one-day embargo):
   - **HAR**: the econometrics benchmark (Corsi 2009).
   - **Gradient boosting + Optuna**: hyperparameters searched by TPE *inside* each training window; exact TreeSHAP
     explanations for every forecast.
   - **StormNet**: dilated temporal convolutions + Transformer self-attention over the last 240 one-minute bars,
     fused with the tabular features. Multi-task heads: heteroscedastic volatility (Gaussian NLL), non-crossing
     range quantiles (pinball loss) and direction. Deep ensemble of seeds for epistemic uncertainty. Trained on
     Apple-silicon GPU.
   - **Stacked ensemble**: non-negative weights refit each year on earlier out-of-sample forecasts.
4. **Uncertainty**: adaptive conformal inference with delayed feedback turns the network's quantiles into intervals
   that keep 90% coverage when volatility jumps.
5. **Regimes**: a 3-state Gaussian HMM read only through filtered (forward) probabilities.

### The one-sentence result
On identical inputs, ML explains about four-fifths of the variation in the *size* of the next 30-minute move and
essentially none of its *direction*: the efficient-market hypothesis, measured.
""")
    st.caption("v2 models were designed after the pre-registered final results (report/report.md); their 2016–2019 "
               "numbers are post hoc.")
