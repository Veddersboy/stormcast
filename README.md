# ⛈ Stormcast

**A weather forecast for the stock market.** Stormcast uses machine learning on 3.1 million one-minute E-mini S&P 500
futures bars to forecast *how much* the market will move in the next 30 minutes, with calibrated intervals,
explanations and model uncertainty. It also shows, with the same machinery, why it cannot forecast *which way*.

| On identical inputs | Result (walk-forward, 2012–2019) |
|---|---|
| **Size** of the next 30-minute move (log realized variance) | R² **0.79**, beating HAR econometrics in every year |
| **Direction** of the next 30-minute move | AUC **0.503**: a coin flip |
| 90% range intervals (adaptive conformal) | **89.5%** coverage, including Feb 2018 and Dec 2018 |
| Position sizing by the forecast | **89%** fewer daily loss-limit breaches at the same average size |

## Try it in two commands (no data needed)
The real market data is licensed, so it isn't included. Stormcast ships with a **synthetic market**: a simulator with the
same statistical structure as E-mini futures (volatility clustering, intraday U-shape, calm/normal/storm regimes, FOMC
bursts, rare crashes, fat tails) whose direction is unpredictable by construction.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (macOS/Linux:
   `curl -LsSf https://astral.sh/uv/install.sh | sh`; Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`).
2. In this folder, run:
   ```bash
   uv run stormcast demo
   ```
   It installs the pinned dependencies (Python 3.11–3.12, PyTorch, scikit-learn…) on first run, then opens the web app at
   http://localhost:8000.

If this copy came with a `sandbox/` folder, it already holds the synthetic market and models trained on it, so the app
opens straight away. Without one (e.g. a fresh git clone), `stormcast demo` generates the market (about 30 seconds) and
trains a quick model set walk-forward. That takes about 10 minutes on an Apple-silicon or CUDA GPU, but an hour or more
on a CPU, because the StormNet deep network is the slow part. `--full` trains with the real-data settings and
`--retrain` starts over.

Everything the demo creates lives in `sandbox/`. Delete that folder to reset.

## Quick start (with the real data)
```bash
uv sync
uv run stormcast serve                       # web app at http://localhost:8000
uv run stormcast forecast "2018-02-05 15:00" # forecast any moment from the terminal
uv run stormcast replay 2018-02-05           # stream a day's forecasts against what happened
```
The web app has seven views (keys `1`–`7`): **Storm replay** (play any day 2012–2019 minute by minute, with the
forecast cone, SHAP drivers, attention weights, regime and a live what-if model), **Model arena**, **Trust**
(conformal coverage and uncertainty calibration), **Risk desk**, **Leak lab** (a live experiment showing how shuffled
cross-validation fakes skill), **Forecast** (ask the engine about any moment) and **How it works**. The 10-minute demo
script is in [`docs/demo.md`](docs/demo.md).

The raw market data is licensed and not in this repository; see [Data](#data).

## How it works
```
1-min ES bars ─┬─ 25 point-in-time features ─┬─ HAR (econometrics baseline)
VIX, FOMC ─────┘                              ├─ Gradient boosting, Optuna-tuned inside each window ── exact TreeSHAP
last 240 bars ───────────────────────────────┴─ StormNet deep ensemble ── attention weights, uncertainty
                                                     │
                       stacked ensemble (weights fit on earlier years only)
                                                     │
          adaptive conformal intervals (delayed feedback) · HMM regimes · position sizing
```
- **StormNet** (`src/stormcast/deep.py`): dilated temporal convolutions and Transformer self-attention over the last 240
  one-minute bars, fused with the tabular features. It is multi-task: heteroscedastic volatility (Gaussian NLL),
  non-crossing range quantiles (pinball loss) and direction. A 3-seed deep ensemble adds epistemic uncertainty. Trains on
  Apple-silicon GPU (MPS), CUDA or CPU.
- **Gradient boosting + Optuna** (`boost.py`): TPE search on a time split *inside* each training window, starting from
  the pre-registered setting. Every forecast has exact TreeSHAP explanations.
- **Stacking** (`stack.py`): non-negative weights refit each year on earlier years' out-of-sample forecasts.
- **Adaptive conformal inference with delayed feedback** (`conformal.py`): forecasts every 5 minutes, outcomes known 30
  minutes later, and the interval only learns from outcomes that have been revealed.
- **Regimes** (`regimes.py`): 3-state Gaussian HMM, filtered (forward-only) probabilities.
- **Engine** (`engine.py`): forecasts any moment using only the models trained before that year. **API + frontend**
  (`server.py`, `web/`): FastAPI and a dependency-free single page (Plotly, vendored).

## Results
Walk-forward by calendar year: each year is forecast by models trained only on earlier years, with a one-day embargo.
Scored on 12 non-overlapping decisions per day (10:00–15:30 ET).

| Model | R² all years | R² 2012–15 (dev) | R² 2016–19 (post hoc) | Diebold–Mariano t vs HAR |
|---|---|---|---|---|
| HAR econometrics | 0.746 | 0.655 | 0.795 | baseline |
| Gradient boosting (pre-registered) | 0.782 | 0.700 | 0.825 | 15.7 |
| Gradient boosting + Optuna | 0.786 | 0.707 | 0.829 | 16.6 |
| StormNet deep ensemble | 0.792 | 0.718 | 0.831 | 21.8 |
| **Stacked ensemble** | **0.794** | **0.719** | **0.834** | **22.1** |

- The stacked ensemble beats both the pre-registered model and HAR in all 8 years. StormNet is the best single model in
  6 of 8 years (it ties gradient boosting in 2016 and trails it in 2019).
- Direction: StormNet's direction head scores AUC 0.503 (95% CI 0.496–0.509). Trading it loses money after costs.
- Intervals: raw network quantiles cover 81.5%; with adaptive conformal inference, 89.5% (2018: 71% → 89%).
- Uncertainty: 90.0% of outcomes fall inside the model's own 90% band (aleatoric + epistemic).
- Risk: at a daily loss limit of 3× the average daily swing, forecast sizing cuts breach days from 63 to 7 of 1,879.

**Honesty note.** Stormcast began as a pre-registered study (below) whose final test, 2016–2019, was run exactly once
from the git tag `final-frozen`. The v2 models (StormNet, tuned GBM, stacking) were built afterwards, so their
2016–2019 numbers are **post hoc**. The development years 2012–2015 are the clean comparison, and the ranking is the
same there.

## The original study (pre-registered)
**What can machine learning forecast in 1-minute S&P 500 futures data, and what can't it?** The plan
(`docs/plan.md`) fixed hypotheses, models and the one-shot final test before any model was fit. Full write-up:
[`report/report.md`](report/report.md).
- **Direction:** best AUC 0.522 (95% CI 0.511–0.533) on 2016–2019; worth 0.01 index points per trade against a 0.70
  cost.
- **Volatility:** gradient boosting R² 0.83, beating HAR in every year (Diebold–Mariano p < 10⁻³⁰).
- **Intervals:** adaptive conformal keeps 90% coverage through the February 2018 shock; model-based intervals fall to
  33–46%.
- **Regimes:** an HMM separates calm from stormy days (η² = 0.51) but says nothing about direction (η² = 0.0001).
- **Risk:** forecast-based sizing cuts daily loss-limit breaches from 4.0% of days to 0.2%.
- **Leak lab:** six planted mistakes; two of them turn a coin flip into AUC 0.76 and 0.96.

## Reproduce
```bash
uv sync
uv run python scripts/00_export_data.py   # needs the AlphaBench data folder (ALPHABENCH_DIR, default ~/alphabench)
uv run pytest                             # point-in-time, StormNet causality, conformal, stacking, HMM tests
scripts/run_all.sh                        # pre-registered study, development years 2012-2015
scripts/run_all.sh --final                # pre-registered final years: only from a clean checkout of tag final-frozen
scripts/run_all.sh --v2                   # Stormcast v2 models (about 1.5 hours on an M2 Pro)
```

| Script | What it does |
|---|---|
| `01`–`08` | The pre-registered study: direction, volatility, intervals, regimes, leak lab, risk, figures, post-hoc diagnostics |
| `09_dense_dataset.py` | Decisions every 5 minutes (150k rows) for v2 training; identical to the study's dataset on its slots |
| `10_v2_stormnet.py` | Walk-forward StormNet deep ensemble; saves each year's model |
| `11_v2_gbm.py` | Walk-forward gradient boosting with Optuna and TreeSHAP |
| `12_v2_combine.py` | HAR baseline, stacking, conformal intervals, metrics, Diebold–Mariano tests |

## Data
- ES one-minute bars, 2010-06-07 to 2019-05-21, from the companion trading project AlphaBench (Databento GLBX.MDP3,
  forward roll-adjusted). Only AlphaBench's exploration window is used; the loader refuses later data.
- Cboe VIX daily closes, prior day only. FOMC statement dates from federalreserve.gov (`docs/sources/`).

Raw data are licensed and not in the repository. Trained models (`models/`) and predictions are not committed either;
`scripts/run_all.sh --v2` rebuilds them.

Layout: `src/stormcast/` (library, engine, API, CLI), `web/` (frontend), `app/` (Streamlit analysis view), `scripts/`,
`tests/`, `results/` (metrics), `report/`, `docs/`.
