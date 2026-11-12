# Stormcast — pre-registered analysis plan

*Written and committed before any model was fit or any target was looked at. Changes after this commit are listed in
the amendments section at the bottom, with the reason and date.*

## Question
In 1-minute E-mini S&P 500 futures (ES) data, **what can machine learning forecast and what can't it?** Using the same
inputs, the same models and the same validation, we forecast two things about the next 30 minutes:
1. **Direction:** will the price be higher or lower? (classification)
2. **Volatility / "market weather":** how much will the price move, in either direction? (regression, plus prediction
   intervals)

The companion trading project (AlphaBench) found no edge in direction after costs. This project asks why: is the data
simply unpredictable, or is direction specifically the hard part?

## Data
- ES 1-minute OHLCV bars (Databento GLBX.MDP3 `ohlcv-1m`, continuous front contract, forward roll adjustment), processed
  by AlphaBench (`data/clean/ES_dev.parquet`). **Only the AlphaBench exploration window, 2010-06-07 to 2019-05-21, is
  exported.** AlphaBench's confirmation window (2019-05-29 on) and sealed holdout (2023-03-30 on) are never read; the
  loader refuses any bar after 2019-05-21. Databento-flagged degraded days are dropped.
- Cboe VIX daily closes (free, official). Point-in-time rule: day D uses only the close of the prior trading day.
- FOMC statement dates from federalreserve.gov (scheduled meetings, known in advance; `docs/sources/fomc_statements.csv`).
- Raw data are not committed (licence); the export script rebuilds everything from the AlphaBench files.

## Samples and targets
- Decision points: every 30 minutes in the cash session, 10:00 to 15:30 ET (12 per day). Information: every bar that
  closed before the decision time. Horizon: the next 30 one-minute bars. Horizons never overlap.
- Direction target: 1 if the close 30 minutes later is above the last close before the decision, else 0.
- Volatility target: log realized variance of the next 30 one-minute log returns.
- Range target (intervals): log of the high-low range of the next 30 minutes, in index points.
- A decision point is dropped if fewer than 25 of the 30 bars in its look-back or horizon exist (early closes, gaps).

## Features (all computed from data before the decision time; truncation-tested)
Recent volatility (last 30 min, today so far, overnight session, prior day, prior 5 and 22 days — the HAR components),
the same-slot volatility profile over the prior 20 days, today's range so far, prior-day VIX close and its change, the
overnight gap, returns over 5 / 30 / 60 minutes and since the open, prior-day return, distance from VWAP, position in
today's range, relative volume for the time of day, time of day, day of week, an FOMC-day flag, and two pure-noise
columns (sanity check).

## Validation
- **Walk-forward by calendar year**, expanding window: each test year is predicted by models trained only on earlier
  data, with a 1-trading-day embargo. All scaling and fitting happens inside the training window.
- **Development years:** 2012, 2013, 2014, 2015. Used to build and debug the code.
- **Final years:** 2016, 2017, 2018, 2019 (to 2019-05-21). Run **once**, after the code is frozen and tagged
  `final-frozen`. Results are reported whatever they are; no changes afterwards.

## Models (fixed hyperparameters, no tuning)
| Task | Models |
|---|---|
| Volatility | persistence (last 30 min), time-of-day profile, HAR-intraday linear regression, ridge (all features), gradient boosting, MLP |
| Direction | base rate, logistic regression, gradient boosting, MLP, 1D-CNN on the raw last 120 one-minute returns and volumes |
| Range intervals (90%) | quantile gradient boosting; + split conformal (CQR); + adaptive conformal (ACI) |
| Regimes | 3-state Gaussian HMM on daily volatility, return and VIX, filtered (forward) probabilities only |

Gradient boosting: sklearn HistGradientBoosting, 300 iterations, learning rate 0.05, 15 leaves, min 200 samples per leaf,
L2 = 1.0. MLP: (64, 32), alpha 1e-3, early stopping on the last 10% of the training window. CNN: 2 conv layers
(16, 32 channels, kernel 5), global average pooling, dropout 0.2, Adam 1e-3, weight decay 1e-4, up to 15 epochs with
early stopping on the last 10% of the training window (by time). Seeds fixed.

## Hypotheses and how they are judged (final years, pooled unless stated)
- **H1 (direction is unpredictable):** every direction model's AUC is within 0.48–0.52 and its 95% day-block bootstrap
  interval contains 0.50. Log-loss skill vs the base rate is ≤ 0. Translated into trades (long if p > 0.5, else short,
  hold 30 minutes), the mean gross move captured is below the 0.70-point round-trip cost used by AlphaBench.
- **H2 (volatility is predictable):** the best volatility model has out-of-sample R² ≥ 0.40 on log realized variance.
- **H3 (does ML beat the classical model?):** gradient boosting vs HAR-intraday, Diebold–Mariano test on daily-averaged
  squared errors (Newey–West, 5 lags). Reported either way; "ML helps" only if p < 0.05 and the sign agrees in at least
  3 of 4 final years.
- **H4 (intervals you can trust):** adaptive conformal intervals reach 90% ± 1.5% coverage, and their worst rolling
  coverage (250 decision points) is higher than that of static conformal intervals.
- **H5 (leaks look like skill):** planted leaks (centered window, off-by-one bar, overlapping labels with shuffled K-fold,
  smoothed HMM states, full-sample scaling, best-of-many selection) are run on the development years only; we expect
  the first four to push AUC well above 0.55 while the honest pipeline stays near 0.50, full-sample scaling to change
  little, and best-of-many selection to regress to 0.50 on fresh years.
- **H6 (regimes):** the filtered HMM state separates next-day volatility strongly but not next-day direction.
- **H7 (forecasts are useful for risk):** sizing a constant-direction position by the volatility forecast makes risk per
  trade more stable than constant sizing (lower kurtosis of standardized P&L, fewer days breaching a fixed daily loss
  limit at equal average size).

## Sanity checks (stop rules)
- Label shuffle: models trained on shuffled training labels must score AUC ≈ 0.50.
- Noise features must rank in the bottom quarter of permutation importance for the volatility models.
- **Too good = bug:** an honest direction AUC above 0.55 stops the analysis until the cause is found.

## Amendments
All made on 2026-11-04 during development (years 2012-2015 only), before the final years were run.

- **A1 — HMM fitting.** The first fit on raw daily features gave poorly separated states (the return column's scale is
  about 1/1000 of the others). The HMM inputs are now standardized with training-window statistics, and EM is run from
  5 starting seeds (best training log-likelihood kept). States are ordered by mean log volatility.
- **A2 — what "separates strongly" means for H6.** Not quantified in the original plan. Fixed as: Kruskal–Wallis
  p < 0.001 across states for next-day log volatility, and p > 0.05 for next-day return (effect sizes, eta squared,
  reported too). Fixed after seeing the development-year numbers, so H6 on the final years is the test that counts.
- **A3 — leak lab, supplementary L6b.** L6 as planned barely moves (best-of-200 AUC 0.511 on the selection years vs
  0.512 on fresh years): with ~6,000 test points per two years the null AUC spread is tiny. Added, after seeing L6, a
  supplementary demonstration in trading terms: the same 200 models as long/short rules, best one picked inside each
  calendar quarter by Sharpe, then run in the next quarter. Labelled as added post hoc wherever reported.
- **A4 — details not stated in the plan, now fixed:** logistic regression C = 0.1 (as in AlphaBench); CNN early-stopping
  patience 3 epochs; MLP patience 10 epochs; conformal calibration set = last 20% of each training window by time; ACI
  quantile window = calibration-set size, reset each test year; risk sizing uses units ∝ 1 / (forecast SD in points),
  rescaled to an average of 1 contract per scheme, and H7's daily loss limit is 3 × the mean absolute daily P&L of
  constant sizing (2× and 4× also reported).
