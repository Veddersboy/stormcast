# Stormcast: 10-minute demo script

**Story in one line:** the same machine learning that cannot tell you *which way* the market will go can tell you
*how much* it will move, and that second answer is genuinely useful.

## Before you start (5 minutes ahead)
```bash
uv run stormcast serve        # opens http://localhost:8000
```
- Open **Leak lab** (key 5) once and click **Run the experiment**. The result is cached to disk, so during the demo the
  click is instant.
- Back on **Storm replay** (key 1), select **Volmageddon** and press ⏮ (or ← until 10:00). Press **F** for full screen.
- Keys: 1–7 switch views, space plays/pauses, ←/→ step 5 minutes, R hides/reveals what happened next.
- Have a terminal ready in the repo for the closing CLI shot.
- Browser zoom ~90% so the replay tab fits on one screen.

## Run of show

| Time | Where | What to do | What to say |
|---|---|---|---|
| 0:00 | Header | Point at the two left KPI cards | "Two questions about the next 30 minutes of S&P 500 futures: how much will it move, and which way? Same data, same features, same models. Size: R² 0.79. Direction: AUC 0.503, a coin flip. That gap is the whole talk." |
| 1:00 | Storm replay, 5 Feb 2018 | Press R to hide the outcome, then space to play (2× speed) | "This is a real day, replayed. Every forecast comes from models trained only on data before 2018, so nothing here has seen this day." Let it play: the cone widens through the afternoon. |
| 2:00 | Same | Pause around 14:30 to 15:00 | "The market fell about 4% in an afternoon. By 14:35 the report reads Storm warning: volatility forecast 19% against a usual 8%, position size cut to 0.44×. Then the crash breaks even the 90% envelope, and the app marks that miss honestly. Notice the regime model, which only knows yesterday, still said 'normal'; the intraday model is what reacted." Point at **P(up) = 0.50**: "Notice it never pretends to know the direction." |
| 2:45 | SHAP panel | Read the top two bars | "Why does it expect a storm? The last 30 minutes were violent, VIX was high, it's late in the day. Every driver has an economic reason; the two pure-noise features we planted rank last." |
| 3:15 | Attention panel (below) | Scroll down | "StormNet is a convolution + Transformer network reading the last four hours of minute bars. These are its attention weights: which part of the window it summarized." |
| 3:45 | Calm summer day preset | Switch day | "Contrast: a calm July day. Narrow cone, sunny report, and the suggested size goes above 1×. That is the product: a weather forecast you can size risk with." |
| 4:15 | What-if panel | Drag VIX to 40, then the last-30-minute volatility to ×4 | "Live inference: change the inputs, the forecast responds the way a trader would expect." |
| 5:00 | Model arena | Leaderboard + R² by year | "Five forecasters, one honest walk-forward test: HAR is the econometrics benchmark, then gradient boosting tuned by Optuna *inside* each training window, the StormNet deep ensemble, and a stacked ensemble refit each year on earlier out-of-sample forecasts. The stacked ensemble beats both the pre-registered model and HAR in all eight years; StormNet alone is the best single model in six of eight. Diebold–Mariano tests say the gain over HAR is not luck." |
| 6:00 | Can we trust it? | Rolling coverage chart | "A forecast needs error bars. Raw network quantiles drift when volatility jumps; adaptive conformal inference fixes coverage at 90% through 2015, Feb 2018 and Dec 2018. Right chart: the model's own uncertainty is calibrated: when it says it's unsure, it is." |
| 7:00 | Risk desk | Slide the loss limit from 3 to 2 | "What is a volatility forecast worth? Hold the same average position, sized by the forecast. Constant-size risk swings with the market; forecast-sized risk is flat, and at a 3× limit, breach days drop from 63 to 7 over 1,879 trading days." |
| 8:00 | Leak lab | Click **Run the experiment** | "Now, why do people believe ML predicts direction? Same random forest, same data. Shuffled cross-validation: AUC 0.88. Honest walk-forward: 0.50. Overlapping labels leak through the shuffle." |
| 9:00 | Leak lab, bottom chart | Point at the two lines | "Our honest direction head: statistically faint, and after a 0.70-point trading cost it bleeds money. Efficient markets, measured." |
| 9:30 | Forecast view (key 6) | Click "Volmageddon, 3 pm" (or run `uv run stormcast forecast "2018-02-05 15:00"` in a terminal) | "And it's a tool, not a notebook: one command gives the forecast, interval, regime and drivers for any moment, using only the models that existed then." |
| 10:00 | | | "ML can't call the market's direction. It can forecast its weather, and that's what risk management needs." |

## If something goes wrong
- App slow on first load: the one-minute bars (3.1M rows) load once on the first replay request and stay cached.
- Backup view: `uv run stormcast app` opens the Streamlit analysis version of the same results.
- Leak experiment not cached: it takes about 20 seconds; talk over the spinner about overlapping labels.
- Backup: `uv run stormcast replay 2018-02-05 --every 15` streams the same day in the terminal.

## Likely questions
- **Isn't 2016–2019 your test set?** Yes. The pre-registered study (report/report.md) used it exactly once. The v2 models
  were built afterwards, so their 2016–2019 numbers are labelled post hoc; 2012–2015 are the clean comparison, and the
  deep ensemble also wins there.
- **Could you trade this?** Not for direction. For risk: position sizing, stops and daily-limit management.
- **Why not predict price directly with a bigger network?** We tried (the 1D-CNN in the original study, the direction
  head here). The signal is not there at this horizon after costs.
