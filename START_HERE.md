# Start here

Stormcast is a machine-learning "weather forecast" for S&P 500 futures. It predicts how much the market will move in
the next 30 minutes (and shows that it can't predict which way), with a web app for exploring and demoing it. See
`README.md` for the full story and `docs/demo.md` for a 10-minute demo script.

**The real market data is licensed and is not included.** This copy runs on a built-in synthetic market with the same
statistical structure. `sandbox/` already contains that market and a full set of models trained on it, so nothing has
to be trained before the app opens.

## For you (2 minutes)
1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. In this folder: `uv run stormcast demo`, then open http://localhost:8000 (it opens by itself).
3. Keys: `1`–`7` switch views, space plays a day, `←`/`→` step, `R` hides/reveals the outcome, `F` full screen.

## For the coding assistant helping set this up
Work through these in order and report back after each step. The machine has a capable GPU, so use it.

1. **Environment.** Run `uv sync`, then `uv run pytest`. Expect all tests to pass or skip. Tests that need the licensed
   data (point-in-time truncation, API tests over real results) skip automatically. Python must be 3.11 or 3.12
   (uv handles this).
2. **GPU check.** Run `uv run python -c "from stormcast.deep import device; print(device())"`.
   - It should print `cuda` (NVIDIA) or `mps` (Apple silicon).
   - If it prints `cpu` on a machine with an NVIDIA GPU, the installed torch wheel is CPU-only, which is common on
     Windows. Install the CUDA build of the same version (torch 2.5.1) from the PyTorch index, e.g. by adding a
     `[tool.uv.sources]` / `[[tool.uv.index]]` entry for `https://download.pytorch.org/whl/cu121` in `pyproject.toml`.
     Re-run the check.
   - `STORMCAST_DEVICE=cuda|mps|cpu` forces a device.
3. **Open the app on the shipped models.** Run `uv run stormcast demo` (add `--port 8010` if 8000 is taken). Confirm
   each view loads: Storm replay, Model arena, Trust, Risk desk, Leak lab (click "Run the experiment": about 30 seconds
   the first time), Forecast, How it works. A purple banner should say the market is synthetic.
4. **Retrain at full quality on this machine** (optional, about 1 to 2 hours on a good GPU). Run
   `uv run stormcast demo --full --retrain`.
   - This replaces `sandbox/results` and `sandbox/models` with the real-data settings: 5-minute training rows, 3
     StormNet seeds, 15 epochs, 20 Optuna trials, all years 2012–2019.
   - Compare `sandbox/results/v2/summary.json` with the shipped one (keep a copy first). Expected on the synthetic
     market: stacked-ensemble R² around 0.75–0.85, direction AUC around 0.50, interval coverage near 90%.
5. **Do not** try to download or reconstruct the licensed market data, and don't change the pre-registered scripts
   (`scripts/01`–`08`) or their results. They document the original real-data study.

Useful map: `src/stormcast/deep.py` (StormNet), `boost.py` (gradient boosting + Optuna + SHAP), `conformal.py`,
`stack.py`, `synthetic.py` (the simulator), `server.py` + `web/` (the app), `cli.py` (`stormcast` commands),
`scripts/10`–`14` (the v2 pipeline). `uv run stormcast --help` lists the commands.
