"""stormcast command line.

  stormcast forecast "2018-02-05 14:00"   forecast the next 30 minutes at any decision time (10:00-15:30 ET)
  stormcast replay 2018-02-05             stream a day's walk-forward forecasts against what actually happened
  stormcast demo                          no licensed data? run everything on a synthetic market and open the app
  stormcast serve                         open the web app (FastAPI + single-page frontend)
  stormcast app                           open the Streamlit control room (analysis view)
  stormcast train                         run the v2 pipeline: StormNet ensemble, tuned GBM, stacking + conformal
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from stormcast.data import ROOT


def _forecast(a):
    from stormcast.engine import Forecaster
    f = Forecaster().forecast(pd.Timestamp(a.when))
    print(f.report())


def _replay(a):
    from stormcast.engine import annualized_vol
    from stormcast.v2 import RESULTS
    p = pd.read_parquet(RESULTS / "predictions.parquet")
    day = p[p.date == pd.Timestamp(a.day)].sort_values("k")
    if day.empty:
        sys.exit(f"no forecasts for {a.day}")
    step = a.every // 5
    print(f"Replay {a.day}  (walk-forward: models never saw this year)\n")
    print(f"{'time':>5}  {'price':>9}  {'fcst vol':>8}  {'real vol':>8}  {'90% range (pts)':>16}  {'actual':>6}  "
          f"{'P(up)':>5}  {'went':>4}")
    for _, r in day.iloc[::step].iterrows():
        t = pd.Timestamp("09:30") + pd.Timedelta(minutes=int(r.k))
        hit = "✓" if r.covered else "✗"
        print(f"{t:%H:%M}  {r.price:9.2f}  {annualized_vol(r.ensemble):7.1f}%  {annualized_vol(r.y_lrv):7.1f}%  "
              f"{r.range_lo_pts:6.2f} - {r.range_hi_pts:6.2f}  {r.range_pts:6.2f}{hit} {r.stormnet_p_up:6.2f}  "
              f"{'up' if r.y_dir else 'down':>4}", flush=True)
        time.sleep(a.delay)
    s = day[day.standard]
    print(f"\nday: {s.covered.mean():.0%} of 90% intervals covered, forecast/realized vol corr "
          f"{np.corrcoef(s.ensemble, s.y_lrv)[0, 1]:.2f}")


def _app(a):
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "control_room.py"), "--server.port",
           str(a.port), "--theme.base", "dark"]
    subprocess.run(cmd, check=False)


def _serve(a):
    import threading
    import webbrowser

    import uvicorn
    url = f"http://localhost:{a.port}"
    print(f"Stormcast is running at {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("stormcast.server:app", host=a.host, port=a.port, log_level="warning")


DEMO_PROFILES = {
    # quick: ~10 min on an Apple-silicon or CUDA GPU, much longer on CPU; full: the real-data settings (~1.5 h on a GPU)
    "quick": {"years": "2018,2019", "stride": "15", "members": "1", "epochs": "6", "trials": "6"},
    "full": {"years": "", "stride": "5", "members": "3", "epochs": "15", "trials": "20"},
}


def _demo(a):
    """Synthetic market end to end: generate data, train every model walk-forward, open the web app. Everything lives
    in ROOT/sandbox, separate from any real data."""
    import os
    home = ROOT / "sandbox"
    prof = DEMO_PROFILES["full" if a.full else "quick"]
    env = dict(os.environ, STORMCAST_HOME=str(home), STORMCAST_YEARS=prof["years"])
    if a.retrain or not (home / "results" / "v2" / "predictions.parquet").exists():
        env["STORMCAST_STRIDE"] = prof["stride"]  # a trained workspace keeps the stride it was built with
    py = [sys.executable, "-W", "ignore"]

    def run(script, *args):
        print(f"\n== {script} {' '.join(args)}", flush=True)
        subprocess.run([*py, str(ROOT / "scripts" / script), *args], env=env, check=True)

    if a.retrain:
        import shutil
        for sub in ("results", "models"):
            shutil.rmtree(home / sub, ignore_errors=True)
    if not (home / "data" / "dataset_dense.parquet").exists():
        print("Generating the synthetic market (about 30 seconds)…", flush=True)
        run("14_synthetic_market.py")
    if not (home / "results" / "v2" / "predictions.parquet").exists():
        import json

        from stormcast.deep import device
        dev = device().type
        print(f"Training the {'full' if a.full else 'quick'} model set walk-forward on {dev.upper()}; progress is "
              "printed per year.", flush=True)
        if dev == "cpu":
            print("No GPU found: StormNet trains slowly on CPU (expect an hour or more). The zip you were given "
                  "normally includes a pre-trained sandbox/ so this step is skipped.", flush=True)
        (home / "workspace.json").write_text(json.dumps({"stride": int(prof["stride"]), "profile":
                                                          "full" if a.full else "quick"}) + "\n")
        run("10_v2_stormnet.py", "--members", prof["members"], "--epochs", prof["epochs"])
        run("11_v2_gbm.py", "--trials", prof["trials"])
        run("13_v2_regimes.py")
        run("12_v2_combine.py")
    url = f"http://localhost:{a.port}"
    print(f"\nStormcast (synthetic market) is running at {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    subprocess.run([*py, "-m", "uvicorn", "stormcast.server:app", "--port", str(a.port), "--log-level", "warning"],
                   env=env, check=False)


def _train(a):
    for s in ("10_v2_stormnet.py", "11_v2_gbm.py", "13_v2_regimes.py", "12_v2_combine.py"):
        print(f"== {s}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "scripts" / s)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stormcast", description="Machine-learning weather forecasts for ES futures.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast the next 30 minutes at a decision time (ET)")
    f.add_argument("when", help='e.g. "2018-02-05 14:00"')
    f.set_defaults(fn=_forecast)
    r = sub.add_parser("replay", help="stream a day's walk-forward forecasts")
    r.add_argument("day", help="e.g. 2018-02-05")
    r.add_argument("--every", type=int, default=30, choices=[5, 10, 15, 30, 60], help="minutes between rows")
    r.add_argument("--delay", type=float, default=0.15, help="seconds between rows")
    r.set_defaults(fn=_replay)
    w = sub.add_parser("serve", help="start the web app (API + frontend)")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--no-browser", action="store_true")
    w.set_defaults(fn=_serve)
    dm = sub.add_parser("demo", help="run everything on a synthetic market (no licensed data needed) and open the app")
    dm.add_argument("--full", action="store_true", help="train with the full settings (slow) instead of quick mode")
    dm.add_argument("--retrain", action="store_true", help="discard the sandbox's trained models and results first")
    dm.add_argument("--port", type=int, default=8000)
    dm.add_argument("--no-browser", action="store_true")
    dm.set_defaults(fn=_demo)
    s = sub.add_parser("app", help="open the Streamlit control room (analysis view)")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(fn=_app)
    t = sub.add_parser("train", help="run the v2 training pipeline")
    t.set_defaults(fn=_train)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
