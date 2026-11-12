"""v2 (post hoc): walk-forward gradient boosting tuned with Optuna inside each training window, plus exact TreeSHAP
explanations of every test forecast.

Usage: uv run python scripts/11_v2_gbm.py [--trials 20] [--years ...]
Writes results/v2/gbm_predictions.parquet (forecasts + SHAP values), results/v2/gbm_tuning.json; models/v2/Y/gbm.joblib
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.boost import TunedGBM  # noqa: E402
from stormcast.features import ALL_FEATURES  # noqa: E402
from stormcast.v2 import MODELS, RESULTS, V2_YEARS, load_dense, split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--years", type=int, nargs="*", default=list(V2_YEARS))
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dense()
    parts, tuning = [], {}
    for y in a.years:
        t0 = time.time()
        tr, te = split(df, y)
        m = TunedGBM(a.trials).fit(df.iloc[tr])
        test = df.iloc[te]
        out = pd.DataFrame({"row": te, "gbm_tuned": m.predict(test)})
        vals, base = m.explain(test)
        for j, f in enumerate(ALL_FEATURES):
            out[f"shap_{f}"] = vals[:, j]
        out["shap_base"] = base
        parts.append(out)
        d = MODELS / str(y)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(m.model_, d / "gbm.joblib")
        trials = m.study_.trials_dataframe()[["number", "value"]]
        tuning[y] = {"best_params": m.params_, "best_inner_mse": float(m.study_.best_value),
                     "preregistered_inner_mse": float(trials.value.iloc[0]),
                     "trial_mse": trials.value.round(5).tolist(), "seconds": round(time.time() - t0)}
        s = out[df.standard.to_numpy()[te]]
        yv = df.y_lrv.to_numpy()[s.row]
        r2 = 1 - np.sum((yv - s.gbm_tuned) ** 2) / np.sum((yv - yv.mean()) ** 2)
        print(f"year {y}: R2 {r2:.3f} on standard slots; inner MSE {tuning[y]['preregistered_inner_mse']:.4f} -> "
              f"{tuning[y]['best_inner_mse']:.4f}; {tuning[y]['seconds']}s", flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(RESULTS / "gbm_predictions.parquet")
        (RESULTS / "gbm_tuning.json").write_text(json.dumps(tuning, indent=1) + "\n")
