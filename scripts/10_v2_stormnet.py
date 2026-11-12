"""v2 (post hoc): walk-forward StormNet deep ensemble, test years 2012-2019.

For each test year Y: train a deep ensemble on every dense row before Y (one-day embargo), forecast every dense row of
Y, and save the year's model to models/v2/Y/stormnet.pt. Also saves conformity scores on the last 10% of the training
window (standard slots) for the conformal step.

Usage: uv run python scripts/10_v2_stormnet.py [--members 3] [--years 2014 2015]
Writes results/v2/stormnet_predictions.parquet, results/v2/stormnet_attention.npy, results/v2/stormnet_training.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.conformal import cqr_scores  # noqa: E402
from stormcast.deep import Ensemble, window_ends  # noqa: E402
from stormcast.v2 import MODELS, RESULTS, V2_YEARS, bars_and_channels, load_dense, split  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--years", type=int, nargs="*", default=list(V2_YEARS))
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_dense()
    bars, chan = bars_and_channels()
    ends = window_ends(bars, df.decision_utc)
    parts, attn, log = [], [], {}
    for y in a.years:
        t0 = time.time()
        tr, te = split(df, y)
        ens = Ensemble(n=a.members, epochs=a.epochs, lr=1e-3).fit(chan, df.iloc[tr], ends[tr])
        out = ens.predict(chan, df.iloc[te], ends[te])
        out.insert(0, "row", te)
        parts.append(out)
        attn.append(ens.last_attn_.astype(np.float16))
        # calibration scores: standard slots in the early-stopping split (last 10% of the training window)
        cal = tr[int(0.9 * len(tr)):]
        cal = cal[df.standard.to_numpy()[cal]]
        pc = ens.predict(chan, df.iloc[cal], ends[cal])
        scores = cqr_scores(df.y_lrange.to_numpy()[cal], pc.stormnet_q05.to_numpy(), pc.stormnet_q95.to_numpy())
        d = MODELS / str(y)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(ens.state(), d / "stormnet.pt")
        np.save(d / "stormnet_calib_scores.npy", scores)
        log[y] = {"train_rows": int(len(tr)), "test_rows": int(len(te)), "seconds": round(time.time() - t0),
                  "members_best_val": [m.best_val_ for m in ens.members_],
                  "members_history": [m.history_ for m in ens.members_]}
        s = out[df.standard.to_numpy()[te]]
        yv = df.y_lrv.to_numpy()[s.row]
        r2 = 1 - np.sum((yv - s.stormnet) ** 2) / np.sum((yv - yv.mean()) ** 2)
        print(f"year {y}: {len(tr)} train rows, R2 {r2:.3f} on standard slots, {log[y]['seconds']}s", flush=True)
        # checkpoint after every year so a long run can be inspected while it continues
        pd.concat(parts, ignore_index=True).to_parquet(RESULTS / "stormnet_predictions.parquet")
        np.save(RESULTS / "stormnet_attention.npy", np.concatenate(attn))
        (RESULTS / "stormnet_training.json").write_text(json.dumps(log, indent=1) + "\n")
