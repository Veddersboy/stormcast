"""POST HOC (written after the final-year results were seen; not part of the pre-registered plan): where does the
small final-year direction AUC (~0.52) come from, and is it worth anything?
  1. Ties: a 30-minute move of exactly 0 is labelled "down"; AUC with ties removed, and whether a low p(up) predicts a tie.
  2. AUC on small vs large moves (|move| below / above the median).
  3. Gross points per trade on the model's most confident 10% and 20% of decisions.
Usage: uv run python scripts/08_posthoc_direction.py   (reads results/final)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["logistic", "gbm", "mlp", "cnn"]

if __name__ == "__main__":
    out = {}
    for scope in ("dev", "final"):
        p = pd.read_parquet(ROOT / "results" / scope / "direction_predictions.parquet")
        tie = p.move_pts == 0
        nt = p[~tie]
        big = nt.move_pts.abs() > nt.move_pts.abs().median()
        r = {"tie_share": float(tie.mean())}
        for m in MODELS:
            conf = {}
            for q in (0.1, 0.2):
                hi, lo = p[m] >= p[m].quantile(1 - q), p[m] <= p[m].quantile(q)
                conf[f"top_{int(q * 100)}pct_gross_pts"] = float(np.r_[p.move_pts[hi], -p.move_pts[lo]].mean())
            r[m] = {"auc_all": roc_auc_score(p.y_dir, p[m]), "auc_no_ties": roc_auc_score(nt.y_dir, nt[m]),
                    "auc_low_p_predicts_tie": roc_auc_score(tie.astype(int), -p[m]),
                    "auc_small_moves": roc_auc_score(nt.y_dir[~big], nt[m][~big]),
                    "auc_large_moves": roc_auc_score(nt.y_dir[big], nt[m][big]),
                    "corr_p_with_move_pts": float(np.corrcoef(p[m] - 0.5, p.move_pts)[0, 1]), **conf}
        out[scope] = r
    (ROOT / "results" / "final" / "posthoc_direction.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps(out, indent=2, default=float))
