"""Generate the synthetic market and its decision datasets in the active workspace (STORMCAST_HOME).

Refuses to run where real data already exists, so it can never overwrite the licensed export.
Usage: STORMCAST_HOME=sandbox uv run python scripts/14_synthetic_market.py   (or simply: uv run stormcast demo)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stormcast.data import BARS, DATA, load_bars, load_vix  # noqa: E402
from stormcast.features import DENSE, build  # noqa: E402
from stormcast.synthetic import generate  # noqa: E402

if __name__ == "__main__":
    if BARS.exists() and not (DATA / "SYNTHETIC").exists():
        sys.exit(f"{BARS} holds real data; set STORMCAST_HOME to another folder for the synthetic market")
    generate(DATA)
    bars, vix = load_bars(), load_vix()
    build(bars, vix).to_parquet(DATA / "dataset.parquet")
    build(bars, vix, decisions=DENSE).to_parquet(DATA / "dataset_dense.parquet")
    print(f"decision datasets written to {DATA}")
