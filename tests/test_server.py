"""API smoke tests over the real walk-forward results (skipped until scripts/12_v2_combine.py has run)."""
import pytest

from stormcast.v2 import RESULTS

pytestmark = pytest.mark.skipif(not (RESULTS / "predictions.parquet").exists(), reason="run the v2 pipeline first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from stormcast.server import app
    return TestClient(app)


def test_summary_and_frontend(client):
    s = client.get("/api/summary").json()
    assert 0.5 < s["kpi"]["r2"] < 0.95 and 0.45 < s["kpi"]["dir_auc"] < 0.55
    assert len(s["risk"]["days"]) == len(s["risk"]["constant"]) == len(s["risk"]["ensemble"])
    assert "Stormcast" in client.get("/").text and client.get("/app.js").status_code == 200


def test_day_is_walk_forward_and_complete(client):
    date = client.get("/api/summary").json()["dates"][100]
    d = client.get(f"/api/day/{date}").json()
    assert d["year"] == int(date[:4]) and len(d["slots"]) > 50
    s = d["slots"][10]
    assert s["range_lo"] <= s["range_med"] <= s["range_hi"] and len(s["attn"]) == 60
    # the first slot's 240-bar window (which already holds 09:30-10:00) plus the rest of the session to 16:00
    assert len(d["bars"]["t"]) >= 240 + 350 and sum(d["bars"]["rth"]) >= 380


def test_whatif_responds_to_inputs(client):
    date = client.get("/api/summary").json()["dates"][100]
    calm = client.post("/api/whatif", json={"date": date, "k": 180, "vix": 12, "vol_mult": 0.5}).json()
    storm = client.post("/api/whatif", json={"date": date, "k": 180, "vix": 40, "vol_mult": 3}).json()
    assert storm["new"] > calm["new"]


def test_unknown_day_is_404(client):
    assert client.get("/api/day/2011-01-03").status_code == 404
