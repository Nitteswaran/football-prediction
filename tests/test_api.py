"""API integration tests — run only once a model bundle exists."""
import pytest

import config

bundle_exists = (config.ARTIFACTS_DIR / "model_bundle.joblib").exists() \
    and config.STATE_SNAPSHOT.exists()

pytestmark = pytest.mark.skipif(not bundle_exists, reason="model not trained yet")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_teams(client):
    teams = client.get("/api/teams").json()["teams"]
    assert "Brazil" in teams and "Germany" in teams
    assert len(teams) > 150


def test_predict_probabilities_sum_to_one(client, monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)  # billing off -> unlocked
    r = client.post("/api/predict", json={"home_team": "Brazil",
                                          "away_team": "Germany"})
    assert r.status_code == 200
    d = r.json()
    p = d["probabilities"]
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-3
    grid_total = sum(sum(row) for row in d["scoreline_grid"])
    assert abs(grid_total - 1.0) < 1e-6
    assert d["expected_goals"]["home"] > 0


def test_predict_rejects_unknown_team(client):
    r = client.post("/api/predict", json={"home_team": "Narnia",
                                          "away_team": "Brazil"})
    assert r.status_code == 404


def test_predict_locked_without_payment(client, monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    r = client.post("/api/predict", json={"home_team": "Brazil",
                                          "away_team": "Germany"})
    assert r.status_code == 200
    d = r.json()
    assert d["locked"] is True
    assert "probabilities" not in d and "scoreline_grid" not in d


def test_checkout_unavailable_without_key(client, monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    r = client.post("/api/checkout")
    assert r.status_code == 503


def test_rankings(client):
    rk = client.get("/api/rankings?top=10").json()["rankings"]
    assert len(rk) == 10
    assert rk[0]["elo"] >= rk[-1]["elo"]


# --- billing base-url resolution (no model bundle required) ---------------
from types import SimpleNamespace

from fastapi import HTTPException

from api import billing


def _fake_request(base="http://testserver/"):
    return SimpleNamespace(base_url=base)


def test_public_base_url_blank_falls_back(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "   ")
    assert billing._public_base_url(_fake_request()) == "http://testserver"


def test_public_base_url_is_absolute(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://pitchsense.fun")
    assert billing._public_base_url(_fake_request()) == "https://pitchsense.fun"


def test_public_base_url_rejects_relative(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "pitchsense.fun")  # no scheme
    with pytest.raises(HTTPException) as exc:
        billing._public_base_url(_fake_request(base="pitchsense.fun"))
    assert exc.value.status_code == 500
