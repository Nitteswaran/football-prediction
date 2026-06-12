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


# --- unlock codes & activation cap ----------------------------------------
@pytest.fixture
def clean_billing(monkeypatch):
    """Billing enabled with a fresh in-memory activation counter."""
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy")
    monkeypatch.setattr(billing, "_activations", billing.defaultdict(int))
    monkeypatch.setattr(billing, "ACTIVATION_CAP", 3)
    return billing


def test_code_format(clean_billing):
    code = clean_billing._new_code()
    assert billing._CODE_RE.fullmatch(code)


def test_token_signed_and_verified(clean_billing):
    b = clean_billing
    code = b._new_code()
    tok, status = b._issue_token(code, None)
    assert status == "ok" and b.device_unlocked(tok)
    # tampered token must not verify
    assert not b.device_unlocked(tok[:-1] + ("A" if tok[-1] != "A" else "B"))
    assert not b.device_unlocked("PS-AAAA-AAAA.forged")


def test_issue_is_idempotent(clean_billing):
    b = clean_billing
    code = b._new_code()
    tok, _ = b._issue_token(code, None)
    tok2, status = b._issue_token(code, tok)   # same token re-presented
    assert tok2 == tok and status == "ok"
    assert b._activations[code] == 1           # no extra slot consumed


def test_activation_cap_enforced(clean_billing):
    b = clean_billing
    code = b._new_code()
    for _ in range(3):
        assert b._issue_token(code, None)[1] == "ok"
    token, status = b._issue_token(code, None)
    assert token is None and status == "capped"


def test_redeem_rejects_malformed_code(client, clean_billing):
    # bad format is rejected before any Stripe call
    r = client.post("/api/redeem", json={"code": "not-a-code"})
    assert r.status_code == 404
