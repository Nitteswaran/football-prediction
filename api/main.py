"""FastAPI service exposing predictions, rankings and tournament simulations.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
or:
    python -m api.main
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from api import billing

logger = logging.getLogger(__name__)

FRONTEND_DIR = config.PROJECT_ROOT / "frontend"
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from prediction.predictor import Predictor
    _state["predictor"] = Predictor()
    logger.info("Predictor loaded: %d teams", len(_state["predictor"].teams()))
    yield
    _state.clear()


app = FastAPI(title="Football Prediction API", version="1.0.0", lifespan=lifespan)

# The frontend is served by this app (same origin), so CORS is only needed
# for explicitly whitelisted extra origins.
_cors_origins = [o.strip() for o in
                 os.environ.get("PITCHSENSE_CORS_ORIGINS", "").split(",")
                 if o.strip()]
if _cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_cors_origins,
                       allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Unlock-Token"])
app.include_router(billing.router)


class PredictRequest(BaseModel):
    home_team: str = Field(..., examples=["Brazil"])
    away_team: str = Field(..., examples=["Germany"])
    neutral: bool = True
    tournament: str = "FIFA World Cup"
    country: str = ""


def _predictor():
    p = _state.get("predictor")
    if p is None:
        raise HTTPException(503, "model not loaded")
    return p


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in _state}


@app.get("/api/teams")
def teams():
    return {"teams": _predictor().teams()}


@app.get("/api/rankings")
def rankings(top: int = 80):
    return {"rankings": _predictor().elo_table(top=top)}


@app.post("/api/predict")
def predict(req: PredictRequest, request: Request,
            x_unlock_token: str | None = Header(default=None)):
    billing.predict_limiter.check(request)
    p = _predictor()
    known = set(p.teams())
    for t in (req.home_team, req.away_team):
        if t not in known:
            raise HTTPException(404, f"unknown team: {t}")
    if req.home_team == req.away_team:
        raise HTTPException(400, "teams must differ")
    # paywall: insights are only computed and returned for paid sessions —
    # the locked response carries no real numbers for the client to reveal
    if not billing.is_unlocked(x_unlock_token):
        return {"home_team": req.home_team, "away_team": req.away_team,
                "locked": True}
    pred = p.predict(req.home_team, req.away_team, neutral=req.neutral,
                     tournament=req.tournament, country=req.country)
    return {**pred.as_dict(), "locked": False}


@app.get("/api/worldcup")
def worldcup():
    """Latest cached 100k-run World Cup simulation."""
    path = config.REPORTS_DIR / "worldcup2026_simulation.json"
    if not path.exists():
        raise HTTPException(404, "simulation not yet generated; run "
                                 "`python -m simulation.engine`")
    return json.loads(path.read_text())


@app.get("/api/evaluation")
def evaluation():
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "evaluation not yet generated")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
