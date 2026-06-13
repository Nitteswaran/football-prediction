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

import html

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from api import billing, news

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
app.include_router(news.router)


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
    if not billing.device_unlocked(x_unlock_token):
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
SITE_URL = os.environ.get("PUBLIC_BASE_URL", "https://pitchsense.fun").rstrip("/")
_index_html: str | None = None


def _wc_picks_html() -> str:
    """Server-render the model's top World Cup contenders so crawlers and
    answer engines see real content, not an empty JS shell."""
    path = config.REPORTS_DIR / "worldcup2026_simulation.json"
    if not path.exists():
        return ""
    teams = json.loads(path.read_text()).get("teams", [])
    top = sorted(teams, key=lambda t: t.get("champion", 0), reverse=True)[:6]
    items = "".join(
        f"<li><span>{i}. {html.escape(t['team'])}</span>"
        f"<span>{t.get('champion', 0) * 100:.1f}% to win</span></li>"
        for i, t in enumerate(top, 1))
    return f"<ol class='wc-picks'>{items}</ol>" if items else ""


def _render_index() -> str:
    global _index_html
    if _index_html is None:
        tpl = (FRONTEND_DIR / "index.html").read_text()
        _index_html = tpl.replace("<!--WC_PICKS-->", _wc_picks_html())
    return _index_html


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return HTMLResponse(_render_index())

    @app.get("/terms", include_in_schema=False)
    def terms():
        return FileResponse(FRONTEND_DIR / "terms.html")

    @app.get("/privacy", include_in_schema=False)
    def privacy():
        return FileResponse(FRONTEND_DIR / "privacy.html")

    @app.get("/robots.txt", include_in_schema=False)
    def robots():
        return PlainTextResponse(
            f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")

    @app.get("/sitemap.xml", include_in_schema=False)
    def sitemap():
        pages = ["/", "/terms", "/privacy"]
        urls = "".join(f"<url><loc>{SITE_URL}{p}</loc>"
                       f"<changefreq>{'daily' if p == '/' else 'yearly'}</changefreq>"
                       f"</url>" for p in pages)
        xml = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
               f"{urls}</urlset>")
        return Response(content=xml, media_type="application/xml")


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
