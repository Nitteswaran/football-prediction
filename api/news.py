"""Football / World Cup news feed + model betting insights, for paid users.

News is aggregated from public RSS feeds (no API key needed) and cached in
memory. Betting insights are derived from the project's own World Cup
simulation — we never invent odds or tips. Access requires a valid unlock
token, same as predictions.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from fastapi import APIRouter, Header, Request

import config
from api import billing

logger = logging.getLogger(__name__)

router = APIRouter()
news_limiter = billing.RateLimiter(limit=30, window_s=60)

# Public football RSS feeds — no key required.
FEEDS = [
    ("BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
    ("Sky Sports", "https://www.skysports.com/rss/12040"),
    ("Guardian", "https://www.theguardian.com/football/rss"),
    ("ESPN", "https://www.espn.com/espn/rss/soccer/news"),
    ("Yahoo Sports", "https://sports.yahoo.com/soccer/rss/"),
]
_CACHE_TTL = 900  # 15 minutes
_cache: dict = {"at": 0.0, "items": []}

_TAG_RE = re.compile(r"<[^>]+>")
_WC_RE = re.compile(r"world cup", re.IGNORECASE)

# Extra name variants headlines use, beyond the literal results-dataset name.
TEAM_ALIASES: dict[str, list[str]] = {
    "South Korea": ["South Korea", "Korea Republic"],
    "North Korea": ["North Korea", "Korea DPR"],
    "United States": ["United States", "USA", "USMNT", "US men"],
    "Netherlands": ["Netherlands", "Dutch", "Holland"],
    "Czech Republic": ["Czech Republic", "Czechia"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"],
    "DR Congo": ["DR Congo", "Congo DR", "DRC"],
    "Cape Verde": ["Cape Verde", "Cabo Verde"],
    "Saudi Arabia": ["Saudi Arabia", "Saudi"],
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia"],
    "Republic of Ireland": ["Republic of Ireland", "Ireland"],
}
_pattern_cache: dict[str, re.Pattern] = {}


def _team_pattern(team: str) -> re.Pattern:
    if team not in _pattern_cache:
        names = TEAM_ALIASES.get(team, [team])
        alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        _pattern_cache[team] = re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)
    return _pattern_cache[team]


def _mentions(item: dict, team: str) -> bool:
    pat = _team_pattern(team)
    return bool(pat.search(item["title"]) or pat.search(item.get("summary", "")))


def _clean(text: str) -> str:
    import html
    text = _TAG_RE.sub("", text or "")
    return html.unescape(text).strip()


def _parse_date(raw: str) -> float:
    try:
        return parsedate_to_datetime(raw).timestamp()
    except Exception:
        return 0.0


def _fetch_feed(name: str, url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Pitchsense/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        root = ET.fromstring(resp.read())
    items = []
    for it in root.iter("item"):
        title = _clean(it.findtext("title", ""))
        link = (it.findtext("link", "") or "").strip()
        if not title or not link:
            continue
        summary = _clean(it.findtext("description", ""))
        pub = it.findtext("pubDate", "") or ""
        items.append({
            "title": title,
            "link": link,
            "summary": summary[:240],
            "source": name,
            "published": pub,
            "_ts": _parse_date(pub),
            "world_cup": bool(_WC_RE.search(title) or _WC_RE.search(summary)),
        })
    return items


def get_news() -> list[dict]:
    now = time.time()
    if _cache["items"] and now - _cache["at"] < _CACHE_TTL:
        return _cache["items"]
    items: list[dict] = []
    for name, url in FEEDS:
        try:
            items += _fetch_feed(name, url)
        except Exception:
            logger.warning("news feed failed: %s", name)
    # dedupe by title, World Cup first, then most recent
    seen, deduped = set(), []
    for it in items:
        key = it["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    deduped.sort(key=lambda x: (x["world_cup"], x["_ts"]), reverse=True)
    for it in deduped:
        it.pop("_ts", None)
    if deduped:                       # only refresh cache on a successful fetch
        _cache.update(at=now, items=deduped)
    return _cache["items"]


def get_insights() -> list[dict]:
    """Top World Cup title contenders from our own simulation — the model's
    view, not a tip."""
    path = config.REPORTS_DIR / "worldcup2026_simulation.json"
    if not path.exists():
        return []
    teams = json.loads(path.read_text()).get("teams", [])
    top = sorted(teams, key=lambda t: t.get("champion", 0), reverse=True)[:6]
    return [{"team": t["team"], "champion": t.get("champion", 0),
             "reach_final": t.get("final", 0)} for t in top]


@router.get("/api/news")
def news(request: Request, x_unlock_token: str | None = Header(default=None)):
    news_limiter.check(request)
    if not billing.device_unlocked(x_unlock_token):
        return {"locked": True}
    return {"locked": False, "items": get_news()[:24], "insights": get_insights()}


@router.get("/api/matchnews")
def matchnews(home: str, away: str, request: Request,
              x_unlock_token: str | None = Header(default=None)):
    """Headlines that actually name each side of a fixture, per team."""
    news_limiter.check(request)
    if not billing.device_unlocked(x_unlock_token):
        return {"locked": True}
    items = get_news()
    used: set[str] = set()

    def pick(team: str, n: int = 4) -> list[dict]:
        out = []
        for it in items:
            if it["title"] in used:
                continue
            if _mentions(it, team):
                used.add(it["title"])
                out.append({k: it[k] for k in ("title", "link", "source")})
                if len(out) >= n:
                    break
        return out

    return {"locked": False, "home": home, "away": away,
            "home_news": pick(home), "away_news": pick(away)}
