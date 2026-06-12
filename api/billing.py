"""Stripe billing: one-time $5 unlock of full prediction insights.

No user accounts: payment happens on Stripe Checkout, and the paid Checkout
session id doubles as the unlock token. The frontend stores it in
localStorage and sends it as the X-Unlock-Token header; we verify it against
Stripe once and cache the result.

If STRIPE_SECRET_KEY is not set, billing is disabled and everything is free
(local development / test mode).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request

import config

logger = logging.getLogger(__name__)

PRICE_ID = os.environ.get("PITCHSENSE_STRIPE_PRICE",
                          "price_1ThQREGpqZjDsnwvYaKgzRvH")
PAID_SESSIONS_FILE = config.PROCESSED_DIR / "paid_sessions.json"

router = APIRouter()

_paid: set[str] = set()
if PAID_SESSIONS_FILE.exists():
    _paid.update(json.loads(PAID_SESSIONS_FILE.read_text()))


class RateLimiter:
    """Sliding-window per-client limiter (per worker process)."""

    def __init__(self, limit: int, window_s: float):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, request: Request) -> None:
        key = request.client.host if request.client else "?"
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if not q:
            # opportunistic cleanup so idle clients don't accumulate
            for k in [k for k, v in self._hits.items() if not v and k != key]:
                del self._hits[k]
        if len(q) >= self.limit:
            raise HTTPException(429, "too many requests, slow down")
        q.append(now)


checkout_limiter = RateLimiter(limit=10, window_s=60)
unlock_limiter = RateLimiter(limit=20, window_s=60)
predict_limiter = RateLimiter(limit=60, window_s=60)


def billing_enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY"))


def _stripe():
    import stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    return stripe


def _plausible_token(token: str) -> bool:
    return token.startswith("cs_") and len(token) < 200


def is_unlocked(token: str | None) -> bool:
    if not billing_enabled():
        return True
    if not token or not _plausible_token(token):
        return False
    if token in _paid:
        return True
    try:
        session = _stripe().checkout.Session.retrieve(token)
    except Exception:
        # invalid/foreign session id; never log the token itself
        return False
    if session.payment_status == "paid":
        _paid.add(token)
        PAID_SESSIONS_FILE.write_text(json.dumps(sorted(_paid)))
        return True
    return False


def _public_base_url(request: Request) -> str:
    # Behind a proxy/CDN the Host header is not trustworthy; production must
    # pin the public origin via PUBLIC_BASE_URL. A present-but-blank value
    # (easy to do in a hosting dashboard) is treated as unset.
    configured = (os.environ.get("PUBLIC_BASE_URL") or "").strip()
    base = (configured or str(request.base_url)).rstrip("/")
    if not re.match(r"^https?://[^/\s]+$", base):
        # fail loudly with the offending value in logs, never a cryptic
        # Stripe "Not a valid URL" error
        logger.error("resolved public base url is not absolute: %r "
                     "(PUBLIC_BASE_URL=%r)", base, configured)
        raise HTTPException(500, "server misconfigured: PUBLIC_BASE_URL must be "
                                 "a full https://… origin")
    return base


@router.post("/api/checkout")
def create_checkout(request: Request):
    checkout_limiter.check(request)
    if not billing_enabled():
        raise HTTPException(503, "billing not configured (set STRIPE_SECRET_KEY)")
    base = _public_base_url(request)
    try:
        session = _stripe().checkout.Session.create(
            mode="payment",
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=base + "/?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base + "/?canceled=1",
        )
    except Exception:
        logger.exception("stripe checkout session creation failed")
        raise HTTPException(502, "payment provider unavailable, try again")
    return {"url": session.url}


@router.get("/api/unlock")
def unlock(session_id: str, request: Request):
    unlock_limiter.check(request)
    return {"unlocked": is_unlocked(session_id)}
