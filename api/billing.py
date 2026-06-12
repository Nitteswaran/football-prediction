"""Stripe billing: one-time $5 unlock of full prediction insights.

No user accounts, and durable against the free host wiping local files.

Flow:
1. At checkout we mint a short unlock code (PS-XXXX-XXXX) and stamp it into the
   Stripe payment metadata. Stripe is therefore the durable source of truth —
   a code stays valid even if this server restarts and loses all local state.
2. On return (or via /api/redeem on another browser) we verify the code is
   backed by a paid Stripe payment, then issue a self-verifying device token:
   `CODE.<hmac>` signed with a key derived from STRIPE_SECRET_KEY. Predictions
   check the signature locally — no Stripe round-trip, and a paying user is
   never locked out, even across rebuilds.
3. A best-effort, in-memory activation counter caps how many device tokens we
   *issue* per code (default 3). It deters casual sharing; because it lives in
   memory it can reset on restart, so the cap is soft by design.

If STRIPE_SECRET_KEY is not set, billing is disabled and everything is free
(local development).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

PRICE_ID = os.environ.get("PITCHSENSE_STRIPE_PRICE",
                          "price_1ThQREGpqZjDsnwvYaKgzRvH")
ACTIVATION_CAP = int(os.environ.get("PITCHSENSE_ACTIVATION_CAP", "3"))

# Code alphabet excludes easily-confused characters (0/O, 1/I/L).
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_RE = re.compile(r"PS-[A-Z0-9]{4}-[A-Z0-9]{4}")

router = APIRouter()

# best-effort, in-memory: code -> number of device tokens issued
_activations: dict[str, int] = defaultdict(int)


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
redeem_limiter = RateLimiter(limit=10, window_s=60)
predict_limiter = RateLimiter(limit=60, window_s=60)


def billing_enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY"))


def _stripe():
    import stripe
    stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
    return stripe


# ---------------------------------------------------------------------------
# Codes & self-verifying device tokens
# ---------------------------------------------------------------------------
def _new_code() -> str:
    body = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(8))
    return f"PS-{body[:4]}-{body[4:]}"


def _sign_key() -> bytes:
    # derived from the (stable, secret) Stripe key so tokens survive restarts
    # without an extra configured secret; never logged or returned
    return hashlib.sha256(b"pitchsense-unlock|"
                          + os.environ["STRIPE_SECRET_KEY"].encode()).digest()


def _make_token(code: str) -> str:
    sig = hmac.new(_sign_key(), code.encode(), hashlib.sha256).digest()
    return code + "." + base64.urlsafe_b64encode(sig)[:24].decode()


def device_unlocked(token: str | None) -> bool:
    """Predict-time check: is this a token we signed? Stateless, so it keeps
    working across restarts (a paid user is never locked out)."""
    if not billing_enabled():
        return True
    if not token or "." not in token:
        return False
    code = token.split(".", 1)[0]
    if not _CODE_RE.fullmatch(code):
        return False
    return hmac.compare_digest(token, _make_token(code))


def _issue_token(code: str, existing: str | None) -> tuple[str | None, str]:
    """Issue a device token for a code, honoring the soft activation cap.
    Returns (token, status) where status is one of: ok, capped."""
    token = _make_token(code)
    if existing and hmac.compare_digest(existing, token):
        return token, "ok"                   # idempotent re-activation
    if _activations[code] >= ACTIVATION_CAP:
        return None, "capped"
    _activations[code] += 1
    return token, "ok"


# ---------------------------------------------------------------------------
# Stripe lookups (durable source of truth for codes)
# ---------------------------------------------------------------------------
def _plausible_session(session_id: str) -> bool:
    return session_id.startswith("cs_") and len(session_id) < 200


def _session_code(session_id: str) -> str | None:
    """Code stamped on a paid Checkout Session, or None if unpaid/invalid."""
    if not _plausible_session(session_id):
        return None
    try:
        session = _stripe().checkout.Session.retrieve(session_id)
    except Exception:
        return None                          # invalid/foreign id; don't log it
    if session.payment_status != "paid":
        return None
    return (session.metadata or {}).get("unlock_code")


def _code_is_paid(code: str) -> bool:
    """True if a succeeded payment carries this unlock code in its metadata."""
    # code is regex-validated by the caller, so it is safe to interpolate
    try:
        res = _stripe().PaymentIntent.search(
            query=f"metadata['unlock_code']:'{code}' AND status:'succeeded'",
            limit=1)
    except Exception:
        logger.exception("stripe payment search failed")
        return False
    return len(res.data) > 0


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
class RedeemBody(BaseModel):
    code: str


@router.post("/api/checkout")
def create_checkout(request: Request):
    checkout_limiter.check(request)
    if not billing_enabled():
        raise HTTPException(503, "billing not configured (set STRIPE_SECRET_KEY)")
    base = _public_base_url(request)
    code = _new_code()
    try:
        session = _stripe().checkout.Session.create(
            mode="payment",
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=base + "/?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base + "/?canceled=1",
            metadata={"unlock_code": code},
            payment_intent_data={"metadata": {"unlock_code": code}},
        )
    except Exception:
        logger.exception("stripe checkout session creation failed")
        raise HTTPException(502, "payment provider unavailable, try again")
    return {"url": session.url}


@router.get("/api/unlock")
def unlock(session_id: str, request: Request,
           x_unlock_token: str | None = Header(default=None)):
    """Called on return from Stripe: confirm payment, surface the code, and
    activate the purchasing browser as the first device."""
    unlock_limiter.check(request)
    if not billing_enabled():
        return {"unlocked": True}
    code = _session_code(session_id)
    if not code:
        return {"unlocked": False}
    token, status = _issue_token(code, x_unlock_token)
    if status == "capped":
        return {"unlocked": False, "error": "device_limit", "code": code}
    return {"unlocked": True, "code": code, "device_token": token,
            "devices_used": _activations[code], "cap": ACTIVATION_CAP}


@router.post("/api/redeem")
def redeem(body: RedeemBody, request: Request,
           x_unlock_token: str | None = Header(default=None)):
    """Restore an unlock on a new browser/device by entering the code."""
    redeem_limiter.check(request)
    if not billing_enabled():
        return {"unlocked": True}
    code = body.code.strip().upper()
    if not _CODE_RE.fullmatch(code) or not _code_is_paid(code):
        raise HTTPException(404, "unknown unlock code")
    token, status = _issue_token(code, x_unlock_token)
    if status == "capped":
        raise HTTPException(409, f"this code is already active on its maximum "
                                 f"of {ACTIVATION_CAP} devices")
    return {"unlocked": True, "device_token": token,
            "devices_used": _activations[code], "cap": ACTIVATION_CAP}
