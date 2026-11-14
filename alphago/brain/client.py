"""Low-level BRAIN HTTP client: login (incl. Persona biometric flow), session expiry, retries, long-polling.

Protocol notes (community-verified, BRAIN publishes no public spec):
- POST /authentication with basic auth -> 201 + session cookie. GET /authentication -> {"token": {"expiry": secs}}.
- Persona: 401 with `WWW-Authenticate: persona` and a `Location`; user verifies in the browser, then re-POST the
  Location URL until 201. Older variant: 201 body {"inquiry": id} -> POST /authentication/persona {"inquiry": id}.
- Long-running resources answer with a `Retry-After` header until they are ready.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin

import httpx

from ..config import BrainSettings
from .errors import (
    AUTH_DETAIL_MESSAGES,
    AuthError,
    BrainError,
    MaybeDelivered,
    PersonaRequired,
    describe_error_body,
    detail_code,
)
from .quota import QuotaTracker
from .ratelimit import RateGate, bucket_for
from .slots import SlotManager

log = logging.getLogger(__name__)

CredentialsProvider = Callable[[], "tuple[str, str] | None"]
StateStore = Callable[[str, Any], Awaitable[None]]


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class BrainClient:
    RATE_LIMIT_RETRIES = 12           # 429 retries per request (each after a cooldown that grows to 2 minutes)

    def __init__(self, cfg: BrainSettings, credentials: CredentialsProvider) -> None:
        self.cfg = cfg
        self._credentials = credentials
        self._http: httpx.AsyncClient | None = None
        self._login_lock = asyncio.Lock()
        self.quota = QuotaTracker()
        self.slots = SlotManager(cfg.max_slots)
        self.gate = RateGate(getattr(cfg, "catalog_rps", 2.0))
        self.on_throttle: Callable[[str, float], None] | None = None   # (bucket, cooldown seconds) for progress UIs
        self.state = "logged_out"           # logged_out | logged_in | persona_required | error
        self.user_id: str | None = None
        self.permissions: list[str] = []
        self.expiry_at: float | None = None
        self.persona_url: str | None = None
        self._persona_inquiry: tuple[str, str] | None = None   # (post_url, inquiry_id) for the legacy flow
        self.last_error: str | None = None
        self.on_state_change: Callable[[str], None] | None = None
        # sign-in lockout guard: once BRAIN refused the stored login, nothing signs in on its own again until the
        # person does (a loop of refused sign-ins can lock the account); silent sign-ins are also at most hourly
        self.silent_off = False
        self.last_silent_login = 0.0
        self.persona_expired = False             # the identity check timed out on Persona's side (410)
        # observers see every response: (method, path, status, json body sent, final?, response or None) —
        # the Health page's counters and the audit log
        self.observers: list[Callable[..., None]] = []

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.cfg.base_url,
                timeout=httpx.Timeout(self.cfg.request_timeout_s, connect=20.0),
                headers={"Accept": "application/json;version=2.0", "User-Agent": "AlphaGo/0.1"},
                follow_redirects=False,
            )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("BrainClient.start() not called")
        return self._http

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            if self.on_state_change:
                try:
                    self.on_state_change(state)
                except Exception:
                    log.exception("state listener failed")

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.expiry_at - time.time()) if self.expiry_at else 0.0

    @property
    def logged_in(self) -> bool:
        return self.state == "logged_in" and self.seconds_left > 0

    # ------------------------------------------------------------ authentication

    def _apply_auth_body(self, body: dict[str, Any]) -> None:
        token = body.get("token") or {}
        expiry = token.get("expiry")
        if isinstance(expiry, (int, float)):
            self.expiry_at = time.time() + float(expiry)
        user = body.get("user") or {}
        if user.get("id"):
            self.user_id = str(user["id"])
        perms = body.get("permissions")
        if isinstance(perms, list):
            self.permissions = [str(p) for p in perms]

    async def login(self, email: str | None = None, password: str | None = None) -> None:
        """Authenticate. Raises PersonaRequired when biometric verification is needed."""
        await self.start()
        if email and password:
            creds = (email, password)
        else:
            creds = self._credentials()
        if not creds:
            self._set_state("logged_out")
            raise AuthError("No BRAIN credentials saved. Enter them on the 'Sync with BRAIN' page.")
        if email and password:
            self.silent_off = False            # the person typed them: they may be tried again
        self.persona_expired = False
        async with self._login_lock:
            captcha = await self._solve_captcha()
            if captcha:
                response = await self.http.post("/authentication", auth=creds, json={"captcha": captcha})
            else:
                response = await self.http.post("/authentication", auth=creds)
            await self._handle_auth_response(response)

    async def _solve_captcha(self) -> str | None:
        """BRAIN's ALTCHA proof-of-work for sign-in, solved in a worker thread. None when BRAIN offers none (the
        plain sign-in is then sent as before)."""
        from .altcha import AltchaError, Challenge, solve
        try:
            response = await self.http.get("/captcha")
        except Exception as exc:                     # no captcha endpoint reachable: sign in without one
            log.debug("no captcha: %s", exc)
            return None
        body = self._json(response)
        if response.status_code != 200 or not isinstance(body, dict) or "challenge" not in body:
            return None
        try:
            solution = await asyncio.to_thread(solve, Challenge.from_payload(body))
        except AltchaError as exc:
            log.warning("BRAIN captcha unsolvable: %s", exc)
            return None
        log.info("BRAIN sign-in captcha solved in %d ms", solution.took_ms)
        return solution.encode()

    async def resume_session(self) -> bool:
        """Restore the session cookie kept from the last run; True when BRAIN still accepts it (no sign-in, no
        captcha, nothing counted toward a lockout). A dead cookie is cleared."""
        from . import session_store
        await self.start()
        if not session_store.load(self.http.cookies):
            return False
        try:
            left = await self.refresh_session()
        except httpx.HTTPError:
            return False
        if left > self.cfg.relogin_threshold_s:
            self._set_state("logged_in")
            log.info("BRAIN session resumed (%.0f s left)", left)
            return True
        session_store.clear()
        self.http.cookies.clear()
        self._set_state("logged_out")
        return False

    async def _handle_auth_response(self, response: httpx.Response) -> None:
        if response.status_code in (200, 201):
            body = self._json(response) or {}
            if "inquiry" in body and "user" not in body:
                inquiry = str(body["inquiry"])
                base = str(response.url).rstrip("/")
                self.persona_url = f"{base}/persona?inquiry={inquiry}"
                self._persona_inquiry = (f"{base}/persona", inquiry)
                self._set_state("persona_required")
                raise PersonaRequired(self.persona_url)
            self._apply_auth_body(body)
            if self.expiry_at is None:
                await self.refresh_session()
            self.persona_url = None
            self._persona_inquiry = None
            self.last_error = None
            self._set_state("logged_in")
            self._keep_cookies()
            log.info("BRAIN login ok (user %s, %.0f s session)", self.user_id, self.seconds_left)
            return
        if response.status_code == 401 and response.headers.get("WWW-Authenticate", "").lower() == "persona":
            self.persona_url = urljoin(str(response.url), response.headers.get("Location", ""))
            self._persona_inquiry = None
            self._set_state("persona_required")
            raise PersonaRequired(self.persona_url)
        body = self._json(response)
        code = detail_code(body)
        self.last_error = AUTH_DETAIL_MESSAGES.get(code) or describe_error_body(body) or f"HTTP {response.status_code}"
        if isinstance(body, dict) and "captcha" in body and not code:
            self.last_error = "BRAIN rejected the sign-in captcha. Please try again."     # not a refusal
        if response.status_code in (400, 401, 403) and code and code != "INQUIRY_INCOMPLETE":
            self.silent_off = True
        if response.status_code in (401, 403):
            self._set_state("error")
            raise AuthError(f"Login rejected: {self.last_error}", response.status_code, body)
        self._set_state("error")
        raise BrainError(f"Login failed: {self.last_error}", response.status_code, body)

    async def complete_persona(self, timeout_s: float = 600, interval_s: float = 5.0) -> bool:
        """Poll until the user finished the browser verification. Returns True when logged in."""
        if not self.persona_url:
            return self.logged_in
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if self._persona_inquiry:
                    url, inquiry = self._persona_inquiry
                    response = await self.http.post(url, json={"inquiry": inquiry})
                else:
                    response = await self.http.post(self.persona_url)
            except httpx.HTTPError as exc:
                log.warning("persona poll failed: %s", exc)
                await asyncio.sleep(interval_s)
                continue
            if response.status_code in (200, 201):
                body = self._json(response) or {}
                if body.get("user") or body.get("token"):
                    self._apply_auth_body(body)
                await self.refresh_session()
                self.persona_url = None
                self._persona_inquiry = None
                self._set_state("logged_in")
                self._keep_cookies()
                return True
            if response.status_code == 410:
                # Persona expired the check unfinished: asking again cannot succeed. A new check starts only
                # when the person presses "Start a new check" (a fresh sign-in), never on its own.
                self.persona_expired = True
                self.last_error = "This identity check expired before it was finished. Start a new one."
                return False
            # INQUIRY_INCOMPLETE (or anything else): keep waiting on the same inquiry. Signing in again would
            # mint a new one and strand the check the person is part-way through in the browser.
            await asyncio.sleep(interval_s)
        return False

    async def refresh_session(self) -> float:
        """GET /authentication -> remaining seconds. Marks the session dead on 401."""
        response = await self.http.get("/authentication")
        if response.status_code == 200:
            self._apply_auth_body(self._json(response) or {})
            return self.seconds_left
        if response.status_code == 401:
            self.expiry_at = None
            self._set_state("logged_out")
        return 0.0

    def _keep_cookies(self) -> None:
        from . import session_store
        try:
            session_store.save(self.http.cookies)
        except Exception:
            log.exception("could not keep the BRAIN session")

    async def logout(self) -> None:
        if self._http is not None:
            try:
                await self.http.delete("/authentication")
            except httpx.HTTPError:
                pass
        self.expiry_at = None
        from . import session_store
        session_store.clear()                    # signed out on purpose: do not resume it on the next start
        self.silent_off = True                   # and do not sign back in on our own
        self._set_state("logged_out")

    def may_sign_in_silently(self, min_gap_s: float = 3600.0) -> bool:
        """A background sign-in is allowed: the stored login was not refused, and none ran in the last hour."""
        if self.silent_off or self.state == "persona_required":
            return False
        return time.time() - self.last_silent_login >= min_gap_s

    async def silent_login(self) -> bool:
        """Sign in in the background if the lockout guard allows it. True when a sign-in was attempted."""
        if not self.may_sign_in_silently():
            return False
        self.last_silent_login = time.time()
        await self.login()
        return True

    async def ensure_session(self) -> None:
        if self._login_lock.locked():       # a concurrent request is signing in: wait and reuse its session
            async with self._login_lock:
                pass
        if self.state == "persona_required":
            raise PersonaRequired(self.persona_url or "")
        if not self.logged_in or self.seconds_left < self.cfg.relogin_threshold_s:
            await self.login()

    # ------------------------------------------------------------ requests

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Any = None,
        json: Any = None,
        retries: int = 6,
        retry_statuses: tuple[int, ...] = (429, 500, 502, 504),
        authenticated: bool = True,
        idempotent: bool | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """One API call with re-login on 401 and backoff on throttling / transient errors.

        idempotent=False (default for POST) marks a request that must never run twice: a transport error that may
        have reached BRAIN (anything but a failed connect) raises MaybeDelivered instead of resending."""
        if idempotent is None:
            idempotent = method.upper() != "POST"
        await self.start()
        if authenticated:
            await self.ensure_session()
        relogged = False
        attempt = 0
        throttles = 0
        bucket = bucket_for(url)
        while True:
            await self.gate.acquire(bucket)
            try:
                response = await self.http.request(
                    method, url, params=params, json=json, headers=headers,
                    **({"timeout": httpx.Timeout(timeout, connect=20.0)} if timeout else {}))
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if not idempotent and not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                                                            httpx.PoolTimeout)):
                    self._observe(method, url, 0, json, True, None)
                    raise MaybeDelivered(f"No answer from BRAIN to {method} {url} ({exc.__class__.__name__}); "
                                         "it may have arrived, so it is not resent.") from exc
                attempt += 1
                self._observe(method, url, 0, json, attempt > retries, None)
                if attempt > retries:
                    raise BrainError(f"Network error talking to BRAIN: {exc}") from exc
                await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))
                continue
            self.quota.update_from_headers(response.headers)
            status = response.status_code
            # 429 = "API rate limit exceeded": wait out a shared cooldown, slow the bucket down, and retry patiently
            # (up to RATE_LIMIT_RETRIES, separate from the transient-error budget) instead of failing the job
            rate_limited = status == 429 and 429 in retry_statuses and throttles < self.RATE_LIMIT_RETRIES
            final = not ((status == 401 and authenticated and not relogged) or rate_limited
                         or (status in retry_statuses and status != 429 and attempt < retries))
            self._observe(method, url, status, json, final, response)
            if status == 401 and authenticated and not relogged:
                relogged = True
                await self.login()
                continue
            if rate_limited:
                throttles += 1
                wait = self.gate.throttled(bucket, _retry_after(response))
                log.warning("BRAIN rate limit on %s %s: pausing %s requests for %.0f s (retry %d/%d)",
                            method, url, bucket, wait, throttles, self.RATE_LIMIT_RETRIES)
                if self.on_throttle:
                    try:
                        self.on_throttle(bucket, wait)
                    except Exception:
                        log.exception("throttle listener failed")
                continue                      # acquire() waits for the cooldown
            if status in retry_statuses and status != 429 and attempt < retries:
                attempt += 1
                wait = _retry_after(response)
                if wait is None:
                    wait = min(60.0, 2 ** attempt + random.random())
                await asyncio.sleep(min(120.0, max(1.0, wait)))
                continue
            if status < 400:
                self.gate.succeeded(bucket)
            return response

    def _observe(self, method: str, url: str, status: int, body: Any, final: bool,
                 response: httpx.Response | None) -> None:
        for fn in list(self.observers):
            try:
                fn(method, str(url), status, body, final, response)
            except Exception:
                log.exception("BRAIN request observer failed")

    async def get_json(self, url: str, params: Any = None) -> Any:
        response = await self.request("GET", url, params=params)
        return self.raise_for(response)

    def raise_for(self, response: httpx.Response) -> Any:
        body = self._json(response)
        if response.status_code >= 400:
            raise BrainError(
                f"{response.request.method} {response.request.url.path} -> {response.status_code}: "
                f"{describe_error_body(body)}",
                response.status_code,
                body,
            )
        return body

    async def poll(
        self,
        url: str,
        *,
        max_wait_s: float | None = None,
        on_progress: Callable[[float | None], None] | None = None,
        method: str = "GET",
    ) -> httpx.Response:
        """Repeat a request while BRAIN answers with Retry-After (long-running resource)."""
        deadline = time.time() + (max_wait_s or self.cfg.max_poll_minutes * 60)
        while True:
            response = await self.request(method, url, retry_statuses=(429, 500, 502, 504))
            wait = _retry_after(response)
            if wait is None or response.status_code >= 400 and response.status_code != 503:
                return response
            if on_progress is not None:
                body = self._json(response)
                progress = body.get("progress") if isinstance(body, dict) else None
                try:
                    on_progress(float(progress) if progress is not None else None)
                except Exception:
                    log.exception("progress callback failed")
            if time.time() > deadline:
                raise BrainError(f"Timed out waiting for {url}")
            await asyncio.sleep(min(30.0, max(0.5, wait)))
