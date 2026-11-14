"""ALTCHA proof-of-work for BRAIN's sign-in (`GET /captcha`), ported from alpha-harness.

BRAIN gates `POST /authentication` behind ALTCHA: it sends a target SHA-256 digest and a salt, and the client
hashes `salt + n` for increasing integers until the digest matches. The answer goes in the sign-in body as
`{"captcha": base64(json)}`. Solving is pure CPU (up to ~1e6 hashes), so it runs in a worker thread.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

ALGORITHM = "SHA-256"
DEFAULT_MAX_NUMBER = 1_000_000


class AltchaError(RuntimeError):
    """A challenge that is malformed or cannot be solved."""


@dataclass(frozen=True)
class Challenge:
    algorithm: str
    challenge: str
    salt: str
    signature: str = ""
    maxnumber: int = DEFAULT_MAX_NUMBER

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Challenge:
        try:
            algorithm = str(payload["algorithm"]).upper()
            challenge = str(payload["challenge"])
            salt = str(payload["salt"])
        except KeyError as exc:
            raise AltchaError(f"Challenge is missing required field {exc.args[0]!r}") from exc
        if algorithm != ALGORITHM:
            raise AltchaError(f"Unsupported ALTCHA algorithm {algorithm!r}; expected {ALGORITHM}")
        try:
            maxnumber = int(payload.get("maxnumber", DEFAULT_MAX_NUMBER))
        except (TypeError, ValueError) as exc:
            raise AltchaError(f"Challenge has a non-integer maxnumber {payload.get('maxnumber')!r}") from exc
        return cls(algorithm, challenge, salt, str(payload.get("signature", "")), maxnumber)


@dataclass(frozen=True)
class Solution:
    challenge: Challenge
    number: int
    took_ms: int

    def to_payload(self) -> dict[str, Any]:
        return {"algorithm": self.challenge.algorithm, "challenge": self.challenge.challenge, "number": self.number,
                "salt": self.challenge.salt, "signature": self.challenge.signature, "took": self.took_ms}

    def encode(self) -> str:
        """The value of the sign-in body's `captcha` field: base64 of the compact, key-sorted JSON."""
        compact = json.dumps(self.to_payload(), separators=(",", ":"), sort_keys=True)
        return base64.b64encode(compact.encode("utf-8")).decode("ascii")


_DIGITS = tuple(str(n).encode("ascii") for n in range(100))
_TWO_DIGITS = tuple(f"{n:02d}".encode("ascii") for n in range(100))


def solve(challenge: Challenge) -> Solution:
    """Find the integer whose digest matches. Raises AltchaError when none exists up to `maxnumber`.

    The salt is hashed once and copied, and past 99 the leading digits are hashed once per hundred numbers and
    finished with the last two: the same digest as the whole number's, since SHA-256 reads a stream."""
    salted = hashlib.sha256(challenge.salt.encode("utf-8"))
    try:
        target = bytes.fromhex(challenge.challenge)
    except ValueError:
        target = b""
    started = time.monotonic()

    def found(number: int) -> Solution:
        return Solution(challenge, number, int((time.monotonic() - started) * 1000))

    limit = challenge.maxnumber
    for number in range(min(limit + 1, 100)):
        attempt = salted.copy()
        attempt.update(_DIGITS[number])
        if attempt.digest() == target:
            return found(number)
    for hundreds in range(1, limit // 100 + 1):
        prefix = salted.copy()
        prefix.update(str(hundreds).encode("ascii"))
        base = hundreds * 100
        for last in range(min(100, limit - base + 1)):
            attempt = prefix.copy()
            attempt.update(_TWO_DIGITS[last])
            if attempt.digest() == target:
                return found(base + last)
    raise AltchaError(f"No solution for the {challenge.algorithm} challenge within maxnumber={challenge.maxnumber}; "
                      "BRAIN's captcha scheme may have changed.")
