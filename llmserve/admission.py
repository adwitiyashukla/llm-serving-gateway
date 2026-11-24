from __future__ import annotations

import threading
import time
from collections.abc import Sequence as SequenceABC

from llmserve.config import AdmissionConfig
from llmserve.types import RequestError

ANONYMOUS = "anonymous"


class TokenBucket:
    def __init__(self, capacity: float, refill_per_second: float, now: float | None = None) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = capacity
        self.updated = now if now is not None else time.monotonic()

    def take(self, amount: float, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        elapsed = max(0.0, current - self.updated)
        self.updated = current
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        if self.tokens < amount:
            return False
        self.tokens -= amount
        return True


class AdmissionController:
    def __init__(self, config: AdmissionConfig, api_keys: SequenceABC[str] = ()) -> None:
        self.config = config
        self.api_keys = tuple(api_keys)
        self._requests: dict[str, TokenBucket] = {}
        self._tokens: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def authorize(self, header: str | None) -> str:
        if not self.api_keys:
            return ANONYMOUS
        if not header:
            raise RequestError(401, "missing Authorization header", "unauthorized")
        value = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
        if value not in self.api_keys:
            raise RequestError(401, "invalid api key", "unauthorized")
        return value

    def check(self, key: str, prompt_tokens: int, max_tokens: int, now: float | None = None) -> None:
        cost = float(prompt_tokens + max_tokens)
        with self._lock:
            requests = self._requests.get(key)
            if requests is None:
                requests = TokenBucket(
                    float(self.config.burst_requests), self.config.requests_per_minute / 60.0, now
                )
                self._requests[key] = requests
            tokens = self._tokens.get(key)
            if tokens is None:
                tokens = TokenBucket(
                    float(self.config.tokens_per_minute), self.config.tokens_per_minute / 60.0, now
                )
                self._tokens[key] = tokens
            if not requests.take(1.0, now):
                raise RequestError(429, "request rate limit exceeded", "rate_limit_exceeded")
            if not tokens.take(cost, now):
                raise RequestError(429, "token rate limit exceeded", "rate_limit_exceeded")
