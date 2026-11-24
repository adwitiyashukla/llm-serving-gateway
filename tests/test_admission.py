from __future__ import annotations

import pytest

from llmserve.admission import ANONYMOUS, AdmissionController, TokenBucket
from llmserve.config import AdmissionConfig
from llmserve.types import RequestError

CONFIG = AdmissionConfig(requests_per_minute=60, tokens_per_minute=600, burst_requests=2)


def test_bucket_allows_up_to_capacity():
    bucket = TokenBucket(2.0, 1.0, now=0.0)
    assert bucket.take(1.0, now=0.0)
    assert bucket.take(1.0, now=0.0)
    assert not bucket.take(1.0, now=0.0)


def test_bucket_refills_over_time():
    bucket = TokenBucket(2.0, 1.0, now=0.0)
    bucket.take(2.0, now=0.0)
    assert not bucket.take(1.0, now=0.0)
    assert bucket.take(1.0, now=1.5)


def test_bucket_never_exceeds_capacity():
    bucket = TokenBucket(2.0, 1.0, now=0.0)
    bucket.take(2.0, now=0.0)
    assert bucket.take(2.0, now=100.0)
    assert not bucket.take(1.0, now=100.0)


def test_anonymous_when_no_keys_configured():
    controller = AdmissionController(CONFIG)
    assert controller.authorize(None) == ANONYMOUS


def test_bearer_token_is_accepted():
    controller = AdmissionController(CONFIG, ("secret",))
    assert controller.authorize("Bearer secret") == "secret"
    assert controller.authorize("secret") == "secret"


def test_missing_and_wrong_keys_are_rejected():
    controller = AdmissionController(CONFIG, ("secret",))
    with pytest.raises(RequestError) as missing:
        controller.authorize(None)
    assert missing.value.status_code == 401
    with pytest.raises(RequestError):
        controller.authorize("Bearer wrong")


def test_request_rate_limit():
    controller = AdmissionController(CONFIG)
    controller.check("key", 1, 1, now=0.0)
    controller.check("key", 1, 1, now=0.0)
    with pytest.raises(RequestError) as error:
        controller.check("key", 1, 1, now=0.0)
    assert error.value.status_code == 429


def test_token_budget_limit():
    controller = AdmissionController(
        AdmissionConfig(requests_per_minute=6000, tokens_per_minute=10, burst_requests=100)
    )
    with pytest.raises(RequestError):
        controller.check("key", 100, 100, now=0.0)


def test_budgets_are_per_key():
    controller = AdmissionController(CONFIG)
    controller.check("one", 1, 1, now=0.0)
    controller.check("one", 1, 1, now=0.0)
    controller.check("two", 1, 1, now=0.0)
