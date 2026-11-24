from __future__ import annotations

from llmserve.caches import (
    CachedResponse,
    ExactCache,
    SemanticCache,
    context_digest,
    context_messages,
    guard_digest,
    query_text,
    request_digest,
)
from llmserve.types import Message, SamplingParams

PARAMS = SamplingParams(max_tokens=8)
MESSAGES = (Message("system", "be terse"), Message("user", "what is the refund window"))


def response(text: str = "answer") -> CachedResponse:
    return CachedResponse(text=text, completion_tokens=3, finish_reason="stop", backend="fast")


def test_request_digest_depends_on_every_message():
    other = (MESSAGES[0], Message("user", "something else"))
    assert request_digest("fast", MESSAGES, PARAMS) != request_digest("fast", other, PARAMS)


def test_request_digest_depends_on_sampling():
    hotter = SamplingParams(max_tokens=8, temperature=0.7)
    assert request_digest("fast", MESSAGES, PARAMS) != request_digest("fast", MESSAGES, hotter)


def test_guard_separates_backends():
    assert guard_digest("fast", PARAMS) != guard_digest("quality", PARAMS)


def test_context_messages_drops_the_final_user_turn():
    assert context_messages(MESSAGES) == [MESSAGES[0]]


def test_query_text_returns_the_final_user_turn():
    assert query_text(MESSAGES) == "what is the refund window"


def test_context_digest_ignores_the_query():
    other = (MESSAGES[0], Message("user", "a different question"))
    assert context_digest("fast", MESSAGES, PARAMS) == context_digest("fast", other, PARAMS)


def test_exact_cache_hit_and_miss():
    cache = ExactCache(max_entries=4)
    assert cache.get("a") is None
    cache.put("a", response())
    assert cache.get("a").text == "answer"
    assert cache.stats()["hits"] == 1


def test_exact_cache_evicts_least_recently_used():
    cache = ExactCache(max_entries=2)
    cache.put("a", response("first"))
    cache.put("b", response("second"))
    cache.get("a")
    cache.put("c", response("third"))
    assert cache.get("b") is None
    assert cache.get("a") is not None


def test_disabled_exact_cache_never_stores():
    cache = ExactCache(max_entries=4, enabled=False)
    cache.put("a", response())
    assert cache.get("a") is None


def test_semantic_cache_requires_similarity():
    cache = SemanticCache(max_entries=4, threshold=0.99)
    cache.put("guard", [1.0, 0.0, 0.0], response("stored"))
    assert cache.get("guard", [1.0, 0.0, 0.0])[0].text == "stored"
    assert cache.get("guard", [0.0, 1.0, 0.0]) is None


def test_semantic_cache_respects_guard():
    cache = SemanticCache(max_entries=4, threshold=0.9)
    cache.put("guard-a", [1.0, 0.0], response("a"))
    assert cache.get("guard-b", [1.0, 0.0]) is None


def test_semantic_cache_is_disabled_for_sampled_requests():
    cache = SemanticCache(max_entries=4, max_temperature=0.0)
    assert cache.eligible(SamplingParams(max_tokens=4, temperature=0.0))
    assert not cache.eligible(SamplingParams(max_tokens=4, temperature=0.7))


def test_semantic_cache_evicts_and_rebuilds():
    cache = SemanticCache(max_entries=2, threshold=0.99)
    cache.put("guard", [1.0, 0.0], response("first"))
    cache.put("guard", [0.0, 1.0], response("second"))
    cache.put("guard", [0.6, 0.8], response("third"))
    assert len(cache) == 2
    assert cache.get("guard", [1.0, 0.0]) is None
    assert cache.get("guard", [0.0, 1.0])[0].text == "second"


def test_semantic_cache_ignores_dimension_mismatch():
    cache = SemanticCache(max_entries=4, threshold=0.5)
    cache.put("guard", [1.0, 0.0], response())
    assert cache.get("guard", [1.0, 0.0, 0.0]) is None
