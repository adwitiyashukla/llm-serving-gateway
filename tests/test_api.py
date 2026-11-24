from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llmserve.api import create_app


@pytest.fixture
def client(base_config):
    with TestClient(create_app(base_config)) as test_client:
        yield test_client


def body(text: str = "how do I reset my password", **extra):
    payload = {
        "model": "auto",
        "messages": [
            {"role": "system", "content": "answer in one sentence"},
            {"role": "user", "content": text},
        ],
        "max_tokens": 6,
    }
    payload.update(extra)
    return payload


def test_health(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_lists_backends_and_auto(client):
    ids = {entry["id"] for entry in client.get("/v1/models").json()["data"]}
    assert {"fast", "quality", "auto"} <= ids


def test_chat_completion_shape(client):
    response = client.post("/v1/chat/completions", json=body())
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["usage"]["completion_tokens"] == 6
    assert payload["usage"]["total_tokens"] > payload["usage"]["completion_tokens"]
    assert payload["llmserve"]["backend"] == "fast"


def test_streaming_returns_sse_chunks(client):
    with client.stream("POST", "/v1/chat/completions", json=body(stream=True)) as response:
        assert response.status_code == 200
        chunks = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    parsed = [json.loads(chunk) for chunk in chunks[:-1]]
    assert any(item["choices"][0]["delta"].get("content") for item in parsed)
    final = parsed[-1]
    assert final["choices"][0]["finish_reason"] in ("stop", "length")
    assert final["llmserve"]["completion_tokens"] == 6


def test_empty_messages_rejected(client):
    response = client.post("/v1/chat/completions", json={"model": "auto", "messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_max_tokens_above_limit_rejected(client):
    response = client.post("/v1/chat/completions", json=body(max_tokens=10000))
    assert response.status_code == 400


def test_unknown_model_returns_404(client):
    response = client.post("/v1/chat/completions", json=body(model="vendor/unknown"))
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "model_not_found"


def test_invalid_top_p_rejected(client):
    response = client.post("/v1/chat/completions", json=body(top_p=0.0))
    assert response.status_code == 400


def test_metrics_endpoint_is_prometheus_text(client):
    client.post("/v1/chat/completions", json=body())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llmserve_requests_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")


def test_stats_endpoint(client):
    client.post("/v1/chat/completions", json=body())
    stats = client.get("/stats").json()
    assert stats["metrics"]["requests"]["ok"] == 1
    assert stats["models"]["loaded"] == ["fast"]


def test_api_key_is_enforced(base_config):
    guarded = base_config.with_overrides({"server": {"api_keys": ["secret"]}})
    with TestClient(create_app(guarded)) as client:
        assert client.post("/v1/chat/completions", json=body()).status_code == 401
        allowed = client.post("/v1/chat/completions", json=body(), headers={"Authorization": "Bearer secret"})
        assert allowed.status_code == 200


def test_rate_limit_returns_429(base_config):
    limited = base_config.with_overrides(
        {"admission": {"requests_per_minute": 60, "tokens_per_minute": 100000, "burst_requests": 1}}
    )
    with TestClient(create_app(limited)) as client:
        assert client.post("/v1/chat/completions", json=body()).status_code == 200
        response = client.post("/v1/chat/completions", json=body("a different question"))
        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_exceeded"
