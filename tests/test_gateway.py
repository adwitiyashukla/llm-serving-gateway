from __future__ import annotations

import asyncio
import copy

import pytest

from llmserve.config import Config
from llmserve.gateway import Gateway, new_request
from llmserve.types import Message, RequestError, SamplingParams
from tests.conftest import BASE

SYSTEM = Message("system", "you answer support questions in one short sentence")


@pytest.fixture
def gateway(base_config):
    engine = Gateway(base_config)
    engine.start()
    yield engine
    engine.stop()


def ask(text: str, max_tokens: int = 6, model: str = "auto"):
    return new_request(model, [SYSTEM, Message("user", text)], SamplingParams(max_tokens=max_tokens))


async def test_single_completion(gateway):
    result = await gateway.complete(ask("where is my invoice"))
    assert result.backend == "fast"
    assert result.completion_tokens == 6
    assert result.source == "generated"
    assert result.prompt_tokens > 0


async def test_repeat_request_is_served_from_cache(gateway):
    first = await gateway.complete(ask("where is my invoice"))
    second = await gateway.complete(ask("where is my invoice"))
    assert second.source == "cache"
    assert second.text == first.text
    assert gateway.metrics.snapshot()["cache_hits"]["exact"] == 1


async def test_near_duplicate_uses_semantic_cache(gateway):
    await gateway.complete(ask("where is my invoice"))
    result = await gateway.complete(ask("where is my invoice "))
    assert result.source == "cache"
    assert gateway.metrics.snapshot()["cache_hits"]["semantic"] == 1


async def test_sampled_requests_bypass_the_semantic_cache(gateway):
    hot = new_request(
        "auto", [SYSTEM, Message("user", "hot question")], SamplingParams(max_tokens=4, temperature=0.8)
    )
    await gateway.complete(hot)
    again = new_request(
        "auto", [SYSTEM, Message("user", "hot question ")], SamplingParams(max_tokens=4, temperature=0.8)
    )
    result = await gateway.complete(again)
    assert result.source == "generated"


async def test_concurrent_requests_share_a_batch(gateway):
    prompts = [ask(f"question {index}", 6) for index in range(4)]
    results = await asyncio.gather(*[gateway.complete(request) for request in prompts])
    assert all(result.completion_tokens == 6 for result in results)
    assert gateway.scheduler.mean_occupancy() > 1.0


async def test_shared_system_prompt_hits_the_prefix_cache(gateway):
    await gateway.complete(ask("first question"))
    result = await gateway.complete(ask("second question"))
    assert result.cached_prefix_tokens > 0


async def test_long_prompt_routes_to_quality(gateway):
    long_question = "please explain " * 60
    result = await gateway.complete(ask(long_question, 4))
    assert result.backend == "quality"


async def test_explicit_backend_is_honoured(gateway):
    result = await gateway.complete(ask("pick this backend", 4, model="quality"))
    assert result.backend == "quality"


async def test_unknown_model_raises(gateway):
    with pytest.raises(RequestError) as error:
        await gateway.complete(ask("hello", 4, model="nope"))
    assert error.value.status_code == 404


async def test_max_tokens_over_limit_is_rejected(gateway):
    with pytest.raises(RequestError):
        gateway.build_params(9999, 0.0, 1.0, (), 0)


async def test_prompt_over_limit_is_rejected(base_config):
    data = copy.deepcopy(BASE)
    data["engine"]["max_prompt_tokens"] = 12
    engine = Gateway(Config.from_dict(data))
    engine.start()
    try:
        with pytest.raises(RequestError) as error:
            await engine.complete(ask("a fairly long question that will not fit", 4))
        assert error.value.status_code == 400
    finally:
        engine.stop()


async def test_queue_full_returns_backpressure(base_config):
    data = copy.deepcopy(BASE)
    data["engine"]["max_queue_depth"] = 1
    engine = Gateway(Config.from_dict(data))
    engine.start()
    engine.scheduler.stop()
    try:
        from llmserve.scheduler import Stream
        from llmserve.types import Sequence

        loop = asyncio.get_running_loop()
        for _ in range(2):
            sequence = Sequence(
                request_id="x", backend_name="fast", prompt_ids=[1, 2, 3], params=SamplingParams(max_tokens=2)
            )
            try:
                engine.scheduler.submit(sequence, Stream(loop))
            except RequestError as error:
                assert error.status_code == 429
                return
        raise AssertionError("expected the queue to reject the second submission")
    finally:
        engine.stop()


async def test_stop_releases_loaded_models(base_config):
    engine = Gateway(base_config)
    engine.start()
    assert engine.manager.loaded_names() == ["fast"]
    engine.stop()
    assert engine.manager.loaded_names() == []


async def test_stats_expose_component_health(gateway):
    await gateway.complete(ask("stats question"))
    stats = gateway.stats()
    assert stats["models"]["loaded"] == ["fast"]
    assert stats["metrics"]["requests"]["ok"] == 1
    assert "fast" in stats["prefix_cache"]
    assert stats["queue_depth"] == 0


async def test_timeout_aborts_the_request(base_config):
    data = copy.deepcopy(BASE)
    data["server"]["request_timeout_s"] = 0.001
    engine = Gateway(Config.from_dict(data))
    engine.start()
    try:
        with pytest.raises(RequestError) as error:
            await engine.complete(ask("this will time out", 8))
        assert error.value.status_code == 504
    finally:
        engine.stop()
