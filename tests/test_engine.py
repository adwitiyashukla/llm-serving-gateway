from __future__ import annotations

import torch

from llmserve.config import EngineConfig
from llmserve.engine import Engine, sample_token
from llmserve.kvcache import PrefixCache, assemble, batch_inputs
from llmserve.types import Message, SamplingParams, Sequence, SequenceState

ENGINE_CONFIG = EngineConfig(max_batch_size=4, default_max_tokens=8, hard_max_tokens=32)


def make_sequence(backend, prompt: str, max_tokens: int = 8, request_id: str = "r"):
    ids = backend.encode_chat([Message("system", "you are a terse assistant"), Message("user", prompt)])
    return Sequence(
        request_id=request_id,
        backend_name=backend.name,
        prompt_ids=ids,
        params=SamplingParams(max_tokens=max_tokens),
    )


def run_alone(backend, prompt: str, max_tokens: int = 8):
    engine = Engine(backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    sequence = make_sequence(backend, prompt, max_tokens)
    output = engine.prefill(sequence)
    while not output.finished:
        output = engine.decode_step([sequence])[0]
    return sequence


def test_greedy_sampling_picks_argmax():
    logits = torch.tensor([0.1, 5.0, 0.3])
    assert sample_token(logits, SamplingParams(max_tokens=1), torch.Generator()) == 1


def test_sampling_is_reproducible():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    params = SamplingParams(max_tokens=1, temperature=1.0)
    first = torch.Generator()
    first.manual_seed(11)
    second = torch.Generator()
    second.manual_seed(11)
    assert sample_token(logits, params, first) == sample_token(logits, params, second)


def test_top_p_restricts_choices():
    logits = torch.tensor([10.0, -10.0, -10.0])
    params = SamplingParams(max_tokens=1, temperature=1.0, top_p=0.5)
    generator = torch.Generator()
    generator.manual_seed(3)
    assert sample_token(logits, params, generator) == 0


def test_prefill_then_decode_produces_tokens(synthetic_backend):
    sequence = run_alone(synthetic_backend, "hello there", 6)
    assert sequence.completion_tokens == 6
    assert sequence.finish_reason == "length"
    assert sequence.state is SequenceState.FINISHED
    assert sequence.first_token_at > 0


def test_max_tokens_is_respected(synthetic_backend):
    sequence = run_alone(synthetic_backend, "hello there", 3)
    assert sequence.completion_tokens == 3


def test_prefix_cache_reuse_matches_cold_prefill(synthetic_backend):
    cold = Engine(synthetic_backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    warm_cache = PrefixCache(block_tokens=8, max_blocks=64)
    warm = Engine(synthetic_backend, ENGINE_CONFIG, warm_cache)

    first = make_sequence(synthetic_backend, "shared question one")
    warm.prefill(first)

    reference = make_sequence(synthetic_backend, "shared question two")
    cold.prefill(reference)
    reused = make_sequence(synthetic_backend, "shared question two")
    warm.prefill(reused)

    assert reused.cached_prefix_tokens > 0
    difference = max(
        float((left[0] - right[0]).abs().max())
        for left, right in zip(reference.cache, reused.cache, strict=True)
    )
    assert difference < 1e-5
    assert reference.output_ids == reused.output_ids


def test_batched_decode_matches_single_sequence(synthetic_backend):
    engine = Engine(synthetic_backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    long_sequence = make_sequence(synthetic_backend, "a much longer question about batching", 8, "long")
    solo = make_sequence(synthetic_backend, "short one", 8, "solo")
    batched = make_sequence(synthetic_backend, "short one", 8, "batched")
    for sequence in (long_sequence, solo, batched):
        engine.prefill(sequence)

    past, width = assemble([solo.cache])
    inputs = batch_inputs([solo.next_token], [solo.length], width)
    alone_logits, _ = synthetic_backend.forward(*inputs, past)

    past, width = assemble([long_sequence.cache, batched.cache])
    inputs = batch_inputs(
        [long_sequence.next_token, batched.next_token], [long_sequence.length, batched.length], width
    )
    together_logits, _ = synthetic_backend.forward(*inputs, past)

    difference = float((alone_logits[0, -1] - together_logits[1, -1]).abs().max())
    assert difference < 1e-4


def test_sequence_joining_mid_flight_keeps_its_own_state(synthetic_backend):
    engine = Engine(synthetic_backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    first = make_sequence(synthetic_backend, "first prompt", 8, "first")
    engine.prefill(first)
    engine.decode_step([first])
    late = make_sequence(synthetic_backend, "late prompt", 8, "late")
    engine.prefill(late)
    engine.decode_step([first, late])
    assert first.completion_tokens == 3
    assert late.completion_tokens == 2
    assert first.length == first.prompt_tokens + 2
    assert late.length == late.prompt_tokens + 1


def test_stop_string_finishes_sequence(synthetic_backend):
    engine = Engine(synthetic_backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    sequence = make_sequence(synthetic_backend, "stopping test", 16)
    output = engine.prefill(sequence)
    while not output.finished:
        output = engine.decode_step([sequence])[0]
    letter = sequence.text[:1]
    if not letter:
        return
    stopping = make_sequence(synthetic_backend, "stopping test", 16)
    stopping.params = SamplingParams(max_tokens=16, stop=(letter,))
    engine_two = Engine(synthetic_backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    output = engine_two.prefill(stopping)
    while not output.finished:
        output = engine_two.decode_step([stopping])[0]
    assert stopping.finish_reason == "stop"
    assert letter not in stopping.text


def test_mean_occupancy_tracks_batch_sizes(synthetic_backend):
    engine = Engine(synthetic_backend, ENGINE_CONFIG, PrefixCache(enabled=False))
    first = make_sequence(synthetic_backend, "one", 4, "a")
    second = make_sequence(synthetic_backend, "two", 4, "b")
    engine.prefill(first)
    engine.prefill(second)
    engine.decode_step([first, second])
    engine.decode_step([first])
    assert engine.mean_occupancy() == 1.5
