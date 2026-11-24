from __future__ import annotations

from llmserve.metrics import Histogram, Metrics
from llmserve.types import SamplingParams, Sequence


def make_sequence(prompt_tokens: int = 5, output_tokens: int = 3) -> Sequence:
    sequence = Sequence(
        request_id="r",
        backend_name="fast",
        prompt_ids=list(range(prompt_tokens)),
        params=SamplingParams(max_tokens=output_tokens),
    )
    sequence.output_ids = list(range(output_tokens))
    sequence.arrival = 100.0
    sequence.first_token_at = 100.5
    sequence.finished_at = 102.0
    return sequence


def test_histogram_counts_and_quantiles():
    histogram = Histogram(buckets=(1.0, 2.0))
    for value in (0.5, 1.5, 3.0):
        histogram.observe(value)
    assert histogram.counts == [1, 1, 1]
    assert histogram.count == 3
    assert histogram.quantile(0.5) == 1.5


def test_empty_histogram_quantile():
    assert Histogram().quantile(0.9) == 0.0


def test_completion_updates_token_counters():
    metrics = Metrics()
    metrics.observe_completion(make_sequence())
    snapshot = metrics.snapshot()
    assert snapshot["prompt_tokens"] == 5
    assert snapshot["completion_tokens"] == 3
    assert snapshot["ttft_p50"] == 0.5
    assert snapshot["latency_p50"] == 2.0


def test_counters_group_by_label():
    metrics = Metrics()
    metrics.observe_request("ok")
    metrics.observe_request("ok")
    metrics.observe_request("error")
    metrics.observe_cache_hit("exact")
    metrics.observe_route("fast")
    metrics.observe_rejection("queue_full")
    snapshot = metrics.snapshot()
    assert snapshot["requests"] == {"ok": 2, "error": 1}
    assert snapshot["cache_hits"] == {"exact": 1}
    assert snapshot["routes"] == {"fast": 1}
    assert snapshot["rejections"] == {"queue_full": 1}


def test_mean_batch_size():
    metrics = Metrics()
    metrics.observe_batch(4)
    metrics.observe_batch(2)
    assert metrics.mean_batch() == 3.0


def test_prometheus_output_is_well_formed():
    metrics = Metrics()
    metrics.observe_request("ok")
    metrics.observe_completion(make_sequence())
    metrics.observe_batch(2)
    metrics.observe_prefix_hit(16)
    text = metrics.render()
    assert 'llmserve_requests_total{status="ok"} 1' in text
    assert "llmserve_prefix_cache_tokens_total 16" in text
    assert 'llmserve_ttft_seconds_bucket{le="+Inf"} 1' in text
    assert "llmserve_request_seconds_count 1" in text
    assert text.endswith("\n")
