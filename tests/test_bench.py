from __future__ import annotations

import asyncio

from llmserve.bench import (
    BenchResult,
    BenchSettings,
    LocalServer,
    RequestRecord,
    _agreement,
    _first_answers,
    aggregate_runs,
    build_prompts,
    format_table,
    format_verification,
    percentile,
    run_workload,
    summarize,
    write_results,
)


def record(prompt: str, text: str, ok: bool = True, source: str = "generated") -> RequestRecord:
    return RequestRecord(ok, 0.1, 0.2, 4, 8, source, "fast", "", prompt, text)


def result(label: str, throughput: float, ttft: float) -> BenchResult:
    return BenchResult(
        label=label,
        requests=10,
        errors=0,
        wall_s=1.0,
        ttft_p50=ttft,
        ttft_p95=ttft,
        ttft_p99=ttft,
        latency_p50=0.2,
        latency_p95=0.3,
        latency_p99=0.4,
        generated_p50=0.2,
        generated_p95=0.3,
        output_tokens=100,
        output_tokens_per_s=throughput,
        requests_per_s=5.0,
        cache_served=2,
        mean_batch_size=3.0,
        prefix_hit_tokens=64,
    )


def test_percentiles():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile([], 0.5) == 0.0


def test_prompts_are_deterministic():
    settings = BenchSettings(requests=30)
    assert build_prompts(settings) == build_prompts(settings)


def test_prompt_mix_contains_duplicates():
    prompts = build_prompts(BenchSettings(requests=60, duplicate_fraction=0.3, near_duplicate_fraction=0.0))
    assert len(set(prompts)) < len(prompts)


def test_unique_workload_has_no_duplicates():
    prompts = build_prompts(BenchSettings(requests=20, duplicate_fraction=0.0, near_duplicate_fraction=0.0))
    assert len(set(prompts)) == len(prompts)


def test_first_answers_keeps_the_earliest_response():
    answers = _first_answers(
        BenchResult(
            label="x",
            requests=2,
            errors=0,
            wall_s=1.0,
            ttft_p50=0.0,
            ttft_p95=0.0,
            ttft_p99=0.0,
            latency_p50=0.0,
            latency_p95=0.0,
            latency_p99=0.0,
            generated_p50=0.0,
            generated_p95=0.0,
            output_tokens=0,
            output_tokens_per_s=0.0,
            requests_per_s=0.0,
            cache_served=0,
            mean_batch_size=0.0,
            prefix_hit_tokens=0,
            records=[record("a", "first"), record("a", "second")],
        )
    )
    assert answers == {"a": "first"}


def test_agreement_counts_only_known_prompts():
    agreed, compared = _agreement({"a": "yes"}, [record("a", "yes"), record("a", "no"), record("b", "yes")])
    assert (agreed, compared) == (1, 2)


def test_aggregate_runs_picks_median_and_reports_spread():
    runs = [result("x", 100.0, 0.1), result("x", 120.0, 0.1), result("x", 110.0, 0.1)]
    merged = aggregate_runs(runs)
    assert merged.output_tokens_per_s == 110.0
    assert merged.repeats == 3
    assert merged.throughput_spread == round(20.0 / 110.0, 4)


def test_summarize_reports_speedups():
    results = [result("all features", 200.0, 0.1), result("no continuous batching", 100.0, 0.4)]
    ratios = summarize(results)
    assert ratios["throughput vs no continuous batching"] == 2.0
    assert ratios["ttft p50 vs no continuous batching"] == 0.25


def test_table_has_one_row_per_variant():
    text = format_table([result("all features", 200.0, 0.1), result("no prefix cache", 150.0, 0.2)])
    lines = text.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("variant")


def test_verification_report_formatting():
    report = {
        "requests": 10,
        "batched_vs_serial_compared": 10,
        "batched_vs_serial_agreement": 1.0,
        "thresholds": [
            {
                "threshold": 0.99,
                "cache_served": 3,
                "agreement": 1.0,
                "requests_per_s": 5.0,
                "all_e2e_p50": 0.2,
            }
        ],
    }
    text = format_verification(report)
    assert "batched vs serial output agreement: 1.0" in text
    assert "0.99" in text


def test_results_file_is_written(tmp_path):
    path = tmp_path / "results.json"
    write_results(str(path), [result("all features", 200.0, 0.1)], BenchSettings(requests=10))
    assert "all features" in path.read_text()


def test_workload_runs_against_a_local_server(base_config):
    settings = BenchSettings(
        requests=6, rate=0.0, max_tokens=4, duplicate_fraction=0.5, near_duplicate_fraction=0.0
    )
    with LocalServer(base_config) as server:
        outcome = asyncio.run(run_workload(server.url, settings, "smoke"))
    assert outcome.requests == 6
    assert outcome.errors == 0
    assert outcome.output_tokens > 0
    assert outcome.server_stats["metrics"]["requests"]
