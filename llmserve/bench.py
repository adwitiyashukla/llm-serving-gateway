from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from llmserve.api import create_app
from llmserve.config import Config

SYSTEM_PROMPT = (
    "You are a support assistant for a cloud storage product. "
    "Answer in at most three sentences. Never invent account details. "
    "If the user asks about billing, tell them to check the billing page first. "
    "If the user reports data loss, ask for the workspace id before anything else. "
    "Always keep a calm and factual tone."
)

TOPICS = (
    "my upload keeps failing at ninety percent",
    "how do I share a folder with a client",
    "the mobile app shows an old version of my file",
    "I was charged twice this month",
    "can I restore a file I deleted last week",
    "sync stopped working after I changed my password",
    "how much storage does the team plan include",
    "a shared link expired before the date I set",
    "the desktop client uses too much memory",
    "I need to move a workspace to another owner",
    "why does search miss files inside zip archives",
    "can I set a retention policy for old versions",
)


NEAR_DUPLICATE_FORMS = (
    lambda text: f"  {text.upper()}  ",
    lambda text: f"{text}?",
    lambda text: f"please {text}",
)


@dataclass
class BenchSettings:
    requests: int = 60
    rate: float = 6.0
    max_tokens: int = 64
    duplicate_fraction: float = 0.2
    near_duplicate_fraction: float = 0.1
    seed: int = 7
    model: str = "auto"
    timeout_s: float = 300.0


@dataclass
class RequestRecord:
    ok: bool
    ttft: float
    total: float
    completion_tokens: int
    prompt_tokens: int
    source: str
    backend: str
    status: str = ""
    prompt: str = ""
    text: str = ""


@dataclass
class BenchResult:
    label: str
    requests: int
    errors: int
    wall_s: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    generated_p50: float
    generated_p95: float
    output_tokens: int
    output_tokens_per_s: float
    requests_per_s: float
    cache_served: int
    mean_batch_size: float
    prefix_hit_tokens: int
    repeats: int = 1
    throughput_spread: float = 0.0
    server_stats: dict[str, Any] = field(default_factory=dict)
    records: list[RequestRecord] = field(default_factory=list)


def build_prompts(settings: BenchSettings) -> list[str]:
    rng = random.Random(settings.seed)
    prompts: list[str] = []
    for index in range(settings.requests):
        roll = rng.random()
        if prompts and roll < settings.duplicate_fraction:
            prompts.append(rng.choice(prompts))
        elif prompts and roll < settings.duplicate_fraction + settings.near_duplicate_fraction:
            base = rng.choice(prompts)
            prompts.append(NEAR_DUPLICATE_FORMS[index % len(NEAR_DUPLICATE_FORMS)](base))
        else:
            topic = TOPICS[index % len(TOPICS)]
            prompts.append(f"{topic} (ticket {index})")
    return prompts


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


async def _one_request(
    client: httpx.AsyncClient, url: str, prompt: str, settings: BenchSettings
) -> RequestRecord:
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": settings.max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    started = time.perf_counter()
    ttft = 0.0
    completion_tokens = 0
    prompt_tokens = 0
    source = "generated"
    backend = ""
    pieces: list[str] = []
    try:
        async with client.stream("POST", url, json=payload, timeout=settings.timeout_s) as response:
            if response.status_code != 200:
                await response.aread()
                return RequestRecord(
                    False, 0.0, 0.0, 0, 0, "error", "", str(response.status_code), prompt, ""
                )
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                if "error" in chunk:
                    return RequestRecord(
                        False, 0.0, 0.0, 0, 0, "error", "", chunk["error"].get("type", "error"), prompt, ""
                    )
                choice = chunk["choices"][0]
                content = choice["delta"].get("content")
                if content:
                    if ttft == 0.0:
                        ttft = time.perf_counter() - started
                    pieces.append(content)
                extension = chunk.get("llmserve")
                if extension:
                    completion_tokens = extension.get("completion_tokens", 0)
                    prompt_tokens = extension.get("prompt_tokens", 0)
                    source = extension.get("source", "generated")
                    backend = extension.get("backend", "")
    except Exception as error:
        return RequestRecord(False, 0.0, 0.0, 0, 0, "error", "", type(error).__name__, prompt, "")
    total = time.perf_counter() - started
    return RequestRecord(
        True,
        ttft or total,
        total,
        completion_tokens,
        prompt_tokens,
        source,
        backend,
        "",
        prompt,
        "".join(pieces),
    )


async def run_workload(base_url: str, settings: BenchSettings, label: str) -> BenchResult:
    prompts = build_prompts(settings)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    rng = random.Random(settings.seed + 1)
    limits = httpx.Limits(
        max_connections=settings.requests + 8, max_keepalive_connections=settings.requests + 8
    )
    async with httpx.AsyncClient(limits=limits) as client:
        started = time.perf_counter()
        tasks = []
        elapsed = 0.0
        for prompt in prompts:
            if settings.rate > 0:
                elapsed += rng.expovariate(settings.rate)
                delay = started + elapsed - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(_one_request(client, url, prompt, settings)))
        records = await asyncio.gather(*tasks)
        wall = time.perf_counter() - started
        stats = {}
        try:
            response = await client.get(f"{base_url.rstrip('/')}/stats", timeout=30.0)
            stats = response.json()
        except Exception:
            stats = {}

    ok = [record for record in records if record.ok]
    generated = [record for record in ok if record.source != "cache"]
    output_tokens = sum(record.completion_tokens for record in ok)
    return BenchResult(
        label=label,
        requests=len(records),
        errors=len(records) - len(ok),
        wall_s=round(wall, 3),
        ttft_p50=round(percentile([r.ttft for r in generated], 0.5), 4),
        ttft_p95=round(percentile([r.ttft for r in generated], 0.95), 4),
        ttft_p99=round(percentile([r.ttft for r in generated], 0.99), 4),
        latency_p50=round(percentile([r.total for r in ok], 0.5), 4),
        latency_p95=round(percentile([r.total for r in ok], 0.95), 4),
        latency_p99=round(percentile([r.total for r in ok], 0.99), 4),
        generated_p50=round(percentile([r.total for r in generated], 0.5), 4),
        generated_p95=round(percentile([r.total for r in generated], 0.95), 4),
        output_tokens=output_tokens,
        output_tokens_per_s=round(output_tokens / wall, 2) if wall else 0.0,
        requests_per_s=round(len(ok) / wall, 3) if wall else 0.0,
        cache_served=sum(1 for record in ok if record.source == "cache"),
        mean_batch_size=float(stats.get("mean_batch_size", 0.0)),
        prefix_hit_tokens=int(stats.get("metrics", {}).get("prefix_hit_tokens", 0)),
        server_stats=stats,
        records=list(records),
    )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalServer:
    def __init__(self, config: Config, local_files_only: bool = False) -> None:
        import uvicorn

        from llmserve.api import build_gateway

        self.port = free_port()
        app = create_app(config, gateway=build_gateway(config, local_files_only=local_files_only))
        settings = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False
        )
        self._server = uvicorn.Server(settings)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LocalServer:
        self._thread.start()
        deadline = time.time() + 600
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                return self
            time.sleep(0.05)
        raise RuntimeError("server failed to start")

    def __exit__(self, *exc: Any) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=60)


VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("all features", {}),
    ("no continuous batching", {"engine": {"max_batch_size": 1, "max_prefills_per_step": 1}}),
    ("no prefix cache", {"prefix_cache": {"enabled": False}}),
    ("no response cache", {"exact_cache": {"enabled": False}, "semantic_cache": {"enabled": False}}),
    (
        "no gateway features",
        {
            "engine": {"max_batch_size": 1, "max_prefills_per_step": 1},
            "prefix_cache": {"enabled": False},
            "exact_cache": {"enabled": False},
            "semantic_cache": {"enabled": False},
        },
    ),
)


def aggregate_runs(runs: Sequence[BenchResult]) -> BenchResult:
    ordered = sorted(runs, key=lambda item: item.output_tokens_per_s)
    median = ordered[len(ordered) // 2]
    best = ordered[-1].output_tokens_per_s
    worst = ordered[0].output_tokens_per_s
    median.repeats = len(runs)
    median.throughput_spread = (
        round((best - worst) / median.output_tokens_per_s, 4) if median.output_tokens_per_s else 0.0
    )
    return median


def run_ablation(
    config: Config,
    settings: BenchSettings,
    variants: Sequence[tuple[str, dict[str, Any]]] = VARIANTS,
    repeats: int = 1,
    local_files_only: bool = False,
) -> list[BenchResult]:
    results = []
    for label, overrides in variants:
        variant_config = config.with_overrides(overrides) if overrides else config
        runs = []
        for _ in range(max(1, repeats)):
            with LocalServer(variant_config, local_files_only=local_files_only) as server:
                runs.append(asyncio.run(run_workload(server.url, settings, label)))
        results.append(aggregate_runs(runs))
    return results


NO_CACHE_OVERRIDE: dict[str, Any] = {
    "exact_cache": {"enabled": False},
    "semantic_cache": {"enabled": False},
}


def _first_answers(result: BenchResult) -> dict[str, str]:
    answers: dict[str, str] = {}
    for record in result.records:
        if record.ok and record.prompt not in answers:
            answers[record.prompt] = record.text
    return answers


def _agreement(expected: dict[str, str], records: Sequence[RequestRecord]) -> tuple[int, int]:
    compared = 0
    agreed = 0
    for record in records:
        if not record.ok or record.prompt not in expected:
            continue
        compared += 1
        if expected[record.prompt] == record.text:
            agreed += 1
    return agreed, compared


def verify_cache(
    config: Config,
    settings: BenchSettings,
    thresholds: Sequence[float] = (),
    local_files_only: bool = False,
) -> dict[str, Any]:
    serial = config.with_overrides(
        {**NO_CACHE_OVERRIDE, "engine": {"max_batch_size": 1, "max_prefills_per_step": 1}}
    )
    with LocalServer(serial, local_files_only=local_files_only) as server:
        reference = asyncio.run(run_workload(server.url, settings, "serial reference"))
    with LocalServer(config.with_overrides(NO_CACHE_OVERRIDE), local_files_only=local_files_only) as server:
        batched = asyncio.run(run_workload(server.url, settings, "batched no cache"))

    serial_answers = _first_answers(reference)
    batched_answers = _first_answers(batched)
    batch_agreed, batch_compared = _agreement(serial_answers, batched.records)

    levels = tuple(thresholds) if thresholds else (config.semantic_cache.threshold,)
    rows = []
    for threshold in levels:
        variant = config.with_overrides({"semantic_cache": {"threshold": threshold}})
        with LocalServer(variant, local_files_only=local_files_only) as server:
            cached = asyncio.run(run_workload(server.url, settings, f"threshold {threshold}"))
        served = [record for record in cached.records if record.ok and record.source == "cache"]
        agreed, compared = _agreement(batched_answers, served)
        rows.append(
            {
                "threshold": threshold,
                "cache_served": len(served),
                "compared": compared,
                "agreement": round(agreed / compared, 4) if compared else 1.0,
                "requests_per_s": cached.requests_per_s,
                "all_e2e_p50": cached.latency_p50,
            }
        )
    return {
        "requests": settings.requests,
        "batched_vs_serial_compared": batch_compared,
        "batched_vs_serial_agreement": round(batch_agreed / batch_compared, 4) if batch_compared else 1.0,
        "thresholds": rows,
    }


def format_verification(report: dict[str, Any]) -> str:
    lines = [
        f"requests: {report['requests']}",
        f"batched vs serial output agreement: {report['batched_vs_serial_agreement']} "
        f"over {report['batched_vs_serial_compared']} requests",
        "",
        "threshold  served  agreement  req/s  all e2e p50",
        "---------  ------  ---------  -----  -----------",
    ]
    for row in report["thresholds"]:
        lines.append(
            f"{row['threshold']:<9}  {row['cache_served']:<6}  {row['agreement']:<9}  "
            f"{row['requests_per_s']:<5}  {row['all_e2e_p50']}"
        )
    return "\n".join(lines)


def format_table(results: Sequence[BenchResult]) -> str:
    headers = (
        "variant",
        "req/s",
        "out tok/s",
        "ttft p50",
        "ttft p95",
        "gen e2e p50",
        "gen e2e p95",
        "all e2e p50",
        "cached",
        "batch",
        "runs",
        "spread",
    )
    rows = [
        (
            result.label,
            f"{result.requests_per_s:.2f}",
            f"{result.output_tokens_per_s:.1f}",
            f"{result.ttft_p50:.3f}",
            f"{result.ttft_p95:.3f}",
            f"{result.generated_p50:.3f}",
            f"{result.generated_p95:.3f}",
            f"{result.latency_p50:.3f}",
            str(result.cache_served),
            f"{result.mean_batch_size:.2f}",
            str(result.repeats),
            f"{result.throughput_spread * 100:.1f}%",
        )
        for result in results
    ]
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in rows), default=0)) for i in range(len(headers))
    ]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def write_results(path: str, results: Sequence[BenchResult], settings: BenchSettings) -> None:
    payload = {
        "settings": asdict(settings),
        "results": [asdict(result) for result in results],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize(results: Sequence[BenchResult]) -> dict[str, float]:
    table = {result.label: result for result in results}
    summary: dict[str, float] = {}
    base = table.get("all features")
    if base is None:
        return summary
    for label, result in table.items():
        if label == "all features":
            continue
        if result.output_tokens_per_s > 0:
            summary[f"throughput vs {label}"] = round(
                base.output_tokens_per_s / result.output_tokens_per_s, 3
            )
        if result.ttft_p50 > 0:
            summary[f"ttft p50 vs {label}"] = round(base.ttft_p50 / result.ttft_p50, 3)
    return summary
