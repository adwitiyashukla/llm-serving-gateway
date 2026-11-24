from __future__ import annotations

import threading
from typing import Any

from llmserve.types import Sequence

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


class Histogram:
    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS) -> None:
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.total = 0.0
        self.count = 0
        self.values: list[float] = []

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        self.values.append(value)
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1
                return
        self.counts[-1] += 1

    def quantile(self, fraction: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[position]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[str, int] = {}
        self.rejections: dict[str, int] = {}
        self.cache_hits: dict[str, int] = {}
        self.routes: dict[str, int] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prefix_hit_tokens = 0
        self.batch_steps = 0
        self.batch_total = 0
        self.ttft = Histogram()
        self.latency = Histogram()

    def _bump(self, table: dict[str, int], key: str, amount: int = 1) -> None:
        table[key] = table.get(key, 0) + amount

    def observe_request(self, status: str) -> None:
        with self._lock:
            self._bump(self.requests, status)

    def observe_rejection(self, reason: str) -> None:
        with self._lock:
            self._bump(self.rejections, reason)

    def observe_cache_hit(self, kind: str) -> None:
        with self._lock:
            self._bump(self.cache_hits, kind)

    def observe_route(self, backend: str) -> None:
        with self._lock:
            self._bump(self.routes, backend)

    def observe_batch(self, size: int) -> None:
        with self._lock:
            self.batch_steps += 1
            self.batch_total += size

    def observe_prefix_hit(self, tokens: int) -> None:
        with self._lock:
            self.prefix_hit_tokens += tokens

    def observe_completion(self, sequence: Sequence) -> None:
        with self._lock:
            self.prompt_tokens += sequence.prompt_tokens
            self.completion_tokens += sequence.completion_tokens
            if sequence.first_token_at:
                self.ttft.observe(sequence.first_token_at - sequence.arrival)
            if sequence.finished_at:
                self.latency.observe(sequence.finished_at - sequence.arrival)

    def observe_cached_completion(self, latency: float) -> None:
        with self._lock:
            self.ttft.observe(latency)
            self.latency.observe(latency)

    def mean_batch(self) -> float:
        with self._lock:
            if not self.batch_steps:
                return 0.0
            return self.batch_total / self.batch_steps

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": dict(self.requests),
                "rejections": dict(self.rejections),
                "cache_hits": dict(self.cache_hits),
                "routes": dict(self.routes),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "prefix_hit_tokens": self.prefix_hit_tokens,
                "batch_steps": self.batch_steps,
                "mean_batch_size": round(self.batch_total / self.batch_steps, 3) if self.batch_steps else 0.0,
                "ttft_p50": round(self.ttft.quantile(0.5), 4),
                "ttft_p95": round(self.ttft.quantile(0.95), 4),
                "ttft_p99": round(self.ttft.quantile(0.99), 4),
                "latency_p50": round(self.latency.quantile(0.5), 4),
                "latency_p95": round(self.latency.quantile(0.95), 4),
                "latency_p99": round(self.latency.quantile(0.99), 4),
            }

    def render(self) -> str:
        with self._lock:
            lines = []
            lines.append("# TYPE llmserve_requests_total counter")
            for status, value in sorted(self.requests.items()):
                lines.append(f'llmserve_requests_total{{status="{status}"}} {value}')
            lines.append("# TYPE llmserve_rejections_total counter")
            for reason, value in sorted(self.rejections.items()):
                lines.append(f'llmserve_rejections_total{{reason="{reason}"}} {value}')
            lines.append("# TYPE llmserve_cache_hits_total counter")
            for kind, value in sorted(self.cache_hits.items()):
                lines.append(f'llmserve_cache_hits_total{{kind="{kind}"}} {value}')
            lines.append("# TYPE llmserve_routed_total counter")
            for backend, value in sorted(self.routes.items()):
                lines.append(f'llmserve_routed_total{{backend="{backend}"}} {value}')
            lines.append("# TYPE llmserve_prompt_tokens_total counter")
            lines.append(f"llmserve_prompt_tokens_total {self.prompt_tokens}")
            lines.append("# TYPE llmserve_completion_tokens_total counter")
            lines.append(f"llmserve_completion_tokens_total {self.completion_tokens}")
            lines.append("# TYPE llmserve_prefix_cache_tokens_total counter")
            lines.append(f"llmserve_prefix_cache_tokens_total {self.prefix_hit_tokens}")
            lines.append("# TYPE llmserve_batch_size gauge")
            mean = self.batch_total / self.batch_steps if self.batch_steps else 0.0
            lines.append(f"llmserve_batch_size {mean:.4f}")
            for name, histogram in (("ttft_seconds", self.ttft), ("request_seconds", self.latency)):
                lines.append(f"# TYPE llmserve_{name} histogram")
                cumulative = 0
                for index, edge in enumerate(histogram.buckets):
                    cumulative += histogram.counts[index]
                    lines.append(f'llmserve_{name}_bucket{{le="{edge}"}} {cumulative}')
                cumulative += histogram.counts[-1]
                lines.append(f'llmserve_{name}_bucket{{le="+Inf"}} {cumulative}')
                lines.append(f"llmserve_{name}_sum {histogram.total:.6f}")
                lines.append(f"llmserve_{name}_count {histogram.count}")
            return "\n".join(lines) + "\n"
