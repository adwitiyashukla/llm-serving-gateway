from __future__ import annotations

import asyncio
import gc
import time
import uuid
from collections.abc import AsyncIterator
from collections.abc import Sequence as SequenceABC
from typing import Any

import torch

from llmserve.admission import AdmissionController
from llmserve.backends import build_backend
from llmserve.caches import (
    CachedResponse,
    ExactCache,
    SemanticCache,
    context_digest,
    normalize_query,
    query_text,
    request_digest,
)
from llmserve.config import Config
from llmserve.metrics import Metrics
from llmserve.models import ModelManager
from llmserve.router import Router, prompt_text
from llmserve.scheduler import Scheduler, Stream
from llmserve.types import (
    Completion,
    GenerationRequest,
    Message,
    RequestError,
    SamplingParams,
    Sequence,
)


class Gateway:
    def __init__(self, config: Config, factory: Any = build_backend, local_files_only: bool = False) -> None:
        self.config = config
        self.metrics = Metrics()
        self.manager = ModelManager(config, factory, local_files_only)
        self.scheduler = Scheduler(config, self.manager, self.metrics)
        self.exact_cache = ExactCache(config.exact_cache.max_entries, config.exact_cache.enabled)
        self.semantic_cache = SemanticCache(
            config.semantic_cache.max_entries,
            config.semantic_cache.threshold,
            config.semantic_cache.max_temperature,
            config.semantic_cache.enabled,
        )
        self.admission = AdmissionController(config.admission, config.server.api_keys)
        self.router = Router(
            config.router,
            config.backend_names,
            {backend.name: backend.model for backend in config.backends},
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        threads = self.config.engine.torch_threads
        if threads and torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        self.manager.preload()
        self.scheduler.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.scheduler.stop()
        for name in self.manager.loaded_names():
            self.manager.unload(name)
        gc.collect()
        self._started = False

    def build_params(
        self, max_tokens: int | None, temperature: float, top_p: float, stop: SequenceABC[str], seed: int
    ) -> SamplingParams:
        limit = self.config.engine.default_max_tokens if max_tokens is None else int(max_tokens)
        if limit < 1:
            raise RequestError(400, "max_tokens must be at least 1")
        if limit > self.config.engine.hard_max_tokens:
            raise RequestError(400, f"max_tokens must be <= {self.config.engine.hard_max_tokens}")
        if temperature < 0.0:
            raise RequestError(400, "temperature must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise RequestError(400, "top_p must be in (0, 1]")
        return SamplingParams(
            max_tokens=limit,
            temperature=float(temperature),
            top_p=float(top_p),
            stop=tuple(stop),
            seed=int(seed),
        )

    def estimate_tokens(self, messages: SequenceABC[Message]) -> int:
        engine = self.manager.get(self.config.router.default, set())
        return len(engine.backend.encode_chat(messages))

    def select_backend(self, request: GenerationRequest) -> tuple[str, list[int]]:
        text = prompt_text(request.messages)
        estimated = self.estimate_tokens(request.messages)
        name = self.router.route(request.model, text, estimated, request.params.max_tokens)
        engine = self.manager.get(name, set())
        prompt_ids = engine.backend.encode_chat(request.messages)
        if len(prompt_ids) > self.config.engine.max_prompt_tokens:
            raise RequestError(
                400,
                f"prompt has {len(prompt_ids)} tokens, limit is {self.config.engine.max_prompt_tokens}",
                "context_length_exceeded",
            )
        if len(prompt_ids) + request.params.max_tokens >= engine.backend.context_window:
            raise RequestError(
                400, "prompt plus max_tokens exceeds the context window", "context_length_exceeded"
            )
        return name, prompt_ids

    def _lookup_cache(self, backend_name: str, request: GenerationRequest) -> CachedResponse | None:
        key = request_digest(backend_name, request.messages, request.params)
        hit = self.exact_cache.get(key)
        if hit is not None:
            self.metrics.observe_cache_hit("exact")
            return hit
        if not self.semantic_cache.eligible(request.params):
            return None
        guard, vector = self._semantic_key(backend_name, request)
        found = self.semantic_cache.get(guard, vector)
        if found is None:
            return None
        self.metrics.observe_cache_hit("semantic")
        return found[0]

    def _semantic_key(self, backend_name: str, request: GenerationRequest) -> tuple[str, list[float]]:
        engine = self.manager.get(backend_name, set())
        query_ids = engine.backend.encode(normalize_query(query_text(request.messages)))
        guard = context_digest(backend_name, request.messages, request.params)
        return guard, engine.backend.embed(query_ids)

    def _store_cache(self, backend_name: str, request: GenerationRequest, response: CachedResponse) -> None:
        self.exact_cache.put(request_digest(backend_name, request.messages, request.params), response)
        if not self.semantic_cache.eligible(request.params):
            return
        guard, vector = self._semantic_key(backend_name, request)
        self.semantic_cache.put(guard, vector, response)

    async def stream(self, request: GenerationRequest) -> AsyncIterator[tuple[str, Any]]:
        started = time.perf_counter()
        backend_name, prompt_ids = self.select_backend(request)
        self.metrics.observe_route(backend_name)

        cached = self._lookup_cache(backend_name, request)
        if cached is not None:
            elapsed = time.perf_counter() - started
            self.metrics.observe_cached_completion(elapsed)
            self.metrics.observe_request("cached")
            if cached.text:
                yield "delta", cached.text
            yield (
                "done",
                Completion(
                    text=cached.text,
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=cached.completion_tokens,
                    finish_reason=cached.finish_reason,
                    backend=cached.backend,
                    source="cache",
                    ttft_s=elapsed,
                    total_s=elapsed,
                ),
            )
            return

        sequence = Sequence(
            request_id=request.request_id,
            backend_name=backend_name,
            prompt_ids=list(prompt_ids),
            params=request.params,
        )
        loop = asyncio.get_running_loop()
        channel = Stream(loop)
        self.scheduler.submit(sequence, channel)

        deadline = started + self.config.server.request_timeout_s
        pieces: list[str] = []
        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                item = await asyncio.wait_for(channel.get(), timeout=remaining)
                if item is None:
                    break
                if isinstance(item, RequestError):
                    self.metrics.observe_request("error")
                    raise item
                pieces.append(item)
                yield "delta", item
        except TimeoutError as error:
            self.scheduler.abort(request.request_id)
            self.metrics.observe_request("timeout")
            raise RequestError(504, "request timed out", "timeout") from error
        except asyncio.CancelledError:
            self.scheduler.abort(request.request_id)
            raise

        text = "".join(pieces)
        finish_reason = sequence.finish_reason or "stop"
        completion = Completion(
            text=text,
            prompt_tokens=sequence.prompt_tokens,
            completion_tokens=sequence.completion_tokens,
            finish_reason=finish_reason,
            backend=backend_name,
            cached_prefix_tokens=sequence.cached_prefix_tokens,
            ttft_s=max(0.0, sequence.first_token_at - sequence.arrival),
            total_s=max(0.0, sequence.finished_at - sequence.arrival),
        )
        self._store_cache(
            backend_name,
            request,
            CachedResponse(
                text=text,
                completion_tokens=sequence.completion_tokens,
                finish_reason=finish_reason,
                backend=backend_name,
            ),
        )
        self.metrics.observe_request("ok")
        yield "done", completion

    async def complete(self, request: GenerationRequest) -> Completion:
        result: Completion | None = None
        async for kind, payload in self.stream(request):
            if kind == "done":
                result = payload
        if result is None:
            raise RequestError(500, "generation produced no result", "internal_error")
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.snapshot(),
            "models": self.manager.stats(),
            "exact_cache": self.exact_cache.stats(),
            "semantic_cache": self.semantic_cache.stats(),
            "prefix_cache": {
                name: self.manager.get(name, set()).prefix_cache.stats()
                for name in self.manager.loaded_names()
            },
            "queue_depth": self.scheduler.queue_depth(),
            "running": self.scheduler.running_count(),
            "mean_batch_size": round(self.scheduler.mean_occupancy(), 3),
        }


def new_request(
    model: str,
    messages: SequenceABC[Message],
    params: SamplingParams,
    api_key: str = "",
) -> GenerationRequest:
    return GenerationRequest(
        request_id=uuid.uuid4().hex,
        model=model,
        messages=tuple(messages),
        params=params,
        api_key=api_key,
    )
