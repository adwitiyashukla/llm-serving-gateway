from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from llmserve.backends import Backend, build_backend
from llmserve.config import Config
from llmserve.engine import Engine
from llmserve.kvcache import PrefixCache
from llmserve.types import RequestError


class ModelManager:
    def __init__(
        self,
        config: Config,
        factory: Callable[[Any, bool], Backend] = build_backend,
        local_files_only: bool = False,
    ) -> None:
        self.config = config
        self.factory = factory
        self.local_files_only = local_files_only
        self._engines: OrderedDict[str, Engine] = OrderedDict()
        self._lock = threading.RLock()
        self.loads = 0
        self.evictions = 0
        self.load_seconds = 0.0

    def loaded_names(self) -> list[str]:
        with self._lock:
            return list(self._engines.keys())

    def loaded_gb(self) -> float:
        with self._lock:
            return sum(self.config.backend(name).footprint_gb for name in self._engines)

    def preload(self) -> None:
        for spec in self.config.backends:
            if spec.resident:
                self.get(spec.name, set())

    def get(self, name: str, in_use: set[str] | None = None) -> Engine:
        with self._lock:
            engine = self._engines.get(name)
            if engine is not None:
                self._engines.move_to_end(name)
                return engine
            spec = self.config.backend(name)
            protected = set(in_use or set())
            protected.add(name)
            self._make_room(spec.footprint_gb, protected)
            started = time.perf_counter()
            backend = self.factory(spec, self.local_files_only)
            engine = Engine(
                backend,
                self.config.engine,
                PrefixCache(
                    block_tokens=self.config.prefix_cache.block_tokens,
                    max_blocks=self.config.prefix_cache.max_blocks,
                    enabled=self.config.prefix_cache.enabled,
                ),
            )
            self._engines[name] = engine
            self.loads += 1
            self.load_seconds += time.perf_counter() - started
            return engine

    def _make_room(self, needed_gb: float, protected: set[str]) -> None:
        budget = self.config.memory.budget_gb
        while self.loaded_gb() + needed_gb > budget:
            victim = None
            for name in self._engines:
                if name not in protected:
                    victim = name
                    break
            if victim is None:
                raise RequestError(
                    503,
                    f"cannot load backend within memory budget of {budget} GB",
                    "insufficient_memory",
                )
            self.unload(victim)

    def unload(self, name: str) -> None:
        with self._lock:
            engine = self._engines.pop(name, None)
            if engine is None:
                return
            engine.prefix_cache.clear()
            engine.backend.unload()
            self.evictions += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "loaded": list(self._engines.keys()),
                "loaded_gb": round(self.loaded_gb(), 3),
                "budget_gb": self.config.memory.budget_gb,
                "loads": self.loads,
                "evictions": self.evictions,
                "load_seconds": round(self.load_seconds, 3),
            }
