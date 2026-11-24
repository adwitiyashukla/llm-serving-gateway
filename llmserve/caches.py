from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass

import numpy as np

from llmserve.types import Message, SamplingParams


@dataclass(frozen=True)
class CachedResponse:
    text: str
    completion_tokens: int
    finish_reason: str
    backend: str


def guard_digest(backend: str, params: SamplingParams) -> str:
    payload = f"{backend}|{params.max_tokens}|{params.temperature}|{params.top_p}|{'|'.join(params.stop)}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


def context_digest(backend: str, messages: SequenceABC[Message], params: SamplingParams) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(guard_digest(backend, params).encode("utf-8"))
    for message in context_messages(messages):
        digest.update(message.role.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message.content.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


def context_messages(messages: SequenceABC[Message]) -> list[Message]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return list(messages[:index]) + list(messages[index + 1 :])
    return list(messages)


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def query_text(messages: SequenceABC[Message]) -> str:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return messages[index].content
    return "\n".join(message.content for message in messages)


def request_digest(backend: str, messages: SequenceABC[Message], params: SamplingParams) -> str:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(guard_digest(backend, params).encode("utf-8"))
    for message in messages:
        digest.update(message.role.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(message.content.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


class ExactCache:
    def __init__(self, max_entries: int = 512, enabled: bool = True) -> None:
        self.max_entries = max_entries
        self.enabled = enabled
        self._entries: OrderedDict[str, CachedResponse] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> CachedResponse | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: str, response: CachedResponse) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = response
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}


class SemanticCache:
    def __init__(
        self,
        max_entries: int = 512,
        threshold: float = 0.985,
        max_temperature: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.max_entries = max_entries
        self.threshold = threshold
        self.max_temperature = max_temperature
        self.enabled = enabled
        self._entries: OrderedDict[int, tuple[str, np.ndarray, CachedResponse]] = OrderedDict()
        self._matrices: dict[str, tuple[np.ndarray, list[int]]] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def eligible(self, params: SamplingParams) -> bool:
        return self.enabled and params.temperature <= self.max_temperature

    def _rebuild(self, guard: str) -> None:
        rows = [
            (key, vector) for key, (entry_guard, vector, _) in self._entries.items() if entry_guard == guard
        ]
        if not rows:
            self._matrices.pop(guard, None)
            return
        keys = [key for key, _ in rows]
        matrix = np.stack([vector for _, vector in rows])
        self._matrices[guard] = (matrix, keys)

    def get(self, guard: str, vector: SequenceABC[float]) -> tuple[CachedResponse, float] | None:
        if not self.enabled or not vector:
            return None
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return None
        query = query / norm
        with self._lock:
            bundle = self._matrices.get(guard)
            if bundle is None:
                self.misses += 1
                return None
            matrix, keys = bundle
            if matrix.shape[1] != query.shape[0]:
                self.misses += 1
                return None
            scores = matrix @ query
            best = int(np.argmax(scores))
            score = float(scores[best])
            if score < self.threshold:
                self.misses += 1
                return None
            key = keys[best]
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[2], score

    def put(self, guard: str, vector: SequenceABC[float], response: CachedResponse) -> None:
        if not self.enabled or not vector:
            return
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            return
        array = array / norm
        with self._lock:
            key = self._next_id
            self._next_id += 1
            self._entries[key] = (guard, array, response)
            dirty = {guard}
            while len(self._entries) > self.max_entries:
                _, (old_guard, _, _) = self._entries.popitem(last=False)
                dirty.add(old_guard)
            for name in dirty:
                self._rebuild(name)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}
