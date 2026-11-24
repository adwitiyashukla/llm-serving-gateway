from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch

LayerCache = tuple[torch.Tensor, torch.Tensor]
SequenceCache = list[LayerCache]
ROOT = b"llmserve"


def cache_length(cache: SequenceCache) -> int:
    if not cache:
        return 0
    return int(cache[0][0].shape[1])


def clone_cache(cache: SequenceCache) -> SequenceCache:
    return [(key.clone(), value.clone()) for key, value in cache]


def slice_cache(cache: SequenceCache, start: int, end: int) -> SequenceCache:
    return [(key[:, start:end, :].clone(), value[:, start:end, :].clone()) for key, value in cache]


def concat_caches(chunks: Sequence[SequenceCache]) -> SequenceCache:
    if not chunks:
        return []
    layers = len(chunks[0])
    merged: SequenceCache = []
    for index in range(layers):
        keys = torch.cat([chunk[index][0] for chunk in chunks], dim=1)
        values = torch.cat([chunk[index][1] for chunk in chunks], dim=1)
        merged.append((keys, values))
    return merged


def assemble(caches: Sequence[SequenceCache]) -> tuple[Any, int]:
    from transformers.cache_utils import DynamicCache

    width = max(cache_length(cache) for cache in caches)
    layers = len(caches[0])
    legacy = []
    for index in range(layers):
        keys = []
        values = []
        for cache in caches:
            key, value = cache[index]
            padding = width - int(key.shape[1])
            if padding:
                pad_key = torch.zeros(key.shape[0], padding, key.shape[2], dtype=key.dtype)
                pad_value = torch.zeros(value.shape[0], padding, value.shape[2], dtype=value.dtype)
                key = torch.cat([pad_key, key], dim=1)
                value = torch.cat([pad_value, value], dim=1)
            keys.append(key)
            values.append(value)
        legacy.append((torch.stack(keys), torch.stack(values)))
    return DynamicCache.from_legacy_cache(tuple(legacy)), width


def split(past: Any, lengths: Sequence[int]) -> list[SequenceCache]:
    legacy = past.to_legacy_cache()
    width = int(legacy[0][0].shape[2])
    caches: list[SequenceCache] = []
    for row, length in enumerate(lengths):
        start = width - length
        caches.append([(key[row, :, start:, :], value[row, :, start:, :]) for key, value in legacy])
    return caches


def batch_inputs(
    tokens: Sequence[int], lengths: Sequence[int], width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = len(tokens)
    input_ids = torch.tensor([[token] for token in tokens], dtype=torch.long)
    attention_mask = torch.zeros(rows, width + 1, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, width - length :] = 1
    position_ids = torch.tensor([[length] for length in lengths], dtype=torch.long)
    return input_ids, attention_mask, position_ids


class PrefixCache:
    def __init__(self, block_tokens: int = 16, max_blocks: int = 512, enabled: bool = True) -> None:
        self.block_tokens = block_tokens
        self.max_blocks = max_blocks
        self.enabled = enabled
        self._blocks: OrderedDict[bytes, SequenceCache] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._blocks)

    def clear(self) -> None:
        self._blocks.clear()

    @staticmethod
    def _key(parent: bytes, tokens: Sequence[int]) -> bytes:
        digest = hashlib.blake2b(parent, digest_size=16)
        digest.update(b"".join(token.to_bytes(4, "little") for token in tokens))
        return digest.digest()

    def lookup(self, token_ids: Sequence[int]) -> tuple[int, SequenceCache | None]:
        if not self.enabled or len(token_ids) <= self.block_tokens:
            self.misses += 1
            return 0, None
        parent = ROOT
        matched: list[bytes] = []
        chunks: list[SequenceCache] = []
        index = 0
        limit = len(token_ids) - 1
        while index + self.block_tokens <= limit:
            key = self._key(parent, token_ids[index : index + self.block_tokens])
            entry = self._blocks.get(key)
            if entry is None:
                break
            matched.append(key)
            chunks.append(entry)
            parent = key
            index += self.block_tokens
        for key in reversed(matched):
            self._blocks.move_to_end(key)
        if not chunks:
            self.misses += 1
            return 0, None
        self.hits += 1
        self.hit_tokens += index
        return index, clone_cache(concat_caches(chunks))

    def insert(self, token_ids: Sequence[int], cache: SequenceCache) -> None:
        if not self.enabled:
            return
        available = min(len(token_ids), cache_length(cache))
        parent = ROOT
        touched: list[bytes] = []
        index = 0
        while index + self.block_tokens <= available:
            key = self._key(parent, token_ids[index : index + self.block_tokens])
            if key not in self._blocks:
                self._blocks[key] = slice_cache(cache, index, index + self.block_tokens)
            touched.append(key)
            parent = key
            index += self.block_tokens
        for key in reversed(touched):
            self._blocks.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
            self.evictions += 1

    def stats(self) -> dict[str, int]:
        return {
            "blocks": len(self._blocks),
            "hits": self.hits,
            "misses": self.misses,
            "hit_tokens": self.hit_tokens,
            "evictions": self.evictions,
        }
