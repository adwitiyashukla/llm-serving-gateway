from __future__ import annotations

import torch

from llmserve.kvcache import PrefixCache, assemble, batch_inputs, cache_length, concat_caches, split


def fake_cache(length: int, layers: int = 2, heads: int = 2, dim: int = 4, value: float = 1.0):
    return [
        (torch.full((heads, length, dim), value), torch.full((heads, length, dim), value))
        for _ in range(layers)
    ]


def test_assemble_and_split_round_trip():
    caches = [fake_cache(5, value=1.0), fake_cache(3, value=2.0)]
    past, width = assemble(caches)
    assert width == 5
    restored = split(past, [5, 3])
    assert [cache_length(item) for item in restored] == [5, 3]
    assert float(restored[1][0][0][0, 0, 0]) == 2.0


def test_batch_inputs_masks_padding():
    input_ids, mask, positions = batch_inputs([7, 9], [5, 3], 5)
    assert input_ids.tolist() == [[7], [9]]
    assert mask.tolist() == [[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]]
    assert positions.tolist() == [[5], [3]]


def test_concat_caches():
    merged = concat_caches([fake_cache(2), fake_cache(3)])
    assert cache_length(merged) == 5


def test_prefix_cache_miss_then_hit():
    cache = PrefixCache(block_tokens=4, max_blocks=16)
    tokens = list(range(20))
    assert cache.lookup(tokens) == (0, None)
    cache.insert(tokens, fake_cache(20))
    matched, restored = cache.lookup(tokens + [99])
    assert matched == 20
    assert cache_length(restored) == 20


def test_prefix_cache_partial_match():
    cache = PrefixCache(block_tokens=4, max_blocks=16)
    cache.insert(list(range(20)), fake_cache(20))
    matched, _ = cache.lookup(list(range(10)) + [77] * 10)
    assert matched == 8


def test_prefix_cache_never_matches_whole_prompt():
    cache = PrefixCache(block_tokens=4, max_blocks=16)
    tokens = list(range(16))
    cache.insert(tokens, fake_cache(16))
    matched, _ = cache.lookup(tokens)
    assert matched < len(tokens)


def test_prefix_cache_disabled():
    cache = PrefixCache(block_tokens=4, max_blocks=16, enabled=False)
    cache.insert(list(range(20)), fake_cache(20))
    assert cache.lookup(list(range(20))) == (0, None)


def test_prefix_cache_evicts_deepest_blocks_first():
    cache = PrefixCache(block_tokens=4, max_blocks=3)
    cache.insert(list(range(16)), fake_cache(16))
    assert len(cache) == 3
    matched, _ = cache.lookup(list(range(16)) + [1])
    assert matched == 12
    assert cache.stats()["evictions"] == 1
