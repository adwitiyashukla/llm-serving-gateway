from __future__ import annotations

import pytest

from llmserve.config import Config

BASE = {
    "server": {"host": "127.0.0.1", "port": 8123, "request_timeout_s": 60.0},
    "engine": {
        "max_batch_size": 4,
        "max_queue_depth": 16,
        "max_prefills_per_step": 2,
        "max_prompt_tokens": 1024,
        "default_max_tokens": 12,
        "hard_max_tokens": 32,
        "seed": 0,
    },
    "memory": {"budget_gb": 1.0},
    "prefix_cache": {"enabled": True, "block_tokens": 8, "max_blocks": 64},
    "exact_cache": {"enabled": True, "max_entries": 32},
    "semantic_cache": {"enabled": True, "max_entries": 32, "threshold": 0.999, "max_temperature": 0.0},
    "admission": {"requests_per_minute": 6000, "tokens_per_minute": 1000000, "burst_requests": 256},
    "router": {
        "default": "fast",
        "fallback": ["fast"],
        "rules": [{"backend": "quality", "min_prompt_tokens": 400}],
    },
    "backends": [
        {
            "name": "fast",
            "kind": "synthetic",
            "model": "synthetic-fast",
            "footprint_gb": 0.05,
            "resident": True,
            "hidden_size": 32,
            "layers": 2,
        },
        {
            "name": "quality",
            "kind": "synthetic",
            "model": "synthetic-quality",
            "footprint_gb": 0.05,
            "hidden_size": 32,
            "layers": 2,
        },
    ],
}


@pytest.fixture
def base_config() -> Config:
    return Config.from_dict(BASE)


@pytest.fixture(scope="session")
def synthetic_backend():
    from llmserve.backends import SyntheticBackend

    return SyntheticBackend(seed=0, hidden_size=32, layers=2)
