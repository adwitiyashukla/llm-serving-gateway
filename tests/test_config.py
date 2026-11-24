from __future__ import annotations

import copy

import pytest

from llmserve.config import Config, ConfigError
from tests.conftest import BASE


def make(**changes):
    data = copy.deepcopy(BASE)
    for path, value in changes.items():
        parts = path.split(".")
        target = data
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return data


def test_loads_repository_config():
    config = Config.from_toml("config.toml")
    assert config.backend_names == ("fast", "quality")
    assert config.router.default in config.backend_names


def test_round_trip_through_dict(base_config):
    again = Config.from_dict(base_config.to_dict())
    assert again == base_config


def test_overrides_are_validated(base_config):
    changed = base_config.with_overrides({"engine": {"max_batch_size": 2}})
    assert changed.engine.max_batch_size == 2
    assert changed.backend_names == base_config.backend_names


def test_missing_file():
    with pytest.raises(ConfigError):
        Config.from_toml("does-not-exist.toml")


@pytest.mark.parametrize(
    "changes",
    [
        {"engine.max_batch_size": 0},
        {"engine.max_prefills_per_step": 99},
        {"engine.default_max_tokens": 999},
        {"server.port": 0},
        {"semantic_cache.threshold": 1.5},
        {"prefix_cache.block_tokens": 0},
        {"admission.requests_per_minute": 0},
        {"router.default": "missing"},
        {"memory.budget_gb": 0.001},
    ],
)
def test_invalid_values_are_rejected(changes):
    with pytest.raises(ConfigError):
        Config.from_dict(make(**changes))


def test_duplicate_backend_names():
    data = copy.deepcopy(BASE)
    data["backends"][1]["name"] = "fast"
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_rule_needs_a_condition():
    data = copy.deepcopy(BASE)
    data["router"]["rules"] = [{"backend": "quality"}]
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_resident_backends_must_fit_budget():
    data = copy.deepcopy(BASE)
    data["backends"][0]["footprint_gb"] = 0.9
    data["backends"][1]["footprint_gb"] = 0.9
    data["backends"][1]["resident"] = True
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_unknown_backend_lookup(base_config):
    with pytest.raises(ConfigError):
        base_config.backend("nope")
