from __future__ import annotations

import copy

import pytest

from llmserve.config import Config
from llmserve.models import ModelManager
from llmserve.types import RequestError
from tests.conftest import BASE


def tight_config(budget: float = 0.06) -> Config:
    data = copy.deepcopy(BASE)
    data["memory"]["budget_gb"] = budget
    data["backends"][0]["resident"] = False
    return Config.from_dict(data)


def test_lazy_loading(base_config):
    manager = ModelManager(base_config)
    assert manager.loaded_names() == []
    manager.get("fast")
    assert manager.loaded_names() == ["fast"]
    assert manager.loads == 1


def test_repeat_get_reuses_engine(base_config):
    manager = ModelManager(base_config)
    first = manager.get("fast")
    second = manager.get("fast")
    assert first is second
    assert manager.loads == 1


def test_preload_loads_resident_backends(base_config):
    manager = ModelManager(base_config)
    manager.preload()
    assert manager.loaded_names() == ["fast"]


def test_eviction_under_budget():
    manager = ModelManager(tight_config())
    manager.get("fast")
    manager.get("quality")
    assert manager.loaded_names() == ["quality"]
    assert manager.evictions == 1


def test_backend_in_use_is_protected():
    manager = ModelManager(tight_config())
    manager.get("fast")
    with pytest.raises(RequestError) as error:
        manager.get("quality", {"fast"})
    assert error.value.status_code == 503


def test_unload_is_idempotent(base_config):
    manager = ModelManager(base_config)
    manager.get("fast")
    manager.unload("fast")
    manager.unload("fast")
    assert manager.loaded_names() == []
    assert manager.evictions == 1


def test_stats_report_budget(base_config):
    manager = ModelManager(base_config)
    manager.get("fast")
    stats = manager.stats()
    assert stats["loaded"] == ["fast"]
    assert stats["budget_gb"] == base_config.memory.budget_gb
    assert stats["loaded_gb"] == 0.05
