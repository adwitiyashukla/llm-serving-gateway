from __future__ import annotations

import pytest

from llmserve.config import RouterConfig, RouterRule
from llmserve.router import Router, has_code, prompt_text
from llmserve.types import Message, RequestError

NAMES = ("fast", "quality")
MODEL_IDS = {"fast": "vendor/small", "quality": "vendor/large"}


def build(rules: tuple[RouterRule, ...]) -> Router:
    return Router(RouterConfig(default="fast", fallback=("fast",), rules=rules), NAMES, MODEL_IDS)


def test_defaults_when_no_rule_matches():
    router = build((RouterRule(backend="quality", min_prompt_tokens=500),))
    assert router.route("auto", "hello", 10, 32) == "fast"


def test_prompt_length_rule():
    router = build((RouterRule(backend="quality", min_prompt_tokens=100),))
    assert router.route("auto", "hello", 120, 32) == "quality"


def test_keyword_rule_is_case_insensitive():
    router = build((RouterRule(backend="quality", keywords=("Prove",)),))
    assert router.route("auto", "please prove this claim", 10, 32) == "quality"


def test_code_rule():
    router = build((RouterRule(backend="quality", requires_code=True),))
    assert router.route("auto", "```python\nx = 1\n```", 10, 32) == "quality"
    assert router.route("auto", "no code here", 10, 32) == "fast"


def test_max_tokens_rule():
    router = build((RouterRule(backend="quality", min_max_tokens=256),))
    assert router.route("auto", "hello", 10, 300) == "quality"
    assert router.route("auto", "hello", 10, 32) == "fast"


def test_first_matching_rule_wins():
    rules = (
        RouterRule(backend="fast", min_prompt_tokens=0),
        RouterRule(backend="quality", min_prompt_tokens=0),
    )
    assert build(rules).route("auto", "hello", 10, 32) == "fast"


def test_explicit_backend_name_overrides_rules():
    router = build((RouterRule(backend="quality", min_prompt_tokens=0),))
    assert router.route("fast", "hello", 10, 32) == "fast"


def test_model_id_resolves_to_backend():
    router = build(())
    assert router.route("vendor/large", "hello", 10, 32) == "quality"


def test_unknown_model_is_rejected():
    router = build(())
    with pytest.raises(RequestError) as error:
        router.route("vendor/missing", "hello", 10, 32)
    assert error.value.status_code == 404


def test_fallback_chain_has_no_duplicates():
    router = build(())
    assert router.chain("fast") == ["fast"]
    assert router.chain("quality") == ["quality", "fast"]


def test_prompt_text_joins_messages():
    text = prompt_text([Message("system", "one"), Message("user", "two")])
    assert text == "one\ntwo"


def test_has_code_detects_fences():
    assert has_code("```")
    assert not has_code("plain")
