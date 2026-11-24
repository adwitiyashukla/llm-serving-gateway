from __future__ import annotations

from collections.abc import Sequence as SequenceABC

from llmserve.config import RouterConfig, RouterRule
from llmserve.types import Message, RequestError

AUTO_MODELS = ("auto", "llmserve", "default")


def prompt_text(messages: SequenceABC[Message]) -> str:
    return "\n".join(message.content for message in messages)


def has_code(text: str) -> bool:
    return "```" in text


class Router:
    def __init__(
        self, config: RouterConfig, backend_names: SequenceABC[str], model_ids: dict[str, str]
    ) -> None:
        self.config = config
        self.names = tuple(backend_names)
        self.model_ids = dict(model_ids)

    def resolve_model(self, requested: str) -> str | None:
        if not requested or requested.lower() in AUTO_MODELS:
            return None
        if requested in self.names:
            return requested
        for name, model_id in self.model_ids.items():
            if model_id == requested:
                return name
        raise RequestError(404, f"model {requested} is not served by this gateway", "model_not_found")

    def matches(self, rule: RouterRule, text: str, prompt_tokens: int, max_tokens: int) -> bool:
        if rule.min_prompt_tokens is not None and prompt_tokens < rule.min_prompt_tokens:
            return False
        if rule.max_prompt_tokens is not None and prompt_tokens > rule.max_prompt_tokens:
            return False
        if rule.min_max_tokens is not None and max_tokens < rule.min_max_tokens:
            return False
        if rule.requires_code and not has_code(text):
            return False
        if rule.keywords:
            lowered = text.lower()
            if not any(keyword.lower() in lowered for keyword in rule.keywords):
                return False
        return True

    def route(self, requested: str, text: str, prompt_tokens: int, max_tokens: int) -> str:
        explicit = self.resolve_model(requested)
        if explicit is not None:
            return explicit
        for rule in self.config.rules:
            if self.matches(rule, text, prompt_tokens, max_tokens):
                return rule.backend
        return self.config.default

    def chain(self, primary: str) -> list[str]:
        order = [primary]
        for name in self.config.fallback:
            if name not in order:
                order.append(name)
        return order
