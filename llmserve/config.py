from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_DTYPES = ("float32", "bfloat16", "float16")
VALID_KINDS = ("hf", "synthetic")


class ConfigError(Exception):
    pass


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table")
    return value


def _get(section: dict[str, Any], path: str, key: str, default: Any) -> Any:
    return section.get(key, default)


def _as_int(value: Any, path: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path} must be <= {maximum}, got {value}")
    return value


def _as_float(value: Any, path: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path} must be <= {maximum}, got {value}")
    return value


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _as_str(value: Any, path: str, allowed: tuple[str, ...] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non empty string")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"{path} must be one of {', '.join(allowed)}, got {value}")
    return value


def _as_str_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{path} must be a list of strings")
    return list(value)


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_keys: tuple[str, ...] = ()
    request_timeout_s: float = 180.0


@dataclass(frozen=True)
class EngineConfig:
    max_batch_size: int = 8
    max_queue_depth: int = 64
    max_prefills_per_step: int = 1
    max_prompt_tokens: int = 2048
    default_max_tokens: int = 128
    hard_max_tokens: int = 512
    seed: int = 0
    torch_threads: int = 0


@dataclass(frozen=True)
class MemoryConfig:
    budget_gb: float = 6.5


@dataclass(frozen=True)
class PrefixCacheConfig:
    enabled: bool = True
    block_tokens: int = 16
    max_blocks: int = 512


@dataclass(frozen=True)
class ExactCacheConfig:
    enabled: bool = True
    max_entries: int = 512


@dataclass(frozen=True)
class SemanticCacheConfig:
    enabled: bool = True
    max_entries: int = 512
    threshold: float = 0.999
    max_temperature: float = 0.0


@dataclass(frozen=True)
class AdmissionConfig:
    requests_per_minute: int = 600
    tokens_per_minute: int = 120000
    burst_requests: int = 60


@dataclass(frozen=True)
class RouterRule:
    backend: str
    min_prompt_tokens: int | None = None
    max_prompt_tokens: int | None = None
    min_max_tokens: int | None = None
    keywords: tuple[str, ...] = ()
    requires_code: bool = False


@dataclass(frozen=True)
class RouterConfig:
    default: str
    fallback: tuple[str, ...] = ()
    rules: tuple[RouterRule, ...] = ()


@dataclass(frozen=True)
class BackendConfig:
    name: str
    kind: str
    model: str
    dtype: str = "float32"
    footprint_gb: float = 1.0
    resident: bool = False
    hidden_size: int = 64
    layers: int = 2


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    prefix_cache: PrefixCacheConfig = field(default_factory=PrefixCacheConfig)
    exact_cache: ExactCacheConfig = field(default_factory=ExactCacheConfig)
    semantic_cache: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    router: RouterConfig = field(default_factory=lambda: RouterConfig(default="fast"))
    backends: tuple[BackendConfig, ...] = ()

    @property
    def backend_names(self) -> tuple[str, ...]:
        return tuple(backend.name for backend in self.backends)

    def backend(self, name: str) -> BackendConfig:
        for backend in self.backends:
            if backend.name == name:
                return backend
        raise ConfigError(f"unknown backend {name}")

    @staticmethod
    def from_toml(path: str | Path) -> Config:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return Config.from_dict(data)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Config:
        return _build(data)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        merged = _merge(self.to_dict(), overrides)
        return Config.from_dict(merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "api_keys": list(self.server.api_keys),
                "request_timeout_s": self.server.request_timeout_s,
            },
            "engine": {
                "max_batch_size": self.engine.max_batch_size,
                "max_queue_depth": self.engine.max_queue_depth,
                "max_prefills_per_step": self.engine.max_prefills_per_step,
                "max_prompt_tokens": self.engine.max_prompt_tokens,
                "default_max_tokens": self.engine.default_max_tokens,
                "hard_max_tokens": self.engine.hard_max_tokens,
                "seed": self.engine.seed,
                "torch_threads": self.engine.torch_threads,
            },
            "memory": {"budget_gb": self.memory.budget_gb},
            "prefix_cache": {
                "enabled": self.prefix_cache.enabled,
                "block_tokens": self.prefix_cache.block_tokens,
                "max_blocks": self.prefix_cache.max_blocks,
            },
            "exact_cache": {
                "enabled": self.exact_cache.enabled,
                "max_entries": self.exact_cache.max_entries,
            },
            "semantic_cache": {
                "enabled": self.semantic_cache.enabled,
                "max_entries": self.semantic_cache.max_entries,
                "threshold": self.semantic_cache.threshold,
                "max_temperature": self.semantic_cache.max_temperature,
            },
            "admission": {
                "requests_per_minute": self.admission.requests_per_minute,
                "tokens_per_minute": self.admission.tokens_per_minute,
                "burst_requests": self.admission.burst_requests,
            },
            "router": {
                "default": self.router.default,
                "fallback": list(self.router.fallback),
                "rules": [
                    {
                        "backend": rule.backend,
                        "min_prompt_tokens": rule.min_prompt_tokens,
                        "max_prompt_tokens": rule.max_prompt_tokens,
                        "min_max_tokens": rule.min_max_tokens,
                        "keywords": list(rule.keywords),
                        "requires_code": rule.requires_code,
                    }
                    for rule in self.router.rules
                ],
            },
            "backends": [
                {
                    "name": backend.name,
                    "kind": backend.kind,
                    "model": backend.model,
                    "dtype": backend.dtype,
                    "footprint_gb": backend.footprint_gb,
                    "resident": backend.resident,
                    "hidden_size": backend.hidden_size,
                    "layers": backend.layers,
                }
                for backend in self.backends
            ],
        }


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_server(data: dict[str, Any]) -> ServerConfig:
    section = _section(data, "server")
    defaults = ServerConfig()
    return ServerConfig(
        host=_as_str(section.get("host", defaults.host), "server.host"),
        port=_as_int(section.get("port", defaults.port), "server.port", 1, 65535),
        api_keys=tuple(_as_str_list(section.get("api_keys", []), "server.api_keys")),
        request_timeout_s=_as_float(
            section.get("request_timeout_s", defaults.request_timeout_s), "server.request_timeout_s", 0.001
        ),
    )


def _build_engine(data: dict[str, Any]) -> EngineConfig:
    section = _section(data, "engine")
    defaults = EngineConfig()
    engine = EngineConfig(
        max_batch_size=_as_int(
            section.get("max_batch_size", defaults.max_batch_size), "engine.max_batch_size", 1
        ),
        max_queue_depth=_as_int(
            section.get("max_queue_depth", defaults.max_queue_depth), "engine.max_queue_depth", 1
        ),
        max_prefills_per_step=_as_int(
            section.get("max_prefills_per_step", defaults.max_prefills_per_step),
            "engine.max_prefills_per_step",
            1,
        ),
        max_prompt_tokens=_as_int(
            section.get("max_prompt_tokens", defaults.max_prompt_tokens), "engine.max_prompt_tokens", 1
        ),
        default_max_tokens=_as_int(
            section.get("default_max_tokens", defaults.default_max_tokens), "engine.default_max_tokens", 1
        ),
        hard_max_tokens=_as_int(
            section.get("hard_max_tokens", defaults.hard_max_tokens), "engine.hard_max_tokens", 1
        ),
        seed=_as_int(section.get("seed", defaults.seed), "engine.seed", 0),
        torch_threads=_as_int(
            section.get("torch_threads", defaults.torch_threads), "engine.torch_threads", 0
        ),
    )
    if engine.max_prefills_per_step > engine.max_batch_size:
        raise ConfigError("engine.max_prefills_per_step must be <= engine.max_batch_size")
    if engine.default_max_tokens > engine.hard_max_tokens:
        raise ConfigError("engine.default_max_tokens must be <= engine.hard_max_tokens")
    return engine


def _build_router(data: dict[str, Any], names: tuple[str, ...]) -> RouterConfig:
    section = _section(data, "router")
    if "default" not in section:
        raise ConfigError("router.default is required")
    default = _as_str(section["default"], "router.default")
    if default not in names:
        raise ConfigError(f"router.default must name a backend, got {default}")
    fallback = tuple(_as_str_list(section.get("fallback", []), "router.fallback"))
    for name in fallback:
        if name not in names:
            raise ConfigError(f"router.fallback must name backends, got {name}")
    raw_rules = section.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ConfigError("router.rules must be a list of tables")
    rules = []
    for index, raw in enumerate(raw_rules):
        path = f"router.rules[{index}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must be a table")
        if "backend" not in raw:
            raise ConfigError(f"{path}.backend is required")
        backend = _as_str(raw["backend"], f"{path}.backend")
        if backend not in names:
            raise ConfigError(f"{path}.backend must name a backend, got {backend}")
        rule = RouterRule(
            backend=backend,
            min_prompt_tokens=(
                None
                if raw.get("min_prompt_tokens") is None
                else _as_int(raw["min_prompt_tokens"], f"{path}.min_prompt_tokens", 0)
            ),
            max_prompt_tokens=(
                None
                if raw.get("max_prompt_tokens") is None
                else _as_int(raw["max_prompt_tokens"], f"{path}.max_prompt_tokens", 0)
            ),
            min_max_tokens=(
                None
                if raw.get("min_max_tokens") is None
                else _as_int(raw["min_max_tokens"], f"{path}.min_max_tokens", 0)
            ),
            keywords=tuple(_as_str_list(raw.get("keywords", []), f"{path}.keywords")),
            requires_code=_as_bool(raw.get("requires_code", False), f"{path}.requires_code"),
        )
        if (
            rule.min_prompt_tokens is None
            and rule.max_prompt_tokens is None
            and rule.min_max_tokens is None
            and not rule.keywords
            and not rule.requires_code
        ):
            raise ConfigError(f"{path} needs at least one condition")
        if (
            rule.min_prompt_tokens is not None
            and rule.max_prompt_tokens is not None
            and rule.min_prompt_tokens > rule.max_prompt_tokens
        ):
            raise ConfigError(f"{path}.min_prompt_tokens must be <= max_prompt_tokens")
        rules.append(rule)
    return RouterConfig(default=default, fallback=fallback, rules=tuple(rules))


def _build_backends(data: dict[str, Any]) -> tuple[BackendConfig, ...]:
    raw_backends = data.get("backends", [])
    if not isinstance(raw_backends, list) or not raw_backends:
        raise ConfigError("at least one [[backends]] entry is required")
    backends = []
    seen = set()
    for index, raw in enumerate(raw_backends):
        path = f"backends[{index}]"
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must be a table")
        for key in ("name", "kind", "model"):
            if key not in raw:
                raise ConfigError(f"{path}.{key} is required")
        name = _as_str(raw["name"], f"{path}.name")
        if name in seen:
            raise ConfigError(f"duplicate backend name {name}")
        seen.add(name)
        backends.append(
            BackendConfig(
                name=name,
                kind=_as_str(raw["kind"], f"{path}.kind", VALID_KINDS),
                model=_as_str(raw["model"], f"{path}.model"),
                dtype=_as_str(raw.get("dtype", "float32"), f"{path}.dtype", VALID_DTYPES),
                footprint_gb=_as_float(raw.get("footprint_gb", 1.0), f"{path}.footprint_gb", 0.0001),
                resident=_as_bool(raw.get("resident", False), f"{path}.resident"),
                hidden_size=_as_int(raw.get("hidden_size", 64), f"{path}.hidden_size", 8),
                layers=_as_int(raw.get("layers", 2), f"{path}.layers", 1),
            )
        )
    return tuple(backends)


def _build(data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("config root must be a table")
    backends = _build_backends(data)
    names = tuple(backend.name for backend in backends)

    memory_section = _section(data, "memory")
    memory = MemoryConfig(
        budget_gb=_as_float(
            memory_section.get("budget_gb", MemoryConfig().budget_gb), "memory.budget_gb", 0.0001
        )
    )
    smallest = min(backend.footprint_gb for backend in backends)
    if smallest > memory.budget_gb:
        raise ConfigError(
            f"memory.budget_gb {memory.budget_gb} cannot hold the smallest backend footprint {smallest}"
        )
    resident_total = sum(backend.footprint_gb for backend in backends if backend.resident)
    if resident_total > memory.budget_gb:
        raise ConfigError(
            f"resident backends need {resident_total:.2f} GB "
            f"which exceeds memory.budget_gb {memory.budget_gb}"
        )

    prefix_section = _section(data, "prefix_cache")
    prefix_defaults = PrefixCacheConfig()
    prefix = PrefixCacheConfig(
        enabled=_as_bool(prefix_section.get("enabled", prefix_defaults.enabled), "prefix_cache.enabled"),
        block_tokens=_as_int(
            prefix_section.get("block_tokens", prefix_defaults.block_tokens), "prefix_cache.block_tokens", 1
        ),
        max_blocks=_as_int(
            prefix_section.get("max_blocks", prefix_defaults.max_blocks), "prefix_cache.max_blocks", 1
        ),
    )

    exact_section = _section(data, "exact_cache")
    exact_defaults = ExactCacheConfig()
    exact = ExactCacheConfig(
        enabled=_as_bool(exact_section.get("enabled", exact_defaults.enabled), "exact_cache.enabled"),
        max_entries=_as_int(
            exact_section.get("max_entries", exact_defaults.max_entries), "exact_cache.max_entries", 1
        ),
    )

    semantic_section = _section(data, "semantic_cache")
    semantic_defaults = SemanticCacheConfig()
    semantic = SemanticCacheConfig(
        enabled=_as_bool(
            semantic_section.get("enabled", semantic_defaults.enabled), "semantic_cache.enabled"
        ),
        max_entries=_as_int(
            semantic_section.get("max_entries", semantic_defaults.max_entries),
            "semantic_cache.max_entries",
            1,
        ),
        threshold=_as_float(
            semantic_section.get("threshold", semantic_defaults.threshold),
            "semantic_cache.threshold",
            0.0,
            1.0,
        ),
        max_temperature=_as_float(
            semantic_section.get("max_temperature", semantic_defaults.max_temperature),
            "semantic_cache.max_temperature",
            0.0,
        ),
    )

    admission_section = _section(data, "admission")
    admission_defaults = AdmissionConfig()
    admission = AdmissionConfig(
        requests_per_minute=_as_int(
            admission_section.get("requests_per_minute", admission_defaults.requests_per_minute),
            "admission.requests_per_minute",
            1,
        ),
        tokens_per_minute=_as_int(
            admission_section.get("tokens_per_minute", admission_defaults.tokens_per_minute),
            "admission.tokens_per_minute",
            1,
        ),
        burst_requests=_as_int(
            admission_section.get("burst_requests", admission_defaults.burst_requests),
            "admission.burst_requests",
            1,
        ),
    )

    return Config(
        server=_build_server(data),
        engine=_build_engine(data),
        memory=memory,
        prefix_cache=prefix,
        exact_cache=exact,
        semantic_cache=semantic,
        admission=admission,
        router=_build_router(data, names),
        backends=backends,
    )
