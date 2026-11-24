from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any


class RequestError(Exception):
    def __init__(self, status_code: int, message: str, code: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int = 0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    model: str
    messages: tuple[Message, ...]
    params: SamplingParams
    api_key: str = ""
    created: float = field(default_factory=time.time)


class SequenceState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"


@dataclass
class TokenChunk:
    text: str
    token_id: int
    index: int


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    backend: str
    source: str = "generated"
    cached_prefix_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0


@dataclass
class Sequence:
    request_id: str
    backend_name: str
    prompt_ids: list[int]
    params: SamplingParams
    state: SequenceState = SequenceState.WAITING
    cache: Any = None
    length: int = 0
    next_token: int | None = None
    output_ids: list[int] = field(default_factory=list)
    text: str = ""
    arrival: float = field(default_factory=time.perf_counter)
    started: float = 0.0
    first_token_at: float = 0.0
    finished_at: float = 0.0
    finish_reason: str = ""
    error: str = ""
    cached_prefix_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.output_ids)

    @property
    def done(self) -> bool:
        return self.state in (SequenceState.FINISHED, SequenceState.ABORTED)
