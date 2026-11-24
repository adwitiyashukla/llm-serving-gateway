from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmserve.backends import build_backend
from llmserve.config import Config
from llmserve.gateway import Gateway
from llmserve.types import GenerationRequest, Message, RequestError


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = Field(default_factory=list)
    seed: int = 0
    stream: bool = False


def error_response(error: RequestError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"message": error.message, "type": error.code, "code": error.code}},
    )


def build_gateway(config: Config, factory: Any = build_backend, local_files_only: bool = False) -> Gateway:
    return Gateway(config, factory=factory, local_files_only=local_files_only)


def create_app(config: Config, gateway: Gateway | None = None) -> FastAPI:
    engine = gateway if gateway is not None else build_gateway(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="llmserve", version="0.1.0", lifespan=lifespan)
    app.state.gateway = engine
    app.state.config = config

    @app.exception_handler(RequestError)
    async def handle_request_error(request: Request, error: RequestError) -> JSONResponse:
        return error_response(error)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "loaded": engine.manager.loaded_names()}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        created = int(time.time())
        entries = [
            {"id": backend.name, "object": "model", "created": created, "owned_by": "llmserve"}
            for backend in config.backends
        ]
        entries.append({"id": "auto", "object": "model", "created": created, "owned_by": "llmserve"})
        return {"object": "list", "data": entries}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(engine.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return engine.stats()

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatRequest, authorization: str | None = Header(default=None)) -> Any:
        try:
            key = engine.admission.authorize(authorization)
            params = engine.build_params(body.max_tokens, body.temperature, body.top_p, body.stop, body.seed)
            if not body.messages:
                raise RequestError(400, "messages must not be empty")
            request = GenerationRequest(
                request_id=uuid.uuid4().hex,
                model=body.model,
                messages=tuple(Message(role=item.role, content=item.content) for item in body.messages),
                params=params,
                api_key=key,
            )
            prompt_estimate = engine.estimate_tokens(request.messages)
            engine.admission.check(key, prompt_estimate, params.max_tokens)
        except RequestError as error:
            engine.metrics.observe_rejection(error.code)
            return error_response(error)

        if not body.stream:
            try:
                completion = await engine.complete(request)
            except RequestError as error:
                engine.metrics.observe_rejection(error.code)
                return error_response(error)
            return JSONResponse(content=_completion_payload(request, completion))

        async def event_stream() -> AsyncIterator[str]:
            created = int(time.time())
            chunk_id = f"chatcmpl-{request.request_id}"
            try:
                async for kind, payload in engine.stream(request):
                    if kind == "delta":
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {"index": 0, "delta": {"content": payload}, "finish_reason": None}
                                ],
                            }
                        )
                    else:
                        yield _sse(
                            {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": payload.backend,
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": payload.finish_reason}
                                ],
                                "llmserve": _extension(payload),
                            }
                        )
            except RequestError as error:
                engine.metrics.observe_rejection(error.code)
                yield _sse({"error": {"message": error.message, "type": error.code}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _extension(completion: Any) -> dict[str, Any]:
    return {
        "backend": completion.backend,
        "source": completion.source,
        "cached_prefix_tokens": completion.cached_prefix_tokens,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "ttft_s": round(completion.ttft_s, 6),
        "total_s": round(completion.total_s, 6),
    }


def _completion_payload(request: GenerationRequest, completion: Any) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{request.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion.backend,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion.text},
                "finish_reason": completion.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.prompt_tokens + completion.completion_tokens,
        },
        "llmserve": _extension(completion),
    }
