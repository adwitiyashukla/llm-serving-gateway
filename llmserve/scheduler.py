from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict, deque
from typing import Any

from llmserve.config import Config
from llmserve.engine import StepOutput
from llmserve.metrics import Metrics
from llmserve.models import ModelManager
from llmserve.types import FinishReason, RequestError, Sequence, SequenceState

IDLE_SLEEP = 0.002


class Stream:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()

    def push(self, item: Any) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self) -> Any:
        return await self._queue.get()


class Scheduler:
    def __init__(self, config: Config, manager: ModelManager, metrics: Metrics) -> None:
        self.config = config
        self.manager = manager
        self.metrics = metrics
        self._waiting: deque[tuple[Sequence, Stream]] = deque()
        self._running: dict[str, list[tuple[Sequence, Stream]]] = defaultdict(list)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.steps = 0
        self.occupancy_total = 0
        self.occupancy_samples = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="llmserve-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._waiting)

    def running_count(self) -> int:
        with self._lock:
            return sum(len(items) for items in self._running.values())

    def submit(self, sequence: Sequence, stream: Stream) -> None:
        with self._lock:
            if len(self._waiting) >= self.config.engine.max_queue_depth:
                raise RequestError(429, "gateway queue is full", "queue_full")
            self._waiting.append((sequence, stream))
        self._wake.set()

    def abort(self, request_id: str) -> bool:
        with self._lock:
            for index, (sequence, stream) in enumerate(self._waiting):
                if sequence.request_id == request_id:
                    del self._waiting[index]
                    sequence.state = SequenceState.ABORTED
                    sequence.finish_reason = FinishReason.ABORT.value
                    stream.push(None)
                    return True
            for items in self._running.values():
                for sequence, _ in items:
                    if sequence.request_id == request_id:
                        sequence.state = SequenceState.ABORTED
                        return True
        return False

    def _admit(self) -> None:
        limit = self.config.engine.max_batch_size
        prefills = 0
        while prefills < self.config.engine.max_prefills_per_step:
            with self._lock:
                if not self._waiting:
                    return
                sequence, stream = self._waiting[0]
                if len(self._running[sequence.backend_name]) >= limit:
                    return
                self._waiting.popleft()
            if sequence.state is SequenceState.ABORTED:
                stream.push(None)
                continue
            try:
                in_use = set(self._active_backends())
                engine = self.manager.get(sequence.backend_name, in_use)
                output = engine.prefill(sequence)
            except RequestError as error:
                self._fail(sequence, stream, error.message)
                prefills += 1
                continue
            except Exception as error:
                self._fail(sequence, stream, str(error))
                prefills += 1
                continue
            self.metrics.observe_prefix_hit(sequence.cached_prefix_tokens)
            with self._lock:
                self._running[sequence.backend_name].append((sequence, stream))
            self._emit(output, stream)
            prefills += 1

    def _active_backends(self) -> list[str]:
        with self._lock:
            return [name for name, items in self._running.items() if items]

    def _fail(self, sequence: Sequence, stream: Stream, message: str) -> None:
        sequence.state = SequenceState.ABORTED
        sequence.finish_reason = FinishReason.ERROR.value
        sequence.error = message
        stream.push(RequestError(503, message, "backend_error"))
        stream.push(None)

    def _emit(self, output: StepOutput, stream: Stream) -> None:
        sequence = output.sequence
        if output.delta:
            stream.push(output.delta)
        if output.finished:
            self.metrics.observe_completion(sequence)
            stream.push(None)

    def _step_group(self, name: str) -> None:
        with self._lock:
            items = list(self._running[name])
        if not items:
            return
        for sequence, stream in items:
            if sequence.state is SequenceState.ABORTED:
                stream.push(None)
        alive = [(sequence, stream) for sequence, stream in items if sequence.state is SequenceState.RUNNING]
        if not alive:
            with self._lock:
                self._running[name] = []
            return
        engine = self.manager.get(name, set(self._active_backends()))
        try:
            outputs = engine.decode_step([sequence for sequence, _ in alive])
        except Exception as error:
            for sequence, stream in alive:
                self._fail(sequence, stream, str(error))
            with self._lock:
                self._running[name] = []
            return
        self.steps += 1
        self.occupancy_total += len(alive)
        self.occupancy_samples += 1
        self.metrics.observe_batch(len(alive))
        survivors = []
        for output, (sequence, stream) in zip(outputs, alive, strict=True):
            self._emit(output, stream)
            if not output.finished:
                survivors.append((sequence, stream))
        with self._lock:
            self._running[name] = survivors

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._admit()
            names = self._active_backends()
            if not names:
                if self.queue_depth() == 0:
                    self._wake.wait(timeout=0.05)
                    self._wake.clear()
                else:
                    time.sleep(IDLE_SLEEP)
                continue
            for name in names:
                self._step_group(name)

    def mean_occupancy(self) -> float:
        if not self.occupancy_samples:
            return 0.0
        return self.occupancy_total / self.occupancy_samples
