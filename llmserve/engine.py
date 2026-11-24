from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from llmserve.backends import Backend
from llmserve.config import EngineConfig
from llmserve.kvcache import PrefixCache, assemble, batch_inputs, split
from llmserve.types import FinishReason, SamplingParams, Sequence, SequenceState


@dataclass
class StepOutput:
    sequence: Sequence
    delta: str
    finished: bool


def sample_token(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    if params.greedy:
        return int(torch.argmax(logits))
    scaled = logits / max(params.temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)
    if params.top_p < 1.0:
        ordered, indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(ordered, dim=-1)
        keep = cumulative - ordered < params.top_p
        keep[0] = True
        ordered = ordered * keep
        total = float(ordered.sum())
        if total <= 0.0:
            return int(indices[0])
        ordered = ordered / total
        choice = int(torch.multinomial(ordered, 1, generator=generator))
        return int(indices[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


class Engine:
    def __init__(self, backend: Backend, config: EngineConfig, prefix_cache: PrefixCache) -> None:
        self.backend = backend
        self.config = config
        self.prefix_cache = prefix_cache
        self.prefill_calls = 0
        self.decode_steps = 0
        self.batch_occupancy: list[int] = []

    def _generator(self, params: SamplingParams) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + params.seed)
        return generator

    def prefill(self, sequence: Sequence) -> StepOutput:
        ids = sequence.prompt_ids
        matched, cached = self.prefix_cache.lookup(ids)
        if cached is not None and matched > 0:
            past, width = assemble([cached])
            remaining = ids[matched:]
            input_ids = torch.tensor([remaining], dtype=torch.long)
            attention_mask = torch.ones(1, width + len(remaining), dtype=torch.long)
            position_ids = torch.arange(matched, len(ids), dtype=torch.long).unsqueeze(0)
        else:
            past = None
            input_ids = torch.tensor([ids], dtype=torch.long)
            attention_mask = torch.ones(1, len(ids), dtype=torch.long)
            position_ids = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)

        logits, new_past = self.backend.forward(input_ids, attention_mask, position_ids, past)
        self.prefill_calls += 1
        sequence.cache = split(new_past, [len(ids)])[0]
        sequence.length = len(ids)
        sequence.cached_prefix_tokens = matched
        self.prefix_cache.insert(ids, sequence.cache)

        token = sample_token(logits[0, -1], sequence.params, self._generator(sequence.params))
        sequence.state = SequenceState.RUNNING
        sequence.started = time.perf_counter()
        return self._accept(sequence, token)

    def decode_step(self, sequences: list[Sequence]) -> list[StepOutput]:
        if not sequences:
            return []
        caches = [sequence.cache for sequence in sequences]
        lengths = [sequence.length for sequence in sequences]
        tokens = [int(sequence.next_token) for sequence in sequences]
        past, width = assemble(caches)
        input_ids, attention_mask, position_ids = batch_inputs(tokens, lengths, width)
        logits, new_past = self.backend.forward(input_ids, attention_mask, position_ids, past)
        self.decode_steps += 1
        self.batch_occupancy.append(len(sequences))
        grown = [length + 1 for length in lengths]
        updated = split(new_past, grown)
        outputs = []
        for row, sequence in enumerate(sequences):
            sequence.cache = updated[row]
            sequence.length = grown[row]
            token = sample_token(logits[row, -1], sequence.params, self._generator(sequence.params))
            outputs.append(self._accept(sequence, token))
        return outputs

    def _accept(self, sequence: Sequence, token: int) -> StepOutput:
        now = time.perf_counter()
        if not sequence.output_ids:
            sequence.first_token_at = now
        sequence.output_ids.append(token)
        sequence.next_token = token

        if token in self.backend.eos_ids:
            sequence.output_ids.pop()
            return self._finish(sequence, FinishReason.STOP.value, "")

        full = self.backend.decode(sequence.output_ids)
        delta = full[len(sequence.text) :]
        sequence.text = full

        for stop in sequence.params.stop:
            position = sequence.text.find(stop)
            if position >= 0:
                trimmed = sequence.text[:position]
                delta = (
                    trimmed[len(sequence.text) - len(delta) :]
                    if len(trimmed) > len(sequence.text) - len(delta)
                    else ""
                )
                sequence.text = trimmed
                return self._finish(sequence, FinishReason.STOP.value, delta)

        if len(sequence.output_ids) >= sequence.params.max_tokens:
            return self._finish(sequence, FinishReason.LENGTH.value, delta)
        if sequence.length + 1 >= self.backend.context_window:
            return self._finish(sequence, FinishReason.LENGTH.value, delta)
        return StepOutput(sequence=sequence, delta=delta, finished=False)

    def _finish(self, sequence: Sequence, reason: str, delta: str) -> StepOutput:
        sequence.state = SequenceState.FINISHED
        sequence.finish_reason = reason
        sequence.finished_at = time.perf_counter()
        sequence.cache = None
        return StepOutput(sequence=sequence, delta=delta, finished=True)

    def mean_occupancy(self) -> float:
        if not self.batch_occupancy:
            return 0.0
        return sum(self.batch_occupancy) / len(self.batch_occupancy)
