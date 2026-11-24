from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence as SequenceABC
from typing import Any

import torch

from llmserve.types import Message

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
SYNTHETIC_PAD = 0
SYNTHETIC_BOS = 1
SYNTHETIC_EOS = 2
SYNTHETIC_SEP = 3
SYNTHETIC_OFFSET = 4


class Backend(ABC):
    def __init__(self, name: str, model_id: str, footprint_gb: float) -> None:
        self.name = name
        self.model_id = model_id
        self.footprint_gb = footprint_gb

    @abstractmethod
    def encode_chat(self, messages: SequenceABC[Message]) -> list[int]: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, token_ids: SequenceABC[int]) -> str: ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past: Any,
    ) -> tuple[torch.Tensor, Any]: ...

    @abstractmethod
    def embed(self, token_ids: SequenceABC[int]) -> list[float]: ...

    @property
    @abstractmethod
    def eos_ids(self) -> tuple[int, ...]: ...

    @property
    @abstractmethod
    def context_window(self) -> int: ...

    def unload(self) -> None:
        return None


class TorchBackend(Backend):
    def __init__(
        self, name: str, model_id: str, footprint_gb: float, model: Any, eos_ids: tuple[int, ...]
    ) -> None:
        super().__init__(name, model_id, footprint_gb)
        self._model = model
        self._eos_ids = eos_ids
        self._model.eval()

    @property
    def eos_ids(self) -> tuple[int, ...]:
        return self._eos_ids

    @property
    def context_window(self) -> int:
        return int(getattr(self._model.config, "max_position_embeddings", 4096))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past: Any,
    ) -> tuple[torch.Tensor, Any]:
        with torch.inference_mode():
            out = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
        return out.logits.float(), out.past_key_values

    def embed(self, token_ids: SequenceABC[int]) -> list[float]:
        if not token_ids:
            return []
        with torch.inference_mode():
            table = self._model.get_input_embeddings()
            ids = torch.tensor(list(token_ids), dtype=torch.long)
            vec = table(ids).float().mean(dim=0)
            norm = float(torch.linalg.vector_norm(vec))
        if norm == 0.0 or math.isnan(norm):
            return [0.0] * int(vec.shape[0])
        return (vec / norm).tolist()

    def unload(self) -> None:
        self._model = None


class HFBackend(TorchBackend):
    def __init__(
        self,
        name: str,
        model_id: str,
        dtype: str = "float32",
        footprint_gb: float = 1.0,
        local_files_only: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        torch_dtype = DTYPES[dtype]
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, local_files_only=local_files_only
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, local_files_only=local_files_only
            )
        eos = []
        if self.tokenizer.eos_token_id is not None:
            eos.append(int(self.tokenizer.eos_token_id))
        for extra in ("<|im_end|>", "<|endoftext|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(extra)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in eos:
                eos.append(token_id)
        super().__init__(name, model_id, footprint_gb, model, tuple(eos))

    def encode_chat(self, messages: SequenceABC[Message]) -> list[int]:
        payload = [message.as_dict() for message in messages]
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(payload, tokenize=True, add_generation_prompt=True)
        else:
            text = "".join(f"{message.role}: {message.content}\n" for message in messages) + "assistant:"
            ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        return [int(token) for token in ids]

    def encode(self, text: str) -> list[int]:
        return [int(token) for token in self.tokenizer(text, add_special_tokens=False)["input_ids"]]

    def decode(self, token_ids: SequenceABC[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


class SyntheticBackend(TorchBackend):
    def __init__(
        self,
        name: str = "synthetic",
        model_id: str = "synthetic-tiny",
        dtype: str = "float32",
        footprint_gb: float = 0.05,
        seed: int = 0,
        hidden_size: int = 64,
        layers: int = 2,
    ) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM

        generator = torch.random.get_rng_state()
        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=SYNTHETIC_OFFSET + 256,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            pad_token_id=SYNTHETIC_PAD,
            bos_token_id=SYNTHETIC_BOS,
            eos_token_id=SYNTHETIC_EOS,
        )
        model = LlamaForCausalLM(config).to(DTYPES[dtype])
        torch.random.set_rng_state(generator)
        super().__init__(name, model_id, footprint_gb, model, (SYNTHETIC_EOS,))

    def encode(self, text: str) -> list[int]:
        return [SYNTHETIC_OFFSET + byte for byte in text.encode("utf-8")]

    def encode_chat(self, messages: SequenceABC[Message]) -> list[int]:
        ids = [SYNTHETIC_BOS]
        for message in messages:
            ids.extend(self.encode(message.role))
            ids.append(SYNTHETIC_SEP)
            ids.extend(self.encode(message.content))
            ids.append(SYNTHETIC_SEP)
        ids.extend(self.encode("assistant"))
        ids.append(SYNTHETIC_SEP)
        return ids

    def decode(self, token_ids: SequenceABC[int]) -> str:
        payload = bytes(token - SYNTHETIC_OFFSET for token in token_ids if token >= SYNTHETIC_OFFSET)
        return payload.decode("utf-8", errors="replace")


def build_backend(spec: Any, local_files_only: bool = False) -> Backend:
    if spec.kind == "synthetic":
        return SyntheticBackend(
            name=spec.name,
            model_id=spec.model,
            dtype=spec.dtype,
            footprint_gb=spec.footprint_gb,
            hidden_size=spec.hidden_size,
            layers=spec.layers,
        )
    return HFBackend(
        name=spec.name,
        model_id=spec.model,
        dtype=spec.dtype,
        footprint_gb=spec.footprint_gb,
        local_files_only=local_files_only,
    )
