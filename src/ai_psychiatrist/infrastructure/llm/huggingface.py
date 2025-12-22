"""HuggingFace LLM client implementation.

This module provides an optional backend that uses HuggingFace Transformers to run
official model weights (e.g., Google's MedGemma). It is intentionally implemented
with lazy imports so the core project can run without heavyweight ML dependencies.
"""

from __future__ import annotations

import asyncio
import importlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_psychiatrist.config import BackendSettings, LLMBackend, ModelSettings
from ai_psychiatrist.domain.exceptions import LLMError, LLMTimeoutError
from ai_psychiatrist.infrastructure.llm.model_aliases import resolve_model_name
from ai_psychiatrist.infrastructure.llm.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from ai_psychiatrist.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


def _l2_normalize(vec: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return tuple(x / norm for x in vec)
    return tuple(vec)


class MissingHuggingFaceDependenciesError(ImportError):
    """Raised when HuggingFace backend is selected without dependencies installed."""


@dataclass(frozen=True, slots=True)
class _TransformersDeps:
    torch: Any
    transformers: Any
    sentence_transformers: Any


def _load_transformers_deps() -> _TransformersDeps:
    """Import HuggingFace dependencies lazily with a helpful error message."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as e:
        msg = (
            "HuggingFace backend requires optional dependencies. "
            "Install with: `pip install 'ai-psychiatrist[hf]'`"
        )
        raise MissingHuggingFaceDependenciesError(msg) from e

    return _TransformersDeps(
        torch=torch,
        transformers=transformers,
        sentence_transformers=sentence_transformers,
    )


class HuggingFaceClient:
    """HuggingFace Transformers client for chat and embeddings.

    Implements the ChatClient and EmbeddingClient protocols (via matching signatures),
    and also provides `simple_chat` / `simple_embed` convenience methods used by agents.
    """

    def __init__(
        self,
        backend_settings: BackendSettings,
        model_settings: ModelSettings,
    ) -> None:
        self._backend_settings = backend_settings
        self._model_settings = model_settings

        self._chat_models: dict[str, tuple[Any, Any]] = {}
        self._embedding_models: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        """Release loaded models (best-effort).

        HuggingFace models are held in process memory; clearing references allows
        garbage collection to reclaim memory.
        """
        self._chat_models.clear()
        self._embedding_models.clear()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model_id = resolve_model_name(request.model, LLMBackend.HUGGINGFACE)
        model, tokenizer = await self._get_chat_model(model_id)
        prompt = self._render_prompt(tokenizer, request.messages)

        async def _generate_async() -> str:
            return await asyncio.to_thread(
                self._generate_text,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )

        try:
            content = await asyncio.wait_for(_generate_async(), timeout=request.timeout_seconds)
        except TimeoutError as e:
            raise LLMTimeoutError(request.timeout_seconds) from e
        except Exception as e:
            logger.error("HuggingFace chat failed", model=model_id, error=str(e))
            raise LLMError(f"HuggingFace chat failed: {e}") from e

        return ChatResponse(
            content=content,
            model=model_id,
            done=True,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model_id = resolve_model_name(request.model, LLMBackend.HUGGINGFACE)
        model = await self._get_embedding_model(model_id)

        async def _embed_async() -> tuple[float, ...]:
            return await asyncio.to_thread(
                self._encode_embedding,
                model=model,
                text=request.text,
            )

        try:
            embedding = await asyncio.wait_for(_embed_async(), timeout=request.timeout_seconds)
        except TimeoutError as e:
            raise LLMTimeoutError(request.timeout_seconds) from e
        except Exception as e:
            logger.error("HuggingFace embed failed", model=model_id, error=str(e))
            raise LLMError(f"HuggingFace embedding failed: {e}") from e

        if request.dimension is not None:
            embedding = embedding[: request.dimension]

        return EmbeddingResponse(
            embedding=embedding,
            model=model_id,
        )

    async def simple_chat(
        self,
        user_prompt: str,
        system_prompt: str = "",
        model: str | None = None,
        temperature: float = 0.2,
        top_k: int = 20,
        top_p: float = 0.8,
    ) -> str:
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=user_prompt))

        request = ChatRequest(
            messages=messages,
            model=model or self._model_settings.qualitative_model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            timeout_seconds=180,
        )
        response = await self.chat(request)
        return response.content

    async def simple_embed(
        self,
        text: str,
        model: str | None = None,
        dimension: int | None = None,
    ) -> tuple[float, ...]:
        request = EmbeddingRequest(
            text=text,
            model=model or self._model_settings.embedding_model,
            dimension=dimension,
            timeout_seconds=120,
        )
        response = await self.embed(request)
        return response.embedding

    async def _get_chat_model(self, model_id: str) -> tuple[Any, Any]:
        async with self._lock:
            cached = self._chat_models.get(model_id)
            if cached is not None:
                return cached

            deps = _load_transformers_deps()
            torch = deps.torch
            transformers = deps.transformers

            auto_model = transformers.AutoModelForCausalLM
            auto_tokenizer = transformers.AutoTokenizer

            model_kwargs: dict[str, Any] = {
                "torch_dtype": getattr(torch, "bfloat16", None) or torch.float32,
                "low_cpu_mem_usage": True,
            }
            if self._backend_settings.hf_cache_dir is not None:
                model_kwargs["cache_dir"] = str(self._backend_settings.hf_cache_dir)
            if self._backend_settings.hf_token is not None:
                model_kwargs["token"] = self._backend_settings.hf_token

            # Device / sharding
            if self._backend_settings.hf_device == "auto":
                model_kwargs["device_map"] = "auto"

            # Optional quantization (best-effort, may require additional deps)
            quant = self._backend_settings.hf_quantization
            if quant is not None:
                model_kwargs["quantization_config"] = self._build_quantization_config(
                    transformers=transformers,
                    quantization=quant,
                )

            tokenizer_kwargs: dict[str, Any] = {}
            if self._backend_settings.hf_cache_dir is not None:
                tokenizer_kwargs["cache_dir"] = str(self._backend_settings.hf_cache_dir)
            if self._backend_settings.hf_token is not None:
                tokenizer_kwargs["token"] = self._backend_settings.hf_token

            logger.info("Loading HuggingFace chat model", model_id=model_id)
            model = auto_model.from_pretrained(model_id, **model_kwargs).eval()
            tokenizer = auto_tokenizer.from_pretrained(model_id, **tokenizer_kwargs)

            # Manual device placement if not using device_map="auto"
            if self._backend_settings.hf_device != "auto":
                device = self._backend_settings.hf_device
                if hasattr(torch, "device"):
                    model = model.to(torch.device(device))

            self._chat_models[model_id] = (model, tokenizer)
            return model, tokenizer

    async def _get_embedding_model(self, model_id: str) -> Any:
        async with self._lock:
            cached = self._embedding_models.get(model_id)
            if cached is not None:
                return cached

            deps = _load_transformers_deps()
            sentence_transformers = deps.sentence_transformers

            st_cls = sentence_transformers.SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self._backend_settings.hf_cache_dir is not None:
                # SentenceTransformer uses `cache_folder` rather than `cache_dir`
                kwargs["cache_folder"] = str(self._backend_settings.hf_cache_dir)
            if self._backend_settings.hf_device != "auto":
                kwargs["device"] = self._backend_settings.hf_device

            logger.info("Loading HuggingFace embedding model", model_id=model_id)
            model = st_cls(model_id, **kwargs)
            self._embedding_models[model_id] = model
            return model

    @staticmethod
    def _build_quantization_config(*, transformers: Any, quantization: str) -> Any:
        """Build a transformers quantization config.

        This is best-effort: the selected quantization may require optional packages.
        """
        if quantization == "int4":
            try:
                torch_ao = transformers.TorchAoConfig
            except AttributeError as e:
                msg = "int4 quantization requires transformers with TorchAoConfig support"
                raise LLMError(msg) from e
            return torch_ao("int4_weight_only", group_size=128)

        if quantization == "int8":
            try:
                bnb = transformers.BitsAndBytesConfig
            except AttributeError as e:
                msg = "int8 quantization requires transformers BitsAndBytesConfig support"
                raise LLMError(msg) from e
            return bnb(load_in_8bit=True)

        msg = f"Unknown quantization: {quantization}"
        raise ValueError(msg)

    @staticmethod
    def _render_prompt(tokenizer: Any, messages: Sequence[ChatMessage]) -> str:
        """Render a chat prompt for a list of ChatMessage objects."""
        chat = [{"role": m.role, "content": m.content} for m in messages]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            return str(rendered)

        # Fallback: concatenate as simple transcript.
        parts: list[str] = []
        for m in messages:
            parts.append(f"{m.role}: {m.content}")
        parts.append("assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _generate_text(
        *,
        model: Any,
        tokenizer: Any,
        prompt: str,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if input_ids is None:
            raise LLMError("Tokenizer did not produce input_ids")

        # Move tensors to model device if possible.
        if hasattr(model, "device"):
            device = model.device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 1024,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None),
        }

        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "top_p": float(top_p),
                }
            )

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        # Strip the prompt part.
        gen_ids = output_ids[0][input_ids.shape[-1] :]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return str(decoded).strip()

    @staticmethod
    def _encode_embedding(*, model: Any, text: str) -> tuple[float, ...]:
        # SentenceTransformer returns numpy arrays. Normalize at source when available.
        encoded = model.encode([text], normalize_embeddings=True)
        vec = encoded[0].tolist()
        return _l2_normalize(vec)
