"""Centralized configuration using Pydantic Settings.

This module provides validated baseline defaults for the AI Psychiatrist system.
Defaults are derived from the paper's reported settings and subsequent repo fixes/ablations.
All settings can be overridden via environment variables; `.env.example` is the recommended
run configuration.

Paper references:
- Section 2.2: Model configuration (Gemma 3 27B, Qwen 3 8B Embedding)
- Section 2.3.5: Agentic system configuration (Ollama API)
- Appendix D: Hyperparameters (chunk_size=8, step_size=2, N_example=2, dim=4096)
- Appendix F: MedGemma quantitative results (MAE 0.505)
"""

from __future__ import annotations

import math
import os
import warnings
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Skip reading .env file during testing to use code defaults
ENV_FILE = None if os.environ.get("TESTING") else ".env"
ENV_FILE_ENCODING = "utf-8"


class LLMBackend(str, Enum):
    """Supported LLM backends.

    The default backend is Ollama (local HTTP server). A HuggingFace backend
    is supported for accessing official model weights (e.g., MedGemma).
    """

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class EmbeddingBackend(str, Enum):
    """Embedding backend selection (separate from LLM backend)."""

    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class HuggingFaceSettings(BaseSettings):
    """HuggingFace-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="HF_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    default_chat_timeout: int = Field(
        default=180,
        description="Default timeout for chat requests",
    )
    default_embed_timeout: int = Field(
        default=120,
        description="Default timeout for embedding requests",
    )
    max_new_tokens: int = Field(
        default=1024,
        description="Maximum tokens to generate (HuggingFace)",
    )
    quantization_group_size: int = Field(
        default=128,
        description="Group size for int4 quantization",
    )


class ServerSettings(BaseSettings):
    """Server-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    ad_hoc_participant_id: int = Field(
        default=999_999,
        description="Participant ID assigned to ad-hoc text submissions",
    )


class BackendSettings(BaseSettings):
    """LLM backend configuration.

    This selects which runtime implementation to use for chat.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    backend: LLMBackend = Field(
        default=LLMBackend.OLLAMA,
        description="LLM backend implementation",
    )

    # HuggingFace-specific settings (only used when backend=huggingface)
    hf_device: str = Field(
        default="auto",
        description="HuggingFace device selection: auto, cpu, cuda, mps",
    )
    hf_quantization: Literal["int4", "int8"] | None = Field(
        default=None,
        description="Optional HuggingFace quantization: int4 or int8",
    )
    hf_cache_dir: Path | None = Field(
        default=None,
        description="Optional HuggingFace cache directory",
    )
    hf_token: str | None = Field(
        default=None,
        repr=False,
        description="Optional HuggingFace token (prefer huggingface-cli login)",
    )


class EmbeddingBackendSettings(BaseSettings):
    """Embedding backend configuration."""

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    backend: EmbeddingBackend = Field(
        default=EmbeddingBackend.HUGGINGFACE,
        description="Embedding backend (huggingface for FP16, ollama for speed)",
    )


class OllamaSettings(BaseSettings):
    """Ollama server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Ollama server host")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama server port")
    timeout_seconds: int = Field(default=600, ge=10, description="Request timeout")

    @property
    def base_url(self) -> str:
        """Get Ollama API base URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def chat_url(self) -> str:
        """Get chat API endpoint."""
        return f"{self.base_url}/api/chat"

    @property
    def embeddings_url(self) -> str:
        """Get embeddings API endpoint."""
        return f"{self.base_url}/api/embeddings"


class ModelSettings(BaseSettings):
    """LLM model configuration.

    Paper baseline (Section 2.2): Gemma 3 27B for the multi-agent system.

    NOTE: MedGemma 27B (Appendix F) achieves better item-level MAE (0.505 vs 0.619)
    but makes fewer predictions overall (lower coverage / more N/A). Use `gemma3:27b`
    for baseline defaults; override `quantitative_model` to evaluate Appendix F.

    Embeddings (Section 2.2): Qwen 3 8B Embedding. The paper does not specify
    quantization.
    """

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    qualitative_model: str = Field(
        default="gemma3:27b", description="Qualitative agent model (Paper Section 2.2)"
    )
    judge_model: str = Field(
        default="gemma3:27b", description="Judge agent model (Paper Section 2.2)"
    )
    meta_review_model: str = Field(
        default="gemma3:27b", description="Meta-review agent model (Paper Section 2.2)"
    )
    quantitative_model: str = Field(
        default="gemma3:27b",
        description="Quantitative agent model (Paper Section 2.2: Gemma 3 27B)",
    )
    embedding_model: str = Field(
        default="qwen3-embedding:8b",
        description="Embedding model (Paper Section 2.2: Qwen 3 8B Embedding)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Clinical AI: temp=0 for reproducibility (Med-PaLM, medRxiv 2025)",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding and few-shot configuration.

    Paper-reported hyperparameters (Appendix D):
    - chunk_size=8, step_size=2, top_k=2, dimension=4096
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    dimension: int = Field(
        default=4096,
        description="Embedding dimension (Paper Appendix D: 4096 optimal)",
    )
    chunk_size: int = Field(
        default=8,
        ge=2,
        le=20,
        description="Transcript chunk size in lines (Paper Appendix D: 8 optimal)",
    )
    chunk_step: int = Field(
        default=2,
        ge=1,
        description="Sliding window step size (Paper: 2)",
    )
    top_k_references: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of reference examples per item (Paper Appendix D: 2)",
    )
    min_evidence_chars: int = Field(
        default=8,
        description="Minimum characters for valid evidence",
    )
    enable_batch_query_embedding: bool = Field(
        default=True,
        description="Use one batch query embedding call per participant (Spec 37).",
    )
    query_embed_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Timeout for query embedding (single or batch) in seconds (Spec 37).",
    )
    embeddings_file: str = Field(
        default="huggingface_qwen3_8b_paper_train",
        description="Reference embeddings basename (no extension)",
    )
    enable_retrieval_audit: bool = Field(
        default=False,
        description="Enable audit logging for retrieved references (Spec 32)",
    )
    min_reference_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Drop retrieved references below this similarity (0 disables)",
    )
    max_reference_chars_per_item: int = Field(
        default=0,
        ge=0,
        description="Max total reference chunk chars per item (0 disables)",
    )
    enable_item_tag_filter: bool = Field(
        default=False,
        description="Enable filtering reference chunks by PHQ-8 item tags.",
    )
    reference_score_source: Literal["participant", "chunk"] = Field(
        default="participant",
        description=(
            "Source of PHQ-8 scores for retrieved chunks "
            "(participant-level or chunk-level estimate)."
        ),
    )
    allow_chunk_scores_prompt_hash_mismatch: bool = Field(
        default=False,
        description=(
            "Allow loading chunk scores when the scorer prompt hash differs or metadata is missing "
            "(unsafe; Spec 35 circularity control bypass)."
        ),
    )
    enable_reference_validation: bool = Field(
        default=False,
        description="Enable CRAG-style runtime validation of retrieved references (Spec 36).",
    )
    validation_model: str = Field(
        default="",
        description="Model to use for reference validation (required if validation enabled).",
    )
    validation_max_refs_per_item: int = Field(
        default=2,
        ge=1,
        description="Maximum accepted references to keep per item after validation.",
    )


class FeedbackLoopSettings(BaseSettings):
    """Feedback loop configuration.

    Paper (Section 2.3.1): "When an original evaluation score was below four,
    the judge agent triggered an automatic feedback loop."

    Paper-derived: threshold=3 means scores <= 3 (i.e., < 4) trigger refinement.
    """

    model_config = SettingsConfigDict(
        env_prefix="FEEDBACK_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable iterative refinement")
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum feedback iterations (Paper Section 2.3.1: 10)",
    )
    score_threshold: int = Field(
        default=3,
        ge=1,
        le=4,
        description="Scores at or below this trigger refinement (Paper: 3)",
    )
    target_score: int = Field(
        default=4,
        ge=3,
        le=5,
        description="Target score (Paper: all scores >= 4 means no refinement)",
    )


class QuantitativeSettings(BaseSettings):
    """Quantitative assessment configuration.

    DEPRECATED FEATURE WARNING:
    Keyword backfill (`enable_keyword_backfill`) is a flawed heuristic that
    inflates coverage metrics without improving clinical validity. The feature
    matches keywords like "sleep" or "tired" without semantic understanding,
    leading to false positives and misleading results.

    DO NOT ENABLE keyword backfill. The feature is retained only for
    historical comparison and ablation studies. The original paper's
    methodology has fundamental issues (see HYPOTHESIS-FEWSHOT-DESIGN-FLAW.md).
    """

    model_config = SettingsConfigDict(
        env_prefix="QUANTITATIVE_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    enable_keyword_backfill: bool = Field(
        default=False,
        description="DEPRECATED: Do NOT enable. Flawed heuristic retained for ablation only.",
    )
    track_na_reasons: bool = Field(
        default=True,
        description="Track why items return N/A",
    )
    keyword_backfill_cap: int = Field(
        default=3,
        ge=1,
        le=10,
        description="DEPRECATED: Irrelevant since backfill should remain OFF.",
    )


class PydanticAISettings(BaseSettings):
    """Pydantic AI integration settings."""

    model_config = SettingsConfigDict(
        env_prefix="PYDANTIC_AI_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description=(
            "Enable Pydantic AI for TextOutput validation + retries "
            "(quantitative scoring, qualitative assessment, judge, meta-review). "
            "When enabled, ollama_base_url is required."
        ),
    )
    retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry count for validation failures (0 disables retries).",
    )
    timeout_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Timeout for Pydantic AI LLM calls. None = use library default.",
    )


class DataSettings(BaseSettings):
    """Data path configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    base_dir: Path = Field(
        default=Path("data"),
        description="Base data directory",
    )
    transcripts_dir: Path = Field(
        default=Path("data/transcripts"),
        description="Transcript files directory",
    )
    embeddings_path: Path = Field(
        default=Path("data/embeddings/huggingface_qwen3_8b_paper_train.npz"),
        description="Pre-computed embeddings for few-shot (NPZ + .json sidecar).",
    )
    train_csv: Path = Field(
        default=Path("data/train_split_Depression_AVEC2017.csv"),
        description="Training set ground truth",
    )
    dev_csv: Path = Field(
        default=Path("data/dev_split_Depression_AVEC2017.csv"),
        description="Development set ground truth",
    )

    @field_validator("base_dir", "transcripts_dir", mode="after")
    @classmethod
    def warn_if_dir_missing(cls, v: Path) -> Path:
        """Warn if directory doesn't exist (don't create)."""
        if not v.exists():
            warnings.warn(f"Data directory does not exist: {v}", stacklevel=2)
        return v


def resolve_reference_embeddings_path(
    data_settings: DataSettings,
    embedding_settings: EmbeddingSettings,
) -> Path:
    """Resolve the reference embeddings NPZ path used for few-shot retrieval.

    Precedence:
    1) `DATA_EMBEDDINGS_PATH` if explicitly set.
    2) `EMBEDDING_EMBEDDINGS_FILE` resolved under `{DATA_BASE_DIR}/embeddings/`.
    """
    # Explicit full-path override wins (including env-provided values).
    if "embeddings_path" in data_settings.model_fields_set:
        return data_settings.embeddings_path

    return resolve_reference_embeddings_path_from_embeddings_file(
        base_dir=data_settings.base_dir,
        embeddings_file=embedding_settings.embeddings_file,
    )


def resolve_reference_embeddings_path_from_embeddings_file(
    *,
    base_dir: Path,
    embeddings_file: str,
) -> Path:
    """Resolve an embeddings NPZ path from an `embeddings_file` string.

    This mirrors `EMBEDDING_EMBEDDINGS_FILE` resolution logic without consulting
    `DATA_EMBEDDINGS_PATH` precedence (useful for CLI tools that accept an explicit
    embeddings file argument).
    """
    candidate = Path(embeddings_file)

    # Absolute paths are used as-is (ensure .npz suffix).
    if candidate.is_absolute():
        return candidate if candidate.suffix == ".npz" else candidate.with_suffix(".npz")

    # Relative paths with directories are resolved under the data base dir.
    if candidate.parent != Path():
        resolved = base_dir / candidate
        return resolved if resolved.suffix == ".npz" else resolved.with_suffix(".npz")

    # Basename-only: resolve under the embeddings directory.
    return (base_dir / "embeddings" / candidate.name).with_suffix(".npz")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level",
    )
    format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format",
    )
    include_timestamp: bool = Field(default=True)
    include_caller: bool = Field(default=True)
    force_colors: bool | None = Field(
        default=None,
        description=(
            "Force ANSI colors on/off for console logs. None = auto-detect via TTY + NO_COLOR."
        ),
    )


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False, description="Enable hot reload (dev only)")
    workers: int = Field(default=1, ge=1, le=16)
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins (restrict in production)",
    )


class Settings(BaseSettings):
    """Root settings combining all configuration groups."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Nested settings groups
    backend: BackendSettings = Field(default_factory=BackendSettings)
    embedding_config: EmbeddingBackendSettings = Field(default_factory=EmbeddingBackendSettings)
    huggingface: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    feedback: FeedbackLoopSettings = Field(default_factory=FeedbackLoopSettings)
    quantitative: QuantitativeSettings = Field(default_factory=QuantitativeSettings)
    pydantic_ai: PydanticAISettings = Field(default_factory=PydanticAISettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)

    # Feature flags
    enable_few_shot: bool = Field(default=True, description="Enable few-shot mode")
    # NOTE: Default quantitative_model is gemma3:27b (Paper Section 2.2).
    # MedGemma was only evaluated as an ALTERNATIVE in Appendix F (fewer predictions overall).

    @model_validator(mode="after")
    def validate_consistency(self) -> Settings:
        """Validate cross-field consistency."""
        embeddings_path = resolve_reference_embeddings_path(self.data, self.embedding)
        if self.enable_few_shot and not embeddings_path.exists():
            warnings.warn(
                f"Few-shot enabled but embeddings not found: {embeddings_path}",
                stacklevel=2,
            )

        # BUG-027: Keep timeouts aligned across Pydantic AI + legacy fallback paths.
        pydantic_timeout_set = "timeout_seconds" in self.pydantic_ai.model_fields_set
        ollama_timeout_set = "timeout_seconds" in self.ollama.model_fields_set

        if pydantic_timeout_set and not ollama_timeout_set:
            timeout = self.pydantic_ai.timeout_seconds
            if timeout is not None:
                self.ollama.timeout_seconds = max(10, math.ceil(timeout))
        elif ollama_timeout_set and not pydantic_timeout_set:
            self.pydantic_ai.timeout_seconds = float(self.ollama.timeout_seconds)
        elif pydantic_timeout_set and ollama_timeout_set:
            timeout = self.pydantic_ai.timeout_seconds
            if timeout is not None and self.ollama.timeout_seconds != math.ceil(timeout):
                warnings.warn(
                    "OLLAMA_TIMEOUT_SECONDS and PYDANTIC_AI_TIMEOUT_SECONDS differ; "
                    "fallback path may timeout sooner than the primary path.",
                    stacklevel=2,
                )

        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()


def get_model_name(model_settings: ModelSettings | None, model_type: str) -> str:
    """Get model name from settings or fall back to config defaults.

    Args:
        model_settings: Optional settings object
        model_type: One of "qualitative", "quantitative", "judge", "meta_review", "embedding"

    Returns:
        Model name string
    """
    if model_settings is not None:
        return str(getattr(model_settings, f"{model_type}_model"))
    # Fall back to fresh settings instance (reads from env/defaults)
    return str(getattr(ModelSettings(), f"{model_type}_model"))


def get_ollama_settings() -> OllamaSettings:
    """Get Ollama settings (for FastAPI Depends)."""
    return get_settings().ollama


def get_model_settings() -> ModelSettings:
    """Get model settings (for FastAPI Depends)."""
    return get_settings().model


def get_backend_settings() -> BackendSettings:
    """Get backend settings (for FastAPI Depends)."""
    return get_settings().backend
