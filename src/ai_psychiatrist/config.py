"""Centralized configuration using Pydantic Settings.

This module provides paper-optimal defaults for the AI Psychiatrist system.
All settings can be overridden via environment variables.

Paper references:
- Section 2.2: Model configuration (Gemma 3 27B, Qwen 3 8B Embedding)
- Section 2.3.5: Agentic system configuration (Ollama API)
- Appendix D: Hyperparameters (chunk_size=8, step_size=2, N_example=2, dim=4096)
- Appendix F: MedGemma quantitative results (MAE 0.505)
"""

from __future__ import annotations

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
    timeout_seconds: int = Field(default=300, ge=10, le=600, description="Request timeout")

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
    for paper-parity defaults; override `quantitative_model` to evaluate Appendix F.

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

    Paper-optimal hyperparameters (Appendix D):
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
    embeddings_file: str = Field(
        default="huggingface_qwen3_8b_paper_train",
        description="Reference embeddings basename (no extension)",
    )


class FeedbackLoopSettings(BaseSettings):
    """Feedback loop configuration.

    Paper (Section 2.3.1): "When an original evaluation score was below four,
    the judge agent triggered an automatic feedback loop."

    Paper-optimal: threshold=3 means scores <= 3 (i.e., < 4) trigger refinement.
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

    Reference: SPEC-003 for backfill toggle and NA reason tracking.
    """

    model_config = SettingsConfigDict(
        env_prefix="QUANTITATIVE_",
        env_file=ENV_FILE,
        env_file_encoding=ENV_FILE_ENCODING,
        extra="ignore",
    )

    enable_keyword_backfill: bool = Field(
        default=False,
        description="Enable keyword backfill when LLM misses evidence (paper parity: OFF)",
    )
    track_na_reasons: bool = Field(
        default=True,
        description="Track why items return N/A",
    )
    keyword_backfill_cap: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max keyword-matched sentences per domain",
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
        default=False,
        description=(
            "Enable Pydantic AI for quantitative scoring (TextOutput validation + retries)."
        ),
    )
    retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry count for validation failures (0 disables retries).",
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

    candidate = Path(embedding_settings.embeddings_file)

    # Absolute paths are used as-is (ensure .npz suffix).
    if candidate.is_absolute():
        return candidate if candidate.suffix == ".npz" else candidate.with_suffix(".npz")

    # Relative paths with directories are resolved under the data base dir.
    if candidate.parent != Path():
        resolved = data_settings.base_dir / candidate
        return resolved if resolved.suffix == ".npz" else resolved.with_suffix(".npz")

    # Basename-only: resolve under the embeddings directory.
    return (data_settings.base_dir / "embeddings" / candidate.name).with_suffix(".npz")


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
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()


def get_ollama_settings() -> OllamaSettings:
    """Get Ollama settings (for FastAPI Depends)."""
    return get_settings().ollama


def get_model_settings() -> ModelSettings:
    """Get model settings (for FastAPI Depends)."""
    return get_settings().model


def get_backend_settings() -> BackendSettings:
    """Get backend settings (for FastAPI Depends)."""
    return get_settings().backend
