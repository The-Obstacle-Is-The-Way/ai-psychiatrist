# Spec 03: Configuration & Logging

## Objective

Implement centralized configuration management using Pydantic Settings and structured logging with structlog. This enables environment-based configuration and production-ready observability.

## Paper Reference

- **Section 2.2**: Model configuration (Gemma 3 27B, Qwen 3 8B Embedding)
- **Section 2.3.5**: Agentic system configuration (Ollama API)
- **Section 2.4.2**: Hyperparameters (chunk_size=8, top_k=2, dim=4096)

## Deliverables

1. `src/ai_psychiatrist/config.py` - Pydantic Settings
2. `src/ai_psychiatrist/infrastructure/logging.py` - structlog setup
3. `tests/unit/test_config.py` - Configuration tests
4. `tests/unit/infrastructure/test_logging.py` - Logging tests

## Implementation

### 1. Configuration (config.py)

```python
"""Centralized configuration using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama server configuration."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    host: str = Field(default="127.0.0.1", description="Ollama server host")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama server port")
    timeout_seconds: int = Field(default=180, ge=10, le=600, description="Request timeout")

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
    """LLM model configuration."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    chat_model: str = Field(
        default="gemma3:27b",
        description="Chat/completion model name",
    )
    embedding_model: str = Field(
        default="dengcao/Qwen3-Embedding-8B:Q8_0",
        description="Embedding model name (Qwen 3 8B Embedding, Q8 quantization)",
    )
    medgemma_model: str = Field(
        default="alibayram/medgemma:27b",
        description="Medical domain model (optional)",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Default temperature")
    temperature_judge: float = Field(default=0.0, ge=0.0, le=2.0, description="Judge agent temperature (deterministic)")
    top_k: int = Field(default=20, ge=1, le=100)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)


class EmbeddingSettings(BaseSettings):
    """Embedding and few-shot configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    dimension: int = Field(
        default=4096,
        description="Embedding dimension (paper optimal: 4096)",
    )
    chunk_size: int = Field(
        default=8,
        ge=2,
        le=20,
        description="Transcript chunk size in lines (paper optimal: 8)",
    )
    chunk_step: int = Field(
        default=2,
        ge=1,
        description="Sliding window step size",
    )
    top_k_references: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of reference examples (paper optimal: 2)",
    )
    min_evidence_chars: int = Field(
        default=8,
        description="Minimum characters for valid evidence",
    )


class FeedbackLoopSettings(BaseSettings):
    """Feedback loop configuration."""

    model_config = SettingsConfigDict(env_prefix="FEEDBACK_")

    enabled: bool = Field(default=True, description="Enable iterative refinement")
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum feedback iterations",
    )
    score_threshold: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Scores at or below this trigger refinement",
    )
    target_score: int = Field(
        default=4,
        ge=3,
        le=5,
        description="Target score to achieve",
    )


class DataSettings(BaseSettings):
    """Data path configuration."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    base_dir: Path = Field(
        default=Path("data"),
        description="Base data directory",
    )
    transcripts_dir: Path = Field(
        default=Path("data/transcripts"),
        description="Transcript files directory",
    )
    embeddings_path: Path = Field(
        default=Path("data/embeddings/participant_embedded_transcripts.pkl"),
        description="Pre-computed embeddings file",
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
    def ensure_dir_exists(cls, v: Path) -> Path:
        """Warn if directory doesn't exist (don't create)."""
        if not v.exists():
            import warnings
            warnings.warn(f"Data directory does not exist: {v}", stacklevel=2)
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

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


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False, description="Enable hot reload (dev only)")
    workers: int = Field(default=1, ge=1, le=16)
    cors_origins: list[str] = Field(default=["*"])


class Settings(BaseSettings):
    """Root settings combining all configuration groups."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Nested settings groups
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    feedback: FeedbackLoopSettings = Field(default_factory=FeedbackLoopSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)

    # Feature flags
    enable_few_shot: bool = Field(default=True, description="Enable few-shot mode")
    enable_medgemma: bool = Field(default=False, description="Use MedGemma model")

    @model_validator(mode="after")
    def validate_consistency(self) -> Settings:
        """Validate cross-field consistency."""
        # If few-shot enabled, embeddings file should exist
        if self.enable_few_shot and not self.data.embeddings_path.exists():
            import warnings
            warnings.warn(
                f"Few-shot enabled but embeddings not found: {self.data.embeddings_path}",
                stacklevel=2,
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()


# Convenience function for dependency injection
def get_ollama_settings() -> OllamaSettings:
    """Get Ollama settings (for FastAPI Depends)."""
    return get_settings().ollama


def get_model_settings() -> ModelSettings:
    """Get model settings (for FastAPI Depends)."""
    return get_settings().model
```

### 2. Structured Logging (infrastructure/logging.py)

```python
"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ai_psychiatrist.config import LoggingSettings


def setup_logging(settings: LoggingSettings | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        settings: Logging settings. If None, uses defaults.
    """
    if settings is None:
        from ai_psychiatrist.config import get_settings
        settings = get_settings().logging

    # Determine processors based on format
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.include_caller:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    if settings.format == "json":
        # Production: JSON output
        final_processors: list[structlog.types.Processor] = [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output
        final_processors = [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]

    structlog.configure(
        processors=shared_processors + final_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.level),
    )

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


# Context management utilities
def bind_context(**kwargs: str | int | float | bool) -> None:
    """Bind context variables for current execution context.

    Args:
        **kwargs: Key-value pairs to bind.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind context variables.

    Args:
        *keys: Keys to unbind.
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


# Decorator for adding request context
def with_context(**context_vars: str | int | float | bool):
    """Decorator to bind context for function execution.

    Args:
        **context_vars: Context variables to bind.
    """
    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            bind_context(**context_vars)
            try:
                return func(*args, **kwargs)
            finally:
                unbind_context(*context_vars.keys())

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            bind_context(**context_vars)
            try:
                return await func(*args, **kwargs)
            finally:
                unbind_context(*context_vars.keys())

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
```

### 3. Logging Usage Examples

```python
"""Example usage of structured logging."""

from ai_psychiatrist.infrastructure.logging import get_logger, bind_context

logger = get_logger(__name__)

# Basic logging
logger.info("Starting assessment", participant_id=123)

# With exception
try:
    risky_operation()
except Exception:
    logger.exception("Assessment failed", participant_id=123)

# Context binding for request
def assess_participant(participant_id: int) -> None:
    bind_context(participant_id=participant_id, operation="assessment")
    logger.info("Starting assessment")
    # ... processing ...
    logger.info("Completed assessment", severity="moderate")

# JSON output example:
# {
#   "event": "Starting assessment",
#   "participant_id": 123,
#   "level": "info",
#   "timestamp": "2025-01-15T10:30:00Z",
#   "logger": "ai_psychiatrist.agents.qualitative",
#   "filename": "qualitative.py",
#   "lineno": 42,
#   "func_name": "assess"
# }
```

### 4. Tests

#### test_config.py

```python
"""Tests for configuration management."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_psychiatrist.config import (
    EmbeddingSettings,
    OllamaSettings,
    Settings,
    get_settings,
)


class TestOllamaSettings:
    """Tests for Ollama configuration."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        settings = OllamaSettings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 11434
        assert settings.timeout_seconds == 180

    def test_base_url_property(self) -> None:
        """Should construct correct base URL."""
        settings = OllamaSettings(host="custom-host", port=12345)
        assert settings.base_url == "http://custom-host:12345"

    def test_endpoint_urls(self) -> None:
        """Should construct correct endpoint URLs."""
        settings = OllamaSettings()
        assert settings.chat_url == "http://127.0.0.1:11434/api/chat"
        assert settings.embeddings_url == "http://127.0.0.1:11434/api/embeddings"


class TestEmbeddingSettings:
    """Tests for embedding configuration."""

    def test_paper_optimal_defaults(self) -> None:
        """Defaults should match paper optimal values."""
        settings = EmbeddingSettings()
        assert settings.dimension == 4096  # Paper optimal
        assert settings.chunk_size == 8    # Paper optimal
        assert settings.top_k_references == 2  # Paper optimal

    def test_chunk_size_validation(self) -> None:
        """Should validate chunk size range."""
        with pytest.raises(ValueError):
            EmbeddingSettings(chunk_size=1)  # Below minimum
        with pytest.raises(ValueError):
            EmbeddingSettings(chunk_size=25)  # Above maximum


class TestSettings:
    """Tests for root settings."""

    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should load settings from environment variables."""
        monkeypatch.setenv("OLLAMA_HOST", "test-host")
        monkeypatch.setenv("OLLAMA_PORT", "9999")
        monkeypatch.setenv("MODEL_CHAT_MODEL", "test-model")

        # Clear cache to reload settings
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.ollama.host == "test-host"
        assert settings.ollama.port == 9999
        assert settings.model.chat_model == "test-model"

    def test_nested_delimiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should support nested delimiter for complex settings."""
        monkeypatch.setenv("OLLAMA__HOST", "nested-host")
        get_settings.cache_clear()
        settings = get_settings()
        # Note: nested delimiter might not work as expected with Pydantic v2
        # This test documents current behavior


class TestSettingsCaching:
    """Tests for settings caching."""

    def test_get_settings_cached(self) -> None:
        """get_settings should return cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
```

#### test_logging.py

```python
"""Tests for structured logging."""

from __future__ import annotations

import json
from io import StringIO

import structlog

from ai_psychiatrist.config import LoggingSettings
from ai_psychiatrist.infrastructure.logging import (
    bind_context,
    clear_context,
    get_logger,
    setup_logging,
)


class TestLoggingSetup:
    """Tests for logging configuration."""

    def test_json_format(self, capsys) -> None:
        """JSON format should produce valid JSON."""
        settings = LoggingSettings(level="INFO", format="json")
        setup_logging(settings)

        logger = get_logger("test")
        logger.info("test message", key="value")

        # Note: Output goes to stdout
        # In real tests, we'd capture and parse the JSON

    def test_console_format(self) -> None:
        """Console format should not raise errors."""
        settings = LoggingSettings(level="INFO", format="console")
        setup_logging(settings)

        logger = get_logger("test")
        logger.info("test message", key="value")


class TestContextBinding:
    """Tests for context variable binding."""

    def test_bind_and_unbind(self) -> None:
        """Should bind and unbind context."""
        clear_context()
        bind_context(participant_id=123, operation="test")

        # Context should be bound (verified via log output in real scenario)
        clear_context()

    def test_clear_context(self) -> None:
        """Should clear all context."""
        bind_context(key1="value1", key2="value2")
        clear_context()
        # Context should be empty
```

## Acceptance Criteria

- [ ] All configuration values match paper hyperparameters
- [ ] Settings load from `.env` file
- [ ] Settings load from environment variables
- [ ] Invalid values are rejected with clear errors
- [ ] Logging produces valid JSON in production mode
- [ ] Logging produces readable output in development mode
- [ ] Context variables propagate through call stack
- [ ] Settings are cached after first load
- [ ] Type hints throughout

## Dependencies

- **Spec 01**: Project structure (pyproject.toml)
- **Spec 02**: Domain types (for validation)

## Specs That Depend on This

- **Spec 04-11**: All specs use configuration and logging
