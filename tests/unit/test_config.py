"""Tests for configuration management.

Tests verify settings match paper hyperparameters and provide
proper validation, caching, and environment variable loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pytest

from ai_psychiatrist.config import (
    APISettings,
    DataSettings,
    EmbeddingSettings,
    FeedbackLoopSettings,
    LoggingSettings,
    ModelSettings,
    OllamaSettings,
    Settings,
    get_model_settings,
    get_ollama_settings,
    get_settings,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:Data directory does not exist.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Few-shot enabled but embeddings not found.*:UserWarning"),
]


class TestOllamaSettings:
    """Tests for Ollama server configuration."""

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

    def test_chat_url_property(self) -> None:
        """Should construct correct chat endpoint URL."""
        settings = OllamaSettings()
        assert settings.chat_url == "http://127.0.0.1:11434/api/chat"

    def test_embeddings_url_property(self) -> None:
        """Should construct correct embeddings endpoint URL."""
        settings = OllamaSettings()
        assert settings.embeddings_url == "http://127.0.0.1:11434/api/embeddings"

    def test_port_validation_min(self) -> None:
        """Port must be >= 1."""
        with pytest.raises(ValueError):
            OllamaSettings(port=0)

    def test_port_validation_max(self) -> None:
        """Port must be <= 65535."""
        with pytest.raises(ValueError):
            OllamaSettings(port=65536)

    def test_timeout_validation_min(self) -> None:
        """Timeout must be >= 10."""
        with pytest.raises(ValueError):
            OllamaSettings(timeout_seconds=5)

    def test_timeout_validation_max(self) -> None:
        """Timeout must be <= 600."""
        with pytest.raises(ValueError):
            OllamaSettings(timeout_seconds=700)


class TestModelSettings:
    """Tests for model configuration."""

    def test_paper_optimal_defaults(self) -> None:
        """Defaults should match paper Section 2.2.

        NOTE: quantitative_model uses gemma3:27b (not MedGemma) because
        MedGemma produces excessive N/A scores in practice, leading to
        worse total-score MAE despite better item-level MAE in Appendix F.
        """
        settings = ModelSettings()
        assert settings.qualitative_model == "gemma3:27b"
        assert settings.judge_model == "gemma3:27b"
        assert settings.meta_review_model == "gemma3:27b"
        assert settings.quantitative_model == "gemma3:27b"  # Not MedGemma - see docstring
        assert settings.embedding_model == "qwen3-embedding:8b"

    def test_temperature_defaults(self) -> None:
        """Temperature defaults should be set correctly."""
        settings = ModelSettings()
        assert settings.temperature == 0.2
        assert settings.temperature_judge == 0.0  # Deterministic

    def test_temperature_validation(self) -> None:
        """Temperature must be 0-2."""
        with pytest.raises(ValueError):
            ModelSettings(temperature=-0.1)
        with pytest.raises(ValueError):
            ModelSettings(temperature=2.1)

    def test_top_k_validation(self) -> None:
        """top_k must be 1-100."""
        with pytest.raises(ValueError):
            ModelSettings(top_k=0)
        with pytest.raises(ValueError):
            ModelSettings(top_k=101)

    def test_top_p_validation(self) -> None:
        """top_p must be 0-1."""
        with pytest.raises(ValueError):
            ModelSettings(top_p=-0.1)
        with pytest.raises(ValueError):
            ModelSettings(top_p=1.1)


class TestEmbeddingSettings:
    """Tests for embedding configuration."""

    def test_paper_optimal_defaults(self) -> None:
        """Defaults should match paper Appendix D optimal values."""
        settings = EmbeddingSettings()
        assert settings.dimension == 4096  # Paper Appendix D
        assert settings.chunk_size == 8  # Paper Appendix D
        assert settings.chunk_step == 2  # Paper: step_size=2
        assert settings.top_k_references == 2  # Paper Appendix D: N_example=2

    def test_chunk_size_validation_min(self) -> None:
        """chunk_size must be >= 2."""
        with pytest.raises(ValueError):
            EmbeddingSettings(chunk_size=1)

    def test_chunk_size_validation_max(self) -> None:
        """chunk_size must be <= 20."""
        with pytest.raises(ValueError):
            EmbeddingSettings(chunk_size=25)

    def test_top_k_references_validation(self) -> None:
        """top_k_references must be 1-10."""
        with pytest.raises(ValueError):
            EmbeddingSettings(top_k_references=0)
        with pytest.raises(ValueError):
            EmbeddingSettings(top_k_references=15)


class TestFeedbackLoopSettings:
    """Tests for feedback loop configuration."""

    def test_paper_optimal_defaults(self) -> None:
        """Defaults should match paper Section 2.3.1."""
        settings = FeedbackLoopSettings()
        assert settings.enabled is True
        assert settings.max_iterations == 10  # Paper Section 2.3.1
        assert settings.score_threshold == 3  # Scores <= 3 trigger refinement
        assert settings.target_score == 4  # Paper: scores >= 4 means no refinement

    def test_max_iterations_validation(self) -> None:
        """max_iterations must be 1-20."""
        with pytest.raises(ValueError):
            FeedbackLoopSettings(max_iterations=0)
        with pytest.raises(ValueError):
            FeedbackLoopSettings(max_iterations=25)

    def test_score_threshold_validation(self) -> None:
        """score_threshold must be 1-4."""
        with pytest.raises(ValueError):
            FeedbackLoopSettings(score_threshold=0)
        with pytest.raises(ValueError):
            FeedbackLoopSettings(score_threshold=5)


class TestDataSettings:
    """Tests for data path configuration."""

    def test_default_paths(self) -> None:
        """Should have expected default paths."""
        settings = DataSettings()
        assert settings.base_dir == Path("data")
        assert settings.transcripts_dir == Path("data/transcripts")
        assert "embeddings" in str(settings.embeddings_path)


class TestLoggingSettings:
    """Tests for logging configuration."""

    def test_default_values(self) -> None:
        """Should have production-ready defaults."""
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.format == "json"
        assert settings.include_timestamp is True
        assert settings.include_caller is True

    def test_level_validation(self) -> None:
        """Level must be valid log level."""
        valid_levels: tuple[
            Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            ...,
        ] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

        for level in valid_levels:
            settings = LoggingSettings(level=level)
            assert settings.level == level

        with pytest.raises(ValueError):
            LoggingSettings(level="INVALID")  # type: ignore[arg-type]

    def test_format_validation(self) -> None:
        """Format must be 'json' or 'console'."""
        settings_json = LoggingSettings(format="json")
        assert settings_json.format == "json"

        settings_console = LoggingSettings(format="console")
        assert settings_console.format == "console"

        with pytest.raises(ValueError):
            LoggingSettings(format="xml")  # type: ignore[arg-type]


class TestAPISettings:
    """Tests for API server configuration."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        settings = APISettings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.reload is False
        assert settings.workers == 1
        assert settings.cors_origins == ["*"]

    def test_port_validation(self) -> None:
        """Port must be valid."""
        with pytest.raises(ValueError):
            APISettings(port=0)
        with pytest.raises(ValueError):
            APISettings(port=70000)

    def test_workers_validation(self) -> None:
        """Workers must be 1-16."""
        with pytest.raises(ValueError):
            APISettings(workers=0)
        with pytest.raises(ValueError):
            APISettings(workers=20)


class TestSettings:
    """Tests for root settings."""

    def test_creates_nested_settings(self) -> None:
        """Should create all nested settings groups."""
        settings = Settings()
        assert settings.ollama is not None
        assert settings.model is not None
        assert settings.embedding is not None
        assert settings.feedback is not None
        assert settings.data is not None
        assert settings.logging is not None
        assert settings.api is not None

    def test_feature_flags_default(self) -> None:
        """Feature flags should have correct defaults."""
        settings = Settings()
        assert settings.enable_few_shot is True
        # NOTE: enable_medgemma removed - use MODEL__QUANTITATIVE_MODEL directly.
        # Default quantitative_model is already alibayram/medgemma:27b (Paper Appendix F).

    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should load settings from environment variables."""
        monkeypatch.setenv("OLLAMA_HOST", "test-host")
        monkeypatch.setenv("OLLAMA_PORT", "9999")
        monkeypatch.setenv("MODEL_QUANTITATIVE_MODEL", "test-model")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.ollama.host == "test-host"
        assert settings.ollama.port == 9999
        assert settings.model.quantitative_model == "test-model"

        get_settings.cache_clear()

    @pytest.mark.skipif(
        os.environ.get("TESTING") == "1",
        reason="TESTING mode disables .env loading to ensure test isolation",
    )
    def test_loads_from_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Should load nested settings from .env file."""
        data_dir = tmp_path / "data"
        transcripts_dir = data_dir / "transcripts"
        transcripts_dir.mkdir(parents=True)
        embeddings_path = data_dir / "embeddings.pkl"
        embeddings_path.write_bytes(b"")

        dotenv = "\n".join(
            [
                "OLLAMA_HOST=dotenv-host",
                "MODEL_QUALITATIVE_MODEL=dotenv-model",
                f"DATA_BASE_DIR={data_dir}",
                f"DATA_TRANSCRIPTS_DIR={transcripts_dir}",
                f"DATA_EMBEDDINGS_PATH={embeddings_path}",
            ]
        )
        (tmp_path / ".env").write_text(dotenv)

        for key in (
            "OLLAMA_HOST",
            "MODEL_QUALITATIVE_MODEL",
            "DATA_BASE_DIR",
            "DATA_TRANSCRIPTS_DIR",
            "DATA_EMBEDDINGS_PATH",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.chdir(tmp_path)
        settings = Settings()

        assert settings.ollama.host == "dotenv-host"
        assert settings.model.qualitative_model == "dotenv-model"
        assert settings.data.base_dir == data_dir
        assert settings.data.transcripts_dir == transcripts_dir
        assert settings.data.embeddings_path == embeddings_path


class TestSettingsCaching:
    """Tests for settings caching."""

    def test_get_settings_cached(self) -> None:
        """get_settings should return cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
        get_settings.cache_clear()

    def test_cache_clear_works(self) -> None:
        """cache_clear should allow fresh load."""
        get_settings.cache_clear()
        settings1 = get_settings()

        get_settings.cache_clear()
        settings2 = get_settings()

        assert settings1 is not settings2
        get_settings.cache_clear()


class TestConvenienceFunctions:
    """Tests for dependency injection convenience functions."""

    def test_get_ollama_settings(self) -> None:
        """get_ollama_settings should return OllamaSettings."""
        settings = get_ollama_settings()
        assert isinstance(settings, OllamaSettings)

    def test_get_model_settings(self) -> None:
        """get_model_settings should return ModelSettings."""
        settings = get_model_settings()
        assert isinstance(settings, ModelSettings)
