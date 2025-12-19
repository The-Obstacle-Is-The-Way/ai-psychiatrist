"""Tests for project bootstrap."""

import importlib
import subprocess
from pathlib import Path

import pytest

import ai_psychiatrist
from ai_psychiatrist import domain


class TestProjectStructure:
    """Test project structure is correct."""

    def test_pyproject_toml_exists(self) -> None:
        """pyproject.toml should exist at project root."""
        assert Path("pyproject.toml").exists()

    def test_src_layout(self) -> None:
        """src/ai_psychiatrist package should exist."""
        assert Path("src/ai_psychiatrist/__init__.py").exists()

    def test_tests_directory(self) -> None:
        """tests directory should exist with subdirs."""
        assert Path("tests/unit").is_dir()
        assert Path("tests/integration").is_dir()
        assert Path("tests/e2e").is_dir()

    def test_package_subdirectories(self) -> None:
        """Package should have expected subdirectories."""
        subdirs = ["api", "agents", "domain", "services", "infrastructure"]
        for subdir in subdirs:
            assert Path(f"src/ai_psychiatrist/{subdir}/__init__.py").exists()


class TestImports:
    """Test that package can be imported."""

    def test_import_package(self) -> None:
        """Package should be importable with version from metadata."""
        # Version is dynamically read from package metadata (pyproject.toml)
        assert isinstance(ai_psychiatrist.__version__, str)
        assert ai_psychiatrist.__version__  # Not empty

    def test_import_public_packages(self) -> None:
        """Public packages should be importable and expose expected API."""
        for module_name in [
            "ai_psychiatrist.agents",
            "ai_psychiatrist.api",
            "ai_psychiatrist.infrastructure",
            "ai_psychiatrist.services",
        ]:
            module = importlib.import_module(module_name)
            assert module.__doc__

        assert domain.PHQ8Item.NO_INTEREST.value == "NoInterest"
        assert all(hasattr(domain, name) for name in domain.__all__)


class TestMakefile:
    """Test Makefile targets work."""

    @pytest.mark.slow
    def test_make_help(self) -> None:
        """make help should run without error."""
        result = subprocess.run(
            ["make", "help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "help" in result.stdout


class TestEnvExample:
    """Test .env.example contains paper-optimal values."""

    def test_env_example_exists(self) -> None:
        """.env.example should exist."""
        assert Path(".env.example").exists()

    def test_env_example_has_paper_optimal_values(self) -> None:
        """.env.example should contain paper-optimal configuration."""
        content = Path(".env.example").read_text()

        # Paper-optimal models
        assert "gemma3:27b" in content
        assert "alibayram/medgemma:27b" in content
        assert "qwen3-embedding:8b" in content

        # Paper-optimal hyperparameters
        assert "EMBEDDING_DIMENSION=4096" in content
        assert "EMBEDDING_TOP_K_REFERENCES=2" in content
        assert "EMBEDDING_CHUNK_SIZE=8" in content
        assert "FEEDBACK_SCORE_THRESHOLD=3" in content
        assert "FEEDBACK_MAX_ITERATIONS=10" in content
