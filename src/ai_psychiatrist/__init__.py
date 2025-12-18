"""AI Psychiatrist: LLM-based Multi-Agent System for Depression Assessment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-psychiatrist")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
__all__ = ["__version__"]
