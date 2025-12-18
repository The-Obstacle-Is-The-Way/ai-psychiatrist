"""Tests for structured logging.

Tests verify logging setup, context binding, and output formats.
"""

from __future__ import annotations

import logging

import pytest
import structlog

from ai_psychiatrist.config import LoggingSettings
from ai_psychiatrist.infrastructure.logging import (
    bind_context,
    clear_context,
    get_logger,
    setup_logging,
    unbind_context,
    with_context,
)


class TestLoggingSetup:
    """Tests for logging configuration."""

    def test_setup_logging_json_format(self) -> None:
        """JSON format should configure structlog correctly."""
        settings = LoggingSettings(level="INFO", format="json")
        setup_logging(settings)

        logger = structlog.get_logger("test.json")
        logger.info("test message", key="value")

    def test_setup_logging_console_format(self) -> None:
        """Console format should configure structlog correctly."""
        settings = LoggingSettings(level="INFO", format="console")
        setup_logging(settings)

        logger = structlog.get_logger("test.console")
        logger.info("test message", key="value")

    def test_setup_logging_sets_log_level(self) -> None:
        """Should set the correct log level."""
        settings = LoggingSettings(level="WARNING", format="json")
        setup_logging(settings)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_debug_level(self) -> None:
        """Should handle DEBUG level."""
        settings = LoggingSettings(level="DEBUG", format="json")
        setup_logging(settings)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_silences_noisy_libraries(self) -> None:
        """Should silence httpx and httpcore."""
        settings = LoggingSettings(level="DEBUG", format="json")
        setup_logging(settings)

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    def test_setup_logging_without_settings(self) -> None:
        """Should work with default settings when None passed."""
        setup_logging(None)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self) -> None:
        """Should return logger with specified name."""
        setup_logging(None)
        logger = get_logger("test.module")
        assert logger is not None

    def test_get_logger_without_name(self) -> None:
        """Should return logger without name."""
        setup_logging(None)
        logger = get_logger()
        assert logger is not None

    def test_logger_can_log(self) -> None:
        """Logger should be able to log messages."""
        setup_logging(None)
        logger = get_logger("test.can_log")

        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

    def test_logger_accepts_kwargs(self) -> None:
        """Logger should accept keyword arguments."""
        setup_logging(None)
        logger = get_logger("test.kwargs")

        logger.info(
            "message with context",
            participant_id=123,
            operation="test",
            score=0.95,
        )


class TestContextBinding:
    """Tests for context variable binding."""

    def test_bind_context(self) -> None:
        """Should bind context variables."""
        setup_logging(None)
        clear_context()

        bind_context(participant_id=123, operation="test")

    def test_unbind_context(self) -> None:
        """Should unbind context variables."""
        setup_logging(None)
        clear_context()
        bind_context(key1="value1", key2="value2")

        unbind_context("key1", "key2")

    def test_clear_context(self) -> None:
        """Should clear all context variables."""
        setup_logging(None)
        bind_context(key1="value1", key2="value2")

        clear_context()


class TestWithContextDecorator:
    """Tests for with_context decorator."""

    def test_decorator_on_sync_function(self) -> None:
        """Should work with sync functions."""
        setup_logging(None)
        clear_context()

        @with_context(operation="test_op")
        def sync_func() -> str:
            return "result"

        result = sync_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self) -> None:
        """Should work with async functions."""
        setup_logging(None)
        clear_context()

        @with_context(operation="async_op")
        async def async_func() -> str:
            return "async_result"

        result = await async_func()
        assert result == "async_result"

    def test_decorator_with_multiple_context_vars(self) -> None:
        """Should bind multiple context variables."""
        setup_logging(None)
        clear_context()

        @with_context(participant_id=123, operation="assess", model="test")
        def multi_context_func() -> int:
            return 42

        result = multi_context_func()
        assert result == 42

    def test_decorator_cleans_up_on_exception(self) -> None:
        """Should clean up context even if function raises."""
        setup_logging(None)
        clear_context()

        @with_context(operation="failing")
        def failing_func() -> None:
            raise ValueError("intentional error")

        with pytest.raises(ValueError, match="intentional error"):
            failing_func()


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_full_workflow(self) -> None:
        """Test complete logging workflow."""
        settings = LoggingSettings(level="DEBUG", format="json")
        setup_logging(settings)
        clear_context()

        logger = get_logger("test.integration")

        bind_context(participant_id=456, phase="quantitative")
        logger.info("Processing started", items=8)
        logger.info("Processing complete", score=15)

        clear_context()

    def test_multiple_loggers(self) -> None:
        """Multiple loggers should work independently."""
        setup_logging(None)

        logger1 = get_logger("module.one")
        logger2 = get_logger("module.two")

        logger1.info("from logger 1")
        logger2.info("from logger 2")
