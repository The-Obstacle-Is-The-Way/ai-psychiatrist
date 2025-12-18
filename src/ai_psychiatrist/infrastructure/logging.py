"""Structured logging configuration using structlog.

This module provides production-ready logging with:
- JSON output for production (machine-parseable)
- Console output for development (human-readable)
- Context variable binding for request tracing
- Caller information (file, line, function)
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from ai_psychiatrist.config import LoggingSettings


def setup_logging(settings: LoggingSettings | None = None) -> None:
    """Configure structured logging for the application.

    Args:
        settings: Logging settings. If None, uses defaults from config.
    """
    if settings is None:
        from ai_psychiatrist.config import get_settings  # noqa: PLC0415

        settings = get_settings().logging

    # Shared processors for all formats
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    if settings.include_timestamp:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso", utc=True))

    shared_processors.extend(
        [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]
    )

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

    # Format-specific final processors
    if settings.format == "json":
        final_processors: list[structlog.types.Processor] = [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
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
        force=True,  # Override any existing config
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
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_context(**kwargs: str | int | float | bool) -> None:
    """Bind context variables for current execution context.

    These variables will be included in all log messages within the
    current context (thread/coroutine).

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


def with_context(
    **context_vars: str | int | float | bool,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to bind context for function execution.

    Context is automatically cleaned up after function returns,
    even if an exception is raised.

    Args:
        **context_vars: Context variables to bind.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        import asyncio  # noqa: PLC0415
        from functools import wraps  # noqa: PLC0415

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            bind_context(**context_vars)
            try:
                return func(*args, **kwargs)
            finally:
                unbind_context(*context_vars.keys())

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            bind_context(**context_vars)
            try:
                return await func(*args, **kwargs)
            finally:
                unbind_context(*context_vars.keys())

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
