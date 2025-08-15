"""Structured logging utilities with request correlation."""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Optional, Union


class RequestContext:
    """Context for tracking requests across the application."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize request context.

        Args:
            request_id: Unique identifier for the request. Auto-generated if not provided.
            metadata: Additional metadata to include with the request.
        """
        self.request_id = request_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.start_time = time.time()

    def get_duration_ms(self) -> float:
        """Get the duration since context creation in milliseconds."""
        return (time.time() - self.start_time) * 1000


class StructuredLogger:
    """Structured logger with JSON output and request correlation."""

    def __init__(self, name: str):
        """Initialize structured logger.

        Args:
            name: Logger name (typically module name).
        """
        self.name = name
        self.logger = logging.getLogger(name)

    def _format_structured_log(
        self,
        level: str,
        message: str,
        context: Optional[RequestContext] = None,
        **kwargs,
    ) -> str:
        """Format log entry as structured JSON.

        Args:
            level: Log level (INFO, ERROR, etc.).
            message: Log message.
            context: Request context for correlation.
            **kwargs: Additional fields to include.

        Returns:
            JSON-formatted log string.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "module": self.name,
        }

        # Add request context information
        if context:
            log_entry["request_id"] = context.request_id
            log_entry["duration_ms"] = context.get_duration_ms()
            if context.metadata:
                log_entry["metadata"] = context.metadata

        # Add any additional fields, sanitizing sensitive data
        if kwargs:
            sanitized_kwargs = sanitize_log_data(kwargs)
            log_entry.update(sanitized_kwargs)

        return json.dumps(log_entry, default=str)

    def debug(self, message: str, context: Optional[RequestContext] = None, **kwargs):
        """Log debug message."""
        formatted = self._format_structured_log("DEBUG", message, context, **kwargs)
        self.logger.debug(formatted)

    def info(self, message: str, context: Optional[RequestContext] = None, **kwargs):
        """Log info message."""
        formatted = self._format_structured_log("INFO", message, context, **kwargs)
        self.logger.info(formatted)

    def warning(self, message: str, context: Optional[RequestContext] = None, **kwargs):
        """Log warning message."""
        formatted = self._format_structured_log("WARNING", message, context, **kwargs)
        self.logger.warning(formatted)

    def error(
        self,
        message: str,
        context: Optional[RequestContext] = None,
        exc_info: bool = False,
        **kwargs,
    ):
        """Log error message."""
        if exc_info:
            kwargs["exc_info"] = True
        formatted = self._format_structured_log("ERROR", message, context, **kwargs)
        self.logger.error(formatted, exc_info=exc_info)

    def critical(
        self, message: str, context: Optional[RequestContext] = None, **kwargs
    ):
        """Log critical message."""
        formatted = self._format_structured_log("CRITICAL", message, context, **kwargs)
        self.logger.critical(formatted)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_request_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger for the given name.

    Args:
        name: Logger name (typically module name).

    Returns:
        StructuredLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def configure_structured_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Configure structured logging for the application.

    Args:
        config: Logging configuration dictionary.
    """
    config = config or {}

    # Set default logging level
    level = config.get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",  # We handle formatting in StructuredLogger
        force=True,
    )

    # Configure root logger to use JSON format
    root_logger = logging.getLogger()

    # Remove default handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler with JSON formatting
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(handler)


def sanitize_log_data(data: Any) -> Any:
    """Sanitize sensitive data from log entries.

    Args:
        data: Data to sanitize (dict, list, or primitive).

    Returns:
        Sanitized copy of the data.
    """
    if isinstance(data, dict):
        return {key: sanitize_log_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    elif isinstance(data, str):
        return _sanitize_string_value(data)
    else:
        return data


def _sanitize_string_value(value: str) -> str:
    """Sanitize sensitive strings.

    Args:
        value: String value to check.

    Returns:
        Original value or "***" if sensitive.
    """
    sensitive_patterns = [
        r"token",
        r"key",
        r"secret",
        r"password",
        r"auth",
        r"credential",
    ]

    # Check if any sensitive pattern matches the value name (case insensitive)
    for pattern in sensitive_patterns:
        if re.search(pattern, value.lower()):
            return "***"

    return value


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name indicates sensitive data.

    Args:
        key: Dictionary key to check.

    Returns:
        True if the key indicates sensitive data.
    """
    sensitive_patterns = [
        r"token",
        r"key",
        r"secret",
        r"password",
        r"auth",
        r"credential",
    ]

    key_lower = key.lower()
    return any(re.search(pattern, key_lower) for pattern in sensitive_patterns)


def sanitize_log_data(data: Any) -> Any:
    """Sanitize sensitive data from log entries.

    Args:
        data: Data to sanitize (dict, list, or primitive).

    Returns:
        Sanitized copy of the data.
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if _is_sensitive_key(key):
                sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    else:
        return data


def timed_operation(
    context: Optional[RequestContext] = None, operation: str = "operation"
):
    """Decorator to log operation timing.

    Args:
        context: Request context for correlation.
        operation: Operation name for logging.

    Returns:
        Decorator function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_request_logger(func.__module__)
            start_time = time.time()

            logger.info(
                f"Starting {operation}",
                context=context,
                operation=operation,
                function=func.__name__,
            )

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Completed {operation}",
                    context=context,
                    operation=operation,
                    function=func.__name__,
                    duration_ms=duration_ms,
                    status="success",
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                logger.error(
                    f"Failed {operation}",
                    context=context,
                    operation=operation,
                    function=func.__name__,
                    duration_ms=duration_ms,
                    status="error",
                    error_type=type(e).__name__,
                    exc_info=True,
                )

                raise

        return wrapper

    return decorator


class MetricsCollector:
    """Collect and log metrics for operations."""

    def __init__(self, context: Optional[RequestContext] = None):
        """Initialize metrics collector.

        Args:
            context: Request context for correlation.
        """
        self.context = context
        self.metrics: Dict[str, Union[int, float]] = {}
        self.logger = get_request_logger(__name__)

    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric.

        Args:
            metric_name: Name of the metric.
            value: Value to increment by (default: 1).
        """
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
        else:
            self.metrics[metric_name] = value

    def set_gauge(self, metric_name: str, value: Union[int, float]) -> None:
        """Set a gauge metric value.

        Args:
            metric_name: Name of the metric.
            value: Value to set.
        """
        self.metrics[metric_name] = value

    def record_timing(self, metric_name: str, duration_ms: float) -> None:
        """Record a timing metric.

        Args:
            metric_name: Name of the metric.
            duration_ms: Duration in milliseconds.
        """
        self.metrics[metric_name] = duration_ms

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get current metrics.

        Returns:
            Dictionary of current metrics.
        """
        return self.metrics.copy()

    def log_metrics(self, message: str = "Metrics report") -> None:
        """Log current metrics.

        Args:
            message: Log message.
        """
        self.logger.info(message, context=self.context, metrics=self.metrics)
