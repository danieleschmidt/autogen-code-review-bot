"""Structured logging configuration for AutoGen Code Review Bot."""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variable for request tracking
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes structured fields and request context."""

    def __init__(self, service_name: str = "autogen-code-review-bot"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Mask sensitive data in log message
        message = record.getMessage()
        try:
            from .token_security import TokenMasker
            message = TokenMasker.mask_sensitive_data(message)
        except ImportError:
            # Fallback if token_security module is not available
            pass

        log_entry = {
            "timestamp": time.time(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": message,
            "service": self.service_name,
        }

        # Add request ID if available
        req_id = request_id.get()
        if req_id:
            log_entry["request_id"] = req_id

        # Add exception info if present with token masking
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            try:
                from .token_security import TokenMasker
                exception_text = TokenMasker.mask_sensitive_data(exception_text)
            except ImportError:
                pass
            log_entry["exception"] = exception_text

        # Add any extra fields with token masking
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                        'filename', 'module', 'lineno', 'funcName', 'created',
                        'msecs', 'relativeCreated', 'thread', 'threadName',
                        'processName', 'process', 'getMessage', 'exc_info', 'exc_text',
                        'stack_info', 'message')
        }

        # Mask sensitive data in extra fields
        try:
            from .token_security import TokenMasker
            extra_fields = TokenMasker.mask_dict(extra_fields)
        except ImportError:
            # Fallback if token_security module is not available
            pass

        log_entry.update(extra_fields)

        return json.dumps(log_entry, default=str)


def configure_logging(level: str = "INFO", service_name: str = "autogen-code-review-bot") -> None:
    """Configure structured JSON logging for the application."""

    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with structured formatter
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter(service_name))

    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with structured formatting."""
    return logging.getLogger(name)


def set_request_id(req_id: Optional[str] = None) -> str:
    """Set the request ID for the current context."""
    if req_id is None:
        req_id = str(uuid.uuid4())
    request_id.set(req_id)
    return req_id


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return request_id.get()


def log_operation_start(logger: logging.Logger, operation: str, **kwargs: Any) -> Dict[str, Any]:
    """Log the start of an operation with timing context."""
    context = {
        "operation": operation,
        "start_time": time.time(),
        **kwargs
    }
    logger.info("Operation started", extra=context)
    return context


def log_operation_end(logger: logging.Logger, context: Dict[str, Any],
                     success: bool = True, error: Optional[str] = None, **kwargs: Any) -> None:
    """Log the end of an operation with duration and result."""
    duration = time.time() - context["start_time"]
    log_data = {
        **context,
        "duration_seconds": round(duration, 3),
        "success": success,
        **kwargs
    }

    if error:
        log_data["error"] = error

    if success:
        logger.info("Operation completed", extra=log_data)
    else:
        logger.error("Operation failed", extra=log_data)


class ContextLogger:
    """Logger wrapper that automatically includes context in all log entries."""

    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        extra = {**self.context, **kwargs}
        getattr(self.logger, level)(message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log("critical", message, **kwargs)
