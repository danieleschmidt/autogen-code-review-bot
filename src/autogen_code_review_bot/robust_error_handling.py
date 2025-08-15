#!/usr/bin/env python3
"""
Robust error handling and recovery mechanisms for AutoGen Code Review Bot.

This module provides comprehensive error handling, logging, and recovery
strategies for all analysis operations.
"""

import functools
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from .exceptions import AnalysisError
from .logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    operation: str
    component: str
    severity: ErrorSeverity
    timestamp: datetime
    details: dict
    stacktrace: Optional[str] = None


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""

    def __init__(self):
        self.error_history = []
        self.failure_counts = {}
        self.circuit_breakers = {}

    def handle_error(self, error: Exception, context: ErrorContext) -> None:
        """Handle error with appropriate logging and recovery actions."""
        # Record error in history
        self.error_history.append(
            {"error": str(error), "context": context, "timestamp": context.timestamp}
        )

        # Track failure counts
        operation_key = f"{context.component}::{context.operation}"
        self.failure_counts[operation_key] = (
            self.failure_counts.get(operation_key, 0) + 1
        )

        # Log based on severity
        error_message = f"Error in {context.component}.{context.operation}: {error}"

        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(
                error_message,
                extra={
                    "component": context.component,
                    "operation": context.operation,
                    "error_details": context.details,
                    "stacktrace": context.stacktrace,
                },
            )
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(
                error_message,
                extra={
                    "component": context.component,
                    "operation": context.operation,
                    "error_details": context.details,
                },
            )
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(
                error_message,
                extra={"component": context.component, "operation": context.operation},
            )
        else:
            logger.info(
                error_message,
                extra={"component": context.component, "operation": context.operation},
            )

        # Implement circuit breaker if too many failures
        if self.failure_counts.get(operation_key, 0) > 5:
            self.circuit_breakers[operation_key] = True
            logger.error(f"Circuit breaker activated for {operation_key}")

    def is_circuit_open(self, component: str, operation: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        operation_key = f"{component}::{operation}"
        return self.circuit_breakers.get(operation_key, False)

    def reset_circuit(self, component: str, operation: str) -> None:
        """Reset circuit breaker for an operation."""
        operation_key = f"{component}::{operation}"
        self.circuit_breakers[operation_key] = False
        self.failure_counts[operation_key] = 0
        logger.info(f"Circuit breaker reset for {operation_key}")


# Global error handler instance
error_handler = RobustErrorHandler()


def robust_operation(
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    retry_count: int = 0,
    fallback_value: Any = None,
    raise_on_failure: bool = True,
):
    """Decorator for robust error handling of operations."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check circuit breaker
            if error_handler.is_circuit_open(component, operation):
                if fallback_value is not None:
                    logger.warning(
                        f"Circuit breaker open for {component}.{operation}, returning fallback"
                    )
                    return fallback_value
                elif raise_on_failure:
                    raise AnalysisError(
                        f"Circuit breaker open for {component}.{operation}"
                    )
                else:
                    return None

            last_exception = None
            for attempt in range(retry_count + 1):
                try:
                    result = func(*args, **kwargs)

                    # Reset circuit breaker on success after failures
                    operation_key = f"{component}::{operation}"
                    if error_handler.failure_counts.get(operation_key, 0) > 0:
                        error_handler.reset_circuit(component, operation)

                    return result

                except Exception as e:
                    last_exception = e

                    # Create error context
                    context = ErrorContext(
                        operation=operation,
                        component=component,
                        severity=severity,
                        timestamp=datetime.now(timezone.utc),
                        details={
                            "attempt": attempt + 1,
                            "max_attempts": retry_count + 1,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                        stacktrace=traceback.format_exc(),
                    )

                    # Handle the error
                    error_handler.handle_error(e, context)

                    # If not the last attempt, continue retrying
                    if attempt < retry_count:
                        logger.info(
                            f"Retrying {component}.{operation} (attempt {attempt + 2})"
                        )
                        continue

                    # Last attempt failed
                    if fallback_value is not None:
                        logger.warning(
                            f"All attempts failed for {component}.{operation}, returning fallback"
                        )
                        return fallback_value
                    elif raise_on_failure:
                        raise AnalysisError(
                            f"Operation {component}.{operation} failed after {retry_count + 1} attempts: {e}"
                        )
                    else:
                        return None

            return None  # Should never reach here

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    *args,
    component: str = "unknown",
    operation: str = "unknown",
    fallback_value: Any = None,
    **kwargs,
) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = ErrorContext(
            operation=operation,
            component=component,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc),
            details={"function": func.__name__},
            stacktrace=traceback.format_exc(),
        )

        error_handler.handle_error(e, context)

        if fallback_value is not None:
            return fallback_value
        else:
            raise AnalysisError(
                f"Safe execution failed for {component}.{operation}: {e}"
            )


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_input(
    value: Any, validator: Callable[[Any], bool], error_message: str
) -> None:
    """Validate input with custom validator."""
    if not validator(value):
        raise ValidationError(error_message)


def validate_file_path(path: str) -> None:
    """Validate file path input."""
    if not path or not isinstance(path, str):
        raise ValidationError("Path must be a non-empty string")

    if len(path) > 4096:  # Reasonable path length limit
        raise ValidationError("Path is too long")

    # Check for path traversal attempts
    if ".." in path or "//" in path:
        logger.warning(f"Suspicious path detected: {path}")


def validate_config_data(config: dict) -> None:
    """Validate configuration data."""
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")

    required_keys = ["agents", "conversation"]
    for key in required_keys:
        if key not in config:
            raise ValidationError(f"Missing required configuration key: {key}")


class HealthChecker:
    """System health monitoring."""

    def __init__(self):
        self.health_status = {}
        self.last_check = {}

    def check_component_health(self, component: str) -> dict:
        """Check health of a specific component."""
        try:
            health_info = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": component,
            }

            # Check error rates
            operation_key = f"{component}::*"
            failure_count = sum(
                count
                for key, count in error_handler.failure_counts.items()
                if key.startswith(component)
            )

            if failure_count > 10:
                health_info["status"] = "degraded"
                health_info["warnings"] = [f"High failure count: {failure_count}"]
            elif failure_count > 20:
                health_info["status"] = "unhealthy"
                health_info["errors"] = [f"Very high failure count: {failure_count}"]

            # Check circuit breakers
            open_circuits = [
                key
                for key in error_handler.circuit_breakers
                if key.startswith(component) and error_handler.circuit_breakers[key]
            ]

            if open_circuits:
                health_info["status"] = "degraded"
                health_info["circuit_breakers"] = open_circuits

            self.health_status[component] = health_info
            self.last_check[component] = datetime.now(timezone.utc)

            return health_info

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": component,
            }

    def get_overall_health(self) -> dict:
        """Get overall system health."""
        try:
            components = [
                "pr_analysis",
                "security_analysis",
                "style_analysis",
                "performance_analysis",
                "agents",
            ]
            health_checks = {}

            for component in components:
                health_checks[component] = self.check_component_health(component)

            # Determine overall status
            statuses = [check["status"] for check in health_checks.values()]

            if "error" in statuses or "unhealthy" in statuses:
                overall_status = "unhealthy"
            elif "degraded" in statuses:
                overall_status = "degraded"
            else:
                overall_status = "healthy"

            return {
                "overall_status": overall_status,
                "components": health_checks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_summary": {
                    "total_errors": len(error_handler.error_history),
                    "circuit_breakers_open": len(
                        [cb for cb in error_handler.circuit_breakers.values() if cb]
                    ),
                    "recent_failures": sum(error_handler.failure_counts.values()),
                },
            }

        except Exception as e:
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global health checker instance
health_checker = HealthChecker()
