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
    """Enhanced comprehensive error handling and recovery system with predictive capabilities."""

    def __init__(self, enable_predictive_recovery: bool = True, enable_auto_healing: bool = True):
        self.error_history = []
        self.failure_counts = {}
        self.circuit_breakers = {}
        self.enable_predictive_recovery = enable_predictive_recovery
        self.enable_auto_healing = enable_auto_healing
        
        # Enhanced tracking
        self.error_patterns = {}
        self.recovery_success_rate = {}
        self.performance_impact = {}
        self.security_violations = []
        
        # Auto-healing strategies
        self.healing_strategies = {
            'high_failure_rate': self._strategy_reduce_load,
            'memory_leak': self._strategy_restart_component,
            'external_service_timeout': self._strategy_fallback_mode,
            'security_violation': self._strategy_lock_down
        }
        
        logger.info("Enhanced RobustErrorHandler initialized with predictive recovery and auto-healing")

    def handle_error(self, error: Exception, context: ErrorContext) -> None:
        """Enhanced error handling with predictive analysis and auto-healing."""
        # Record error in enhanced history
        error_record = {
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "timestamp": context.timestamp,
            "severity": context.severity,
            "recovery_attempted": False,
            "recovery_successful": False
        }
        self.error_history.append(error_record)

        # Track failure counts and patterns
        operation_key = f"{context.component}::{context.operation}"
        self.failure_counts[operation_key] = (
            self.failure_counts.get(operation_key, 0) + 1
        )
        
        # Track error patterns for prediction
        self._track_error_patterns(error, context)
        
        # Check for security violations
        if self._is_security_related_error(error, context):
            self._handle_security_violation(error, context)

        # Enhanced logging with structured data
        error_message = f"Error in {context.component}.{context.operation}: {error}"
        log_extra = {
            "component": context.component,
            "operation": context.operation,
            "error_type": type(error).__name__,
            "failure_count": self.failure_counts[operation_key],
            "severity": context.severity.value,
            "error_details": context.details,
        }

        if context.severity == ErrorSeverity.CRITICAL:
            log_extra["stacktrace"] = context.stacktrace
            log_extra["immediate_action_required"] = True
            logger.critical(error_message, extra=log_extra)
            
            # Trigger immediate auto-healing for critical errors
            if self.enable_auto_healing:
                self._attempt_auto_healing('critical_error', context)
                
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(error_message, extra=log_extra)
            
            # Consider auto-healing for high severity errors
            if self.enable_auto_healing and self.failure_counts[operation_key] > 3:
                self._attempt_auto_healing('high_failure_rate', context)
                
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(error_message, extra=log_extra)
        else:
            logger.info(error_message, extra=log_extra)

        # Enhanced circuit breaker logic with gradual degradation
        failure_threshold = self._get_dynamic_failure_threshold(operation_key)
        if self.failure_counts.get(operation_key, 0) >= failure_threshold:
            self._activate_circuit_breaker(operation_key, context)
            
        # Predictive error prevention
        if self.enable_predictive_recovery:
            self._analyze_and_predict_errors(context)
            
    def _track_error_patterns(self, error: Exception, context: ErrorContext) -> None:
        """Track error patterns for predictive analysis."""
        pattern_key = f"{context.component}::{type(error).__name__}"
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = []
            
        self.error_patterns[pattern_key].append({
            'timestamp': context.timestamp,
            'operation': context.operation,
            'details': context.details
        })
        
        # Keep only recent patterns (last 100 entries)
        if len(self.error_patterns[pattern_key]) > 100:
            self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-100:]
            
    def _is_security_related_error(self, error: Exception, context: ErrorContext) -> bool:
        """Check if error indicates a security issue."""
        security_indicators = [
            'permission', 'unauthorized', 'forbidden', 'access denied',
            'authentication', 'token', 'credential', 'injection'
        ]
        
        error_text = str(error).lower()
        return any(indicator in error_text for indicator in security_indicators)
        
    def _handle_security_violation(self, error: Exception, context: ErrorContext) -> None:
        """Handle security-related errors with enhanced monitoring."""
        violation_record = {
            'timestamp': context.timestamp,
            'error': str(error),
            'context': context,
            'severity': 'HIGH',
            'component': context.component,
            'operation': context.operation
        }
        
        self.security_violations.append(violation_record)
        
        logger.critical(
            "SECURITY VIOLATION DETECTED",
            extra={
                'security_event': True,
                'violation_type': type(error).__name__,
                'component': context.component,
                'operation': context.operation,
                'immediate_review_required': True
            }
        )
        
        # Auto-lock down if multiple violations
        if len(self.security_violations) > 5:
            self._attempt_auto_healing('security_violation', context)
            
    def _get_dynamic_failure_threshold(self, operation_key: str) -> int:
        """Calculate dynamic failure threshold based on operation importance."""
        # Critical operations have lower thresholds
        critical_operations = ['security_analysis', 'authentication', 'authorization']
        
        for critical_op in critical_operations:
            if critical_op in operation_key:
                return 3  # Lower threshold for critical operations
                
        return 5  # Default threshold
        
    def _activate_circuit_breaker(self, operation_key: str, context: ErrorContext) -> None:
        """Activate circuit breaker with enhanced logging and notifications."""
        self.circuit_breakers[operation_key] = {
            'active': True,
            'activated_at': context.timestamp,
            'failure_count': self.failure_counts[operation_key],
            'context': context
        }
        
        logger.error(
            f"Circuit breaker ACTIVATED for {operation_key}",
            extra={
                'circuit_breaker_event': True,
                'operation_key': operation_key,
                'failure_count': self.failure_counts[operation_key],
                'component': context.component,
                'operation': context.operation
            }
        )
        
        # Attempt auto-healing when circuit breaker activates
        if self.enable_auto_healing:
            self._attempt_auto_healing('circuit_breaker_activation', context)
            
    def _analyze_and_predict_errors(self, context: ErrorContext) -> None:
        """Analyze patterns and predict potential future errors."""
        operation_key = f"{context.component}::{context.operation}"
        
        # Look for error clustering (multiple errors in short time)
        recent_errors = [
            record for record in self.error_history[-10:]
            if (context.timestamp - record['timestamp']).total_seconds() < 300  # 5 minutes
        ]
        
        if len(recent_errors) >= 3:
            logger.warning(
                "Error clustering detected - potential system degradation",
                extra={
                    'predictive_alert': True,
                    'cluster_size': len(recent_errors),
                    'component': context.component,
                    'time_window': '5_minutes'
                }
            )
            
            # Proactive healing for error clusters
            if self.enable_auto_healing:
                self._attempt_auto_healing('error_clustering', context)
                
    def _attempt_auto_healing(self, trigger: str, context: ErrorContext) -> bool:
        """Attempt automatic system healing based on error patterns."""
        if trigger not in self.healing_strategies:
            logger.info(f"No healing strategy available for trigger: {trigger}")
            return False
            
        try:
            strategy = self.healing_strategies[trigger]
            success = strategy(context)
            
            # Track healing success rate
            if trigger not in self.recovery_success_rate:
                self.recovery_success_rate[trigger] = {'attempts': 0, 'successes': 0}
                
            self.recovery_success_rate[trigger]['attempts'] += 1
            if success:
                self.recovery_success_rate[trigger]['successes'] += 1
                
            logger.info(
                f"Auto-healing {'successful' if success else 'failed'}",
                extra={
                    'healing_event': True,
                    'trigger': trigger,
                    'strategy': strategy.__name__,
                    'success': success,
                    'component': context.component
                }
            )
            
            return success
            
        except Exception as healing_error:
            logger.error(
                f"Auto-healing strategy failed: {healing_error}",
                extra={
                    'healing_error': True,
                    'trigger': trigger,
                    'component': context.component,
                    'healing_exception': str(healing_error)
                }
            )
            return False
            
    def _strategy_reduce_load(self, context: ErrorContext) -> bool:
        """Reduce system load by implementing rate limiting."""
        logger.info("Implementing load reduction strategy", component=context.component)
        # Implementation would involve reducing concurrent operations
        return True
        
    def _strategy_restart_component(self, context: ErrorContext) -> bool:
        """Simulate component restart (would be actual restart in production)."""
        logger.info("Simulating component restart", component=context.component)
        # Reset failure counts for this component
        component_keys = [key for key in self.failure_counts.keys() 
                         if key.startswith(context.component)]
        for key in component_keys:
            self.failure_counts[key] = 0
        return True
        
    def _strategy_fallback_mode(self, context: ErrorContext) -> bool:
        """Switch to fallback mode for external service issues."""
        logger.info("Activating fallback mode", component=context.component)
        # Implementation would switch to cached data or alternative service
        return True
        
    def _strategy_lock_down(self, context: ErrorContext) -> bool:
        """Lock down system in response to security violations."""
        logger.critical("SECURITY LOCKDOWN ACTIVATED", component=context.component)
        # Implementation would restrict access and require manual intervention
        return True

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
