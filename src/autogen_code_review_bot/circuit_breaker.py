"""Circuit breaker pattern implementation for enhanced error handling.

This module provides a comprehensive circuit breaker implementation to prevent
cascading failures and improve system resilience. The circuit breaker monitors
failure rates and automatically blocks requests when services are unhealthy.

Features:
- Three states: CLOSED, OPEN, HALF_OPEN  
- Configurable failure thresholds and timeouts
- Exponential backoff with jitter
- Respect for rate limiting headers
- Integration with metrics and logging systems
"""

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from .logging_utils import RequestContext, get_request_logger
from .metrics import get_metrics_registry

logger = get_request_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5           # Number of failures to open circuit
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes needed to close from half-open
    request_timeout: float = 30.0        # Individual request timeout
    max_retries: int = 3                 # Maximum retry attempts
    base_delay: float = 0.5              # Base delay for exponential backoff
    max_delay: float = 60.0              # Maximum delay between retries
    jitter_factor: float = 0.1           # Jitter factor (0.0-1.0)
    monitoring_window: int = 100         # Number of recent requests to track


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation with failure tracking and recovery."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Name for this circuit breaker instance.
            config: Configuration settings.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()

        # Track recent requests for failure rate calculation
        self._recent_requests: deque = deque(maxlen=self.config.monitoring_window)

        # Metrics integration
        self._registry = get_metrics_registry()
        self._state_gauge = self._registry.gauge(
            f"circuit_breaker_state_{name}",
            f"Circuit breaker state for {name} (0=closed, 1=open, 2=half_open)",
            labels=["circuit_breaker"]
        )
        self._failure_counter = self._registry.counter(
            f"circuit_breaker_failures_{name}",
            f"Total failures for circuit breaker {name}",
            labels=["circuit_breaker", "error_type"]
        )

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate from recent requests."""
        with self._lock:
            if not self._recent_requests:
                return 0.0

            failures = sum(1 for success in self._recent_requests if not success)
            return failures / len(self._recent_requests)

    def _record_success(self):
        """Record a successful request."""
        with self._lock:
            self._recent_requests.append(True)
            self._success_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()

    def _record_failure(self, error_type: str = "unknown"):
        """Record a failed request."""
        with self._lock:
            self._recent_requests.append(False)
            self._failure_count += 1
            self._last_failure_time = time.time()

            # Record metrics
            self._failure_counter.increment(labels={
                "circuit_breaker": self.name,
                "error_type": error_type
            })

            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0

            self._state_gauge.set(0, {"circuit_breaker": self.name})

            logger.info(
                f"Circuit breaker '{self.name}' transitioned to CLOSED",
                circuit_breaker=self.name,
                old_state=old_state.value,
                new_state=self._state.value
            )

    def _transition_to_open(self):
        """Transition to OPEN state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.OPEN
            self._success_count = 0

            self._state_gauge.set(1, {"circuit_breaker": self.name})

            logger.warning(
                f"Circuit breaker '{self.name}' transitioned to OPEN",
                circuit_breaker=self.name,
                old_state=old_state.value,
                new_state=self._state.value,
                failure_count=self._failure_count,
                failure_rate=self.failure_rate
            )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.HALF_OPEN
            self._success_count = 0

            self._state_gauge.set(2, {"circuit_breaker": self.name})

            logger.info(
                f"Circuit breaker '{self.name}' transitioned to HALF_OPEN",
                circuit_breaker=self.name,
                old_state=old_state.value,
                new_state=self._state.value
            )

    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state."""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                # Check if enough time has passed to try recovery
                if time.time() - self._last_failure_time >= self.config.recovery_timeout:
                    self._transition_to_half_open()
                    return True
                return False
            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Allow limited requests to test recovery
                return True

            return False

    def call(self, func: Callable, *args, context: Optional[RequestContext] = None, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute.
            *args: Function arguments.
            context: Request context for logging.
            **kwargs: Function keyword arguments.
            
        Returns:
            Function result.
            
        Raises:
            CircuitBreakerError: If circuit breaker is open.
            Exception: If function raises an exception.
        """
        if not self._should_allow_request():
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN, request blocked"
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            error_type = type(e).__name__
            self._record_failure(error_type)

            if context:
                logger.warning(
                    f"Circuit breaker '{self.name}' recorded failure",
                    context=context,
                    circuit_breaker=self.name,
                    error_type=error_type,
                    failure_count=self._failure_count,
                    state=self._state.value
                )

            raise

    def get_stats(self) -> Dict[str, Union[str, int, float]]:
        """Get circuit breaker statistics.
        
        Returns:
            Dictionary with current statistics.
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_rate": self.failure_rate,
                "recent_requests_count": len(self._recent_requests),
                "last_failure_time": self._last_failure_time,
                "time_since_last_failure": time.time() - self._last_failure_time if self._last_failure_time else 0,
            }


class RetryStrategy:
    """Advanced retry strategy with exponential backoff and jitter."""

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize retry strategy.
        
        Args:
            config: Circuit breaker configuration.
        """
        self.config = config

    def calculate_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (0-based).
            retry_after: Optional Retry-After header value.
            
        Returns:
            Delay in seconds.
        """
        if retry_after is not None:
            # Respect server-provided retry delay
            return retry_after

        # Exponential backoff with jitter
        base_delay = self.config.base_delay * (2 ** attempt)
        max_delay = min(base_delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        jitter = max_delay * self.config.jitter_factor * random.random()

        return max_delay + jitter

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if request should be retried.
        
        Args:
            attempt: Current attempt number (0-based).
            error: Exception that occurred.
            
        Returns:
            True if should retry, False otherwise.
        """
        if attempt >= self.config.max_retries:
            return False

        # Import here to avoid circular import
        import requests

        # Don't retry on authentication errors
        if isinstance(error, requests.HTTPError):
            if hasattr(error, 'response') and error.response.status_code in [401, 403]:
                return False

            # Don't retry on client errors (except rate limits)
            if hasattr(error, 'response') and 400 <= error.response.status_code < 500:
                return error.response.status_code == 429  # Retry on rate limits

        # Retry on network errors and server errors
        if isinstance(error, (requests.ConnectionError, requests.Timeout, requests.HTTPError)):
            return True

        return False

    def extract_retry_after(self, response) -> Optional[float]:
        """Extract Retry-After value from response headers.
        
        Args:
            response: HTTP response object.
            
        Returns:
            Retry delay in seconds, or None if not present.
        """
        if not hasattr(response, 'headers'):
            return None

        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                # Retry-After can be a date, but we'll just use default delay
                pass

        # Check for GitHub-specific rate limit headers
        reset_time = response.headers.get('X-RateLimit-Reset')
        if reset_time:
            try:
                reset_timestamp = float(reset_time)
                delay = max(0, reset_timestamp - time.time())
                return delay
            except ValueError:
                pass

        return None


# Global circuit breaker instances
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_breaker_lock = threading.RLock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker instance.
    
    Args:
        name: Circuit breaker name.
        config: Optional configuration.
        
    Returns:
        Circuit breaker instance.
    """
    with _breaker_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def reset_circuit_breaker(name: str) -> None:
    """Reset a circuit breaker to CLOSED state.
    
    Args:
        name: Circuit breaker name.
    """
    with _breaker_lock:
        if name in _circuit_breakers:
            breaker = _circuit_breakers[name]
            with breaker._lock:
                breaker._transition_to_closed()


def get_all_circuit_breaker_stats() -> Dict[str, Dict[str, Union[str, int, float]]]:
    """Get statistics for all circuit breakers.
    
    Returns:
        Dictionary mapping breaker names to their statistics.
    """
    with _breaker_lock:
        return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}
