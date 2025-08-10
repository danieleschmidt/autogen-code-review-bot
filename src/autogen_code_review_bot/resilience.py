#!/usr/bin/env python3
"""
Enterprise Resilience Framework for AutoGen Code Review Bot.

Implements comprehensive error handling, circuit breakers, retries,
bulkheads, and fault tolerance patterns for production reliability.
"""

import asyncio
import functools
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from .circuit_breaker import CircuitBreaker
from .exceptions import (
    AnalysisError,
    SecurityError,
)
from .logging_utils import get_request_logger as get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


class RetryStrategy(Enum):
    """Retry strategies for different failure scenarios."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, AnalysisError, SecurityError
    ])


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent_requests: int = 100
    queue_size: int = 1000
    timeout_seconds: float = 300.0  # 5 minutes
    priority_levels: int = 3


class TimeoutManager:
    """Manages operation timeouts with graceful degradation."""

    def __init__(self):
        self.active_timeouts: Dict[str, asyncio.Task] = {}
        self.logger = get_logger(__name__ + ".TimeoutManager")

    @contextmanager
    def timeout_context(self, timeout_seconds: float, operation_name: str):
        """Context manager for operation timeouts."""
        start_time = time.time()
        timeout_task = None

        try:
            self.logger.debug(f"Starting timeout context for {operation_name}", extra={
                'timeout_seconds': timeout_seconds,
                'operation': operation_name
            })

            yield

        except Exception:
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                self.logger.error(f"Operation {operation_name} timed out", extra={
                    'elapsed_seconds': elapsed,
                    'timeout_seconds': timeout_seconds
                })
                raise TimeoutError(f"Operation {operation_name} timed out after {elapsed:.2f}s")
            raise

        finally:
            elapsed = time.time() - start_time
            self.logger.debug(f"Timeout context completed for {operation_name}", extra={
                'elapsed_seconds': elapsed
            })

    @asynccontextmanager
    async def async_timeout_context(self, timeout_seconds: float, operation_name: str):
        """Async context manager for operation timeouts."""
        start_time = time.time()

        try:
            self.logger.debug(f"Starting async timeout context for {operation_name}", extra={
                'timeout_seconds': timeout_seconds,
                'operation': operation_name
            })

            async with asyncio.timeout(timeout_seconds):
                yield

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"Async operation {operation_name} timed out", extra={
                'elapsed_seconds': elapsed,
                'timeout_seconds': timeout_seconds
            })
            raise TimeoutError(f"Async operation {operation_name} timed out after {elapsed:.2f}s")

        finally:
            elapsed = time.time() - start_time
            self.logger.debug(f"Async timeout context completed for {operation_name}", extra={
                'elapsed_seconds': elapsed
            })


class RetryManager:
    """Manages retry logic with configurable strategies."""

    def __init__(self, timeout_manager: Optional[TimeoutManager] = None):
        self.timeout_manager = timeout_manager or TimeoutManager()
        self.logger = get_logger(__name__ + ".RetryManager")
        self._retry_stats: Dict[str, Dict[str, Any]] = {}

    def retry(self, config: RetryConfig, operation_name: str = None):
        """Decorator for adding retry logic to functions."""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(func, config, operation_name or func.__name__, *args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_retry_async(func, config, operation_name or func.__name__, *args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper

        return decorator

    def _execute_with_retry(self, func: Callable, config: RetryConfig,
                           operation_name: str, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                self.logger.debug(f"Attempting {operation_name} (attempt {attempt}/{config.max_attempts})")

                result = func(*args, **kwargs)

                # Record successful retry if not first attempt
                if attempt > 1:
                    self._record_retry_success(operation_name, attempt)

                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                    self.logger.warning(f"Non-retryable exception in {operation_name}: {e}")
                    raise

                # Don't retry on last attempt
                if attempt >= config.max_attempts:
                    break

                # Calculate delay
                delay = self._calculate_delay(config, attempt)

                self.logger.warning(f"Attempt {attempt} failed for {operation_name}, retrying in {delay:.2f}s", extra={
                    'error': str(e),
                    'attempt': attempt,
                    'max_attempts': config.max_attempts,
                    'delay_seconds': delay
                })

                time.sleep(delay)

        # All retries exhausted
        self._record_retry_failure(operation_name, config.max_attempts)
        self.logger.error(f"All retry attempts exhausted for {operation_name}", extra={
            'max_attempts': config.max_attempts,
            'final_error': str(last_exception)
        })

        raise last_exception

    async def _execute_with_retry_async(self, func: Callable, config: RetryConfig,
                                       operation_name: str, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                self.logger.debug(f"Attempting async {operation_name} (attempt {attempt}/{config.max_attempts})")

                result = await func(*args, **kwargs)

                # Record successful retry if not first attempt
                if attempt > 1:
                    self._record_retry_success(operation_name, attempt)

                return result

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                    self.logger.warning(f"Non-retryable async exception in {operation_name}: {e}")
                    raise

                # Don't retry on last attempt
                if attempt >= config.max_attempts:
                    break

                # Calculate delay
                delay = self._calculate_delay(config, attempt)

                self.logger.warning(f"Async attempt {attempt} failed for {operation_name}, retrying in {delay:.2f}s", extra={
                    'error': str(e),
                    'attempt': attempt,
                    'max_attempts': config.max_attempts,
                    'delay_seconds': delay
                })

                await asyncio.sleep(delay)

        # All retries exhausted
        self._record_retry_failure(operation_name, config.max_attempts)
        self.logger.error(f"All async retry attempts exhausted for {operation_name}", extra={
            'max_attempts': config.max_attempts,
            'final_error': str(last_exception)
        })

        raise last_exception

    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_factor ** (attempt - 1))
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        else:  # IMMEDIATE
            delay = 0.0

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter if enabled
        if config.jitter and delay > 0:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.0, delay)  # Ensure non-negative

        return delay

    def _record_retry_success(self, operation_name: str, attempts: int):
        """Record successful retry metrics."""
        if operation_name not in self._retry_stats:
            self._retry_stats[operation_name] = {'successes': 0, 'failures': 0, 'total_attempts': 0}

        self._retry_stats[operation_name]['successes'] += 1
        self._retry_stats[operation_name]['total_attempts'] += attempts

        metrics.record_counter("retry_successes_total", 1, tags={
            'operation': operation_name,
            'attempts': str(attempts)
        })

    def _record_retry_failure(self, operation_name: str, attempts: int):
        """Record failed retry metrics."""
        if operation_name not in self._retry_stats:
            self._retry_stats[operation_name] = {'successes': 0, 'failures': 0, 'total_attempts': 0}

        self._retry_stats[operation_name]['failures'] += 1
        self._retry_stats[operation_name]['total_attempts'] += attempts

        metrics.record_counter("retry_failures_total", 1, tags={
            'operation': operation_name,
            'attempts': str(attempts)
        })

    def get_retry_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get retry statistics for all operations."""
        return self._retry_stats.copy()


class BulkheadManager:
    """Implements bulkhead pattern for resource isolation."""

    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.queues: Dict[str, asyncio.Queue] = {}
        self.active_requests: Dict[str, int] = {}
        self.locks: Dict[str, threading.Lock] = {}
        self.logger = get_logger(__name__ + ".BulkheadManager")

    def get_bulkhead(self, name: str) -> asyncio.Semaphore:
        """Get or create bulkhead semaphore for resource isolation."""
        if name not in self.semaphores:
            self.semaphores[name] = asyncio.Semaphore(self.config.max_concurrent_requests)
            self.queues[name] = asyncio.Queue(maxsize=self.config.queue_size)
            self.active_requests[name] = 0
            self.locks[name] = threading.Lock()

            self.logger.info(f"Created bulkhead for {name}", extra={
                'max_concurrent': self.config.max_concurrent_requests,
                'queue_size': self.config.queue_size
            })

        return self.semaphores[name]

    @asynccontextmanager
    async def acquire_resource(self, bulkhead_name: str, priority: int = 1):
        """Acquire resource with bulkhead protection."""
        semaphore = self.get_bulkhead(bulkhead_name)

        start_time = time.time()

        try:
            # Wait for available slot with timeout
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=self.config.timeout_seconds
            )

            with self.locks[bulkhead_name]:
                self.active_requests[bulkhead_name] += 1

            wait_time = time.time() - start_time

            self.logger.debug(f"Acquired resource in bulkhead {bulkhead_name}", extra={
                'wait_time_seconds': wait_time,
                'active_requests': self.active_requests[bulkhead_name]
            })

            # Record metrics
            metrics.record_histogram("bulkhead_wait_time_seconds", wait_time, tags={
                'bulkhead': bulkhead_name
            })

            yield

        except asyncio.TimeoutError:
            self.logger.error(f"Bulkhead {bulkhead_name} acquisition timeout", extra={
                'timeout_seconds': self.config.timeout_seconds,
                'wait_time_seconds': time.time() - start_time
            })

            metrics.record_counter("bulkhead_timeouts_total", 1, tags={
                'bulkhead': bulkhead_name
            })

            raise TimeoutError(f"Could not acquire resource in bulkhead {bulkhead_name}")

        finally:
            try:
                semaphore.release()
                with self.locks[bulkhead_name]:
                    self.active_requests[bulkhead_name] -= 1
            except Exception as e:
                self.logger.error(f"Error releasing bulkhead resource: {e}")

    def get_bulkhead_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current bulkhead statistics."""
        stats = {}
        for name in self.semaphores:
            semaphore = self.semaphores[name]
            stats[name] = {
                'active_requests': self.active_requests.get(name, 0),
                'available_slots': semaphore._value,
                'max_concurrent': self.config.max_concurrent_requests,
                'queue_size': self.config.queue_size
            }
        return stats


class HealthMonitor:
    """Monitors system health and provides degradation strategies."""

    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.degradation_strategies: Dict[str, Callable] = {}
        self.logger = get_logger(__name__ + ".HealthMonitor")
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

    def register_health_check(self, name: str, check_func: Callable,
                            interval_seconds: float = 60.0):
        """Register a health check function."""
        self.health_checks[name] = {
            'function': check_func,
            'interval': interval_seconds,
            'last_check': 0,
            'consecutive_failures': 0
        }

        self.health_status[name] = {
            'status': 'unknown',
            'last_check': None,
            'message': 'Not yet checked'
        }

        self.logger.info(f"Registered health check: {name}", extra={
            'interval_seconds': interval_seconds
        })

    def register_degradation_strategy(self, component: str, strategy_func: Callable):
        """Register degradation strategy for component."""
        self.degradation_strategies[component] = strategy_func
        self.logger.info(f"Registered degradation strategy for {component}")

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started health monitoring")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped health monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                current_time = time.time()

                # Run health checks
                for name, check_config in self.health_checks.items():
                    if current_time - check_config['last_check'] >= check_config['interval']:
                        await self._run_health_check(name, check_config)

                # Check for degradation needs
                await self._check_degradation_needs()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on errors

    async def _run_health_check(self, name: str, check_config: Dict[str, Any]):
        """Run individual health check."""
        try:
            check_func = check_config['function']

            # Run health check (support both sync and async)
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            # Update status
            self.health_status[name] = {
                'status': 'healthy' if result.get('healthy', True) else 'unhealthy',
                'last_check': datetime.now(timezone.utc),
                'message': result.get('message', 'OK'),
                'details': result.get('details', {})
            }

            check_config['last_check'] = time.time()
            check_config['consecutive_failures'] = 0

            self.logger.debug(f"Health check {name} completed", extra={
                'status': self.health_status[name]['status'],
                'message': self.health_status[name]['message']
            })

        except Exception as e:
            check_config['consecutive_failures'] += 1

            self.health_status[name] = {
                'status': 'unhealthy',
                'last_check': datetime.now(timezone.utc),
                'message': f'Health check failed: {str(e)}',
                'consecutive_failures': check_config['consecutive_failures']
            }

            self.logger.error(f"Health check {name} failed", extra={
                'error': str(e),
                'consecutive_failures': check_config['consecutive_failures']
            })

    async def _check_degradation_needs(self):
        """Check if any components need degradation."""
        for component, status in self.health_status.items():
            if status['status'] == 'unhealthy':
                consecutive_failures = status.get('consecutive_failures', 0)

                # Trigger degradation after 3 consecutive failures
                if consecutive_failures >= 3 and component in self.degradation_strategies:
                    try:
                        strategy = self.degradation_strategies[component]
                        if asyncio.iscoroutinefunction(strategy):
                            await strategy(status)
                        else:
                            strategy(status)

                        self.logger.warning(f"Applied degradation strategy for {component}")

                    except Exception as e:
                        self.logger.error(f"Degradation strategy failed for {component}: {e}")

    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_count = sum(1 for status in self.health_status.values() if status['status'] == 'healthy')
        total_count = len(self.health_status)

        overall_status = 'healthy' if healthy_count == total_count else 'degraded' if healthy_count > 0 else 'unhealthy'

        return {
            'status': overall_status,
            'healthy_components': healthy_count,
            'total_components': total_count,
            'components': self.health_status.copy(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class ResilienceOrchestrator:
    """Orchestrates all resilience components."""

    def __init__(self,
                 retry_config: Optional[RetryConfig] = None,
                 bulkhead_config: Optional[BulkheadConfig] = None):
        self.retry_config = retry_config or RetryConfig()
        self.bulkhead_config = bulkhead_config or BulkheadConfig()

        self.timeout_manager = TimeoutManager()
        self.retry_manager = RetryManager(self.timeout_manager)
        self.bulkhead_manager = BulkheadManager(self.bulkhead_config)
        self.health_monitor = HealthMonitor()

        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger(__name__ + ".ResilienceOrchestrator")

        # Register default health checks
        self._register_default_health_checks()

    def _register_default_health_checks(self):
        """Register default system health checks."""
        self.health_monitor.register_health_check(
            'memory_usage',
            self._check_memory_usage,
            interval_seconds=30.0
        )

        self.health_monitor.register_health_check(
            'disk_space',
            self._check_disk_space,
            interval_seconds=60.0
        )

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()

            # Consider unhealthy if memory usage > 90%
            healthy = memory.percent < 90

            return {
                'healthy': healthy,
                'message': f'Memory usage: {memory.percent:.1f}%',
                'details': {
                    'percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'total_gb': memory.total / (1024**3)
                }
            }
        except ImportError:
            return {'healthy': True, 'message': 'psutil not available'}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        try:
            import psutil
            disk = psutil.disk_usage('/')

            # Consider unhealthy if disk usage > 85%
            usage_percent = (disk.used / disk.total) * 100
            healthy = usage_percent < 85

            return {
                'healthy': healthy,
                'message': f'Disk usage: {usage_percent:.1f}%',
                'details': {
                    'percent': usage_percent,
                    'free_gb': disk.free / (1024**3),
                    'total_gb': disk.total / (1024**3)
                }
            }
        except ImportError:
            return {'healthy': True, 'message': 'psutil not available'}

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            from .circuit_breaker import CircuitBreakerConfig
            config = CircuitBreakerConfig(name=name)
            self.circuit_breakers[name] = CircuitBreaker(config)

        return self.circuit_breakers[name]

    async def start(self):
        """Start all resilience components."""
        await self.health_monitor.start_monitoring()
        self.logger.info("Resilience orchestrator started")

    async def stop(self):
        """Stop all resilience components."""
        await self.health_monitor.stop_monitoring()
        self.logger.info("Resilience orchestrator stopped")

    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        return {
            'health': self.health_monitor.get_overall_health(),
            'bulkheads': self.bulkhead_manager.get_bulkhead_stats(),
            'retries': self.retry_manager.get_retry_stats(),
            'circuit_breakers': {
                name: cb.get_stats() for name, cb in self.circuit_breakers.items()
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Global resilience orchestrator instance
_resilience_orchestrator: Optional[ResilienceOrchestrator] = None


def get_resilience_orchestrator() -> ResilienceOrchestrator:
    """Get global resilience orchestrator instance."""
    global _resilience_orchestrator
    if _resilience_orchestrator is None:
        _resilience_orchestrator = ResilienceOrchestrator()
    return _resilience_orchestrator


# Convenience decorators
def with_retry(config: Optional[RetryConfig] = None, operation_name: str = None):
    """Decorator for adding retry logic."""
    retry_config = config or RetryConfig()
    return get_resilience_orchestrator().retry_manager.retry(retry_config, operation_name)


def with_circuit_breaker(breaker_name: str):
    """Decorator for adding circuit breaker protection."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cb = get_resilience_orchestrator().get_circuit_breaker(breaker_name)
            return cb.call(func, *args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cb = get_resilience_orchestrator().get_circuit_breaker(breaker_name)
            return await cb.async_call(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


def with_bulkhead(bulkhead_name: str, priority: int = 1):
    """Decorator for adding bulkhead protection (async only)."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            orchestrator = get_resilience_orchestrator()
            async with orchestrator.bulkhead_manager.acquire_resource(bulkhead_name, priority):
                return await func(*args, **kwargs)

        if not asyncio.iscoroutinefunction(func):
            raise TypeError("Bulkhead decorator can only be used with async functions")

        return async_wrapper

    return decorator
