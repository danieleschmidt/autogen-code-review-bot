"""Health monitoring and status checking for the AutoGen Code Review Bot."""

import asyncio
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil
import redis

from .logging_config import get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


class HealthChecker:
    """Comprehensive health monitoring system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize health checker with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.checks: List[Callable] = []
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.checks.extend(
            [
                self._check_memory_usage,
                self._check_disk_space,
                self._check_cpu_usage,
                self._check_required_tools,
                self._check_cache_connectivity,
                self._check_github_connectivity,
            ]
        )

    async def run_all_checks(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Run all registered health checks.

        Args:
            timeout: Maximum time to wait for all checks to complete

        Returns:
            Dictionary containing overall status and individual check results
        """
        start_time = time.time()
        results = []

        # Run checks concurrently with timeout
        try:
            tasks = [
                asyncio.wait_for(self._run_check_safely(check), timeout=5.0)
                for check in self.checks
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Health checks timed out", timeout=timeout)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    HealthCheckResult(
                        name=f"check_{i}",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(result)}",
                        duration_ms=0.0,
                        timestamp=datetime.now(timezone.utc),
                    )
                )
            elif isinstance(result, HealthCheckResult):
                processed_results.append(result)

        # Calculate overall status
        overall_status = self._calculate_overall_status(processed_results)
        total_duration = (time.time() - start_time) * 1000

        # Record metrics
        metrics.record_histogram("health_check_duration_ms", total_duration)
        metrics.record_counter(
            "health_checks_total", 1, tags={"status": overall_status.value}
        )

        return {
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": total_duration,
            "checks": [result.to_dict() for result in processed_results],
            "summary": {
                "total": len(processed_results),
                "healthy": sum(
                    1 for r in processed_results if r.status == HealthStatus.HEALTHY
                ),
                "degraded": sum(
                    1 for r in processed_results if r.status == HealthStatus.DEGRADED
                ),
                "unhealthy": sum(
                    1 for r in processed_results if r.status == HealthStatus.UNHEALTHY
                ),
                "critical": sum(
                    1 for r in processed_results if r.status == HealthStatus.CRITICAL
                ),
            },
        }

    async def _run_check_safely(self, check_func: Callable) -> HealthCheckResult:
        """Run a single health check with error handling.

        Args:
            check_func: Health check function to run

        Returns:
            HealthCheckResult object
        """
        start_time = time.time()
        try:
            # Handle both sync and async check functions
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            if not isinstance(result, HealthCheckResult):
                # Convert basic results to HealthCheckResult
                result = HealthCheckResult(
                    name=check_func.__name__,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(timezone.utc),
                )

            return result

        except Exception as e:
            logger.error(f"Health check {check_func.__name__} failed", error=str(e))
            return HealthCheckResult(
                name=check_func.__name__,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
            )

    def _calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Calculate overall health status from individual check results.

        Args:
            results: List of individual health check results

        Returns:
            Overall health status
        """
        if not results:
            return HealthStatus.UNHEALTHY

        # Count status occurrences
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        # Determine overall status based on worst case
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.UNHEALTHY, 0) > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage."""
        start_time = time.time()

        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Determine status based on memory usage
            if memory_percent < 70:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            elif memory_percent < 85:
                status = HealthStatus.DEGRADED
                message = f"Memory usage elevated: {memory_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_percent:.1f}%"

            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "memory_percent": memory_percent,
                    "memory_available": memory.available,
                    "memory_total": memory.total,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory usage: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        start_time = time.time()

        try:
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            if disk_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            elif disk_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Disk usage elevated: {disk_percent:.1f}%"
            else:
                status = HealthStatus.CRITICAL
                message = f"Disk usage critical: {disk_percent:.1f}%"

            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "disk_percent": disk_percent,
                    "disk_free": disk.free,
                    "disk_total": disk.total,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk space: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
            )

    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        start_time = time.time()

        try:
            # Sample CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1.0)

            if cpu_percent < 70:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            elif cpu_percent < 85:
                status = HealthStatus.DEGRADED
                message = f"CPU usage elevated: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage high: {cpu_percent:.1f}%"

            return HealthCheckResult(
                name="cpu_usage",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
                metadata={"cpu_percent": cpu_percent, "cpu_count": psutil.cpu_count()},
            )

        except Exception as e:
            return HealthCheckResult(
                name="cpu_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check CPU usage: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
            )

    def _check_required_tools(self) -> HealthCheckResult:
        """Check availability of required external tools."""
        start_time = time.time()

        required_tools = ["git", "python3", "ruff", "bandit"]
        missing_tools = []

        for tool in required_tools:
            try:
                subprocess.run(
                    [tool, "--version"], capture_output=True, check=True, timeout=5
                )
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                FileNotFoundError,
            ):
                missing_tools.append(tool)

        if not missing_tools:
            status = HealthStatus.HEALTHY
            message = "All required tools available"
        elif len(missing_tools) < len(required_tools) / 2:
            status = HealthStatus.DEGRADED
            message = f"Some tools missing: {', '.join(missing_tools)}"
        else:
            status = HealthStatus.CRITICAL
            message = f"Critical tools missing: {', '.join(missing_tools)}"

        return HealthCheckResult(
            name="required_tools",
            status=status,
            message=message,
            duration_ms=(time.time() - start_time) * 1000,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "required_tools": required_tools,
                "missing_tools": missing_tools,
                "available_tools": [
                    t for t in required_tools if t not in missing_tools
                ],
            },
        )

    def _check_cache_connectivity(self) -> HealthCheckResult:
        """Check Redis cache connectivity."""
        start_time = time.time()

        try:
            cache_config = self.config.get("cache", {})
            if not cache_config.get("enabled", True):
                return HealthCheckResult(
                    name="cache_connectivity",
                    status=HealthStatus.HEALTHY,
                    message="Cache disabled - skipping connectivity check",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(timezone.utc),
                )

            redis_url = cache_config.get("redis_url", "redis://localhost:6379")
            r = redis.from_url(redis_url, socket_timeout=5)

            # Test connection with ping
            r.ping()

            # Test basic operations
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            r.delete(test_key)

            if value == b"test_value":
                status = HealthStatus.HEALTHY
                message = "Cache connectivity successful"
            else:
                status = HealthStatus.DEGRADED
                message = "Cache operations partially working"

            return HealthCheckResult(
                name="cache_connectivity",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
                metadata={"redis_url": redis_url.split("@")[-1]},  # Hide credentials
            )

        except Exception as e:
            return HealthCheckResult(
                name="cache_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache connection failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
            )

    async def _check_github_connectivity(self) -> HealthCheckResult:
        """Check GitHub API connectivity."""
        start_time = time.time()

        try:
            import aiohttp

            github_config = self.config.get("github", {})
            api_url = github_config.get("api_url", "https://api.github.com")
            timeout = github_config.get("request_timeout", 10)

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(f"{api_url}/rate_limit") as response:
                    if response.status == 200:
                        data = await response.json()
                        rate_limit = data.get("rate", {})
                        remaining = rate_limit.get("remaining", 0)

                        if remaining > 100:
                            status = HealthStatus.HEALTHY
                            message = (
                                f"GitHub API accessible, {remaining} requests remaining"
                            )
                        elif remaining > 10:
                            status = HealthStatus.DEGRADED
                            message = f"GitHub API rate limit low: {remaining} requests remaining"
                        else:
                            status = HealthStatus.UNHEALTHY
                            message = f"GitHub API rate limit critical: {remaining} requests remaining"

                        return HealthCheckResult(
                            name="github_connectivity",
                            status=status,
                            message=message,
                            duration_ms=(time.time() - start_time) * 1000,
                            timestamp=datetime.now(timezone.utc),
                            metadata={
                                "rate_limit_remaining": remaining,
                                "rate_limit_limit": rate_limit.get("limit", 0),
                                "rate_limit_reset": rate_limit.get("reset", 0),
                            },
                        )
                    else:
                        return HealthCheckResult(
                            name="github_connectivity",
                            status=HealthStatus.UNHEALTHY,
                            message=f"GitHub API returned status {response.status}",
                            duration_ms=(time.time() - start_time) * 1000,
                            timestamp=datetime.now(timezone.utc),
                        )

        except Exception as e:
            return HealthCheckResult(
                name="github_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"GitHub API check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
            )


def create_health_checker(config: Optional[Dict[str, Any]] = None) -> HealthChecker:
    """Factory function to create a health checker instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        HealthChecker instance
    """
    return HealthChecker(config)
