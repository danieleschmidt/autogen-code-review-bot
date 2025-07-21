"""Monitoring infrastructure for code review bot operations."""

from __future__ import annotations

import time
import threading
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import json

from .logging_config import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check operation."""
    
    name: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary format."""
        result = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp
        }
        
        if self.response_time_ms is not None:
            result["response_time_ms"] = self.response_time_ms
        
        if self.details:
            result["details"] = self.details
        
        return result


class HealthChecker:
    """Manages and executes health checks."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._lock = threading.RLock()
        
        # Register default system health checks
        self._register_default_checks()
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        with self._lock:
            self.checks[name] = check_func
            logger.debug(f"Registered health check: {name}")
    
    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self.checks:
                del self.checks[name]
                logger.debug(f"Unregistered health check: {name}")
                return True
            return False
    
    def run_all_checks(self, timeout_seconds: float = 30.0) -> Dict[str, HealthCheck]:
        """Run all registered health checks with optional timeout."""
        with self._lock:
            check_functions = self.checks.copy()
        
        results = {}
        
        # Use thread pool for concurrent execution with timeout
        with ThreadPoolExecutor(max_workers=min(len(check_functions), 5)) as executor:
            # Submit all health checks
            future_to_name = {}
            for name, check_func in check_functions.items():
                future = executor.submit(self._run_single_check, name, check_func)
                future_to_name[future] = name
            
            # Collect results with timeout
            for future in as_completed(future_to_name, timeout=timeout_seconds):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=1.0)  # Individual check timeout
                    results[name] = result
                except TimeoutError:
                    logger.warning(f"Health check timeout: {name}")
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message="Health check timeout",
                        response_time_ms=timeout_seconds * 1000
                    )
                except Exception as e:
                    logger.error(f"Health check error: {name}: {e}")
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(e)}"
                    )
        
        logger.info(f"Completed {len(results)} health checks", 
                   extra={"total_checks": len(check_functions), "completed": len(results)})
        
        return results
    
    def _run_single_check(self, name: str, check_func: Callable[[], HealthCheck]) -> HealthCheck:
        """Run a single health check with timing."""
        start_time = time.time()
        
        try:
            result = check_func()
            
            # Add response time if not already set
            if result.response_time_ms is None:
                result.response_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"Health check completed: {name} -> {result.status}")
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check failed: {name}: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                response_time_ms=response_time
            )
    
    def get_overall_status(self, check_results: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system health from individual check results."""
        if not check_results:
            return HealthStatus.UNKNOWN
        
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in check_results.values():
            status_counts[result.status] += 1
        
        # Determine overall status based on worst case
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)
    
    def _check_system_resources(self) -> HealthCheck:
        """Check overall system resource usage."""
        try:
            # Try to get basic system info using available methods
            cpu_percent, memory_percent = self._get_system_stats()
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            elif cpu_percent > 75 or memory_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check system resources: {e}"
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            disk_usage = self._get_disk_usage()
            
            if disk_usage > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {disk_usage:.1f}%"
            elif disk_usage > 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {disk_usage:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal disk usage: {disk_usage:.1f}%"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                details={"disk_usage_percent": disk_usage}
            )
            
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check disk space: {e}"
            )
    
    def _check_memory(self) -> HealthCheck:
        """Check detailed memory information."""
        try:
            memory_info = self._get_memory_info()
            memory_percent = memory_info.get("used_percent", 50.0)
            available_gb = memory_info.get("available_gb", 4.0)
            total_gb = memory_info.get("total_gb", 8.0)
            
            if memory_percent > 95:
                status = HealthStatus.UNHEALTHY
            elif memory_percent > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheck(
                name="memory",
                status=status,
                message=f"Memory usage: {memory_percent:.1f}% ({available_gb:.1f}GB available)",
                details={
                    "total_gb": total_gb,
                    "available_gb": available_gb,
                    "used_percent": memory_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Unable to check memory: {e}"
            )
    
    def _get_system_stats(self) -> tuple[float, float]:
        """Get CPU and memory percentages using fallback methods."""
        try:
            # Try to use psutil if available
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return cpu_percent, memory_percent
        except ImportError:
            # Fallback to basic approximation
            # Use load average as CPU approximation
            try:
                load_avg = os.getloadavg()[0]  # 1-minute load average
                cpu_count = os.cpu_count() or 1
                cpu_percent = min((load_avg / cpu_count) * 100, 100.0)
            except (AttributeError, OSError):
                cpu_percent = 25.0  # Default assumption
            
            # Estimate memory usage (fallback)
            memory_percent = 50.0  # Default assumption
            
            return cpu_percent, memory_percent
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage using fallback methods."""
        try:
            # Try to use psutil if available
            import psutil
            return psutil.disk_usage('/').percent
        except ImportError:
            # Fallback using shutil
            try:
                total, used, free = shutil.disk_usage('/')
                return (used / total) * 100
            except Exception:
                return 50.0  # Default assumption
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information using fallback methods."""
        try:
            # Try to use psutil if available
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / 1024**3,
                "available_gb": memory.available / 1024**3,
                "used_percent": memory.percent
            }
        except ImportError:
            # Fallback estimates
            return {
                "total_gb": 8.0,
                "available_gb": 4.0,
                "used_percent": 50.0
            }


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""
    
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    tags: Optional[Dict[str, str]] = None
    metric_type: str = "gauge"  # gauge, counter, histogram


class MetricsEmitter:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricValue]] = {}
        self._lock = threading.RLock()
        self.max_values_per_metric = 1000  # Limit memory usage
    
    def record_counter(self, name: str, value: Union[int, float] = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (cumulative value)."""
        metric_value = MetricValue(
            value=value,
            tags=tags,
            metric_type="counter"
        )
        
        self._store_metric(name, metric_value)
        logger.debug(f"Recorded counter metric: {name}={value}")
    
    def record_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (point-in-time value)."""
        metric_value = MetricValue(
            value=value,
            tags=tags,
            metric_type="gauge"
        )
        
        self._store_metric(name, metric_value)
        logger.debug(f"Recorded gauge metric: {name}={value}")
    
    def record_histogram(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric (distribution of values)."""
        metric_value = MetricValue(
            value=value,
            tags=tags,
            metric_type="histogram"
        )
        
        self._store_metric(name, metric_value)
        logger.debug(f"Recorded histogram metric: {name}={value}")
    
    def _store_metric(self, name: str, metric_value: MetricValue) -> None:
        """Store a metric value with memory management."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(metric_value)
            
            # Limit memory usage by keeping only recent values
            if len(self.metrics[name]) > self.max_values_per_metric:
                self.metrics[name] = self.metrics[name][-self.max_values_per_metric:]
    
    def get_metrics(self, since_timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Get current metrics, optionally filtered by timestamp."""
        with self._lock:
            result = {}
            
            for name, values in self.metrics.items():
                if not values:
                    continue
                
                # Filter by timestamp if specified
                filtered_values = values
                if since_timestamp:
                    filtered_values = [v for v in values if v.timestamp >= since_timestamp]
                
                if not filtered_values:
                    continue
                
                # Get latest value and aggregate statistics
                latest = filtered_values[-1]
                
                result[name] = {
                    "value": latest.value,
                    "timestamp": latest.timestamp,
                    "type": latest.metric_type,
                    "count": len(filtered_values)
                }
                
                if latest.tags:
                    result[name]["tags"] = latest.tags
                
                # Add aggregated statistics for multi-value metrics
                if len(filtered_values) > 1:
                    values_only = [v.value for v in filtered_values]
                    result[name]["stats"] = {
                        "min": min(values_only),
                        "max": max(values_only),
                        "avg": sum(values_only) / len(values_only),
                        "sum": sum(values_only) if latest.metric_type == "counter" else None
                    }
            
            return result
    
    def clear_metrics(self, older_than_hours: Optional[float] = None) -> int:
        """Clear metrics, optionally keeping recent ones."""
        cleared_count = 0
        cutoff_time = time.time() - (older_than_hours * 3600) if older_than_hours else 0
        
        with self._lock:
            if older_than_hours is None:
                # Clear all metrics
                cleared_count = sum(len(values) for values in self.metrics.values())
                self.metrics.clear()
            else:
                # Clear old metrics
                for name in list(self.metrics.keys()):
                    original_count = len(self.metrics[name])
                    self.metrics[name] = [
                        v for v in self.metrics[name] 
                        if v.timestamp >= cutoff_time
                    ]
                    cleared_count += original_count - len(self.metrics[name])
                    
                    # Remove empty metric entries
                    if not self.metrics[name]:
                        del self.metrics[name]
        
        logger.info(f"Cleared {cleared_count} metric values", extra={"cutoff_hours": older_than_hours})
        return cleared_count


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""
    
    name: str
    target: float  # Target percentage (e.g., 99.9)
    measurement_window_hours: float  # Time window for measurement
    description: str = ""
    
    def __post_init__(self):
        """Validate SLO definition."""
        if not (0 <= self.target <= 100):
            raise ValueError("SLO target must be between 0 and 100")
        if self.measurement_window_hours <= 0:
            raise ValueError("Measurement window must be positive")


@dataclass
class SLIMeasurement:
    """A single SLI measurement."""
    
    success: bool
    timestamp: float = field(default_factory=time.time)
    value: Optional[float] = None  # For latency-based SLIs
    metadata: Optional[Dict[str, Any]] = None


class SLITracker:
    """Tracks Service Level Indicators and calculates SLO compliance."""
    
    def __init__(self):
        self.slos: Dict[str, SLODefinition] = {}
        self.measurements: Dict[str, List[SLIMeasurement]] = {}
        self._lock = threading.RLock()
    
    def register_slo(self, slo: SLODefinition) -> None:
        """Register a Service Level Objective."""
        with self._lock:
            self.slos[slo.name] = slo
            if slo.name not in self.measurements:
                self.measurements[slo.name] = []
        
        logger.info(f"Registered SLO: {slo.name} (target: {slo.target}%)")
    
    def record_measurement(
        self, 
        slo_name: str, 
        success: bool, 
        value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an SLI measurement."""
        if slo_name not in self.slos:
            logger.warning(f"Recording measurement for unregistered SLO: {slo_name}")
            return
        
        measurement = SLIMeasurement(
            success=success,
            value=value,
            metadata=metadata
        )
        
        with self._lock:
            if slo_name not in self.measurements:
                self.measurements[slo_name] = []
            
            self.measurements[slo_name].append(measurement)
            
            # Keep only measurements within the window
            self._cleanup_old_measurements(slo_name)
        
        logger.debug(f"Recorded SLI measurement: {slo_name} success={success}")
    
    def calculate_sli(self, slo_name: str) -> Optional[float]:
        """Calculate current SLI value as percentage."""
        if slo_name not in self.slos:
            return None
        
        with self._lock:
            measurements = self.measurements.get(slo_name, [])
            if not measurements:
                return None
            
            # Calculate within the measurement window
            window_hours = self.slos[slo_name].measurement_window_hours
            cutoff_time = time.time() - (window_hours * 3600)
            
            recent_measurements = [
                m for m in measurements if m.timestamp >= cutoff_time
            ]
            
            if not recent_measurements:
                return None
            
            successful = sum(1 for m in recent_measurements if m.success)
            total = len(recent_measurements)
            
            return (successful / total) * 100 if total > 0 else None
    
    def is_slo_met(self, slo_name: str) -> Optional[bool]:
        """Check if SLO is currently being met."""
        current_sli = self.calculate_sli(slo_name)
        if current_sli is None:
            return None
        
        target = self.slos[slo_name].target
        return current_sli >= target
    
    def get_slo_status(self, slo_name: str) -> Dict[str, Any]:
        """Get detailed SLO status information."""
        if slo_name not in self.slos:
            return {"error": "SLO not found"}
        
        slo = self.slos[slo_name]
        current_sli = self.calculate_sli(slo_name)
        is_met = self.is_slo_met(slo_name)
        
        with self._lock:
            measurement_count = len(self.measurements.get(slo_name, []))
        
        return {
            "name": slo_name,
            "target": slo.target,
            "current_sli": current_sli,
            "is_met": is_met,
            "measurement_count": measurement_count,
            "window_hours": slo.measurement_window_hours,
            "description": slo.description
        }
    
    def _cleanup_old_measurements(self, slo_name: str) -> None:
        """Remove measurements outside the measurement window."""
        if slo_name not in self.slos:
            return
        
        window_hours = self.slos[slo_name].measurement_window_hours
        cutoff_time = time.time() - (window_hours * 3600)
        
        measurements = self.measurements.get(slo_name, [])
        self.measurements[slo_name] = [
            m for m in measurements if m.timestamp >= cutoff_time
        ]


class MonitoringServer:
    """Central monitoring server coordinating health checks, metrics, and SLIs."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.health_checker = HealthChecker()
        self.metrics_emitter = MetricsEmitter()
        self.sli_tracker = SLITracker()
        
        # Register default SLOs
        self._register_default_slos()
        
        logger.info(f"Monitoring server initialized on port {port}")
    
    def _register_default_slos(self) -> None:
        """Register default SLOs for the system."""
        default_slos = [
            SLODefinition(
                name="system_availability",
                target=99.5,
                measurement_window_hours=24,
                description="System should be available 99.5% of the time"
            ),
            SLODefinition(
                name="api_response_time",
                target=95.0,
                measurement_window_hours=1,
                description="95% of API requests should complete within acceptable time"
            )
        ]
        
        for slo in default_slos:
            self.sli_tracker.register_slo(slo)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        # Run health checks
        health_results = self.health_checker.run_all_checks()
        overall_health = self.health_checker.get_overall_status(health_results)
        
        # Get metrics
        metrics = self.metrics_emitter.get_metrics()
        
        # Get SLO status
        slo_status = {}
        for slo_name in self.sli_tracker.slos:
            slo_status[slo_name] = self.sli_tracker.get_slo_status(slo_name)
        
        return {
            "timestamp": time.time(),
            "overall_health": overall_health,
            "health_checks": {name: check.to_dict() for name, check in health_results.items()},
            "metrics": metrics,
            "slos": slo_status
        }


def create_health_endpoint(health_checker: HealthChecker) -> Callable[[], Dict[str, Any]]:
    """Create a health endpoint function for web frameworks."""
    def health_endpoint():
        """Health endpoint that returns system health status."""
        try:
            results = health_checker.run_all_checks()
            overall_status = health_checker.get_overall_status(results)
            
            return {
                "status": overall_status,
                "timestamp": time.time(),
                "checks": {name: check.to_dict() for name, check in results.items()}
            }
        except Exception as e:
            logger.error(f"Health endpoint error: {e}")
            return {
                "status": HealthStatus.UNKNOWN,
                "timestamp": time.time(),
                "error": str(e)
            }
    
    return health_endpoint


def get_system_health() -> Dict[str, Any]:
    """Get basic system health information."""
    try:
        # Use the same fallback methods as HealthChecker
        checker = HealthChecker()
        cpu_percent, memory_percent = checker._get_system_stats()
        disk_usage = checker._get_disk_usage()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_usage": disk_usage,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "error": str(e),
            "timestamp": time.time()
        }