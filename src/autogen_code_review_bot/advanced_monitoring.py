"""
Advanced monitoring and observability for AutoGen Code Review Bot.

Implements comprehensive monitoring with metrics, tracing, alerting,
and health checks for enterprise production environments.
"""

import json
import psutil
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .global_config import get_config
from .logging_config import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class HealthStatus:
    """Health check status information."""
    
    service: str
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: datetime
    details: Dict[str, Any]
    response_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """Comprehensive metrics collection and management."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self._start_time = time.time()
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_total = Counter(
            'autogen_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'autogen_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Analysis metrics
        self.analysis_total = Counter(
            'autogen_analysis_total',
            'Total number of analyses',
            ['type', 'status'],
            registry=self.registry
        )
        
        self.analysis_duration = Histogram(
            'autogen_analysis_duration_seconds',
            'Analysis duration in seconds',
            ['type'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'autogen_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'autogen_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'autogen_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Application metrics
        self.active_sessions = Gauge(
            'autogen_active_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'autogen_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'autogen_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'autogen_errors_total',
            'Total number of errors',
            ['type', 'severity'],
            registry=self.registry
        )
        
        # Business metrics
        self.reviews_completed = Counter(
            'autogen_reviews_completed_total',
            'Total completed code reviews',
            ['result'],
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            'autogen_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record HTTP request metrics."""
        self.request_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_analysis(self, analysis_type: str, status: str, duration: float):
        """Record analysis metrics."""
        self.analysis_total.labels(type=analysis_type, status=status).inc()
        self.analysis_duration.labels(type=analysis_type).observe(duration)
    
    def record_error(self, error_type: str, severity: str):
        """Record error metrics."""
        self.errors_total.labels(type=error_type, severity=severity).inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_review_completion(self, result: str):
        """Record completed code review."""
        self.reviews_completed.labels(result=result).inc()
    
    def update_system_metrics(self):
        """Update system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)
            
            # Uptime
            uptime = time.time() - self._start_time
            self.uptime_seconds.set(uptime)
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class HealthChecker:
    """Health check management for services and dependencies."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system", self._check_system_health)
        self.register_check("memory", self._check_memory_health)
        self.register_check("disk", self._check_disk_health)
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info(f"Health check registered: {name}")
    
    def run_check(self, name: str) -> HealthStatus:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthStatus(
                service=name,
                status="unhealthy",
                timestamp=datetime.now(timezone.utc),
                details={"error": f"Unknown health check: {name}"}
            )
        
        start_time = time.time()
        try:
            with tracer.start_as_current_span(f"health_check_{name}") as span:
                result = self.checks[name]()
                response_time = (time.time() - start_time) * 1000
                
                status = HealthStatus(
                    service=name,
                    status="healthy",
                    timestamp=datetime.now(timezone.utc),
                    details=result,
                    response_time_ms=response_time
                )
                
                span.set_status(Status(StatusCode.OK))
                return status
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus(
                service=name,
                status="unhealthy",
                timestamp=datetime.now(timezone.utc),
                details={"error": str(e)},
                response_time_ms=response_time
            )
            
            logger.error(f"Health check failed for {name}: {e}")
            return status
    
    def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        all_checks = self.run_all_checks()
        
        # Determine overall status
        unhealthy_checks = [name for name, status in all_checks.items() 
                          if status.status == "unhealthy"]
        degraded_checks = [name for name, status in all_checks.items() 
                         if status.status == "degraded"]
        
        if unhealthy_checks:
            overall_status = "unhealthy"
        elif degraded_checks:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return HealthStatus(
            service="overall",
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            details={
                "total_checks": len(all_checks),
                "healthy_checks": len([s for s in all_checks.values() if s.status == "healthy"]),
                "degraded_checks": len(degraded_checks),
                "unhealthy_checks": len(unhealthy_checks),
                "unhealthy_services": unhealthy_checks,
                "degraded_services": degraded_checks
            }
        )
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            load_avg = psutil.getloadavg()
            cpu_count = psutil.cpu_count()
            
            return {
                "load_average": load_avg,
                "cpu_count": cpu_count,
                "load_per_cpu": load_avg[0] / cpu_count if cpu_count > 0 else 0
            }
        except Exception as e:
            raise Exception(f"System health check failed: {e}")
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            if memory.percent > 90:
                raise Exception(f"High memory usage: {memory.percent}%")
            
            return {
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "swap_percent": swap.percent
            }
        except Exception as e:
            raise Exception(f"Memory health check failed: {e}")
    
    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent > 85:
                raise Exception(f"High disk usage: {disk_percent:.1f}%")
            
            return {
                "disk_percent": disk_percent,
                "disk_free_gb": disk.free / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }
        except Exception as e:
            raise Exception(f"Disk health check failed: {e}")


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self):
        self.config = get_config()
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_counts = defaultdict(int)
    
    def send_alert(self, alert_type: str, message: str, severity: str = "warning", 
                   metadata: Dict[str, Any] = None):
        """Send an alert notification."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        self.alert_history.append(alert)
        self.alert_counts[alert_type] += 1
        
        logger.warning(f"Alert {severity}: {alert_type} - {message}")
        
        # Send to external systems (webhook, email, Slack, etc.)
        self._send_to_external_systems(alert)
    
    def _send_to_external_systems(self, alert: Dict[str, Any]):
        """Send alert to external notification systems."""
        # This would integrate with actual notification systems
        # For now, we'll just log the alert
        
        if alert["severity"] == "critical":
            # In production, this would send immediate notifications
            logger.critical(f"CRITICAL ALERT: {alert}")
        
        # Example webhook integration
        webhook_url = self.config.features.get("alert_webhook_url")
        if webhook_url:
            try:
                response = requests.post(
                    webhook_url,
                    json=alert,
                    timeout=5,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Failed to send alert to webhook: {e}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for the specified time period."""
        cutoff_time = datetime.now(timezone.utc).replace(
            hour=datetime.now(timezone.utc).hour - hours
        )
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in recent_alerts:
            severity_counts[alert["severity"]] += 1
            type_counts[alert["type"]] += 1
        
        return {
            "total_alerts": len(recent_alerts),
            "by_severity": dict(severity_counts),
            "by_type": dict(type_counts),
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }


class PerformanceMonitor:
    """Performance monitoring and profiling."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.slow_operations_threshold = 5.0  # seconds
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation: str, duration: float):
        """Record operation execution time."""
        self.operation_times[operation].append(duration)
        
        # Keep only last 1000 measurements per operation
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]
        
        # Alert on slow operations
        if duration > self.slow_operations_threshold:
            alert_manager.send_alert(
                "slow_operation",
                f"Slow operation detected: {operation} took {duration:.2f}s",
                severity="warning",
                metadata={"operation": operation, "duration": duration}
            )
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "avg_duration": sum(times) / len(times),
                    "min_duration": min(times),
                    "max_duration": max(times),
                    "total_duration": sum(times)
                }
        
        return stats


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_operation_time(self.operation_name, duration)


# Global monitoring instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
alert_manager = AlertManager()
performance_monitor = PerformanceMonitor()


def monitor_function(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            with performance_monitor.time_operation(name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def health_check_endpoint():
    """HTTP endpoint for health checks."""
    overall_health = health_checker.get_overall_health()
    return {
        "status": overall_health.status,
        "timestamp": overall_health.timestamp.isoformat(),
        "details": overall_health.details
    }


def metrics_endpoint():
    """HTTP endpoint for Prometheus metrics."""
    metrics_collector.update_system_metrics()
    return metrics_collector.get_metrics()


def alerts_endpoint():
    """HTTP endpoint for alert summary."""
    return alert_manager.get_alert_summary()


def performance_endpoint():
    """HTTP endpoint for performance statistics."""
    return performance_monitor.get_performance_stats()