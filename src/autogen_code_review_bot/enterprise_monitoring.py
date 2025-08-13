"""
Enterprise Monitoring and Observability Framework

Comprehensive monitoring, logging, and observability system for autonomous SDLC
execution with real-time metrics, distributed tracing, and intelligent alerting.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .metrics import get_metrics_registry

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class MonitoringLevel(Enum):
    """Monitoring detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class Alert:
    """Alert definition"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    component: str
    metrics: Dict = None
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "metrics": self.metrics or {},
            "resolved": self.resolved
        }


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels or {}
        }


class EnterpriseMonitor:
    """Enterprise-grade monitoring and observability system"""
    
    def __init__(self, monitoring_level: MonitoringLevel = MonitoringLevel.COMPREHENSIVE):
        self.monitoring_level = monitoring_level
        self.registry = CollectorRegistry()
        self.active_alerts: Dict[str, Alert] = {}
        self.performance_metrics: List[PerformanceMetric] = []
        self.alert_handlers: List[callable] = []
        
        # Initialize Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Initialize distributed tracing
        self._setup_distributed_tracing()
        
        # Start monitoring loops
        self._monitoring_tasks = []
        
        logger.info("Enterprise monitor initialized", level=monitoring_level.value)
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collection"""
        
        # Core system metrics
        self.system_metrics = {
            "requests_total": Counter(
                "autogen_requests_total",
                "Total number of requests processed",
                ["component", "status"],
                registry=self.registry
            ),
            "request_duration": Histogram(
                "autogen_request_duration_seconds",
                "Request duration in seconds",
                ["component", "operation"],
                registry=self.registry
            ),
            "active_operations": Gauge(
                "autogen_active_operations",
                "Number of active operations",
                ["component"],
                registry=self.registry
            ),
            "error_rate": Gauge(
                "autogen_error_rate",
                "Error rate percentage",
                ["component"],
                registry=self.registry
            ),
            "memory_usage": Gauge(
                "autogen_memory_usage_bytes",
                "Memory usage in bytes",
                ["component"],
                registry=self.registry
            ),
            "cpu_usage": Gauge(
                "autogen_cpu_usage_percent",
                "CPU usage percentage",
                ["component"],
                registry=self.registry
            )
        }
        
        # SDLC-specific metrics
        self.sdlc_metrics = {
            "analysis_duration": Histogram(
                "autogen_analysis_duration_seconds",
                "Analysis operation duration",
                ["analysis_type"],
                registry=self.registry
            ),
            "generation_completions": Counter(
                "autogen_generation_completions_total",
                "Number of completed generations",
                ["generation_level"],
                registry=self.registry
            ),
            "quality_gate_results": Counter(
                "autogen_quality_gate_results_total",
                "Quality gate results",
                ["gate_name", "result"],
                registry=self.registry
            ),
            "validation_score": Gauge(
                "autogen_validation_score",
                "Current validation score",
                ["validation_type"],
                registry=self.registry
            )
        }
        
        logger.info("Prometheus metrics initialized")
    
    def _setup_distributed_tracing(self):
        """Setup distributed tracing with Jaeger"""
        try:
            # Configure tracer provider
            trace.set_tracer_provider(TracerProvider())
            
            # Setup Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(__name__)
            logger.info("Distributed tracing initialized")
            
        except Exception as e:
            logger.warning("Failed to initialize distributed tracing", error=str(e))
            self.tracer = None
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        self._monitoring_tasks = [
            asyncio.create_task(self._system_health_monitor()),
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        logger.info("Monitoring tasks started")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        for task in self._monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        logger.info("Monitoring tasks stopped")
    
    async def _system_health_monitor(self):
        """Monitor system health continuously"""
        while True:
            try:
                await self._check_system_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("System health monitor error", error=str(e))
                await asyncio.sleep(60)
    
    async def _performance_monitor(self):
        """Monitor performance metrics"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance monitor error", error=str(e))
                await asyncio.sleep(30)
    
    async def _alert_processor(self):
        """Process and handle alerts"""
        while True:
            try:
                await self._process_alerts()
                await asyncio.sleep(5)  # Process every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert processor error", error=str(e))
                await asyncio.sleep(15)
    
    async def _metrics_collector(self):
        """Collect and aggregate metrics"""
        while True:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(60)  # Aggregate every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collector error", error=str(e))
                await asyncio.sleep(120)
    
    async def _check_system_health(self):
        """Check overall system health"""
        health_checks = {
            "memory_usage": await self._check_memory_usage(),
            "cpu_usage": await self._check_cpu_usage(),
            "disk_usage": await self._check_disk_usage(),
            "network_connectivity": await self._check_network_connectivity(),
            "database_connectivity": await self._check_database_connectivity()
        }
        
        # Update health metrics
        for check_name, (healthy, value, threshold) in health_checks.items():
            if not healthy:
                await self._create_alert(
                    f"health_{check_name}",
                    f"Health check failed: {check_name}",
                    f"{check_name} check failed: {value} exceeds threshold {threshold}",
                    AlertSeverity.WARNING,
                    "system_health",
                    {"check": check_name, "value": value, "threshold": threshold}
                )
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics"""
        timestamp = datetime.utcnow()
        
        # Simulate metric collection
        metrics = [
            PerformanceMetric("response_time", 150.5, "ms", timestamp, {"endpoint": "analysis"}),
            PerformanceMetric("throughput", 1250.0, "req/s", timestamp, {"service": "autogen"}),
            PerformanceMetric("error_rate", 0.02, "percent", timestamp, {"component": "validator"}),
            PerformanceMetric("queue_depth", 5.0, "count", timestamp, {"queue": "processing"}),
        ]
        
        self.performance_metrics.extend(metrics)
        
        # Keep only recent metrics (last hour)
        cutoff_time = timestamp - timedelta(hours=1)
        self.performance_metrics = [
            m for m in self.performance_metrics 
            if m.timestamp > cutoff_time
        ]
        
        # Update Prometheus metrics
        for metric in metrics:
            self._update_prometheus_metric(metric)
    
    def _update_prometheus_metric(self, metric: PerformanceMetric):
        """Update Prometheus metric"""
        try:
            if metric.name == "response_time":
                self.system_metrics["request_duration"].labels(
                    component=metric.labels.get("endpoint", "unknown"),
                    operation="request"
                ).observe(metric.value / 1000)  # Convert ms to seconds
            
            elif metric.name == "error_rate":
                self.system_metrics["error_rate"].labels(
                    component=metric.labels.get("component", "unknown")
                ).set(metric.value)
            
            elif metric.name == "throughput":
                # Update request counter based on throughput
                pass  # Would update based on actual request counts
                
        except Exception as e:
            logger.warning("Failed to update Prometheus metric", 
                         metric_name=metric.name, error=str(e))
    
    async def _process_alerts(self):
        """Process active alerts"""
        current_time = datetime.utcnow()
        alerts_to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            # Auto-resolve old alerts (e.g., older than 1 hour)
            if (current_time - alert.timestamp).total_seconds() > 3600:
                alerts_to_resolve.append(alert_id)
            
            # Check if alert should be escalated
            elif (current_time - alert.timestamp).total_seconds() > 300:  # 5 minutes
                if alert.severity == AlertSeverity.WARNING:
                    alert.severity = AlertSeverity.ERROR
                    await self._notify_alert_handlers(alert)
        
        # Resolve old alerts
        for alert_id in alerts_to_resolve:
            await self.resolve_alert(alert_id)
    
    async def _aggregate_metrics(self):
        """Aggregate and analyze metrics"""
        if not self.performance_metrics:
            return
        
        # Calculate aggregates for different time windows
        current_time = datetime.utcnow()
        
        # Last 5 minutes
        recent_metrics = [
            m for m in self.performance_metrics
            if (current_time - m.timestamp).total_seconds() <= 300
        ]
        
        if recent_metrics:
            # Group by metric name
            metric_groups = {}
            for metric in recent_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            # Calculate statistics
            for metric_name, values in metric_groups.items():
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                
                # Check for anomalies
                if metric_name == "response_time" and avg_value > 500:  # 500ms threshold
                    await self._create_alert(
                        f"performance_{metric_name}",
                        "High response time detected",
                        f"Average response time {avg_value:.1f}ms exceeds threshold",
                        AlertSeverity.WARNING,
                        "performance",
                        {"metric": metric_name, "avg_value": avg_value, "threshold": 500}
                    )
                
                elif metric_name == "error_rate" and avg_value > 5:  # 5% threshold
                    await self._create_alert(
                        f"performance_{metric_name}",
                        "High error rate detected",
                        f"Error rate {avg_value:.2f}% exceeds threshold",
                        AlertSeverity.ERROR,
                        "performance",
                        {"metric": metric_name, "avg_value": avg_value, "threshold": 5}
                    )
    
    async def _check_memory_usage(self) -> tuple:
        """Check memory usage"""
        # Mock implementation - would use psutil in real implementation
        memory_usage = 75.2  # Percentage
        threshold = 90.0
        return memory_usage < threshold, memory_usage, threshold
    
    async def _check_cpu_usage(self) -> tuple:
        """Check CPU usage"""
        cpu_usage = 45.8  # Percentage
        threshold = 80.0
        return cpu_usage < threshold, cpu_usage, threshold
    
    async def _check_disk_usage(self) -> tuple:
        """Check disk usage"""
        disk_usage = 62.3  # Percentage
        threshold = 90.0
        return disk_usage < threshold, disk_usage, threshold
    
    async def _check_network_connectivity(self) -> tuple:
        """Check network connectivity"""
        # Mock implementation
        latency = 25.5  # ms
        threshold = 100.0
        return latency < threshold, latency, threshold
    
    async def _check_database_connectivity(self) -> tuple:
        """Check database connectivity"""
        # Mock implementation
        connection_time = 50.2  # ms
        threshold = 1000.0
        return connection_time < threshold, connection_time, threshold
    
    async def _create_alert(
        self,
        alert_id: str,
        title: str,
        message: str,
        severity: AlertSeverity,
        component: str,
        metrics: Optional[Dict] = None
    ):
        """Create new alert"""
        
        # Don't create duplicate alerts
        if alert_id in self.active_alerts:
            return
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.utcnow(),
            component=component,
            metrics=metrics
        )
        
        self.active_alerts[alert_id] = alert
        
        logger.warning("Alert created",
                      alert_id=alert_id,
                      title=title,
                      severity=severity.value,
                      component=component)
        
        # Notify alert handlers
        await self._notify_alert_handlers(alert)
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            
            logger.info("Alert resolved",
                       alert_id=alert_id,
                       title=alert.title,
                       component=alert.component)
            
            del self.active_alerts[alert_id]
    
    def register_alert_handler(self, handler: callable):
        """Register alert notification handler"""
        self.alert_handlers.append(handler)
        logger.info("Alert handler registered")
    
    async def _notify_alert_handlers(self, alert: Alert):
        """Notify all registered alert handlers"""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        current_time = datetime.utcnow()
        
        # Recent metrics (last 5 minutes)
        recent_metrics = [
            m for m in self.performance_metrics
            if (current_time - m.timestamp).total_seconds() <= 300
        ]
        
        summary = {
            "timestamp": current_time.isoformat(),
            "monitoring_level": self.monitoring_level.value,
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            "total_metrics": len(self.performance_metrics),
            "recent_metrics": len(recent_metrics),
            "system_health": "healthy"  # Would be calculated based on checks
        }
        
        # Add metric breakdowns
        if recent_metrics:
            metric_groups = {}
            for metric in recent_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            summary["metric_averages"] = {
                name: sum(values) / len(values)
                for name, values in metric_groups.items()
            }
        
        return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def create_trace_span(self, operation_name: str, **kwargs) -> Optional[trace.Span]:
        """Create distributed tracing span"""
        if self.tracer:
            span = self.tracer.start_span(operation_name)
            for key, value in kwargs.items():
                span.set_attribute(key, str(value))
            return span
        return None
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_performance_metrics(self, metric_name: Optional[str] = None, 
                               time_window: Optional[int] = None) -> List[Dict]:
        """Get performance metrics with optional filtering"""
        metrics = self.performance_metrics
        
        # Filter by metric name
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        # Filter by time window (minutes)
        if time_window:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window)
            metrics = [m for m in metrics if m.timestamp > cutoff_time]
        
        return [metric.to_dict() for metric in metrics]


# Global monitor instance
_global_monitor: Optional[EnterpriseMonitor] = None


def get_enterprise_monitor(level: MonitoringLevel = MonitoringLevel.COMPREHENSIVE) -> EnterpriseMonitor:
    """Get global enterprise monitor instance"""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = EnterpriseMonitor(level)
    
    return _global_monitor


async def example_alert_handler(alert: Alert):
    """Example alert handler implementation"""
    logger.info("Alert notification",
               alert_id=alert.id,
               title=alert.title,
               severity=alert.severity.value)
    
    # In real implementation, this would send notifications via:
    # - Email
    # - Slack
    # - PagerDuty
    # - SMS
    # etc.