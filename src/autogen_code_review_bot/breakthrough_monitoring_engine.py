"""
Breakthrough Monitoring Engine

Advanced monitoring system with real-time metrics collection, 
intelligent alerting, and breakthrough performance optimization.
"""

import asyncio
import json
import time
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import logging
import statistics
from collections import defaultdict, deque
import threading
import psutil
import socket
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    BREAKTHROUGH_INDICATOR = "breakthrough_indicator"


class AlertSeverity(Enum):
    """Alert severity levels"""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BREAKTHROUGH = "breakthrough"


class MonitoringStatus(Enum):
    """Monitoring system status"""
    
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected"""
    
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    collection_interval: float = 1.0  # seconds
    retention_period: int = 86400  # 24 hours in seconds
    aggregation_functions: List[str] = field(default_factory=lambda: ["avg", "min", "max"])
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    breakthrough_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """Single metric data point"""
    
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class Alert:
    """System alert with context"""
    
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }


@dataclass
class PerformanceProfile:
    """System performance profile"""
    
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    custom_metrics: Dict[str, float]
    timestamp: datetime
    breakthrough_indicators: Dict[str, float] = field(default_factory=dict)
    
    def calculate_overall_health(self) -> float:
        """Calculate overall system health score"""
        health_factors = {
            "cpu_health": max(0, 1.0 - (self.cpu_usage / 100.0)),
            "memory_health": max(0, 1.0 - (self.memory_usage / 100.0)),
            "io_health": 0.8,  # Simplified for demo
            "network_health": 0.9,  # Simplified for demo
        }
        
        return statistics.mean(health_factors.values())


class BreakthroughMonitoringEngine:
    """Advanced monitoring engine with breakthrough detection"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.monitoring_data_path = self.repo_path / ".monitoring_data"
        self.monitoring_data_path.mkdir(exist_ok=True)
        
        # Core monitoring state
        self.status = MonitoringStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Alert system
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Performance monitoring
        self.performance_history: deque = deque(maxlen=1000)
        self.breakthrough_detectors: Dict[str, Callable[[List[float]], bool]] = {}
        
        # Async monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Thread safety
        self.metrics_lock = threading.Lock()
        
        # Initialize built-in metrics
        self._setup_builtin_metrics()
        
        logger.info(f"Breakthrough monitoring engine initialized for {self.repo_path}")
    
    def _setup_builtin_metrics(self):
        """Setup essential system metrics"""
        
        builtin_metrics = [
            MetricDefinition(
                name="system.cpu_usage",
                metric_type=MetricType.GAUGE,
                description="CPU usage percentage",
                unit="percent",
                collection_interval=1.0,
                alert_thresholds={"warning": 80.0, "critical": 95.0},
                breakthrough_thresholds={"efficiency_breakthrough": 30.0}  # Low CPU usage breakthrough
            ),
            MetricDefinition(
                name="system.memory_usage",
                metric_type=MetricType.GAUGE,
                description="Memory usage percentage",
                unit="percent",
                collection_interval=1.0,
                alert_thresholds={"warning": 85.0, "critical": 95.0},
                breakthrough_thresholds={"memory_optimization": 50.0}
            ),
            MetricDefinition(
                name="application.response_time",
                metric_type=MetricType.TIMER,
                description="Application response time",
                unit="milliseconds",
                collection_interval=0.1,
                alert_thresholds={"warning": 500.0, "critical": 1000.0},
                breakthrough_thresholds={"speed_breakthrough": 100.0}  # Sub-100ms response
            ),
            MetricDefinition(
                name="application.throughput",
                metric_type=MetricType.RATE,
                description="Requests per second",
                unit="rps",
                collection_interval=1.0,
                breakthrough_thresholds={"throughput_breakthrough": 1000.0}
            ),
            MetricDefinition(
                name="quality.test_coverage",
                metric_type=MetricType.GAUGE,
                description="Test coverage percentage",
                unit="percent",
                collection_interval=60.0,
                breakthrough_thresholds={"coverage_excellence": 95.0}
            ),
            MetricDefinition(
                name="quality.security_score",
                metric_type=MetricType.GAUGE,
                description="Security posture score",
                unit="score",
                collection_interval=300.0,
                breakthrough_thresholds={"security_excellence": 0.95}
            ),
            MetricDefinition(
                name="development.velocity",
                metric_type=MetricType.RATE,
                description="Development velocity",
                unit="story_points_per_day",
                collection_interval=3600.0,
                breakthrough_thresholds={"velocity_breakthrough": 20.0}
            )
        ]
        
        for metric_def in builtin_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_definition: MetricDefinition):
        """Register a new metric for collection"""
        self.metric_definitions[metric_definition.name] = metric_definition
        logger.debug(f"Registered metric: {metric_definition.name}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value"""
        
        if name not in self.metric_definitions:
            logger.warning(f"Recording metric '{name}' without definition")
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.metrics_lock:
            self.metrics[name].append(point)
        
        # Check for alerts
        self._check_metric_alerts(name, value, point.timestamp)
        
        # Check for breakthrough detection
        self._check_breakthrough_detection(name, value)
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("Starting breakthrough monitoring engine")
        self.status = MonitoringStatus.RUNNING
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._system_metrics_collector()),
            asyncio.create_task(self._performance_profiler()),
            asyncio.create_task(self._breakthrough_detector()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._metrics_aggregator()),
            asyncio.create_task(self._health_checker())
        ]
        
        self.monitoring_tasks.extend(monitoring_tasks)
        
        logger.info(f"Started {len(monitoring_tasks)} monitoring tasks")
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        # Cleanup
        await self._shutdown_monitoring_tasks()
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        logger.info("Stopping breakthrough monitoring engine")
        self.status = MonitoringStatus.SHUTDOWN
        self.shutdown_event.set()
    
    async def _system_metrics_collector(self):
        """Collect system performance metrics"""
        
        while self.status == MonitoringStatus.RUNNING:
            try:
                # CPU usage
                cpu_usage = psutil.cpu_percent(interval=None)
                self.record_metric("system.cpu_usage", cpu_usage)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("system.memory_usage", memory.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.record_metric("system.disk_read_bytes", disk_io.read_bytes)
                    self.record_metric("system.disk_write_bytes", disk_io.write_bytes)
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    self.record_metric("system.network_bytes_sent", network_io.bytes_sent)
                    self.record_metric("system.network_bytes_recv", network_io.bytes_recv)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(5.0)
    
    async def _performance_profiler(self):
        """Profile application performance"""
        
        while self.status == MonitoringStatus.RUNNING:
            try:
                # Create performance profile
                profile = PerformanceProfile(
                    cpu_usage=psutil.cpu_percent(),
                    memory_usage=psutil.virtual_memory().percent,
                    disk_io={"read_rate": 0.0, "write_rate": 0.0},  # Simplified
                    network_io={"send_rate": 0.0, "recv_rate": 0.0},  # Simplified
                    custom_metrics={},
                    timestamp=datetime.utcnow()
                )
                
                # Calculate breakthrough indicators
                profile.breakthrough_indicators = self._calculate_breakthrough_indicators(profile)
                
                self.performance_history.append(profile)
                
                # Record breakthrough metrics
                for indicator, value in profile.breakthrough_indicators.items():
                    self.record_metric(f"breakthrough.{indicator}", value)
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Performance profiling error: {e}")
                await asyncio.sleep(10.0)
    
    async def _breakthrough_detector(self):
        """Detect breakthrough performance patterns"""
        
        while self.status == MonitoringStatus.RUNNING:
            try:
                # Analyze recent metrics for breakthrough patterns
                for metric_name in self.metric_definitions:
                    if metric_name in self.metrics:
                        recent_values = [
                            point.value for point in list(self.metrics[metric_name])[-100:]  # Last 100 points
                        ]
                        
                        if len(recent_values) >= 10:  # Minimum data points
                            breakthrough_detected = self._detect_breakthrough_pattern(metric_name, recent_values)
                            if breakthrough_detected:
                                await self._handle_breakthrough_detection(metric_name, recent_values)
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Breakthrough detection error: {e}")
                await asyncio.sleep(60.0)
    
    async def _alert_processor(self):
        """Process and manage alerts"""
        
        while self.status == MonitoringStatus.RUNNING:
            try:
                # Check for alert resolutions
                resolved_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    if not alert.resolved:
                        # Check if alert condition still exists
                        if self._should_resolve_alert(alert):
                            alert.resolved = True
                            alert.resolution_timestamp = datetime.utcnow()
                            resolved_alerts.append(alert_id)
                            
                            logger.info(f"Alert resolved: {alert.title}")
                            
                            # Notify callbacks
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert)
                                except Exception as e:
                                    logger.error(f"Alert callback error: {e}")
                
                # Move resolved alerts to history
                for alert_id in resolved_alerts:
                    alert = self.active_alerts.pop(alert_id)
                    self.alert_history.append(alert)
                
                # Cleanup old alert history
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.alert_history = [
                    alert for alert in self.alert_history 
                    if alert.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(10.0)
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30.0)
    
    async def _metrics_aggregator(self):
        """Aggregate and persist metrics"""
        
        while self.status == MonitoringStatus.RUNNING:
            try:
                # Aggregate metrics every 60 seconds
                aggregated_metrics = {}
                
                with self.metrics_lock:
                    for metric_name, points in self.metrics.items():
                        if not points:
                            continue
                        
                        # Get last minute of data
                        cutoff_time = datetime.utcnow() - timedelta(minutes=1)
                        recent_points = [
                            point for point in points 
                            if point.timestamp > cutoff_time
                        ]
                        
                        if recent_points:
                            values = [point.value for point in recent_points]
                            
                            aggregated_metrics[metric_name] = {
                                "avg": statistics.mean(values),
                                "min": min(values),
                                "max": max(values),
                                "count": len(values),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                
                # Save aggregated metrics
                if aggregated_metrics:
                    await self._persist_aggregated_metrics(aggregated_metrics)
                
                await asyncio.sleep(60.0)
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(120.0)
    
    async def _health_checker(self):
        """Monitor system health"""
        
        while self.status == MonitoringStatus.RUNNING:
            try:
                # Calculate overall health score
                if self.performance_history:
                    latest_profile = self.performance_history[-1]
                    health_score = latest_profile.calculate_overall_health()
                    
                    self.record_metric("system.health_score", health_score)
                    
                    # Generate health alert if needed
                    if health_score < 0.5:  # Critical health threshold
                        await self._generate_health_alert(health_score, latest_profile)
                
                await asyncio.sleep(30.0)
                
            except Exception as e:
                logger.error(f"Health checking error: {e}")
                await asyncio.sleep(60.0)
    
    def _check_metric_alerts(self, metric_name: str, value: float, timestamp: datetime):
        """Check if metric value triggers alerts"""
        
        if metric_name not in self.metric_definitions:
            return
        
        metric_def = self.metric_definitions[metric_name]
        
        for threshold_type, threshold_value in metric_def.alert_thresholds.items():
            alert_id = f"{metric_name}_{threshold_type}"
            
            # Check if alert should be triggered
            should_alert = False
            if threshold_type in ["warning", "error", "critical"]:
                should_alert = value >= threshold_value
            
            if should_alert and alert_id not in self.active_alerts:
                # Create new alert
                severity = AlertSeverity.WARNING
                if threshold_type == "critical":
                    severity = AlertSeverity.CRITICAL
                elif threshold_type == "error":
                    severity = AlertSeverity.ERROR
                
                alert = Alert(
                    alert_id=alert_id,
                    title=f"{metric_name} {threshold_type} threshold exceeded",
                    description=f"Metric {metric_name} value {value} exceeded {threshold_type} threshold {threshold_value}",
                    severity=severity,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=threshold_value,
                    timestamp=timestamp
                )
                
                self.active_alerts[alert_id] = alert
                
                logger.warning(f"Alert triggered: {alert.title}")
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
    
    def _check_breakthrough_detection(self, metric_name: str, value: float):
        """Check for breakthrough detection in metric"""
        
        if metric_name not in self.metric_definitions:
            return
        
        metric_def = self.metric_definitions[metric_name]
        
        for breakthrough_type, breakthrough_threshold in metric_def.breakthrough_thresholds.items():
            # Simple breakthrough detection (could be more sophisticated)
            breakthrough_detected = False
            
            if "efficiency" in breakthrough_type or "optimization" in breakthrough_type:
                # For efficiency metrics, lower values are breakthrough
                breakthrough_detected = value <= breakthrough_threshold
            else:
                # For performance metrics, higher values are breakthrough
                breakthrough_detected = value >= breakthrough_threshold
            
            if breakthrough_detected:
                self._handle_immediate_breakthrough(metric_name, breakthrough_type, value, breakthrough_threshold)
    
    def _handle_immediate_breakthrough(self, metric_name: str, breakthrough_type: str, value: float, threshold: float):
        """Handle immediate breakthrough detection"""
        
        alert_id = f"breakthrough_{metric_name}_{breakthrough_type}"
        
        if alert_id not in self.active_alerts:
            alert = Alert(
                alert_id=alert_id,
                title=f"Breakthrough detected: {breakthrough_type}",
                description=f"Metric {metric_name} achieved breakthrough performance: {value} (threshold: {threshold})",
                severity=AlertSeverity.BREAKTHROUGH,
                metric_name=metric_name,
                current_value=value,
                threshold_value=threshold,
                timestamp=datetime.utcnow()
            )
            
            self.active_alerts[alert_id] = alert
            
            logger.info(f"🚀 Breakthrough detected: {alert.title}")
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Breakthrough callback error: {e}")
    
    def _detect_breakthrough_pattern(self, metric_name: str, values: List[float]) -> bool:
        """Detect breakthrough patterns in metric history"""
        
        if len(values) < 20:  # Need sufficient data
            return False
        
        # Analyze trend
        recent_values = values[-10:]  # Last 10 values
        historical_values = values[-30:-10]  # Previous 20 values
        
        if not historical_values:
            return False
        
        recent_avg = statistics.mean(recent_values)
        historical_avg = statistics.mean(historical_values)
        
        # Check for significant improvement
        if historical_avg > 0:
            improvement_ratio = (recent_avg - historical_avg) / historical_avg
            
            # Different breakthrough criteria for different metrics
            if "response_time" in metric_name or "latency" in metric_name:
                # For latency metrics, lower is better
                return improvement_ratio < -0.2  # 20% improvement (reduction)
            elif "throughput" in metric_name or "velocity" in metric_name:
                # For throughput metrics, higher is better
                return improvement_ratio > 0.3  # 30% improvement
            elif "cpu_usage" in metric_name or "memory_usage" in metric_name:
                # For resource usage, lower is better
                return improvement_ratio < -0.15  # 15% improvement (reduction)
            else:
                # General improvement threshold
                return improvement_ratio > 0.25  # 25% improvement
        
        return False
    
    async def _handle_breakthrough_detection(self, metric_name: str, values: List[float]):
        """Handle breakthrough pattern detection"""
        
        recent_avg = statistics.mean(values[-10:])
        historical_avg = statistics.mean(values[-30:-10])
        improvement = ((recent_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
        
        alert_id = f"pattern_breakthrough_{metric_name}"
        
        if alert_id not in self.active_alerts:
            alert = Alert(
                alert_id=alert_id,
                title=f"Performance breakthrough pattern detected",
                description=f"Metric {metric_name} shows sustained breakthrough improvement: {improvement:.1f}%",
                severity=AlertSeverity.BREAKTHROUGH,
                metric_name=metric_name,
                current_value=recent_avg,
                threshold_value=historical_avg,
                timestamp=datetime.utcnow(),
                metadata={"improvement_percentage": improvement}
            )
            
            self.active_alerts[alert_id] = alert
            
            logger.info(f"🎯 Breakthrough pattern detected: {alert.title}")
            
            # Record breakthrough metric
            self.record_metric(f"breakthrough.pattern.{metric_name}", abs(improvement))
    
    def _calculate_breakthrough_indicators(self, profile: PerformanceProfile) -> Dict[str, float]:
        """Calculate breakthrough performance indicators"""
        
        indicators = {}
        
        # Efficiency breakthrough (low resource usage with high performance)
        efficiency_score = (100 - profile.cpu_usage) / 100.0 * (100 - profile.memory_usage) / 100.0
        indicators["efficiency_breakthrough"] = efficiency_score
        
        # Stability breakthrough (consistent performance)
        if len(self.performance_history) > 10:
            recent_cpu = [p.cpu_usage for p in list(self.performance_history)[-10:]]
            cpu_stability = max(0, 1.0 - (statistics.stdev(recent_cpu) / 100.0)) if len(recent_cpu) > 1 else 1.0
            indicators["stability_breakthrough"] = cpu_stability
        
        # Overall health breakthrough
        health_score = profile.calculate_overall_health()
        if health_score > 0.9:
            indicators["health_breakthrough"] = health_score
        
        # Resource optimization breakthrough
        if profile.cpu_usage < 30 and profile.memory_usage < 50:
            indicators["resource_optimization"] = 1.0 - ((profile.cpu_usage + profile.memory_usage) / 200.0)
        
        return indicators
    
    def _should_resolve_alert(self, alert: Alert) -> bool:
        """Check if alert should be resolved"""
        
        if alert.metric_name not in self.metrics:
            return True  # No recent data, resolve alert
        
        # Get recent metric values
        recent_points = list(self.metrics[alert.metric_name])[-5:]  # Last 5 points
        if not recent_points:
            return True
        
        recent_values = [point.value for point in recent_points]
        recent_avg = statistics.mean(recent_values)
        
        # Check if alert condition is resolved
        if alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            # For regular alerts, check if value is below threshold
            return recent_avg < alert.threshold_value * 0.95  # 5% buffer
        elif alert.severity == AlertSeverity.BREAKTHROUGH:
            # For breakthrough alerts, they auto-resolve after some time
            time_since_alert = datetime.utcnow() - alert.timestamp
            return time_since_alert > timedelta(minutes=30)  # Auto-resolve after 30 minutes
        
        return False
    
    async def _generate_health_alert(self, health_score: float, profile: PerformanceProfile):
        """Generate system health alert"""
        
        alert_id = "system_health_critical"
        
        if alert_id not in self.active_alerts:
            alert = Alert(
                alert_id=alert_id,
                title="Critical system health alert",
                description=f"System health score critically low: {health_score:.3f}",
                severity=AlertSeverity.CRITICAL,
                metric_name="system.health_score",
                current_value=health_score,
                threshold_value=0.5,
                timestamp=datetime.utcnow(),
                metadata={
                    "cpu_usage": profile.cpu_usage,
                    "memory_usage": profile.memory_usage
                }
            )
            
            self.active_alerts[alert_id] = alert
            logger.critical(f"System health critical: {alert.description}")
    
    async def _persist_aggregated_metrics(self, metrics: Dict[str, Any]):
        """Persist aggregated metrics to storage"""
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H")
        metrics_file = self.monitoring_data_path / f"metrics_{timestamp}.json"
        
        # Load existing data if file exists
        existing_data = {}
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing metrics file: {e}")
        
        # Update with new metrics
        existing_data.update(metrics)
        
        # Save updated data
        try:
            with open(metrics_file, "w") as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    async def _shutdown_monitoring_tasks(self):
        """Shutdown all monitoring tasks"""
        
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.monitoring_tasks.clear()
        logger.info("All monitoring tasks shutdown")
    
    # Public API methods
    
    def get_metric_history(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MetricPoint]:
        """Get metric history for time range"""
        
        if metric_name not in self.metrics:
            return []
        
        points = list(self.metrics[metric_name])
        
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        return points
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_breakthrough_metrics(self) -> Dict[str, Any]:
        """Get current breakthrough metrics"""
        
        breakthrough_metrics = {}
        
        # Get breakthrough indicators from recent performance profiles
        if self.performance_history:
            latest_profile = self.performance_history[-1]
            breakthrough_metrics.update(latest_profile.breakthrough_indicators)
        
        # Get breakthrough alerts
        breakthrough_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.BREAKTHROUGH
        ]
        
        breakthrough_metrics["active_breakthroughs"] = len(breakthrough_alerts)
        breakthrough_metrics["breakthrough_alerts"] = [alert.to_dict() for alert in breakthrough_alerts]
        
        return breakthrough_metrics
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "monitoring_status": self.status.value,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "active_alerts": len(self.active_alerts),
            "registered_metrics": len(self.metric_definitions),
            "collected_metrics": len(self.metrics),
            "performance_profiles": len(self.performance_history),
            "breakthrough_detections": len([
                alert for alert in self.active_alerts.values() 
                if alert.severity == AlertSeverity.BREAKTHROUGH
            ])
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)  # Last 24 hours
        
        report = {
            "report_generated": end_time.isoformat(),
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "system_status": self.get_system_status(),
            "metric_summary": {},
            "alert_summary": {
                "active_alerts": len(self.active_alerts),
                "resolved_alerts": len([a for a in self.alert_history if a.resolution_timestamp and a.resolution_timestamp > start_time])
            },
            "breakthrough_summary": self.get_breakthrough_metrics(),
            "performance_analysis": {}
        }
        
        # Generate metric summaries
        for metric_name in self.metric_definitions:
            history = self.get_metric_history(metric_name, start_time, end_time)
            
            if history:
                values = [point.value for point in history]
                
                report["metric_summary"][metric_name] = {
                    "data_points": len(values),
                    "avg_value": statistics.mean(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        # Performance analysis
        if self.performance_history:
            recent_profiles = [
                p for p in self.performance_history 
                if p.timestamp > start_time
            ]
            
            if recent_profiles:
                cpu_values = [p.cpu_usage for p in recent_profiles]
                memory_values = [p.memory_usage for p in recent_profiles]
                health_values = [p.calculate_overall_health() for p in recent_profiles]
                
                report["performance_analysis"] = {
                    "avg_cpu_usage": statistics.mean(cpu_values),
                    "avg_memory_usage": statistics.mean(memory_values),
                    "avg_health_score": statistics.mean(health_values),
                    "performance_trend": "improving" if health_values[-1] > health_values[0] else "declining" if len(health_values) > 1 else "stable"
                }
        
        return report


# Global monitoring engine instance
_global_monitoring_engine: Optional[BreakthroughMonitoringEngine] = None


def get_monitoring_engine(repo_path: str = ".") -> BreakthroughMonitoringEngine:
    """Get global monitoring engine instance"""
    global _global_monitoring_engine
    
    if _global_monitoring_engine is None:
        _global_monitoring_engine = BreakthroughMonitoringEngine(repo_path)
    
    return _global_monitoring_engine


async def demo_breakthrough_monitoring():
    """Demonstrate breakthrough monitoring capabilities"""
    
    engine = get_monitoring_engine()
    
    print("📊 Breakthrough Monitoring Engine Demo")
    print("=" * 50)
    
    # Setup alert callback
    def alert_handler(alert: Alert):
        emoji = "🚨" if alert.severity == AlertSeverity.CRITICAL else "⚠️" if alert.severity == AlertSeverity.WARNING else "🚀"
        print(f"{emoji} Alert: {alert.title} - {alert.description}")
    
    engine.add_alert_callback(alert_handler)
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(engine.start_monitoring())
    
    print("✅ Monitoring system started")
    
    # Simulate some metrics
    print("\n📈 Simulating metrics collection...")
    
    for i in range(20):
        # Simulate improving performance
        response_time = max(50, 200 - i * 5)  # Improving response time
        throughput = min(1500, 500 + i * 25)  # Increasing throughput
        
        engine.record_metric("application.response_time", response_time)
        engine.record_metric("application.throughput", throughput)
        engine.record_metric("quality.test_coverage", min(98, 70 + i * 1.5))
        
        # Simulate breakthrough moment
        if i == 15:
            engine.record_metric("application.response_time", 45)  # Breakthrough response time
            engine.record_metric("application.throughput", 1200)  # Breakthrough throughput
        
        await asyncio.sleep(0.1)
    
    print("✅ Metrics simulation completed")
    
    # Wait a bit for processing
    await asyncio.sleep(2.0)
    
    # Check system status
    status = engine.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Monitoring Status: {status['monitoring_status']}")
    print(f"   Active Alerts: {status['active_alerts']}")
    print(f"   Registered Metrics: {status['registered_metrics']}")
    print(f"   Breakthrough Detections: {status['breakthrough_detections']}")
    
    # Check breakthrough metrics
    breakthrough_metrics = engine.get_breakthrough_metrics()
    print(f"\n🚀 Breakthrough Metrics:")
    for metric, value in breakthrough_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    # Generate report
    report = engine.generate_monitoring_report()
    print(f"\n📋 Monitoring Report Generated:")
    print(f"   Metric Summaries: {len(report['metric_summary'])}")
    print(f"   Alert Summary: {report['alert_summary']}")
    print(f"   Breakthrough Summary: Available")
    
    # Stop monitoring
    await engine.stop_monitoring()
    await monitoring_task
    
    print("\n🎯 Breakthrough Monitoring Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_breakthrough_monitoring())