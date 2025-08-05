"""Comprehensive metrics collection and monitoring system.

This module provides a complete metrics infrastructure for monitoring application
performance, including counters, gauges, histograms, and exporters for various
formats (JSON, Prometheus).

Key features:
- Thread-safe metrics collection
- Multiple metric types (Counter, Gauge, Histogram)
- Label support for dimensional metrics
- Export to multiple formats
- Integration with existing logging system
- Performance optimized for high-throughput scenarios
"""

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
from collections import defaultdict
from functools import wraps

from .logging_utils import RequestContext, get_request_logger


logger = get_request_logger(__name__)


class MetricType:
    """Enum for metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricSample:
    """A single metric sample with labels."""
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseMetric:
    """Base class for all metrics."""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Initialize base metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            labels: List of label names for this metric.
        """
        self.name = name
        self.description = description
        self.labels = labels or []
        self._lock = threading.RLock()
        
    def _validate_labels(self, labels: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Validate and normalize labels.
        
        Args:
            labels: Label dictionary to validate.
            
        Returns:
            Validated label dictionary.
        """
        if labels is None:
            return {}
            
        if not isinstance(labels, dict):
            raise ValueError("Labels must be a dictionary")
            
        # Ensure all label values are strings
        return {str(k): str(v) for k, v in labels.items()}


class Counter(BaseMetric):
    """Counter metric that only increases."""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Initialize counter metric."""
        super().__init__(name, description, labels)
        self._values: Dict[tuple, int] = defaultdict(int)
        
    @property
    def value(self) -> int:
        """Get total value across all label combinations."""
        with self._lock:
            return sum(self._values.values())
            
    def increment(self, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter.
        
        Args:
            value: Value to increment by (must be positive).
            labels: Labels for this increment.
        """
        if value < 0:
            raise ValueError("Counter can only be incremented with positive values")
            
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._values[label_key] += value
            
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get value for specific label combination.
        
        Args:
            labels: Labels to get value for.
            
        Returns:
            Value for the label combination.
        """
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            return self._values[label_key]
            
    def get_all_samples(self) -> List[MetricSample]:
        """Get all samples with their labels.
        
        Returns:
            List of metric samples.
        """
        with self._lock:
            samples = []
            for label_key, value in self._values.items():
                labels = dict(label_key)
                samples.append(MetricSample(value=value, labels=labels))
            return samples


class Gauge(BaseMetric):
    """Gauge metric that can increase or decrease."""
    
    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Initialize gauge metric."""
        super().__init__(name, description, labels)
        self._values: Dict[tuple, float] = defaultdict(float)
        
    @property
    def value(self) -> float:
        """Get current value (sum across all label combinations)."""
        with self._lock:
            return sum(self._values.values())
            
    def set(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value.
        
        Args:
            value: Value to set.
            labels: Labels for this value.
        """
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._values[label_key] = float(value)
            
    def increment(self, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge value.
        
        Args:
            value: Value to increment by.
            labels: Labels for this increment.
        """
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            self._values[label_key] += value
            
    def decrement(self, value: Union[int, float] = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge value.
        
        Args:
            value: Value to decrement by.
            labels: Labels for this decrement.
        """
        self.increment(-value, labels)
        
    def get_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get value for specific label combination.
        
        Args:
            labels: Labels to get value for.
            
        Returns:
            Value for the label combination.
        """
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            return self._values[label_key]
            
    def get_all_samples(self) -> List[MetricSample]:
        """Get all samples with their labels.
        
        Returns:
            List of metric samples.
        """
        with self._lock:
            samples = []
            for label_key, value in self._values.items():
                labels = dict(label_key)
                samples.append(MetricSample(value=value, labels=labels))
            return samples


class Histogram(BaseMetric):
    """Histogram metric for tracking distributions."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
    
    def __init__(self, name: str, description: str, buckets: Optional[List[float]] = None, 
                 labels: Optional[List[str]] = None):
        """Initialize histogram metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            buckets: Histogram buckets (sorted).
            labels: List of label names.
        """
        super().__init__(name, description, labels)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[tuple, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._observation_counts: Dict[tuple, int] = defaultdict(int)
        
    @property
    def count(self) -> int:
        """Get total number of observations."""
        with self._lock:
            return sum(self._observation_counts.values())
            
    @property
    def sum(self) -> float:
        """Get sum of all observed values."""
        with self._lock:
            return sum(self._sums.values())
            
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value.
        
        Args:
            value: Value to observe.
            labels: Labels for this observation.
        """
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            # Update sum and count
            self._sums[label_key] += value
            self._observation_counts[label_key] += 1
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
                    
    def get_bucket_counts(self, labels: Optional[Dict[str, str]] = None) -> Dict[float, int]:
        """Get bucket counts for specific labels.
        
        Args:
            labels: Labels to get counts for.
            
        Returns:
            Dictionary mapping bucket upper bounds to counts.
        """
        labels = self._validate_labels(labels)
        label_key = tuple(sorted(labels.items()))
        
        with self._lock:
            return dict(self._counts[label_key])
            
    def get_all_samples(self) -> List[MetricSample]:
        """Get all samples including bucket counts.
        
        Returns:
            List of metric samples.
        """
        with self._lock:
            samples = []
            
            for label_key in self._observation_counts.keys():
                labels = dict(label_key)
                
                # Add count and sum samples
                samples.append(MetricSample(
                    value=self._observation_counts[label_key],
                    labels={**labels, "type": "count"}
                ))
                samples.append(MetricSample(
                    value=self._sums[label_key],
                    labels={**labels, "type": "sum"}
                ))
                
                # Add bucket samples
                for bucket, count in self._counts[label_key].items():
                    bucket_labels = {**labels, "le": str(bucket)}
                    samples.append(MetricSample(
                        value=count,
                        labels=bucket_labels
                    ))
                    
            return samples


class Timer:
    """Utility for timing operations and recording to histogram."""
    
    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        """Initialize timer.
        
        Args:
            histogram: Histogram to record timing to.
            labels: Labels for the timing.
        """
        self.histogram = histogram
        self.labels = labels
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.histogram.observe(duration, self.labels)
            
    @classmethod
    def decorator(cls, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        """Create a decorator for timing functions.
        
        Args:
            histogram: Histogram to record timing to.
            labels: Labels for the timing.
            
        Returns:
            Decorator function.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with cls(histogram, labels):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class MetricsRegistry:
    """Central registry for all metrics."""
    
    def __init__(self):
        """Initialize metrics registry."""
        self._metrics: Dict[str, BaseMetric] = {}
        self._lock = threading.RLock()
        
    def counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Counter:
        """Get or create a counter metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            labels: List of label names.
            
        Returns:
            Counter metric.
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric
                
            counter = Counter(name, description, labels)
            self._metrics[name] = counter
            return counter
            
    def gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Gauge:
        """Get or create a gauge metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            labels: List of label names.
            
        Returns:
            Gauge metric.
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric
                
            gauge = Gauge(name, description, labels)
            self._metrics[name] = gauge
            return gauge
            
    def histogram(self, name: str, description: str, buckets: Optional[List[float]] = None,
                  labels: Optional[List[str]] = None) -> Histogram:
        """Get or create a histogram metric.
        
        Args:
            name: Metric name.
            description: Metric description.
            buckets: Histogram buckets.
            labels: List of label names.
            
        Returns:
            Histogram metric.
        """
        with self._lock:
            if name in self._metrics:
                metric = self._metrics[name]
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {name} already exists as {type(metric).__name__}")
                return metric
                
            histogram = Histogram(name, description, buckets, labels)
            self._metrics[name] = histogram
            return histogram
            
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get metric by name.
        
        Args:
            name: Metric name.
            
        Returns:
            Metric or None if not found.
        """
        with self._lock:
            return self._metrics.get(name)
            
    def get_all_metrics(self) -> Dict[str, BaseMetric]:
        """Get all registered metrics.
        
        Returns:
            Dictionary of all metrics.
        """
        with self._lock:
            return self._metrics.copy()


class MetricsExporter:
    """Base class for metrics exporters."""
    
    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics from registry.
        
        Args:
            registry: Metrics registry to export.
            
        Returns:
            Exported metrics as string.
        """
        raise NotImplementedError


class JSONExporter(MetricsExporter):
    """JSON metrics exporter."""
    
    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics in JSON format.
        
        Args:
            registry: Metrics registry to export.
            
        Returns:
            JSON string of metrics.
        """
        metrics_data = {}
        
        for name, metric in registry.get_all_metrics().items():
            metric_info = {
                "name": metric.name,
                "description": metric.description,
                "type": type(metric).__name__.lower(),
                "samples": []
            }
            
            if isinstance(metric, (Counter, Gauge)):
                metric_info["value"] = metric.value
                for sample in metric.get_all_samples():
                    metric_info["samples"].append({
                        "value": sample.value,
                        "labels": sample.labels,
                        "timestamp": sample.timestamp
                    })
            elif isinstance(metric, Histogram):
                metric_info["count"] = metric.count
                metric_info["sum"] = metric.sum
                metric_info["buckets"] = metric.buckets
                for sample in metric.get_all_samples():
                    metric_info["samples"].append({
                        "value": sample.value,
                        "labels": sample.labels,
                        "timestamp": sample.timestamp
                    })
                    
            metrics_data[name] = metric_info
            
        return json.dumps(metrics_data, indent=2)


class PrometheusExporter(MetricsExporter):
    """Prometheus metrics exporter."""
    
    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics in Prometheus format.
        
        Args:
            registry: Metrics registry to export.
            
        Returns:
            Prometheus format string.
        """
        lines = []
        
        for name, metric in registry.get_all_metrics().items():
            # Add help and type comments
            lines.append(f"# HELP {name} {metric.description}")
            
            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
                for sample in metric.get_all_samples():
                    labels_str = self._format_labels(sample.labels)
                    lines.append(f"{name}{labels_str} {sample.value}")
                    
            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
                for sample in metric.get_all_samples():
                    labels_str = self._format_labels(sample.labels)
                    lines.append(f"{name}{labels_str} {sample.value}")
                    
            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")
                for sample in metric.get_all_samples():
                    if "type" in sample.labels:
                        if sample.labels["type"] == "count":
                            labels = {k: v for k, v in sample.labels.items() if k != "type"}
                            labels_str = self._format_labels(labels)
                            lines.append(f"{name}_count{labels_str} {sample.value}")
                        elif sample.labels["type"] == "sum":
                            labels = {k: v for k, v in sample.labels.items() if k != "type"}
                            labels_str = self._format_labels(labels)
                            lines.append(f"{name}_sum{labels_str} {sample.value}")
                    elif "le" in sample.labels:
                        labels_str = self._format_labels(sample.labels)
                        lines.append(f"{name}_bucket{labels_str} {sample.value}")
                        
        return "\n".join(lines) + "\n"
        
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus format.
        
        Args:
            labels: Labels dictionary.
            
        Returns:
            Formatted labels string.
        """
        if not labels:
            return ""
            
        formatted_labels = []
        for key, value in sorted(labels.items()):
            # Escape quotes in values
            escaped_value = str(value).replace('"', '\\"')
            formatted_labels.append(f'{key}="{escaped_value}"')
            
        return "{" + ",".join(formatted_labels) + "}"


# Global metrics registry
_global_registry: Optional[MetricsRegistry] = None
_registry_lock = threading.RLock()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry.
    
    Returns:
        Global metrics registry instance.
    """
    global _global_registry
    
    with _registry_lock:
        if _global_registry is None:
            _global_registry = MetricsRegistry()
        return _global_registry


def record_operation_metrics(operation: str, duration_ms: float, status: str = "success",
                           context: Optional[RequestContext] = None) -> None:
    """Record metrics for an operation.
    
    Args:
        operation: Operation name.
        duration_ms: Duration in milliseconds.
        status: Operation status (success, error, timeout, etc.).
        context: Request context for correlation.
    """
    registry = get_metrics_registry()
    
    # Record operation duration
    duration_histogram = registry.histogram(
        "operation_duration_seconds",
        "Duration of operations in seconds",
        labels=["operation", "status"]
    )
    duration_histogram.observe(duration_ms / 1000.0, {
        "operation": operation,
        "status": status
    })
    
    # Record operation count
    operation_counter = registry.counter(
        "operation_total",
        "Total number of operations",
        labels=["operation", "status"]
    )
    operation_counter.increment(labels={
        "operation": operation,
        "status": status
    })
    
    # Log metrics with context
    if context:
        logger.info(
            f"Operation metrics recorded: {operation}",
            context=context,
            operation=operation,
            duration_ms=duration_ms,
            status=status
        )


def with_metrics(operation: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to automatically record metrics for a function.
    
    Args:
        operation: Operation name for metrics.
        labels: Additional labels for metrics.
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                # Merge labels
                final_labels = labels.copy() if labels else {}
                final_labels.update({"operation": operation, "status": status})
                
                record_operation_metrics(operation, duration_ms, status)
                
        return wrapper
    return decorator


# Convenience functions for common metrics
def get_request_duration_histogram() -> Histogram:
    """Get or create request duration histogram."""
    registry = get_metrics_registry()
    return registry.histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        labels=["method", "endpoint", "status"]
    )


def get_error_counter() -> Counter:
    """Get or create error counter."""
    registry = get_metrics_registry()
    return registry.counter(
        "errors_total",
        "Total number of errors",
        labels=["error_type", "module"]
    )


def get_active_connections_gauge() -> Gauge:
    """Get or create active connections gauge."""
    registry = get_metrics_registry()
    return registry.gauge(
        "active_connections",
        "Number of active connections",
        labels=["connection_type"]
    )