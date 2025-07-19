"""Tests for comprehensive metrics collection system."""

import time
import threading
from unittest.mock import patch, MagicMock
from io import StringIO
import json

import pytest

from autogen_code_review_bot.metrics import (
    MetricsRegistry,
    GlobalMetrics,
    Counter,
    Gauge,
    Histogram,
    Timer,
    MetricsExporter,
    PrometheusExporter,
    JSONExporter,
    get_metrics_registry,
    record_operation_metrics,
    with_metrics,
)
from autogen_code_review_bot.logging_utils import RequestContext


class TestMetricsRegistry:
    """Test centralized metrics registry."""
    
    def test_metrics_registry_creation(self):
        """Test creating a metrics registry."""
        registry = MetricsRegistry()
        
        assert registry is not None
        assert len(registry.get_all_metrics()) == 0
        
    def test_register_counter(self):
        """Test registering a counter metric."""
        registry = MetricsRegistry()
        
        counter = registry.counter("requests_total", "Total number of requests")
        
        assert counter is not None
        assert counter.name == "requests_total"
        assert counter.description == "Total number of requests"
        assert counter.value == 0
        
    def test_register_gauge(self):
        """Test registering a gauge metric."""
        registry = MetricsRegistry()
        
        gauge = registry.gauge("active_connections", "Number of active connections")
        
        assert gauge is not None
        assert gauge.name == "active_connections"
        assert gauge.description == "Number of active connections"
        assert gauge.value == 0
        
    def test_register_histogram(self):
        """Test registering a histogram metric."""
        registry = MetricsRegistry()
        
        histogram = registry.histogram("response_time", "Response time distribution")
        
        assert histogram is not None
        assert histogram.name == "response_time"
        assert histogram.description == "Response time distribution"
        assert len(histogram.buckets) > 0
        
    def test_get_existing_metric(self):
        """Test getting an existing metric returns same instance."""
        registry = MetricsRegistry()
        
        counter1 = registry.counter("test_metric", "Test metric")
        counter2 = registry.counter("test_metric", "Test metric")
        
        assert counter1 is counter2
        
    def test_list_all_metrics(self):
        """Test listing all registered metrics."""
        registry = MetricsRegistry()
        
        registry.counter("counter1", "Counter 1")
        registry.gauge("gauge1", "Gauge 1")
        registry.histogram("histogram1", "Histogram 1")
        
        metrics = registry.get_all_metrics()
        assert len(metrics) == 3
        assert "counter1" in metrics
        assert "gauge1" in metrics
        assert "histogram1" in metrics


class TestCounter:
    """Test counter metric implementation."""
    
    def test_counter_initialization(self):
        """Test counter starts at zero."""
        counter = Counter("test_counter", "Test counter")
        
        assert counter.value == 0
        assert counter.name == "test_counter"
        assert counter.description == "Test counter"
        
    def test_counter_increment(self):
        """Test incrementing counter."""
        counter = Counter("test_counter", "Test counter")
        
        counter.increment()
        assert counter.value == 1
        
        counter.increment(5)
        assert counter.value == 6
        
    def test_counter_increment_with_labels(self):
        """Test incrementing counter with labels."""
        counter = Counter("test_counter", "Test counter")
        
        counter.increment(labels={"method": "GET", "status": "200"})
        counter.increment(labels={"method": "GET", "status": "200"})
        counter.increment(labels={"method": "POST", "status": "201"})
        
        assert counter.get_value({"method": "GET", "status": "200"}) == 2
        assert counter.get_value({"method": "POST", "status": "201"}) == 1
        
    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        counter = Counter("test_counter", "Test counter")
        
        def increment_worker():
            for _ in range(100):
                counter.increment()
                
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        assert counter.value == 1000


class TestGauge:
    """Test gauge metric implementation."""
    
    def test_gauge_initialization(self):
        """Test gauge starts at zero."""
        gauge = Gauge("test_gauge", "Test gauge")
        
        assert gauge.value == 0
        assert gauge.name == "test_gauge"
        
    def test_gauge_set_value(self):
        """Test setting gauge value."""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.set(42.5)
        assert gauge.value == 42.5
        
        gauge.set(100)
        assert gauge.value == 100
        
    def test_gauge_increment_decrement(self):
        """Test incrementing and decrementing gauge."""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.increment()
        assert gauge.value == 1
        
        gauge.increment(5)
        assert gauge.value == 6
        
        gauge.decrement(2)
        assert gauge.value == 4
        
    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.set(10, labels={"region": "us-east"})
        gauge.set(20, labels={"region": "us-west"})
        
        assert gauge.get_value({"region": "us-east"}) == 10
        assert gauge.get_value({"region": "us-west"}) == 20


class TestHistogram:
    """Test histogram metric implementation."""
    
    def test_histogram_initialization(self):
        """Test histogram initializes with buckets."""
        histogram = Histogram("test_histogram", "Test histogram")
        
        assert histogram.name == "test_histogram"
        assert len(histogram.buckets) > 0
        assert histogram.count == 0
        assert histogram.sum == 0
        
    def test_histogram_observe(self):
        """Test observing values in histogram."""
        histogram = Histogram("test_histogram", "Test histogram")
        
        histogram.observe(0.5)
        histogram.observe(1.5)
        histogram.observe(5.0)
        
        assert histogram.count == 3
        assert histogram.sum == 7.0
        
    def test_histogram_buckets(self):
        """Test histogram bucket distribution."""
        buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        histogram = Histogram("test_histogram", "Test histogram", buckets=buckets)
        
        histogram.observe(0.3)  # Should go in 0.5 bucket
        histogram.observe(1.2)  # Should go in 2.5 bucket
        histogram.observe(8.0)  # Should go in 10.0 bucket
        
        bucket_counts = histogram.get_bucket_counts()
        assert bucket_counts[0.5] >= 1  # 0.3 observation
        assert bucket_counts[2.5] >= 1  # 1.2 observation
        assert bucket_counts[10.0] >= 1  # 8.0 observation


class TestTimer:
    """Test timer utility for measuring operation duration."""
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        histogram = Histogram("test_duration", "Test duration")
        
        with Timer(histogram):
            time.sleep(0.01)  # 10ms
            
        assert histogram.count == 1
        assert histogram.sum > 0.005  # Should be at least 5ms
        
    def test_timer_decorator(self):
        """Test timer as decorator."""
        histogram = Histogram("function_duration", "Function duration")
        
        @Timer.decorator(histogram)
        def test_function():
            time.sleep(0.01)
            return "result"
            
        result = test_function()
        
        assert result == "result"
        assert histogram.count == 1
        assert histogram.sum > 0.005


class TestGlobalMetrics:
    """Test global metrics convenience functions."""
    
    def test_global_metrics_registry(self):
        """Test global metrics registry."""
        registry = get_metrics_registry()
        
        assert registry is not None
        assert isinstance(registry, MetricsRegistry)
        
        # Should return same instance
        registry2 = get_metrics_registry()
        assert registry is registry2
        
    def test_record_operation_metrics(self):
        """Test recording operation metrics."""
        context = RequestContext("test-123")
        
        record_operation_metrics(
            operation="github_api_request",
            duration_ms=150.5,
            status="success",
            context=context
        )
        
        registry = get_metrics_registry()
        metrics = registry.get_all_metrics()
        
        # Should have created appropriate metrics
        assert "operation_duration_seconds" in metrics
        assert "operation_total" in metrics
        
    def test_with_metrics_decorator(self):
        """Test with_metrics decorator."""
        @with_metrics(operation="test_operation")
        def test_function(x, y):
            time.sleep(0.01)
            if x == 0:
                raise ValueError("Division by zero")
            return y / x
            
        # Test successful operation
        result = test_function(2, 10)
        assert result == 5.0
        
        # Test failed operation
        try:
            test_function(0, 10)
        except ValueError:
            pass
            
        registry = get_metrics_registry()
        metrics = registry.get_all_metrics()
        
        assert "operation_duration_seconds" in metrics
        assert "operation_total" in metrics


class TestMetricsExporters:
    """Test metrics export functionality."""
    
    def test_json_exporter(self):
        """Test JSON metrics exporter."""
        registry = MetricsRegistry()
        
        counter = registry.counter("test_counter", "Test counter")
        gauge = registry.gauge("test_gauge", "Test gauge")
        
        counter.increment(5)
        gauge.set(42)
        
        exporter = JSONExporter()
        output = exporter.export(registry)
        
        data = json.loads(output)
        assert "test_counter" in data
        assert "test_gauge" in data
        assert data["test_counter"]["value"] == 5
        assert data["test_gauge"]["value"] == 42
        
    def test_prometheus_exporter(self):
        """Test Prometheus metrics exporter."""
        registry = MetricsRegistry()
        
        counter = registry.counter("requests_total", "Total requests", labels=["method", "status"])
        gauge = registry.gauge("active_connections", "Active connections")
        
        counter.increment(labels={"method": "GET", "status": "200"})
        counter.increment(labels={"method": "POST", "status": "201"})
        gauge.set(10)
        
        exporter = PrometheusExporter()
        output = exporter.export(registry)
        
        assert "# HELP requests_total Total requests" in output
        assert "# TYPE requests_total counter" in output
        assert 'requests_total{method="GET",status="200"} 1' in output
        assert 'requests_total{method="POST",status="201"} 1' in output
        assert "active_connections 10" in output


class TestMetricsIntegration:
    """Test metrics integration with existing modules."""
    
    def test_github_integration_metrics(self):
        """Test metrics collection in GitHub integration."""
        # This would test actual integration - placeholder for now
        registry = get_metrics_registry()
        
        # Simulate GitHub API call metrics
        record_operation_metrics(
            operation="github_api_call",
            duration_ms=250,
            status="success",
            context=RequestContext("github-test")
        )
        
        metrics = registry.get_all_metrics()
        assert len(metrics) > 0
        
    def test_pr_analysis_metrics(self):
        """Test metrics collection in PR analysis."""
        registry = get_metrics_registry()
        
        # Simulate PR analysis metrics
        record_operation_metrics(
            operation="pr_analysis",
            duration_ms=5000,
            status="success",
            context=RequestContext("pr-test")
        )
        
        metrics = registry.get_all_metrics()
        assert len(metrics) > 0
        
    def test_error_rate_calculation(self):
        """Test error rate metrics calculation."""
        registry = MetricsRegistry()
        
        # Simulate operations with different outcomes
        for _ in range(90):
            record_operation_metrics("test_op", 100, "success")
        for _ in range(10):
            record_operation_metrics("test_op", 150, "error")
            
        total_counter = registry.get_metric("operation_total")
        error_rate = (
            total_counter.get_value({"operation": "test_op", "status": "error"}) /
            total_counter.get_value({"operation": "test_op"})
        ) * 100
        
        assert error_rate == 10.0  # 10% error rate


class TestMetricsPerformance:
    """Test metrics system performance."""
    
    def test_high_throughput_metrics(self):
        """Test metrics system under high load."""
        registry = MetricsRegistry()
        counter = registry.counter("high_throughput", "High throughput test")
        
        start_time = time.time()
        
        # Simulate high throughput
        for _ in range(10000):
            counter.increment()
            
        duration = time.time() - start_time
        
        assert counter.value == 10000
        assert duration < 1.0  # Should complete in under 1 second
        
    def test_memory_usage(self):
        """Test metrics system memory usage."""
        registry = MetricsRegistry()
        
        # Create many metrics
        for i in range(1000):
            counter = registry.counter(f"metric_{i}", f"Metric {i}")
            counter.increment(i)
            
        metrics = registry.get_all_metrics()
        assert len(metrics) == 1000
        
        # Memory usage should be reasonable
        # This is a basic check - in practice you'd use memory profiling tools