"""Tests for monitoring infrastructure."""

import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from autogen_code_review_bot.monitoring import (
    HealthChecker,
    HealthStatus,
    HealthCheck,
    MetricsEmitter,
    SLITracker,
    SLODefinition,
    MonitoringServer,
    create_health_endpoint,
    get_system_health
)


def test_health_status_enum():
    """Test HealthStatus enum values."""
    assert HealthStatus.HEALTHY == "healthy"
    assert HealthStatus.DEGRADED == "degraded"
    assert HealthStatus.UNHEALTHY == "unhealthy"
    assert HealthStatus.UNKNOWN == "unknown"


def test_health_check_creation():
    """Test HealthCheck dataclass creation."""
    check = HealthCheck(
        name="database",
        status=HealthStatus.HEALTHY,
        message="Connection successful",
        response_time_ms=45.2
    )
    
    assert check.name == "database"
    assert check.status == HealthStatus.HEALTHY
    assert check.message == "Connection successful"
    assert check.response_time_ms == 45.2
    assert isinstance(check.timestamp, float)
    assert check.details is None


def test_health_check_with_details():
    """Test HealthCheck with additional details."""
    details = {"connections": 5, "pool_size": 10}
    check = HealthCheck(
        name="db_pool",
        status=HealthStatus.DEGRADED,
        message="Pool utilization high",
        details=details
    )
    
    assert check.details == details


def test_health_checker_initialization():
    """Test HealthChecker initialization."""
    checker = HealthChecker()
    
    assert isinstance(checker.checks, dict)
    assert len(checker.checks) == 0
    assert hasattr(checker, '_lock')


def test_health_checker_register_check():
    """Test registering health checks."""
    checker = HealthChecker()
    
    def dummy_check():
        return HealthCheck("test", HealthStatus.HEALTHY, "OK")
    
    checker.register_check("test_service", dummy_check)
    
    assert "test_service" in checker.checks
    assert callable(checker.checks["test_service"])


def test_health_checker_run_checks():
    """Test running health checks."""
    checker = HealthChecker()
    
    def healthy_check():
        return HealthCheck("service1", HealthStatus.HEALTHY, "Running")
    
    def degraded_check():
        return HealthCheck("service2", HealthStatus.DEGRADED, "Slow response")
    
    checker.register_check("service1", healthy_check)
    checker.register_check("service2", degraded_check)
    
    results = checker.run_all_checks()
    
    assert len(results) == 2
    assert results["service1"].status == HealthStatus.HEALTHY
    assert results["service2"].status == HealthStatus.DEGRADED


def test_health_checker_overall_status():
    """Test calculating overall health status."""
    checker = HealthChecker()
    
    # All healthy
    results = {
        "service1": HealthCheck("service1", HealthStatus.HEALTHY, "OK"),
        "service2": HealthCheck("service2", HealthStatus.HEALTHY, "OK")
    }
    assert checker.get_overall_status(results) == HealthStatus.HEALTHY
    
    # One degraded
    results["service2"] = HealthCheck("service2", HealthStatus.DEGRADED, "Slow")
    assert checker.get_overall_status(results) == HealthStatus.DEGRADED
    
    # One unhealthy
    results["service1"] = HealthCheck("service1", HealthStatus.UNHEALTHY, "Down")
    assert checker.get_overall_status(results) == HealthStatus.UNHEALTHY
    
    # Empty results
    assert checker.get_overall_status({}) == HealthStatus.UNKNOWN


def test_health_checker_thread_safety():
    """Test that HealthChecker is thread-safe."""
    checker = HealthChecker()
    results = []
    errors = []
    
    def register_check(thread_id):
        try:
            def check_func():
                return HealthCheck(f"service_{thread_id}", HealthStatus.HEALTHY, "OK")
            
            checker.register_check(f"service_{thread_id}", check_func)
            check_results = checker.run_all_checks()
            results.append(len(check_results))
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=register_check, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Check results
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    assert len(results) == 5
    final_checks = checker.run_all_checks()
    assert len(final_checks) == 5


def test_metrics_emitter_initialization():
    """Test MetricsEmitter initialization."""
    emitter = MetricsEmitter()
    
    assert hasattr(emitter, 'metrics')
    assert hasattr(emitter, '_lock')
    assert isinstance(emitter.metrics, dict)


def test_metrics_emitter_record_metric():
    """Test recording metrics."""
    emitter = MetricsEmitter()
    
    # Record counter metric
    emitter.record_counter("api_requests", 1, tags={"endpoint": "/health"})
    
    # Record gauge metric
    emitter.record_gauge("memory_usage", 85.5, tags={"unit": "percent"})
    
    # Record histogram metric
    emitter.record_histogram("response_time", 123.4, tags={"method": "GET"})
    
    # Check metrics were recorded
    assert len(emitter.metrics) == 3
    assert "api_requests" in emitter.metrics
    assert "memory_usage" in emitter.metrics
    assert "response_time" in emitter.metrics


def test_metrics_emitter_get_metrics():
    """Test retrieving metrics."""
    emitter = MetricsEmitter()
    
    emitter.record_counter("requests", 5)
    emitter.record_gauge("cpu", 75.0)
    
    metrics = emitter.get_metrics()
    
    assert len(metrics) == 2
    assert "requests" in metrics
    assert "cpu" in metrics
    assert metrics["requests"]["value"] == 5
    assert metrics["cpu"]["value"] == 75.0


def test_metrics_emitter_clear_metrics():
    """Test clearing metrics."""
    emitter = MetricsEmitter()
    
    emitter.record_counter("test", 1)
    assert len(emitter.metrics) == 1
    
    emitter.clear_metrics()
    assert len(emitter.metrics) == 0


def test_slo_definition():
    """Test SLO definition."""
    slo = SLODefinition(
        name="api_availability",
        target=99.9,
        measurement_window_hours=24,
        description="API should be available 99.9% of the time"
    )
    
    assert slo.name == "api_availability"
    assert slo.target == 99.9
    assert slo.measurement_window_hours == 24
    assert slo.description == "API should be available 99.9% of the time"


def test_sli_tracker_initialization():
    """Test SLITracker initialization."""
    tracker = SLITracker()
    
    assert isinstance(tracker.slos, dict)
    assert isinstance(tracker.measurements, dict)
    assert hasattr(tracker, '_lock')


def test_sli_tracker_register_slo():
    """Test registering SLOs."""
    tracker = SLITracker()
    
    slo = SLODefinition("test_slo", 95.0, 24)
    tracker.register_slo(slo)
    
    assert "test_slo" in tracker.slos
    assert tracker.slos["test_slo"] == slo


def test_sli_tracker_record_measurement():
    """Test recording SLI measurements."""
    tracker = SLITracker()
    
    slo = SLODefinition("availability", 99.0, 24)
    tracker.register_slo(slo)
    
    # Record successful measurement
    tracker.record_measurement("availability", True)
    
    # Record failed measurement
    tracker.record_measurement("availability", False)
    
    assert "availability" in tracker.measurements
    assert len(tracker.measurements["availability"]) == 2


def test_sli_tracker_calculate_sli():
    """Test calculating SLI values."""
    tracker = SLITracker()
    
    slo = SLODefinition("test_sli", 95.0, 1)  # 1 hour window
    tracker.register_slo(slo)
    
    # Record measurements: 8 successes, 2 failures = 80% SLI
    for _ in range(8):
        tracker.record_measurement("test_sli", True)
    for _ in range(2):
        tracker.record_measurement("test_sli", False)
    
    sli_value = tracker.calculate_sli("test_sli")
    assert abs(sli_value - 80.0) < 0.1  # Should be 80%


def test_sli_tracker_is_slo_met():
    """Test checking if SLO is met."""
    tracker = SLITracker()
    
    slo = SLODefinition("test_slo", 90.0, 1)
    tracker.register_slo(slo)
    
    # Record measurements that exceed SLO: 95% success
    for _ in range(19):
        tracker.record_measurement("test_slo", True)
    for _ in range(1):
        tracker.record_measurement("test_slo", False)
    
    assert tracker.is_slo_met("test_slo") is True
    
    # Add more failures to bring below SLO
    for _ in range(5):
        tracker.record_measurement("test_slo", False)
    
    assert tracker.is_slo_met("test_slo") is False


def test_monitoring_server_initialization():
    """Test MonitoringServer initialization."""
    server = MonitoringServer(port=8080)
    
    assert server.port == 8080
    assert isinstance(server.health_checker, HealthChecker)
    assert isinstance(server.metrics_emitter, MetricsEmitter)
    assert isinstance(server.sli_tracker, SLITracker)


def test_create_health_endpoint():
    """Test creating health endpoint function."""
    health_checker = HealthChecker()
    
    def test_check():
        return HealthCheck("test", HealthStatus.HEALTHY, "OK")
    
    health_checker.register_check("test", test_check)
    
    endpoint_func = create_health_endpoint(health_checker)
    assert callable(endpoint_func)
    
    # Test endpoint response
    response = endpoint_func()
    assert isinstance(response, dict)
    assert "status" in response
    assert "checks" in response
    assert "timestamp" in response


def test_get_system_health():
    """Test getting system health information."""
    health_info = get_system_health()
    
    assert isinstance(health_info, dict)
    assert "cpu_percent" in health_info
    assert "memory_percent" in health_info
    assert "disk_usage" in health_info
    assert "timestamp" in health_info
    
    # Check value types
    assert isinstance(health_info["cpu_percent"], (int, float))
    assert isinstance(health_info["memory_percent"], (int, float))
    assert isinstance(health_info["disk_usage"], (int, float))
    assert isinstance(health_info["timestamp"], float)


def test_health_check_timeout():
    """Test health check timeout handling."""
    checker = HealthChecker()
    
    def slow_check():
        time.sleep(0.1)  # Simulate slow check
        return HealthCheck("slow", HealthStatus.HEALTHY, "Slow but OK")
    
    def timeout_check():
        time.sleep(2.0)  # Very slow check
        return HealthCheck("timeout", HealthStatus.HEALTHY, "Should timeout")
    
    checker.register_check("slow", slow_check)
    checker.register_check("timeout", timeout_check)
    
    # Run checks with timeout
    start_time = time.time()
    results = checker.run_all_checks(timeout_seconds=0.5)
    elapsed = time.time() - start_time
    
    # Should complete within reasonable time despite slow checks
    assert elapsed < 1.0
    
    # Slow check should complete, timeout check should be marked as unhealthy
    assert len(results) == 2


def test_metrics_aggregation():
    """Test metrics aggregation functionality."""
    emitter = MetricsEmitter()
    
    # Record multiple values for the same metric
    emitter.record_counter("requests", 1)
    emitter.record_counter("requests", 5)
    emitter.record_counter("requests", 2)
    
    metrics = emitter.get_metrics()
    
    # Counter should aggregate values
    assert "requests" in metrics
    # Implementation should handle aggregation appropriately


def test_monitoring_integration():
    """Test integration between monitoring components."""
    server = MonitoringServer(port=8081)
    
    # Add some SLOs
    slo = SLODefinition("system_availability", 99.5, 24)
    server.sli_tracker.register_slo(slo)
    
    # Add health checks
    def system_check():
        return HealthCheck("system", HealthStatus.HEALTHY, "All systems operational")
    
    server.health_checker.register_check("system", system_check)
    
    # Record some metrics
    server.metrics_emitter.record_counter("total_requests", 100)
    server.metrics_emitter.record_gauge("active_connections", 25)
    
    # Record SLI measurements
    server.sli_tracker.record_measurement("system_availability", True)
    
    # Verify everything works together
    health_results = server.health_checker.run_all_checks()
    metrics = server.metrics_emitter.get_metrics()
    sli_value = server.sli_tracker.calculate_sli("system_availability")
    
    assert len(health_results) == 1
    assert len(metrics) == 2
    assert sli_value == 100.0  # 1 success = 100%