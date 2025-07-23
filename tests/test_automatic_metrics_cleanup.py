"""Test automatic metrics cleanup functionality."""

import time
import threading
from unittest.mock import patch
import pytest

from autogen_code_review_bot.monitoring import MetricsEmitter


class TestAutomaticMetricsCleanup:
    """Test automatic metrics cleanup to prevent memory leaks."""
    
    def test_automatic_cleanup_initialization(self):
        """Test that automatic cleanup can be enabled during initialization."""
        emitter = MetricsEmitter(auto_cleanup_enabled=True, auto_cleanup_hours=1.0, auto_cleanup_interval=60)
        
        assert emitter.auto_cleanup_enabled is True
        assert emitter.auto_cleanup_hours == 1.0
        assert emitter.auto_cleanup_interval == 60
        assert emitter._cleanup_thread is not None
        assert emitter._cleanup_thread.daemon is True
        
        # Clean up
        emitter.stop_auto_cleanup()
    
    def test_disable_automatic_cleanup_by_default(self):
        """Test that automatic cleanup is disabled by default for backward compatibility."""
        emitter = MetricsEmitter()
        
        assert emitter.auto_cleanup_enabled is False
        assert emitter._cleanup_thread is None
    
    def test_manual_start_stop_auto_cleanup(self):
        """Test manual start and stop of automatic cleanup."""
        emitter = MetricsEmitter()
        
        # Start automatic cleanup
        emitter.start_auto_cleanup(cleanup_hours=0.5, interval_seconds=30)
        assert emitter.auto_cleanup_enabled is True
        assert emitter._cleanup_thread is not None
        assert emitter._cleanup_thread.is_alive()
        
        # Stop automatic cleanup
        emitter.stop_auto_cleanup()
        assert emitter.auto_cleanup_enabled is False
        # Thread should stop gracefully
        emitter._cleanup_thread.join(timeout=1)
        assert not emitter._cleanup_thread.is_alive()
    
    def test_automatic_cleanup_removes_old_metrics(self):
        """Test that automatic cleanup actually removes old metrics."""
        emitter = MetricsEmitter()
        
        # Add some metrics
        emitter.record_counter("test_metric", 1)
        emitter.record_gauge("test_gauge", 100)
        
        # Verify metrics exist
        metrics = emitter.get_metrics()
        assert "test_metric" in metrics
        assert "test_gauge" in metrics
        
        # Manually simulate time passing by modifying timestamps
        old_time = time.time() - 3600  # 1 hour ago
        for values in emitter.metrics.values():
            for value in values:
                value.timestamp = old_time
        
        # Run cleanup (should remove metrics older than 0.5 hours)
        emitter._cleanup_old_metrics(cleanup_hours=0.5)
        
        # Verify old metrics were removed
        metrics = emitter.get_metrics()
        assert len(metrics) == 0
    
    def test_automatic_cleanup_preserves_recent_metrics(self):
        """Test that automatic cleanup preserves recent metrics."""
        emitter = MetricsEmitter()
        
        # Add some old metrics
        old_time = time.time() - 3600  # 1 hour ago
        emitter.record_counter("old_metric", 1)
        for values in emitter.metrics.values():
            for value in values:
                value.timestamp = old_time
        
        # Add some recent metrics
        emitter.record_counter("recent_metric", 2)
        emitter.record_gauge("recent_gauge", 200)
        
        # Run cleanup (should remove metrics older than 0.5 hours)
        emitter._cleanup_old_metrics(cleanup_hours=0.5)
        
        # Verify recent metrics are preserved
        metrics = emitter.get_metrics()
        assert "recent_metric" in metrics
        assert "recent_gauge" in metrics
        assert "old_metric" not in metrics
    
    def test_cleanup_thread_safety(self):
        """Test that cleanup is thread-safe with concurrent metric recording."""
        emitter = MetricsEmitter()
        
        # Flag to control test duration
        stop_test = threading.Event()
        
        def record_metrics():
            """Record metrics continuously."""
            counter = 0
            while not stop_test.is_set():
                emitter.record_counter("concurrent_metric", counter)
                counter += 1
                time.sleep(0.01)
        
        def run_cleanup():
            """Run cleanup continuously."""
            while not stop_test.is_set():
                emitter._cleanup_old_metrics(cleanup_hours=1.0)
                time.sleep(0.02)
        
        # Start concurrent operations
        record_thread = threading.Thread(target=record_metrics)
        cleanup_thread = threading.Thread(target=run_cleanup)
        
        record_thread.start()
        cleanup_thread.start()
        
        # Let them run for a short time
        time.sleep(0.1)
        
        # Stop the test
        stop_test.set()
        record_thread.join()
        cleanup_thread.join()
        
        # Verify no crashes and metrics still work
        emitter.record_counter("final_test", 1)
        metrics = emitter.get_metrics()
        assert "final_test" in metrics
    
    def test_cleanup_with_different_metric_types(self):
        """Test cleanup works correctly with different metric types."""
        emitter = MetricsEmitter()
        
        # Add metrics of different types
        emitter.record_counter("test_counter", 1)
        emitter.record_gauge("test_gauge", 100)
        emitter.record_histogram("test_histogram", 50)
        
        # Make all metrics old
        old_time = time.time() - 3600  # 1 hour ago
        for values in emitter.metrics.values():
            for value in values:
                value.timestamp = old_time
        
        # Run cleanup
        emitter._cleanup_old_metrics(cleanup_hours=0.5)
        
        # Verify all old metrics removed regardless of type
        metrics = emitter.get_metrics()
        assert len(metrics) == 0
    
    def test_cleanup_respects_max_values_limit(self):
        """Test that cleanup works with max_values_per_metric limit."""
        emitter = MetricsEmitter()
        emitter.max_values_per_metric = 5
        
        # Add more metrics than the limit
        for i in range(10):
            emitter.record_counter("test_metric", i)
        
        # Should be limited to max_values_per_metric
        assert len(emitter.metrics["test_metric"]) == 5
        
        # Make some metrics old (but not all, since we only keep 5)
        old_time = time.time() - 3600  # 1 hour ago
        for value in emitter.metrics["test_metric"][:3]:
            value.timestamp = old_time
        
        # Run cleanup
        emitter._cleanup_old_metrics(cleanup_hours=0.5)
        
        # Should have removed the old ones
        assert len(emitter.metrics["test_metric"]) == 2
    
    @patch('time.sleep')
    def test_auto_cleanup_thread_loop(self, mock_sleep):
        """Test the automatic cleanup thread loop."""
        emitter = MetricsEmitter()
        
        # Mock the cleanup method to track calls
        cleanup_calls = []
        original_cleanup = emitter._cleanup_old_metrics
        def mock_cleanup(cleanup_hours):
            cleanup_calls.append(cleanup_hours)
            return original_cleanup(cleanup_hours)
        
        emitter._cleanup_old_metrics = mock_cleanup
        
        # Start auto cleanup with short interval for testing
        emitter.start_auto_cleanup(cleanup_hours=1.0, interval_seconds=0.1)
        
        # Let it run for a bit
        time.sleep(0.25)
        
        # Stop cleanup
        emitter.stop_auto_cleanup()
        
        # Verify cleanup was called
        assert len(cleanup_calls) >= 1
        assert all(hours == 1.0 for hours in cleanup_calls)


if __name__ == "__main__":
    pytest.main([__file__])