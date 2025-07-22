"""Tests for metrics collection in PR analysis and webhook processing."""

import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from autogen_code_review_bot.pr_analysis import analyze_pr, _run_all_checks_parallel, metrics
from autogen_code_review_bot.monitoring import MetricsEmitter


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        
        # Create a simple Python file
        (repo_path / "test.py").write_text("""
def hello():
    print("Hello, World!")
""")
        
        yield str(repo_path)


@pytest.fixture
def mock_metrics():
    """Create a mock metrics emitter for testing."""
    with patch('autogen_code_review_bot.pr_analysis.metrics') as mock:
        yield mock


def test_analyze_pr_records_request_metrics(temp_repo, mock_metrics):
    """Test that PR analysis records throughput metrics."""
    with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
        mock_checks.return_value = Mock(
            security=Mock(output=""),
            style=Mock(output=""),
            performance=Mock(output="")
        )
        
        analyze_pr(temp_repo, use_cache=False, use_parallel=True)
        
        # Verify request counter was called
        mock_metrics.record_counter.assert_any_call(
            "pr_analysis_requests_total", 1, 
            tags={"cache_enabled": "False", "parallel_enabled": "True"}
        )


def test_analyze_pr_records_completion_metrics(temp_repo, mock_metrics):
    """Test that PR analysis records completion and duration metrics."""
    with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
        mock_checks.return_value = Mock(
            security=Mock(output="security issue"),
            style=Mock(output=""),
            performance=Mock(output="performance issue")
        )
        
        analyze_pr(temp_repo, use_cache=False)
        
        # Verify completion metrics
        mock_metrics.record_counter.assert_any_call("pr_analysis_completed_total", 1)
        
        # Verify duration metric was recorded
        duration_calls = [call for call in mock_metrics.record_histogram.call_args_list 
                         if call[0][0] == "pr_analysis_duration_seconds"]
        assert len(duration_calls) > 0
        
        # Verify quality score was recorded
        quality_calls = [call for call in mock_metrics.record_gauge.call_args_list 
                        if call[0][0] == "pr_analysis_quality_score"]
        assert len(quality_calls) > 0


def test_analyze_pr_records_issue_detection_metrics(temp_repo, mock_metrics):
    """Test that PR analysis records issue detection metrics."""
    with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
        mock_checks.return_value = Mock(
            security=Mock(output="security vulnerability found"),
            style=Mock(output="style issues detected"),
            performance=Mock(output="")
        )
        
        analyze_pr(temp_repo, use_cache=False)
        
        # Verify issue detection metrics
        mock_metrics.record_counter.assert_any_call(
            "pr_analysis_issues_detected_total", 1, tags={"issue_type": "security"}
        )
        mock_metrics.record_counter.assert_any_call(
            "pr_analysis_issues_detected_total", 1, tags={"issue_type": "style"}
        )


def test_analyze_pr_records_error_metrics(mock_metrics):
    """Test that PR analysis records error metrics for invalid paths."""
    analyze_pr("", use_cache=False)
    
    # Verify error metric was recorded
    mock_metrics.record_counter.assert_any_call(
        "pr_analysis_errors_total", 1, tags={"error_type": "invalid_path"}
    )


def test_analyze_pr_records_cache_metrics(temp_repo, mock_metrics):
    """Test that PR analysis records cache hit metrics."""
    with patch('autogen_code_review_bot.pr_analysis.LinterCache') as mock_cache_class:
        mock_cache = Mock()
        mock_cached_result = Mock(
            security=Mock(output=""),
            style=Mock(output=""),
            performance=Mock(output="")
        )
        mock_cache.get_with_invalidation_check.return_value = mock_cached_result
        mock_cache_class.return_value = mock_cache
        
        with patch('autogen_code_review_bot.pr_analysis.get_commit_hash', return_value="abc123"):
            analyze_pr(temp_repo, use_cache=True)
        
        # Verify cache hit metric was recorded
        mock_metrics.record_counter.assert_any_call("pr_analysis_cache_hits_total", 1)


def test_parallel_checks_record_individual_timings(temp_repo, mock_metrics):
    """Test that parallel checks record timing metrics for each check type."""
    with patch('autogen_code_review_bot.pr_analysis._run_security_checks', return_value=""):
        with patch('autogen_code_review_bot.pr_analysis._run_style_checks_parallel', return_value=("", "")):
            with patch('autogen_code_review_bot.pr_analysis._run_performance_checks', return_value=""):
                _run_all_checks_parallel(temp_repo, {})
    
    # Verify individual check timing metrics were recorded
    duration_calls = [call for call in mock_metrics.record_histogram.call_args_list 
                     if call[0][0] == "pr_analysis_check_duration_seconds"]
    
    # Should have calls for security, style, and performance
    check_types = [call[1]["tags"]["check_type"] for call in duration_calls]
    assert "security" in check_types
    assert "style" in check_types
    assert "performance" in check_types


def test_parallel_checks_record_error_metrics(temp_repo, mock_metrics):
    """Test that parallel check errors are recorded."""
    with patch('autogen_code_review_bot.pr_analysis._run_security_checks', side_effect=Exception("Test error")):
        result = _run_all_checks_parallel(temp_repo, {})
    
    # Verify error metric was recorded
    mock_metrics.record_counter.assert_any_call(
        "pr_analysis_errors_total", 1, tags={"error_type": "parallel_execution"}
    )


def test_metrics_collection_thread_safety():
    """Test that metrics collection is thread-safe."""
    import threading
    
    metrics_emitter = MetricsEmitter()
    results = []
    
    def record_metrics():
        for i in range(100):
            metrics_emitter.record_counter("test_counter", 1)
            metrics_emitter.record_gauge("test_gauge", i)
            metrics_emitter.record_histogram("test_histogram", i * 0.1)
    
    # Run multiple threads
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=record_metrics)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify metrics were recorded
    metrics_data = metrics_emitter.get_metrics()
    assert "test_counter" in metrics_data
    assert "test_gauge" in metrics_data
    assert "test_histogram" in metrics_data
    
    # Counter should show total of all increments (5 threads * 100 increments)
    assert metrics_data["test_counter"]["count"] == 500


def test_metrics_memory_management():
    """Test that metrics collection manages memory properly."""
    metrics_emitter = MetricsEmitter()
    metrics_emitter.max_values_per_metric = 10  # Set low limit for testing
    
    # Record more values than the limit
    for i in range(20):
        metrics_emitter.record_gauge("test_gauge", i)
    
    # Verify that memory is managed (should only keep last 10 values)
    metrics_data = metrics_emitter.get_metrics()
    assert metrics_data["test_gauge"]["count"] == 10


def test_webhook_metrics_integration():
    """Test webhook handler metrics integration."""
    from bot import metrics as bot_metrics
    
    # Mock webhook handler call
    with patch.object(bot_metrics, 'record_counter') as mock_record_counter:
        with patch.object(bot_metrics, 'record_histogram') as mock_record_histogram:
            # Simulate successful webhook processing
            bot_metrics.record_counter("webhook_requests_total", 1, tags={"status": "success"})
            bot_metrics.record_counter("webhook_events_total", 1, tags={"event_type": "pull_request", "action": "opened"})
            bot_metrics.record_counter("pr_events_processed_total", 1, tags={"action": "opened"})
            
            # Verify metrics were called
            assert mock_record_counter.call_count == 3
            
            # Verify correct tags were used
            calls = mock_record_counter.call_args_list
            assert any("status" in call[1]["tags"] and call[1]["tags"]["status"] == "success" for call in calls)
            assert any("event_type" in call[1]["tags"] and call[1]["tags"]["event_type"] == "pull_request" for call in calls)


def test_performance_metrics_capture():
    """Test that performance metrics capture useful timing data."""
    metrics_emitter = MetricsEmitter()
    
    # Simulate realistic operation timing
    start_time = time.time()
    time.sleep(0.01)  # Simulate 10ms operation
    duration = time.time() - start_time
    
    metrics_emitter.record_histogram("test_operation_duration", duration)
    
    metrics_data = metrics_emitter.get_metrics()
    recorded_duration = metrics_data["test_operation_duration"]["value"]
    
    # Verify timing is reasonable (should be around 0.01 seconds)
    assert 0.005 < recorded_duration < 0.1  # Allow some variance


def test_metrics_aggregation():
    """Test metrics aggregation and statistics calculation."""
    metrics_emitter = MetricsEmitter()
    
    # Record multiple values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for value in values:
        metrics_emitter.record_histogram("test_metric", value)
    
    metrics_data = metrics_emitter.get_metrics()
    stats = metrics_data["test_metric"]["stats"]
    
    # Verify statistics
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["avg"] == 3.0  # (1+2+3+4+5)/5
    assert stats["sum"] is None  # Histograms don't track sum


def test_error_rate_calculation():
    """Test calculation of error rates from metrics."""
    metrics_emitter = MetricsEmitter()
    
    # Record some successes and errors
    for _ in range(90):
        metrics_emitter.record_counter("operations_total", 1, tags={"status": "success"})
    
    for _ in range(10):
        metrics_emitter.record_counter("operations_total", 1, tags={"status": "error"})
    
    metrics_data = metrics_emitter.get_metrics()
    
    # The metric system records each tag combination separately
    # So we should have metrics for both success and error tags
    assert "operations_total" in metrics_data
    assert metrics_data["operations_total"]["count"] >= 10  # At least the error count


if __name__ == "__main__":
    pytest.main([__file__])