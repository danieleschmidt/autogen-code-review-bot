"""Tests for code duplication refactoring in PR analysis."""

import time
from unittest.mock import patch, MagicMock

from autogen_code_review_bot.pr_analysis import _run_timed_check


class TestCodeDuplicationRefactor:
    """Test code duplication refactoring utilities."""

    def test_run_timed_check_basic(self):
        """Test basic functionality of timed check utility."""
        def dummy_func(arg1, arg2):
            return f"result: {arg1} {arg2}"
        
        result = _run_timed_check("test", dummy_func, "hello", "world")
        assert result == "result: hello world"

    def test_run_timed_check_timing(self):
        """Test that timed check records timing metrics."""
        def slow_func():
            time.sleep(0.01)  # Small delay
            return "done"
        
        with patch('autogen_code_review_bot.pr_analysis.metrics') as mock_metrics:
            result = _run_timed_check("test", slow_func)
            
            assert result == "done"
            # Verify metrics were recorded
            mock_metrics.record_histogram.assert_called_once()
            call_args = mock_metrics.record_histogram.call_args
            assert call_args[0][0] == "pr_analysis_check_duration_seconds"
            assert call_args[0][1] > 0  # Duration should be positive
            assert call_args[1]["tags"]["check_type"] == "test"

    def test_run_timed_check_logging(self):
        """Test that timed check produces appropriate logs."""
        def test_func():
            return "test result"
        
        with patch('autogen_code_review_bot.pr_analysis.logger') as mock_logger:
            result = _run_timed_check("security", test_func)
            
            assert result == "test result"
            # Check debug logs were called
            assert mock_logger.debug.call_count == 2
            start_call = mock_logger.debug.call_args_list[0]
            end_call = mock_logger.debug.call_args_list[1]
            
            assert "Starting security analysis" in start_call[0][0]
            assert "Security analysis completed" in end_call[0][0]
            assert "duration_seconds" in end_call[1]["extra"]

    def test_run_timed_check_exception_handling(self):
        """Test that timed check properly handles exceptions."""
        def failing_func():
            raise ValueError("test error")
        
        with patch('autogen_code_review_bot.pr_analysis.logger'):
            try:
                _run_timed_check("test", failing_func)
                assert False, "Should have raised exception"
            except ValueError as e:
                assert str(e) == "test error"

    def test_run_timed_check_different_types(self):
        """Test timed check with different check types."""
        def identity_func(value):
            return value
        
        with patch('autogen_code_review_bot.pr_analysis.metrics') as mock_metrics:
            # Test different check types
            for check_type in ["security", "style", "performance"]:
                result = _run_timed_check(check_type, identity_func, f"{check_type}_result")
                assert result == f"{check_type}_result"
            
            # Verify all calls were made with correct check types
            assert mock_metrics.record_histogram.call_count == 3
            call_types = [call[1]["tags"]["check_type"] for call in mock_metrics.record_histogram.call_args_list]
            assert set(call_types) == {"security", "style", "performance"}

    def test_run_timed_check_no_args(self):
        """Test timed check with function that takes no arguments."""
        def no_arg_func():
            return "no args"
        
        result = _run_timed_check("test", no_arg_func)
        assert result == "no args"

    def test_run_timed_check_many_args(self):
        """Test timed check with many arguments."""
        def many_args_func(a, b, c, d, e):
            return sum([a, b, c, d, e])
        
        result = _run_timed_check("test", many_args_func, 1, 2, 3, 4, 5)
        assert result == 15