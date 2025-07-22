"""Comprehensive tests for GitHub integration error handling."""

import json
import time
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests

from autogen_code_review_bot.github_integration import (
    _request_with_retries,
    get_pull_request_diff,
    post_comment,
    analyze_and_comment,
    GitHubError,
    RateLimitError,
    GitHubConnectionError,
    CircuitBreakerError
)


class TestErrorClassification:
    """Test proper error classification and handling."""
    
    def test_rate_limit_error_detection(self):
        """Test that 429 responses are classified as rate limit errors."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {
                'X-RateLimit-Reset': str(int(time.time()) + 3600),
                'X-RateLimit-Remaining': '0'
            }
            mock_response.raise_for_status.side_effect = requests.HTTPError("429 rate limit")
            mock_request.return_value = mock_response
            
            with pytest.raises(RateLimitError) as exc_info:
                _request_with_retries("GET", "https://api.github.com/test", token="fake")
            
            assert "rate limit exceeded" in str(exc_info.value).lower()
            assert exc_info.value.reset_time is not None

    def test_server_error_retry_logic(self):
        """Test that 5xx errors trigger appropriate retry logic."""
        with patch('requests.request') as mock_request:
            # First call fails with 502, second succeeds
            mock_response_fail = Mock()
            mock_response_fail.status_code = 502
            mock_response_fail.raise_for_status.side_effect = requests.HTTPError("502 bad gateway")
            
            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.raise_for_status.return_value = None
            
            mock_request.side_effect = [mock_response_fail, mock_response_success]
            
            result = _request_with_retries("GET", "https://api.github.com/test", token="fake", retries=2)
            assert result.status_code == 200

    def test_authentication_error_handling(self):
        """Test that 401/403 errors are handled appropriately."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = requests.HTTPError("401 unauthorized")
            mock_request.return_value = mock_response
            
            with pytest.raises(GitHubError) as exc_info:
                _request_with_retries("GET", "https://api.github.com/test", token="invalid")
            
            assert "authentication failed" in str(exc_info.value).lower()

    def test_network_error_handling(self):
        """Test that network errors are properly classified."""
        with patch('requests.request') as mock_request:
            mock_request.side_effect = requests.ConnectionError("Connection failed")
            
            with pytest.raises(GitHubConnectionError):
                _request_with_retries("GET", "https://api.github.com/test", token="fake")


class TestRetryLogic:
    """Test enhanced retry logic for different error types."""
    
    def test_exponential_backoff_calculation(self):
        """Test that exponential backoff is calculated correctly."""
        with patch('time.sleep') as mock_sleep:
            with patch('requests.request') as mock_request:
                mock_request.side_effect = requests.ConnectionError("Network error")
                
                try:
                    _request_with_retries("GET", "https://api.github.com/test", token="fake", retries=3)
                except:
                    pass
                
                # Verify exponential backoff: 0.5, 1.0, 2.0
                expected_sleeps = [0.5, 1.0, 2.0]
                actual_sleeps = [call[0][0] for call in mock_sleep.call_args_list]
                assert actual_sleeps == expected_sleeps

    def test_rate_limit_specific_backoff(self):
        """Test that rate limit errors use GitHub's reset time."""
        reset_time = int(time.time()) + 60  # 1 minute from now
        
        with patch('time.sleep') as mock_sleep:
            with patch('requests.request') as mock_request:
                mock_response = Mock()
                mock_response.status_code = 429
                mock_response.headers = {
                    'X-RateLimit-Reset': str(reset_time),
                    'X-RateLimit-Remaining': '0'
                }
                mock_response.raise_for_status.side_effect = requests.HTTPError("429 rate limit")
                mock_request.return_value = mock_response
                
                try:
                    _request_with_retries("GET", "https://api.github.com/test", token="fake", retries=1)
                except:
                    pass
                
                # Should sleep until rate limit reset (max 60 seconds)
                assert mock_sleep.call_count == 1
                sleep_time = mock_sleep.call_args[0][0]
                assert 0 < sleep_time <= 60

    def test_no_retry_for_client_errors(self):
        """Test that 4xx errors (except 429) don't retry."""
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 not found")
            mock_request.return_value = mock_response
            
            with pytest.raises(GitHubError):
                _request_with_retries("GET", "https://api.github.com/test", token="fake", retries=3)
            
            # Should only make 1 request (no retries for 404)
            assert mock_request.call_count == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after consecutive failures."""
        # This will be implemented as part of the enhancement
        pass
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        # This will be implemented as part of the enhancement
        pass


class TestFallbackModes:
    """Test fallback modes for partial failures."""
    
    def test_comment_posting_fallback_to_summary(self):
        """Test fallback to shorter comment when full comment fails."""
        with patch('autogen_code_review_bot.github_integration._request_with_retries') as mock_request:
            # First call (full comment) fails due to size
            mock_request.side_effect = [
                GitHubError("Comment too large"),
                Mock()  # Second call (summary) succeeds
            ]
            
            with patch('autogen_code_review_bot.pr_analysis.analyze_pr') as mock_analyze:
                mock_result = Mock()
                mock_result.security.output = "Security issues found"
                mock_result.style.output = "Style issues found"
                mock_result.performance.output = "Performance issues found"
                mock_analyze.return_value = mock_result
                
                # Should not raise exception due to fallback
                result = analyze_and_comment("/fake/path", "owner/repo", 123)
                assert result is not None

    def test_partial_analysis_on_github_failure(self):
        """Test that analysis continues even if GitHub operations fail."""
        with patch('autogen_code_review_bot.github_integration.get_pull_request_diff') as mock_diff:
            mock_diff.side_effect = GitHubConnectionError("GitHub unavailable")
            
            with patch('autogen_code_review_bot.pr_analysis.analyze_pr') as mock_analyze:
                mock_analyze.return_value = Mock()
                
                # Should still perform local analysis even if diff fetch fails
                with pytest.raises(GitHubConnectionError):
                    analyze_and_comment("/fake/path", "owner/repo", 123)
                
                # Analysis should still have been attempted
                mock_analyze.assert_called_once()


class TestErrorMessages:
    """Test improved error messages and context."""
    
    def test_rate_limit_error_includes_reset_time(self):
        """Test that rate limit errors include helpful reset time."""
        reset_time = int(time.time()) + 1800  # 30 minutes
        
        with patch('requests.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {
                'X-RateLimit-Reset': str(reset_time),
                'X-RateLimit-Remaining': '0'
            }
            mock_response.raise_for_status.side_effect = requests.HTTPError("429 rate limit")
            mock_request.return_value = mock_response
            
            with pytest.raises(RateLimitError) as exc_info:
                _request_with_retries("GET", "https://api.github.com/test", token="fake")
            
            error_msg = str(exc_info.value)
            assert "rate limit" in error_msg.lower()
            assert str(reset_time) in error_msg or "30 min" in error_msg

    def test_network_error_includes_retry_suggestion(self):
        """Test that network errors include helpful retry suggestions."""
        with patch('requests.request') as mock_request:
            mock_request.side_effect = requests.ConnectionError("Connection timeout")
            
            with pytest.raises(GitHubConnectionError) as exc_info:
                _request_with_retries("GET", "https://api.github.com/test", token="fake")
            
            error_msg = str(exc_info.value)
            assert "connection" in error_msg.lower()
            assert any(word in error_msg.lower() for word in ["retry", "network", "timeout"])


class TestMetricsIntegration:
    """Test that error handling integrates with metrics collection."""
    
    def test_error_metrics_recorded(self):
        """Test that different error types are recorded in metrics."""
        with patch('autogen_code_review_bot.github_integration.metrics') as mock_metrics:
            with patch('requests.request') as mock_request:
                mock_request.side_effect = requests.ConnectionError("Network error")
                
                try:
                    _request_with_retries("GET", "https://api.github.com/test", token="fake")
                except:
                    pass
                
                # Verify error metrics were recorded
                mock_metrics.record_counter.assert_called_with(
                    "github_api_errors_total", 1, 
                    tags={"error_type": "connection_error", "api_operation": "GET"}
                )

    def test_retry_metrics_recorded(self):
        """Test that retry attempts are recorded in metrics."""
        with patch('autogen_code_review_bot.github_integration.metrics') as mock_metrics:
            with patch('requests.request') as mock_request:
                # Fail twice, succeed on third attempt
                mock_request.side_effect = [
                    requests.HTTPError("502 bad gateway"),
                    requests.HTTPError("502 bad gateway"),
                    Mock(status_code=200, raise_for_status=lambda: None)
                ]
                
                _request_with_retries("GET", "https://api.github.com/test", token="fake", retries=3)
                
                # Verify retry metrics were recorded
                retry_calls = [call for call in mock_metrics.record_counter.call_args_list 
                              if "github_api_retries_total" in call[0]]
                assert len(retry_calls) >= 1


if __name__ == "__main__":
    pytest.main([__file__])