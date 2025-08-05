"""Tests for enhanced error handling and circuit breaker functionality."""

import time
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

from src.autogen_code_review_bot.github_integration import (
    _request_with_retries,
    get_pull_request_diff,
    post_comment
)
from src.autogen_code_review_bot.logging_utils import RequestContext


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in CLOSED state allows requests."""
        # Circuit breaker should be in CLOSED state initially
        # and allow requests to proceed normally
        pass  # Will implement after circuit breaker class
        
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in OPEN state rejects requests."""
        # After failure threshold, circuit breaker should open
        # and reject requests without attempting them
        pass  # Will implement after circuit breaker class
        
    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker in HALF_OPEN state allows limited requests."""
        # After timeout, circuit breaker should allow probe requests
        # to test if service has recovered
        pass  # Will implement after circuit breaker class


class TestDifferentiatedErrorHandling:
    """Test different error handling strategies for different error types."""
    
    @patch('requests.request')
    def test_rate_limit_handling(self, mock_request):
        """Test proper handling of GitHub rate limit responses."""
        # Mock rate limit response with Retry-After header
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '60', 'X-RateLimit-Reset': str(int(time.time()) + 60)}
        mock_response.raise_for_status.side_effect = requests.HTTPError("429 Client Error")
        mock_request.return_value = mock_response
        
        context = RequestContext()
        
        with pytest.raises(requests.HTTPError):
            _request_with_retries(
                "GET",
                "https://api.github.com/test",
                token="test_token",
                retries=1,
                context=context
            )
        
        # Should have attempted request once and not retried immediately due to rate limit
        assert mock_request.call_count == 1
        
    @patch('requests.request')
    def test_server_error_retry(self, mock_request):
        """Test retry behavior for server errors (5xx)."""
        # Mock server error responses that should be retried
        mock_response = Mock()
        mock_response.status_code = 502
        mock_response.raise_for_status.side_effect = requests.HTTPError("502 Bad Gateway")
        mock_request.return_value = mock_response
        
        context = RequestContext()
        
        with pytest.raises(requests.HTTPError):
            _request_with_retries(
                "GET", 
                "https://api.github.com/test",
                token="test_token",
                retries=3,
                context=context
            )
        
        # Should have retried all 3 attempts for server errors
        assert mock_request.call_count == 3
        
    @patch('requests.request')
    def test_client_error_no_retry(self, mock_request):
        """Test that client errors (4xx except 429) are not retried."""
        # Mock client error that shouldn't be retried
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_request.return_value = mock_response
        
        context = RequestContext()
        
        with pytest.raises(requests.HTTPError):
            _request_with_retries(
                "GET",
                "https://api.github.com/test", 
                token="test_token",
                retries=3,
                context=context
            )
        
        # Should have attempted only once, no retries for 404
        assert mock_request.call_count == 1
        
    @patch('requests.request')
    def test_network_error_retry(self, mock_request):
        """Test retry behavior for network errors."""
        # Mock network errors that should be retried
        mock_request.side_effect = [
            ConnectionError("Connection failed"),
            ConnectionError("Connection failed"), 
            Mock(status_code=200, text="success")
        ]
        mock_request.return_value.raise_for_status = Mock()
        
        context = RequestContext()
        
        # Should eventually succeed after retries
        result = _request_with_retries(
            "GET",
            "https://api.github.com/test",
            token="test_token", 
            retries=3,
            context=context
        )
        
        assert mock_request.call_count == 3
        assert result.text == "success"


class TestAdvancedRetryLogic:
    """Test advanced retry logic with jitter and backoff."""
    
    @patch('time.sleep')
    @patch('requests.request')
    def test_jittered_exponential_backoff(self, mock_request, mock_sleep):
        """Test that backoff includes jitter to prevent thundering herd."""
        # Mock failures to trigger retries
        mock_request.side_effect = [
            requests.Timeout("Timeout"),
            requests.Timeout("Timeout"),
            Mock(status_code=200, text="success")
        ]
        mock_request.return_value.raise_for_status = Mock()
        
        context = RequestContext()
        
        _request_with_retries(
            "GET", 
            "https://api.github.com/test",
            token="test_token",
            retries=3,
            context=context
        )
        
        # Should have called sleep twice (between retries)
        assert mock_sleep.call_count == 2
        
        # Verify backoff times include jitter (not exact exponential)
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert len(sleep_calls) == 2
        
        # First retry should be around 0.5s with jitter
        assert 0.1 <= sleep_calls[0] <= 1.0
        
        # Second retry should be around 1.0s with jitter  
        assert 0.5 <= sleep_calls[1] <= 2.0
        
    @patch('requests.request')
    def test_respect_retry_after_header(self, mock_request):
        """Test that Retry-After header is respected."""
        # Mock response with Retry-After header
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.headers = {'Retry-After': '30'}
        mock_response.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
        mock_request.return_value = mock_response
        
        context = RequestContext()
        
        with patch('time.sleep') as mock_sleep:
            with pytest.raises(requests.HTTPError):
                _request_with_retries(
                    "GET",
                    "https://api.github.com/test",
                    token="test_token", 
                    retries=2,
                    context=context
                )
            
            # Should respect the 30-second Retry-After header
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            assert 30 in sleep_calls or any(call >= 30 for call in sleep_calls)


class TestFailureTracking:
    """Test failure rate tracking and recovery monitoring."""
    
    def test_failure_rate_calculation(self):
        """Test that failure rates are calculated correctly."""
        # Will implement after circuit breaker class
        pass
        
    def test_failure_threshold_triggering(self):
        """Test that failure threshold triggers circuit opening."""
        # Will implement after circuit breaker class
        pass
        
    def test_recovery_detection(self):
        """Test that service recovery is detected properly."""
        # Will implement after circuit breaker class
        pass


class TestMetricsIntegration:
    """Test enhanced metrics collection for error scenarios."""
    
    @patch('requests.request')
    def test_error_classification_metrics(self, mock_request):
        """Test that errors are properly classified in metrics."""
        from src.autogen_code_review_bot.metrics import get_metrics_registry
        
        # Mock different error types
        mock_request.side_effect = requests.Timeout("Request timeout")
        
        context = RequestContext()
        registry = get_metrics_registry()
        
        # Clear any existing metrics
        registry._metrics.clear()
        
        with pytest.raises(requests.Timeout):
            _request_with_retries(
                "GET",
                "https://api.github.com/test",
                token="test_token",
                retries=1,
                context=context
            )
        
        # Check that error metrics were recorded
        operation_counter = registry.get_metric("operation_total")
        assert operation_counter is not None
        
        # Should have recorded timeout error
        timeout_count = operation_counter.get_value({"operation": "github_api_request", "status": "error"})
        assert timeout_count > 0
        
    def test_circuit_breaker_state_metrics(self):
        """Test that circuit breaker state changes are recorded."""
        # Will implement after circuit breaker class
        pass


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @patch('requests.request')
    def test_get_pull_request_diff_with_retries(self, mock_request):
        """Test get_pull_request_diff with error handling."""
        # Mock initial failure then success
        mock_request.side_effect = [
            requests.ConnectionError("Network error"),
            Mock(status_code=200, text="diff content")
        ]
        mock_request.return_value.raise_for_status = Mock()
        
        context = RequestContext()
        
        result = get_pull_request_diff(
            "owner/repo", 
            123,
            token="test_token",
            context=context
        )
        
        assert result == "diff content"
        assert mock_request.call_count == 2
        
    @patch('requests.request')  
    def test_post_comment_with_circuit_breaker(self, mock_request):
        """Test post_comment with circuit breaker protection."""
        # Mock repeated failures that should open circuit breaker
        mock_request.side_effect = requests.HTTPError("500 Internal Server Error")
        
        context = RequestContext()
        
        # Multiple attempts should eventually be blocked by circuit breaker
        for i in range(5):
            with pytest.raises((requests.HTTPError, Exception)):
                post_comment(
                    "owner/repo",
                    123, 
                    "test comment",
                    token="test_token",
                    context=context
                )
        
        # After circuit opens, requests should be blocked
        # (specific assertion depends on circuit breaker implementation)


class TestErrorRecovery:
    """Test error recovery and service restoration."""
    
    def test_gradual_recovery_after_failures(self):
        """Test that service recovery is handled gracefully."""
        # Will implement after circuit breaker class
        pass
        
    def test_circuit_breaker_reset_after_success(self):
        """Test that circuit breaker resets after successful requests."""
        # Will implement after circuit breaker class  
        pass


if __name__ == "__main__":
    pytest.main([__file__])