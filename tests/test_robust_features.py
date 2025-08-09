"""
Tests for Generation 2 robustness features: health checking, rate limiting, 
validation, error handling, and reliability improvements.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime, timezone

from autogen_code_review_bot.health_checker import (
    HealthChecker, HealthStatus, HealthCheckResult, create_health_checker
)
from autogen_code_review_bot.rate_limiter import (
    InMemoryRateLimiter, RateLimitConfig, RateLimitResult,
    create_rate_limiter, AdaptiveRateLimiter
)
from autogen_code_review_bot.enhanced_validation import (
    SecurityValidator, DataValidator, InputSanitizer,
    validate_webhook_signature
)
from autogen_code_review_bot.exceptions import ValidationError, SecurityError


class TestHealthChecker:
    """Test health monitoring functionality."""
    
    def test_health_check_result_creation(self):
        """Test HealthCheckResult model."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            duration_ms=50.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.duration_ms == 50.0
    
    def test_health_check_result_to_dict(self):
        """Test HealthCheckResult serialization."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.DEGRADED,
            message="Warning",
            duration_ms=100.0,
            timestamp=datetime.now(timezone.utc),
            metadata={"cpu": 75.5}
        )
        
        result_dict = result.to_dict()
        assert result_dict["name"] == "test_check"
        assert result_dict["status"] == "degraded"
        assert result_dict["message"] == "Warning"
        assert result_dict["metadata"]["cpu"] == 75.5
    
    def test_health_checker_creation(self):
        """Test health checker initialization."""
        config = {"test": "value"}
        checker = create_health_checker(config)
        
        assert isinstance(checker, HealthChecker)
        assert checker.config == config
        assert len(checker.checks) > 0  # Should have default checks
    
    @pytest.mark.asyncio
    async def test_health_checker_run_all_checks(self):
        """Test running all health checks."""
        checker = HealthChecker()
        
        with patch.object(checker, 'checks', []):  # Remove default checks for clean test
            # Add a simple mock check
            mock_check = Mock(return_value=HealthCheckResult(
                name="mock_check",
                status=HealthStatus.HEALTHY,
                message="Mock check passed",
                duration_ms=10.0,
                timestamp=datetime.now(timezone.utc)
            ))
            checker.checks.append(mock_check)
            
            result = await checker.run_all_checks()
            
            assert result["status"] == "healthy"
            assert result["summary"]["total"] == 1
            assert result["summary"]["healthy"] == 1
            assert len(result["checks"]) == 1
    
    @patch('psutil.virtual_memory')
    def test_memory_check_healthy(self, mock_memory):
        """Test memory health check - healthy status."""
        mock_memory.return_value = Mock(percent=50.0, available=8000000000, total=16000000000)
        
        checker = HealthChecker()
        result = checker._check_memory_usage()
        
        assert result.status == HealthStatus.HEALTHY
        assert "Memory usage normal" in result.message
        assert result.metadata["memory_percent"] == 50.0
    
    @patch('psutil.virtual_memory') 
    def test_memory_check_degraded(self, mock_memory):
        """Test memory health check - degraded status."""
        mock_memory.return_value = Mock(percent=80.0, available=2000000000, total=10000000000)
        
        checker = HealthChecker()
        result = checker._check_memory_usage()
        
        assert result.status == HealthStatus.DEGRADED
        assert "Memory usage elevated" in result.message
    
    @patch('psutil.disk_usage')
    def test_disk_check_critical(self, mock_disk):
        """Test disk health check - critical status."""
        mock_disk.return_value = Mock(used=950000000, total=1000000000, free=50000000)
        
        checker = HealthChecker()
        result = checker._check_disk_space()
        
        assert result.status == HealthStatus.CRITICAL
        assert "Disk usage critical" in result.message


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limit_config(self):
        """Test rate limit configuration."""
        config = RateLimitConfig(requests=100, window=3600)
        
        assert config.requests == 100
        assert config.window == 3600
        assert config.burst == 100  # Should default to requests
    
    def test_rate_limit_config_with_burst(self):
        """Test rate limit configuration with explicit burst."""
        config = RateLimitConfig(requests=100, window=3600, burst=150)
        
        assert config.requests == 100
        assert config.window == 3600
        assert config.burst == 150
    
    def test_in_memory_rate_limiter_allow(self):
        """Test in-memory rate limiter allows requests within limit."""
        limiter = InMemoryRateLimiter()
        config = RateLimitConfig(requests=5, window=60)
        
        # First request should be allowed
        result = limiter.check_limit("test_key", config)
        
        assert result.allowed is True
        assert result.remaining == 4
        assert result.retry_after is None
    
    def test_in_memory_rate_limiter_deny(self):
        """Test in-memory rate limiter denies requests over limit."""
        limiter = InMemoryRateLimiter()
        config = RateLimitConfig(requests=2, window=60)
        
        # Make requests up to limit
        limiter.check_limit("test_key", config)
        limiter.check_limit("test_key", config)
        
        # Third request should be denied
        result = limiter.check_limit("test_key", config)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
    
    def test_in_memory_rate_limiter_sliding_window(self):
        """Test sliding window behavior."""
        limiter = InMemoryRateLimiter()
        config = RateLimitConfig(requests=2, window=1)  # 1 second window
        
        # Make requests
        result1 = limiter.check_limit("test_key", config)
        result2 = limiter.check_limit("test_key", config)
        result3 = limiter.check_limit("test_key", config)
        
        assert result1.allowed is True
        assert result2.allowed is True
        assert result3.allowed is False
        
        # Wait for window to slide
        time.sleep(1.1)
        
        # Should be allowed again
        result4 = limiter.check_limit("test_key", config)
        assert result4.allowed is True
    
    def test_rate_limiter_factory(self):
        """Test rate limiter factory function."""
        memory_limiter = create_rate_limiter("memory")
        assert isinstance(memory_limiter, InMemoryRateLimiter)
        
        with pytest.raises(ValidationError):
            create_rate_limiter("invalid_backend")
    
    def test_adaptive_rate_limiter(self):
        """Test adaptive rate limiting."""
        base_limiter = InMemoryRateLimiter()
        adaptive_limiter = AdaptiveRateLimiter(base_limiter)
        
        config = RateLimitConfig(requests=10, window=60)
        
        # Test with low load (should not adjust)
        result = adaptive_limiter.check_limit("test_key", config, system_load=0.3)
        assert result.allowed is True
        
        # Test with high load (should reduce limits)
        result = adaptive_limiter.check_limit("test_key", config, system_load=0.9)
        assert result.allowed is True  # First request still allowed


class TestSecurityValidator:
    """Test security validation functionality."""
    
    def test_validate_safe_file_path(self):
        """Test validation of safe file paths."""
        safe_paths = [
            "src/main.py",
            "docs/readme.md",
            "tests/test_file.py"
        ]
        
        for path in safe_paths:
            result = SecurityValidator.validate_file_path(path)
            assert isinstance(result, str)
    
    def test_validate_dangerous_file_path(self):
        """Test rejection of dangerous file paths."""
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\cmd.exe"
        ]
        
        for path in dangerous_paths:
            with pytest.raises((ValidationError, SecurityError)):
                SecurityValidator.validate_file_path(path)
    
    def test_validate_dangerous_extensions(self):
        """Test rejection of dangerous file extensions."""
        dangerous_files = [
            "malware.exe",
            "script.bat", 
            "virus.dll",
            "backdoor.sh"
        ]
        
        for file in dangerous_files:
            with pytest.raises(SecurityError):
                SecurityValidator.validate_file_path(file)
    
    def test_validate_code_content_safe(self):
        """Test validation of safe code content."""
        safe_code = """
        def hello_world():
            print("Hello, World!")
            return "success"
        
        if __name__ == "__main__":
            hello_world()
        """
        
        result = SecurityValidator.validate_code_content(safe_code, "test.py")
        assert result is True
    
    def test_validate_code_content_suspicious(self):
        """Test detection of suspicious code patterns."""
        suspicious_code = """
        import os
        os.system("rm -rf /")  # This should be flagged
        """
        
        # Should not raise exception but log warning
        result = SecurityValidator.validate_code_content(suspicious_code, "dangerous.py")
        assert result is True  # Currently logs but doesn't block
    
    def test_validate_url_safe(self):
        """Test validation of safe URLs."""
        safe_urls = [
            "https://api.github.com/repos/user/repo",
            "http://example.com/webhook",
            "https://hooks.slack.com/webhook"
        ]
        
        for url in safe_urls:
            result = SecurityValidator.validate_url(url)
            assert result == url
    
    def test_validate_url_unsafe(self):
        """Test rejection of unsafe URLs."""
        unsafe_urls = [
            "ftp://internal.server.com/file",
            "file:///etc/passwd",
            "https://localhost:8080/admin",
            "http://127.0.0.1/internal"
        ]
        
        for url in unsafe_urls:
            with pytest.raises((ValidationError, SecurityError)):
                SecurityValidator.validate_url(url)
    
    def test_sanitize_input_basic(self):
        """Test basic input sanitization."""
        dirty_input = "<script>alert('xss')</script>Hello World"
        clean_input = SecurityValidator.sanitize_input(dirty_input)
        
        assert "<script>" not in clean_input
        assert "Hello World" in clean_input
    
    def test_sanitize_input_too_long(self):
        """Test input length validation."""
        long_input = "x" * 2000
        
        with pytest.raises(ValidationError):
            SecurityValidator.sanitize_input(long_input, max_length=1000)


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_validate_json_valid(self):
        """Test validation of valid JSON."""
        valid_json = '{"name": "test", "value": 123, "active": true}'
        result = DataValidator.validate_json(valid_json)
        
        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["active"] is True
    
    def test_validate_json_invalid(self):
        """Test rejection of invalid JSON."""
        invalid_json = '{"name": "test", "value": 123, "active": true'  # Missing closing brace
        
        with pytest.raises(ValidationError):
            DataValidator.validate_json(invalid_json)
    
    def test_validate_json_too_large(self):
        """Test rejection of oversized JSON."""
        large_data = {"data": "x" * 1000000}
        large_json = json.dumps(large_data)
        
        with pytest.raises(ValidationError):
            DataValidator.validate_json(large_json, max_size=1000)
    
    def test_validate_github_webhook_payload(self):
        """Test validation of GitHub webhook payload."""
        valid_payload = {
            "action": "opened",
            "sender": {"login": "testuser"},
            "repository": {"full_name": "testuser/testrepo"}
        }
        
        result = DataValidator.validate_github_webhook_payload(valid_payload)
        assert result == valid_payload
    
    def test_validate_github_webhook_payload_invalid(self):
        """Test rejection of invalid webhook payload."""
        invalid_payloads = [
            {},  # Missing required fields
            {"action": "opened"},  # Missing sender
            {"action": "opened", "sender": {}},  # Invalid sender
            {"action": "opened", "sender": {"login": "user"}, "repository": {}}  # Invalid repo
        ]
        
        for payload in invalid_payloads:
            with pytest.raises(ValidationError):
                DataValidator.validate_github_webhook_payload(payload)


class TestInputSanitizer:
    """Test input sanitization utilities."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        unsafe_filenames = [
            "../../etc/passwd",
            "file<with>unsafe:chars",
            "file|with*wildcards?",
            "   .hidden_file   ",
            ""
        ]
        
        expected_results = [
            "____etc_passwd",
            "file_with_unsafe_chars",
            "file_with_wildcards_",
            "_hidden_file",
            "unnamed_file"
        ]
        
        for unsafe, expected in zip(unsafe_filenames, expected_results):
            result = InputSanitizer.sanitize_filename(unsafe)
            assert result == expected
    
    def test_sanitize_log_message(self):
        """Test log message sanitization."""
        unsafe_message = "Error occurred\nInjected log entry\rAnother injection"
        safe_message = InputSanitizer.sanitize_log_message(unsafe_message)
        
        assert "\n" not in safe_message
        assert "\r" not in safe_message
        assert "\\n" in safe_message
        assert "\\r" in safe_message
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing."""
        sensitive_data = "secret_api_key_12345"
        hashed = InputSanitizer.hash_sensitive_data(sensitive_data)
        
        assert len(hashed) == 8
        assert hashed != sensitive_data
        assert hashed.isalnum()


class TestWebhookSignatureValidation:
    """Test webhook signature validation."""
    
    def test_valid_signature(self):
        """Test valid webhook signature validation."""
        payload = b'{"test": "data"}'
        secret = "my_secret_key"
        
        import hmac
        import hashlib
        signature = 'sha256=' + hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        result = validate_webhook_signature(payload, signature, secret)
        assert result is True
    
    def test_invalid_signature(self):
        """Test invalid webhook signature validation."""
        payload = b'{"test": "data"}'
        secret = "my_secret_key"
        invalid_signature = "sha256=invalid_signature_hash"
        
        result = validate_webhook_signature(payload, invalid_signature, secret)
        assert result is False
    
    def test_missing_signature(self):
        """Test missing signature validation."""
        payload = b'{"test": "data"}'
        secret = "my_secret_key"
        
        result = validate_webhook_signature(payload, "", secret)
        assert result is False


# Test utilities - no fixtures needed since we use datetime directly