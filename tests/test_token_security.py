"""Test token security and masking functionality."""

import pytest
from src.autogen_code_review_bot.token_security import (
    TokenMasker, 
    mask_token_in_url, 
    mask_sensitive_in_text, 
    safe_log_dict, 
    safe_exception_str
)


class TestTokenMasker:
    """Test TokenMasker class functionality."""

    def test_mask_github_tokens(self):
        """Test masking of various GitHub token formats."""
        test_cases = [
            ("ghp_1234567890123456789012345678901234567890", "***GITHUB_TOKEN***"),
            ("ghs_1234567890123456789012345678901234567890", "***GITHUB_TOKEN***"),
            ("gho_1234567890123456789012345678901234567890", "***GITHUB_TOKEN***"),
            ("ghu_1234567890123456789012345678901234567890", "***GITHUB_TOKEN***"),
            ("ghr_1234567890123456789012345678901234567890", "***GITHUB_TOKEN***"),
            ("1234567890abcdef1234567890abcdef12345678", "***TOKEN***"),  # 40-char hex
        ]
        
        for token, expected_mask in test_cases:
            text = f"The token is {token} for authentication"
            result = TokenMasker.mask_sensitive_data(text)
            assert token not in result
            assert expected_mask in result

    def test_mask_authorization_headers(self):
        """Test masking of Authorization headers."""
        test_cases = [
            ("Authorization: token ghp_1234567890123456789012345678901234567890", "Authorization: token ***"),
            ("Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9", "Authorization: Bearer ***"),
            ("authorization: TOKEN abc123", "Authorization: token ***"),
        ]
        
        for auth_header, expected in test_cases:
            result = TokenMasker.mask_sensitive_data(auth_header)
            assert "ghp_" not in result
            assert "eyJ0eXAi" not in result
            assert "abc123" not in result

    def test_mask_basic_auth_urls(self):
        """Test masking of basic auth in URLs."""
        test_cases = [
            ("https://user:pass@github.com/repo", "https://***:***@github.com/repo"),
            ("https://token:x-oauth-basic@github.com/repo", "https://***:***@github.com/repo"),
        ]
        
        for url, expected in test_cases:
            result = TokenMasker.mask_sensitive_data(url)
            assert "user:pass" not in result
            assert "token:x-oauth-basic" not in result
            assert "***:***@" in result

    def test_mask_secrets_in_json(self):
        """Test masking of secrets in JSON-like text."""
        test_cases = [
            ('{"secret": "mysecret123"}', 'secret: "***"'),
            ('"webhook_secret": "abcdef"', 'webhook_secret: "***"'),
            ("password='mypass'", 'password: "***"'),
            ("api_key=sk-1234567890", '***SECRET***'),
        ]
        
        for text, expected_pattern in test_cases:
            result = TokenMasker.mask_sensitive_data(text)
            assert "mysecret123" not in result
            assert "abcdef" not in result
            assert "mypass" not in result
            assert "sk-1234567890" not in result

    def test_mask_url_with_token(self):
        """Test URL masking with specific token."""
        token = "ghp_1234567890123456789012345678901234567890"
        url = f"https://api.github.com/repos/owner/repo?access_token={token}"
        
        result = TokenMasker.mask_url(url, token)
        
        assert token not in result
        assert "***" in result

    def test_mask_url_with_auth(self):
        """Test URL masking with authentication."""
        url = "https://token123:x-oauth-basic@api.github.com/repos/owner/repo"
        
        result = TokenMasker.mask_url(url)
        
        assert "token123" not in result
        assert "***:***@" in result

    def test_mask_dict_sensitive_keys(self):
        """Test masking of sensitive keys in dictionaries."""
        test_dict = {
            "username": "john",
            "password": "secret123",
            "api_token": "token456",
            "webhook_secret": "webhook789",
            "github_key": "key123",
            "auth_header": "Bearer token123",
            "normal_field": "safe_value"
        }
        
        result = TokenMasker.mask_dict(test_dict)
        
        # Sensitive fields should be masked
        assert result["password"] == "***"
        assert result["api_token"] == "***"
        assert result["webhook_secret"] == "***"
        assert result["github_key"] == "***"
        assert result["auth_header"] == "***"
        
        # Non-sensitive fields should remain
        assert result["username"] == "john"
        assert result["normal_field"] == "safe_value"

    def test_mask_nested_dict(self):
        """Test masking of nested dictionaries."""
        test_dict = {
            "config": {
                "github": {
                    "token": "secret_token",
                    "webhook_secret": "webhook_secret"
                },
                "database": {
                    "password": "db_pass",
                    "host": "localhost"
                }
            },
            "public_info": "safe"
        }
        
        result = TokenMasker.mask_dict(test_dict)
        
        assert result["config"]["github"]["token"] == "***"
        assert result["config"]["github"]["webhook_secret"] == "***"
        assert result["config"]["database"]["password"] == "***"
        assert result["config"]["database"]["host"] == "localhost"
        assert result["public_info"] == "safe"

    def test_mask_list_with_sensitive_data(self):
        """Test masking of lists containing sensitive data."""
        test_list = [
            "normal_string",
            {"token": "secret123"},
            ["nested", {"password": "pass123"}],
            "Authorization: Bearer token456"
        ]
        
        result = TokenMasker.mask_list(test_list)
        
        assert result[0] == "normal_string"
        assert result[1]["token"] == "***"
        assert result[2][1]["password"] == "***"
        assert "token456" not in result[3]

    def test_mask_exception_message(self):
        """Test masking of exception messages."""
        try:
            raise ValueError("Authentication failed with token ghp_1234567890123456789012345678901234567890")
        except ValueError as e:
            result = TokenMasker.mask_exception_message(e)
            assert "ghp_" not in result
            assert "***GITHUB_TOKEN***" in result

    def test_mask_logging_args(self):
        """Test masking of logging arguments."""
        args = ("Request failed", {"token": "secret123"})
        kwargs = {"url": "https://token:pass@api.com", "error": "Auth failed with key abc123"}
        
        masked_args, masked_kwargs = TokenMasker.mask_logging_args(*args, **kwargs)
        
        assert masked_args[1]["token"] == "***"
        assert "token:pass" not in masked_kwargs["url"]
        assert "abc123" not in masked_kwargs["error"]

    def test_preserve_original_structure(self):
        """Test that original data structure types are preserved."""
        original_tuple = ("data", {"token": "secret"})
        result = TokenMasker.mask_list(original_tuple)
        assert isinstance(result, tuple)
        
        original_dict = {"nested": {"token": "secret"}}
        result = TokenMasker.mask_dict(original_dict)
        assert isinstance(result, dict)
        assert isinstance(result["nested"], dict)

    def test_handle_non_string_values(self):
        """Test handling of non-string values in data structures."""
        test_dict = {
            "number": 123,
            "boolean": True,
            "none_value": None,
            "list": [1, 2, 3],
            "token": "secret123"  # This should be masked
        }
        
        result = TokenMasker.mask_dict(test_dict)
        
        assert result["number"] == 123
        assert result["boolean"] is True
        assert result["none_value"] is None
        assert result["list"] == [1, 2, 3]
        assert result["token"] == "***"

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test mask_token_in_url
        url = "https://api.github.com/repos/owner/repo"
        token = "ghp_1234567890123456789012345678901234567890"
        result = mask_token_in_url(f"{url}?token={token}", token)
        assert token not in result

        # Test mask_sensitive_in_text
        text = "Password is secret123"
        result = mask_sensitive_in_text(text)
        assert "secret123" not in result

        # Test safe_log_dict
        data = {"password": "secret"}
        result = safe_log_dict(data)
        assert result["password"] == "***"

        # Test safe_exception_str
        try:
            raise ValueError("Error with token abc123")
        except ValueError as e:
            result = safe_exception_str(e)
            assert "abc123" not in result

    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        # Empty string
        assert TokenMasker.mask_sensitive_data("") == ""
        
        # None input (converted to string)
        assert TokenMasker.mask_sensitive_data(None) == "None"
        
        # Empty dict
        assert TokenMasker.mask_dict({}) == {}
        
        # Empty list
        assert TokenMasker.mask_list([]) == []
        
        # Empty URL
        assert TokenMasker.mask_url("") == ""
        assert TokenMasker.mask_url(None) is None

    def test_complex_real_world_scenario(self):
        """Test a complex real-world scenario with mixed sensitive data."""
        log_data = {
            "timestamp": "2025-01-01T00:00:00Z",
            "level": "error",
            "message": "GitHub API request failed",
            "request_details": {
                "url": "https://ghp_abcd1234567890123456789012345678901234@api.github.com/repos/owner/repo",
                "headers": {
                    "Authorization": "token ghp_abcd1234567890123456789012345678901234",
                    "User-Agent": "MyApp/1.0"
                },
                "method": "POST"
            },
            "error": "Authentication failed with token ghp_abcd1234567890123456789012345678901234",
            "context": [
                "Step 1: Generate token",
                {"webhook_secret": "mysecret123"},
                "Step 3: Make request"
            ]
        }
        
        result = TokenMasker.mask_dict(log_data)
        
        # Check that all instances of the token are masked
        assert "ghp_abcd1234567890123456789012345678901234" not in str(result)
        assert "mysecret123" not in str(result)
        
        # Check that structure is preserved
        assert result["timestamp"] == "2025-01-01T00:00:00Z"
        assert result["level"] == "error"
        assert result["request_details"]["method"] == "POST"
        assert result["request_details"]["headers"]["User-Agent"] == "MyApp/1.0"
        
        # Check that tokens are masked
        assert result["request_details"]["headers"]["Authorization"] == "***"
        assert result["context"][1]["webhook_secret"] == "***"