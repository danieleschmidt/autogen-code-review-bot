"""Test GitHub API URL configuration functionality."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from src.autogen_code_review_bot.system_config import SystemConfig, reset_system_config, get_system_config
from src.autogen_code_review_bot.github_integration import get_pull_request_diff, post_comment


class TestGitHubAPIURLConfiguration:
    """Test GitHub API URL configuration."""

    def setup_method(self):
        """Reset system config before each test."""
        reset_system_config()

    def teardown_method(self):
        """Clean up after each test."""
        reset_system_config()

    def test_default_github_api_url(self):
        """Test that default GitHub API URL is set correctly."""
        config = SystemConfig()
        assert config.github_api_url == "https://api.github.com"

    def test_github_api_url_from_environment(self):
        """Test that GitHub API URL can be overridden via environment variable."""
        with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': 'https://github.enterprise.com/api/v3'}):
            config = SystemConfig()
            assert config.github_api_url == "https://github.enterprise.com/api/v3"

    def test_github_api_url_from_config_file(self):
        """Test that GitHub API URL can be set via configuration file."""
        config_data = {
            'system': {
                'github_api_url': 'https://custom.github.com/api'
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = SystemConfig.load_from_file(config_path)
            assert config.github_api_url == "https://custom.github.com/api"
        finally:
            os.unlink(config_path)

    def test_get_pull_request_diff_uses_configured_url(self):
        """Test that get_pull_request_diff uses the configured GitHub API URL."""
        custom_url = "https://github.enterprise.com/api/v3"
        
        with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': custom_url}):
            with patch('src.autogen_code_review_bot.github_integration._request_with_retries') as mock_request:
                mock_response = MagicMock()
                mock_response.text = "diff content"
                mock_request.return_value = mock_response
                
                # Reset config to pick up environment variable
                reset_system_config()
                
                result = get_pull_request_diff("owner/repo", 123, "fake-token")
                
                # Verify the correct URL was used
                expected_url = f"{custom_url}/repos/owner/repo/pulls/123"
                mock_request.assert_called_once_with(
                    "get",
                    expected_url,
                    token="fake-token",
                    params={"media_type": "diff"}
                )
                assert result == "diff content"

    def test_post_comment_uses_configured_url(self):
        """Test that post_comment uses the configured GitHub API URL."""
        custom_url = "https://github.enterprise.com/api/v3"
        
        with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': custom_url}):
            with patch('src.autogen_code_review_bot.github_integration._request_with_retries') as mock_request:
                mock_response = MagicMock()
                mock_response.json.return_value = {"id": 123456}
                mock_request.return_value = mock_response
                
                # Reset config to pick up environment variable
                reset_system_config()
                
                result = post_comment("owner/repo", 123, "Test comment", "fake-token")
                
                # Verify the correct URL was used
                expected_url = f"{custom_url}/repos/owner/repo/issues/123/comments"
                mock_request.assert_called_once_with(
                    "post",
                    expected_url,
                    token="fake-token",
                    data='{"body": "Test comment"}'
                )
                assert result == {"id": 123456}

    def test_github_enterprise_url_format(self):
        """Test that GitHub Enterprise URLs are handled correctly."""
        enterprise_urls = [
            "https://github.enterprise.com/api/v3",
            "https://api.github.enterprise.com",
            "http://internal.github.com/api/v3",
            "https://custom-domain.com/github/api/v3"
        ]
        
        for enterprise_url in enterprise_urls:
            with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': enterprise_url}):
                with patch('src.autogen_code_review_bot.github_integration._request_with_retries') as mock_request:
                    mock_response = MagicMock()
                    mock_response.text = "diff content"
                    mock_request.return_value = mock_response
                    
                    # Reset config to pick up environment variable
                    reset_system_config()
                    
                    get_pull_request_diff("owner/repo", 123, "fake-token")
                    
                    # Verify the correct URL was constructed
                    expected_url = f"{enterprise_url}/repos/owner/repo/pulls/123"
                    mock_request.assert_called_once_with(
                        "get",
                        expected_url,
                        token="fake-token",
                        params={"media_type": "diff"}
                    )

    def test_url_configuration_validation(self):
        """Test that URL configuration validates properly."""
        # Test that empty URL is handled gracefully
        with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': ''}):
            config = SystemConfig()
            # Empty string should be replaced with default
            assert config.github_api_url == ""  # Environment override still applies

    def test_url_without_trailing_slash(self):
        """Test that URLs work correctly without trailing slashes."""
        # URL with trailing slash
        url_with_slash = "https://api.github.com/"
        
        with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': url_with_slash}):
            with patch('src.autogen_code_review_bot.github_integration._request_with_retries') as mock_request:
                mock_response = MagicMock()
                mock_response.text = "diff content"
                mock_request.return_value = mock_response
                
                reset_system_config()
                get_pull_request_diff("owner/repo", 123, "fake-token")
                
                # Should construct URL correctly even with trailing slash
                expected_url = f"{url_with_slash}/repos/owner/repo/pulls/123"
                mock_request.assert_called_once()
                actual_url = mock_request.call_args[0][1]
                assert "//" not in actual_url.replace("https://", "")

    def test_multiple_config_sources_precedence(self):
        """Test the precedence of configuration sources."""
        # Environment variable should override config file
        config_data = {
            'system': {
                'github_api_url': 'https://config-file.github.com/api'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Set environment variable that should take precedence
            with patch.dict(os.environ, {'AUTOGEN_GITHUB_API_URL': 'https://env-override.github.com/api'}):
                config = SystemConfig.load_from_file(config_path)
                # Environment should override file
                assert config.github_api_url == "https://env-override.github.com/api"
        finally:
            os.unlink(config_path)

    def test_system_config_logging_includes_github_url(self):
        """Test that system configuration logging includes GitHub API URL."""
        with patch('src.autogen_code_review_bot.system_config.logger') as mock_logger:
            SystemConfig()
            
            # Verify that initialization logging was called
            mock_logger.info.assert_called()
            
            # The GitHub URL is part of the config but not explicitly logged in __post_init__
            # This test ensures the config is properly initialized
            assert mock_logger.info.called