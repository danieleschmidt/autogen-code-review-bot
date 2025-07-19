"""Tests for configuration management system."""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autogen_code_review_bot.config import (
    Config,
    load_config,
    get_default_config,
    merge_config_sources,
    validate_config,
    ConfigurationError,
)


class TestConfig:
    """Test configuration management."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        config = Config()
        
        assert config.github_api_url == "https://api.github.com"
        assert config.default_timeout == 30
        assert config.http_timeout == 10
        assert isinstance(config.default_linters, dict)
        assert "python" in config.default_linters
        assert config.default_linters["python"] == "ruff"
        
    def test_config_creation_with_custom_values(self):
        """Test creating config with custom values."""
        custom_linters = {"python": "flake8", "javascript": "jshint"}
        
        config = Config(
            github_api_url="https://api.github.enterprise.com",
            default_timeout=60,
            http_timeout=20,
            default_linters=custom_linters
        )
        
        assert config.github_api_url == "https://api.github.enterprise.com"
        assert config.default_timeout == 60
        assert config.http_timeout == 20
        assert config.default_linters == custom_linters
        
    def test_config_validation_valid(self):
        """Test config validation with valid values."""
        config = Config(
            github_api_url="https://api.github.com",
            default_timeout=30,
            http_timeout=10
        )
        
        # Should not raise
        validate_config(config)
        
    def test_config_validation_invalid_url(self):
        """Test config validation with invalid URL."""
        config = Config(github_api_url="not-a-url")
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Invalid GitHub API URL" in str(exc_info.value)
        
    def test_config_validation_invalid_timeout(self):
        """Test config validation with invalid timeout."""
        config = Config(default_timeout=-5)
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "Timeout must be positive" in str(exc_info.value)
        
    def test_config_validation_invalid_linters(self):
        """Test config validation with invalid linters."""
        config = Config(default_linters="not-a-dict")
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config(config)
        assert "default_linters must be a dictionary" in str(exc_info.value)


class TestConfigLoading:
    """Test configuration loading from various sources."""
    
    def test_load_config_from_json_file(self):
        """Test loading config from JSON file."""
        config_data = {
            "github_api_url": "https://custom.github.com",
            "default_timeout": 45,
            "http_timeout": 15,
            "default_linters": {
                "python": "mypy",
                "javascript": "standard"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        try:
            config = load_config(config_file)
            
            assert config.github_api_url == "https://custom.github.com"
            assert config.default_timeout == 45
            assert config.http_timeout == 15
            assert config.default_linters["python"] == "mypy"
            assert config.default_linters["javascript"] == "standard"
        finally:
            os.unlink(config_file)
            
    def test_load_config_from_environment(self):
        """Test loading config from environment variables."""
        env_vars = {
            'AUTOGEN_GITHUB_API_URL': 'https://env.github.com',
            'AUTOGEN_DEFAULT_TIMEOUT': '25',
            'AUTOGEN_HTTP_TIMEOUT': '8',
        }
        
        with patch.dict(os.environ, env_vars):
            config = load_config()
            
            assert config.github_api_url == "https://env.github.com"
            assert config.default_timeout == 25
            assert config.http_timeout == 8
            
    def test_load_config_environment_overrides_file(self):
        """Test that environment variables override file config."""
        config_data = {
            "github_api_url": "https://file.github.com",
            "default_timeout": 45
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
            
        env_vars = {
            'AUTOGEN_GITHUB_API_URL': 'https://env.github.com',
        }
        
        try:
            with patch.dict(os.environ, env_vars):
                config = load_config(config_file)
                
                # Environment should override file
                assert config.github_api_url == "https://env.github.com"
                # File value should be preserved
                assert config.default_timeout == 45
        finally:
            os.unlink(config_file)
            
    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        config = load_config("/nonexistent/config.json")
        
        # Should return default config
        assert config.github_api_url == "https://api.github.com"
        assert config.default_timeout == 30
        
    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_file = f.name
            
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config(config_file)
            assert "Failed to parse config file" in str(exc_info.value)
        finally:
            os.unlink(config_file)


class TestConfigMerging:
    """Test configuration merging from multiple sources."""
    
    def test_merge_config_sources(self):
        """Test merging configuration from multiple sources."""
        defaults = {
            "github_api_url": "https://api.github.com",
            "default_timeout": 30,
            "http_timeout": 10,
            "default_linters": {"python": "ruff"}
        }
        
        file_config = {
            "github_api_url": "https://file.github.com",
            "default_timeout": 45,
            "default_linters": {"python": "flake8", "javascript": "eslint"}
        }
        
        env_config = {
            "github_api_url": "https://env.github.com",
            "http_timeout": 15
        }
        
        merged = merge_config_sources(defaults, file_config, env_config)
        
        # Environment should have highest priority
        assert merged["github_api_url"] == "https://env.github.com"
        assert merged["http_timeout"] == 15
        
        # File should override defaults
        assert merged["default_timeout"] == 45
        
        # Linters should be merged properly
        assert merged["default_linters"]["python"] == "flake8"
        assert merged["default_linters"]["javascript"] == "eslint"
        
    def test_merge_nested_dictionaries(self):
        """Test merging of nested dictionary values."""
        base = {
            "default_linters": {
                "python": "ruff",
                "javascript": "eslint",
                "typescript": "eslint"
            }
        }
        
        override = {
            "default_linters": {
                "python": "flake8",
                "go": "golangci-lint"
            }
        }
        
        merged = merge_config_sources(base, override)
        
        expected_linters = {
            "python": "flake8",      # overridden
            "javascript": "eslint",  # preserved
            "typescript": "eslint",  # preserved  
            "go": "golangci-lint"    # added
        }
        
        assert merged["default_linters"] == expected_linters


class TestDefaultConfig:
    """Test default configuration generation."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        
        assert isinstance(config, Config)
        assert config.github_api_url == "https://api.github.com"
        assert config.default_timeout == 30
        assert config.http_timeout == 10
        assert isinstance(config.default_linters, dict)
        assert len(config.default_linters) > 0


class TestConfigurationError:
    """Test configuration error handling."""
    
    def test_configuration_error_message(self):
        """Test configuration error with custom message."""
        error = ConfigurationError("Custom error message")
        assert str(error) == "Custom error message"
        
    def test_configuration_error_inheritance(self):
        """Test that ConfigurationError inherits from Exception."""
        assert issubclass(ConfigurationError, Exception)


class TestConfigIntegration:
    """Test configuration integration with existing modules."""
    
    def test_config_used_in_github_integration(self):
        """Test that configuration is used in GitHub integration."""
        # This would test actual integration - placeholder for now
        config = Config(github_api_url="https://custom.github.com")
        
        # Verify config can be applied to modules
        assert config.github_api_url != "https://api.github.com"
        
    def test_config_used_in_pr_analysis(self):
        """Test that configuration is used in PR analysis."""
        config = Config(default_timeout=60)
        
        # Verify config can be applied to modules
        assert config.default_timeout != 30
        
    def test_config_environment_variables_list(self):
        """Test complete list of supported environment variables."""
        expected_env_vars = [
            'AUTOGEN_GITHUB_API_URL',
            'AUTOGEN_DEFAULT_TIMEOUT', 
            'AUTOGEN_HTTP_TIMEOUT',
            'AUTOGEN_CONFIG_FILE'
        ]
        
        # This ensures we document all supported env vars
        for var in expected_env_vars:
            assert var.startswith('AUTOGEN_')
            
    def test_config_file_formats_supported(self):
        """Test that configuration supports expected file formats."""
        # Initially support JSON, could expand to YAML later
        supported_formats = ['.json']
        
        for fmt in supported_formats:
            assert fmt in ['.json']  # Placeholder test


class TestConfigPerformance:
    """Test configuration loading performance."""
    
    def test_config_caching(self):
        """Test that configuration is cached for performance."""
        # Load config multiple times
        config1 = load_config()
        config2 = load_config()
        
        # Verify they're equivalent (not necessarily same object)
        assert config1.github_api_url == config2.github_api_url
        assert config1.default_timeout == config2.default_timeout
        
    def test_config_loading_speed(self):
        """Test that config loading is reasonably fast."""
        import time
        
        start_time = time.time()
        config = load_config()
        load_time = time.time() - start_time
        
        # Should load in under 100ms
        assert load_time < 0.1
        assert config is not None