"""Configuration management for autogen-code-review-bot.

This module implements the Twelve-Factor App principle of configuration via environment
variables while also supporting file-based configuration for development and testing.

Environment variables take precedence over file configuration, which takes precedence
over default values.

Supported environment variables:
- AUTOGEN_GITHUB_API_URL: GitHub API base URL
- AUTOGEN_DEFAULT_TIMEOUT: Default command timeout in seconds
- AUTOGEN_HTTP_TIMEOUT: HTTP request timeout in seconds
- AUTOGEN_CONFIG_FILE: Path to configuration file
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""

    pass


@dataclass
class Config:
    """Configuration container for autogen-code-review-bot."""

    # GitHub Integration Settings
    github_api_url: str = "https://api.github.com"

    # Timeout Settings (in seconds)
    default_timeout: int = 30
    http_timeout: int = 10

    # Linter Configuration
    default_linters: Dict[str, str] = field(
        default_factory=lambda: {
            "python": "ruff",
            "javascript": "eslint",
            "typescript": "eslint",
            "ruby": "rubocop",
            "go": "golangci-lint",
            "rust": "clippy",
        }
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        validate_config(self)


def validate_config(config: Config) -> None:
    """Validate configuration values.

    Args:
        config: Configuration to validate.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    # Validate GitHub API URL
    if not isinstance(config.github_api_url, str) or not config.github_api_url.strip():
        raise ConfigurationError("GitHub API URL cannot be empty")

    parsed_url = urlparse(config.github_api_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ConfigurationError(f"Invalid GitHub API URL: {config.github_api_url}")

    if parsed_url.scheme not in ("http", "https"):
        raise ConfigurationError(
            f"GitHub API URL must use HTTP or HTTPS: {config.github_api_url}"
        )

    # Validate timeouts
    if not isinstance(config.default_timeout, int) or config.default_timeout <= 0:
        raise ConfigurationError(
            f"Timeout must be positive integer, got: {config.default_timeout}"
        )

    if not isinstance(config.http_timeout, int) or config.http_timeout <= 0:
        raise ConfigurationError(
            f"HTTP timeout must be positive integer, got: {config.http_timeout}"
        )

    # Validate linters configuration
    if not isinstance(config.default_linters, dict):
        raise ConfigurationError(
            f"default_linters must be a dictionary, got: {type(config.default_linters)}"
        )

    for language, linter in config.default_linters.items():
        if not isinstance(language, str) or not isinstance(linter, str):
            raise ConfigurationError(
                f"Linter mapping must be str->str, got {language}: {linter}"
            )


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Default configuration instance.
    """
    return Config()


def load_config_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file.

    Args:
        file_path: Path to configuration file.

    Returns:
        Dictionary of configuration values.

    Raises:
        ConfigurationError: If file cannot be loaded or parsed.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {}

    try:
        with open(file_path, encoding="utf-8") as f:
            if file_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {file_path.suffix}"
                )
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Failed to parse config file {file_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load config file {file_path}: {e}")


def load_config_from_environment() -> Dict[str, Any]:
    """Load configuration from environment variables.

    Returns:
        Dictionary of configuration values from environment.
    """
    config = {}

    # GitHub API URL
    if "AUTOGEN_GITHUB_API_URL" in os.environ:
        config["github_api_url"] = os.environ["AUTOGEN_GITHUB_API_URL"]

    # Timeout settings
    if "AUTOGEN_DEFAULT_TIMEOUT" in os.environ:
        try:
            config["default_timeout"] = int(os.environ["AUTOGEN_DEFAULT_TIMEOUT"])
        except ValueError:
            raise ConfigurationError(
                f"Invalid AUTOGEN_DEFAULT_TIMEOUT: {os.environ['AUTOGEN_DEFAULT_TIMEOUT']}"
            )

    if "AUTOGEN_HTTP_TIMEOUT" in os.environ:
        try:
            config["http_timeout"] = int(os.environ["AUTOGEN_HTTP_TIMEOUT"])
        except ValueError:
            raise ConfigurationError(
                f"Invalid AUTOGEN_HTTP_TIMEOUT: {os.environ['AUTOGEN_HTTP_TIMEOUT']}"
            )

    # Linter configuration via environment (JSON format)
    if "AUTOGEN_DEFAULT_LINTERS" in os.environ:
        try:
            config["default_linters"] = json.loads(
                os.environ["AUTOGEN_DEFAULT_LINTERS"]
            )
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid AUTOGEN_DEFAULT_LINTERS JSON: {e}")

    return config


def merge_config_sources(*sources: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration from multiple sources.

    Later sources take precedence over earlier ones.
    Nested dictionaries are merged recursively.

    Args:
        *sources: Configuration dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    merged = {}

    for source in sources:
        for key, value in source.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge dictionaries
                merged[key] = merge_config_sources(merged[key], value)
            else:
                # Override with new value
                merged[key] = value

    return merged


# Global config cache to avoid reloading
_config_cache: Optional[Config] = None


def load_config(config_file: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from all sources.

    Configuration is loaded in this precedence order:
    1. Default values
    2. Configuration file (if provided or via AUTOGEN_CONFIG_FILE)
    3. Environment variables (highest precedence)

    Args:
        config_file: Optional path to configuration file.

    Returns:
        Loaded and validated configuration.

    Raises:
        ConfigurationError: If configuration is invalid.
    """
    global _config_cache

    # Use cached config if available (for performance)
    if _config_cache is not None:
        return _config_cache

    # Determine config file to use
    if config_file is None:
        config_file = os.environ.get("AUTOGEN_CONFIG_FILE")

    # Load from all sources
    default_config = get_default_config()
    default_dict = {
        "github_api_url": default_config.github_api_url,
        "default_timeout": default_config.default_timeout,
        "http_timeout": default_config.http_timeout,
        "default_linters": default_config.default_linters,
    }

    file_config = load_config_from_file(config_file) if config_file else {}
    env_config = load_config_from_environment()

    # Merge all sources (environment takes precedence)
    merged_config = merge_config_sources(default_dict, file_config, env_config)

    # Create and validate config object
    try:
        config = Config(**merged_config)
        _config_cache = config
        return config
    except TypeError as e:
        raise ConfigurationError(f"Invalid configuration parameters: {e}")


def clear_config_cache() -> None:
    """Clear the configuration cache.

    This is useful for testing or when configuration needs to be reloaded.
    """
    global _config_cache
    _config_cache = None


# Convenience functions for getting specific config values
def get_github_api_url() -> str:
    """Get GitHub API URL from configuration."""
    return load_config().github_api_url


def get_default_timeout() -> int:
    """Get default timeout from configuration."""
    return load_config().default_timeout


def get_http_timeout() -> int:
    """Get HTTP timeout from configuration."""
    return load_config().http_timeout


def get_default_linters() -> Dict[str, str]:
    """Get default linters mapping from configuration."""
    return load_config().default_linters.copy()


def get_linter_for_language(language: str) -> Optional[str]:
    """Get configured linter for a specific language.

    Args:
        language: Programming language name.

    Returns:
        Linter command name or None if not configured.
    """
    return load_config().default_linters.get(language)
