"""Centralized system configuration management.

This module provides a centralized place for all system configuration values,
replacing hardcoded constants throughout the codebase with configurable parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SystemConfig:
    """Centralized system configuration with all tunable parameters."""

    # Command execution settings
    default_command_timeout: int = 30
    max_command_timeout: int = 300

    # Thread pool configuration
    default_thread_pool_size: int = 3
    max_thread_pool_size: int = 10
    monitoring_thread_pool_size: int = 5

    # File processing limits
    max_files_normal_mode: int = 1000
    max_size_normal_mode: int = 10 * 1024 * 1024  # 10MB
    max_files_streaming_mode: int = 10000
    language_detection_max_files: int = 10000

    # Webhook and caching settings
    webhook_deduplication_ttl: int = 3600  # 1 hour
    cache_cleanup_interval: int = 1800     # 30 minutes

    # Coverage analysis settings
    coverage_timeout: int = 300            # 5 minutes
    coverage_default_threshold: float = 85.0

    # Monitoring and metrics settings
    max_values_per_metric: int = 1000
    metrics_cleanup_interval: int = 3600   # 1 hour
    health_check_timeout: int = 10

    # Performance thresholds
    analysis_timeout_warning: int = 60     # Warn if analysis takes > 60s
    large_repo_threshold_files: int = 5000
    large_repo_threshold_size: int = 50 * 1024 * 1024  # 50MB

    # Retry and circuit breaker settings
    default_retry_attempts: int = 3
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 300     # 5 minutes

    # GitHub API settings
    github_api_url: str = "https://api.github.com"
    github_api_timeout: int = 15
    github_api_rate_limit_buffer: int = 10  # requests to keep in reserve

    # Path validation settings
    max_path_length: int = 4096
    unicode_normalization_max_diff: int = 2

    def __post_init__(self):
        """Validate configuration values after initialization."""
        self._validate_config()
        self._apply_environment_overrides()
        logger.info("System configuration initialized", extra={
            "command_timeout": self.default_command_timeout,
            "thread_pool_size": self.default_thread_pool_size,
            "max_files": self.max_files_normal_mode,
            "coverage_threshold": self.coverage_default_threshold
        })

    def _validate_config(self):
        """Validate that configuration values are reasonable."""
        if self.default_command_timeout <= 0:
            raise ValueError("default_command_timeout must be positive")
        if self.default_thread_pool_size <= 0:
            raise ValueError("default_thread_pool_size must be positive")
        if self.max_files_normal_mode <= 0:
            raise ValueError("max_files_normal_mode must be positive")
        if not 0 <= self.coverage_default_threshold <= 100:
            raise ValueError("coverage_default_threshold must be between 0 and 100")

    def _apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'AUTOGEN_COMMAND_TIMEOUT': ('default_command_timeout', int),
            'AUTOGEN_THREAD_POOL_SIZE': ('default_thread_pool_size', int),
            'AUTOGEN_MAX_FILES': ('max_files_normal_mode', int),
            'AUTOGEN_COVERAGE_THRESHOLD': ('coverage_default_threshold', float),
            'AUTOGEN_WEBHOOK_TTL': ('webhook_deduplication_ttl', int),
            'AUTOGEN_GITHUB_API_URL': ('github_api_url', str),
            'AUTOGEN_GITHUB_TIMEOUT': ('github_api_timeout', int),
            'AUTOGEN_RETRY_ATTEMPTS': ('default_retry_attempts', int),
        }

        for env_var, (attr_name, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    setattr(self, attr_name, converted_value)
                    logger.info(f"Applied environment override: {attr_name} = {converted_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> SystemConfig:
        """Create configuration from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })

    @classmethod
    def load_from_file(cls, config_path: Path | str) -> SystemConfig:
        """Load configuration from YAML file."""
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            logger.info(f"Config file {config_path} not found, using defaults")
            return cls()

        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

            # Extract system config section if it exists
            system_config = config_data.get('system', {})
            logger.info(f"Loaded configuration from {config_path}")
            return cls.from_dict(system_config)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
            return cls()


# Global configuration instance
_config_instance: SystemConfig | None = None


def get_system_config() -> SystemConfig:
    """Get the global system configuration instance."""
    global _config_instance
    if _config_instance is None:
        # Try to load from standard locations
        config_paths = [
            Path.cwd() / "system_config.yaml",
            Path.cwd() / "config" / "system.yaml",
            Path.home() / ".autogen" / "system_config.yaml"
        ]

        for config_path in config_paths:
            if config_path.exists():
                _config_instance = SystemConfig.load_from_file(config_path)
                break

        if _config_instance is None:
            _config_instance = SystemConfig()

    return _config_instance


def reset_system_config():
    """Reset the global configuration instance (useful for testing)."""
    global _config_instance
    _config_instance = None


def set_system_config(config: SystemConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config
