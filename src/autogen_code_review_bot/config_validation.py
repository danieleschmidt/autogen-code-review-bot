"""Configuration validation framework for ensuring reliable config loading."""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field

from .logging_config import get_logger

logger = get_logger(__name__)


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ValidationError(ConfigError):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


@dataclass
class SchemaDefinition:
    """Defines validation schema for configuration sections."""
    
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate(self, config: Dict[str, Any]) -> None:
        """Validate configuration against this schema."""
        # Check required fields
        for field_name, field_def in self.fields.items():
            if field_def.get("required", False) and field_name not in config:
                raise ValidationError(
                    f"Required field '{field_name}' is missing",
                    field=field_name
                )
        
        # Validate each field in the config
        for field_name, value in config.items():
            if field_name not in self.fields:
                # Allow unknown fields but log warning
                logger.warning(f"Unknown configuration field: {field_name}")
                continue
            
            field_def = self.fields[field_name]
            self._validate_field(field_name, value, field_def)
    
    def _validate_field(self, field_name: str, value: Any, field_def: Dict[str, Any]) -> None:
        """Validate a single field against its definition."""
        # Type validation
        expected_type = field_def.get("type")
        if expected_type and not isinstance(value, expected_type):
            raise ValidationError(
                f"Field '{field_name}' must be of type {expected_type.__name__}, got {type(value).__name__}",
                field=field_name,
                value=value
            )
        
        # Numeric constraints
        if isinstance(value, (int, float)):
            min_val = field_def.get("min")
            max_val = field_def.get("max")
            
            if min_val is not None and value < min_val:
                raise ValidationError(
                    f"Field '{field_name}' must be >= {min_val}, got {value}",
                    field=field_name,
                    value=value
                )
            
            if max_val is not None and value > max_val:
                raise ValidationError(
                    f"Field '{field_name}' must be <= {max_val}, got {value}",
                    field=field_name,
                    value=value
                )
        
        # String constraints
        if isinstance(value, str):
            pattern = field_def.get("pattern")
            if pattern and not re.match(pattern, value):
                raise ValidationError(
                    f"Field '{field_name}' does not match required pattern: {pattern}",
                    field=field_name,
                    value=value
                )
            
            allowed_values = field_def.get("allowed")
            if allowed_values and value not in allowed_values:
                raise ValidationError(
                    f"Field '{field_name}' must be one of {allowed_values}, got '{value}'",
                    field=field_name,
                    value=value
                )
            
            min_length = field_def.get("min_length")
            if min_length is not None and len(value) < min_length:
                raise ValidationError(
                    f"Field '{field_name}' must be at least {min_length} characters long",
                    field=field_name,
                    value=value
                )
        
        # List constraints
        if isinstance(value, list):
            item_type = field_def.get("item_type")
            if item_type:
                for i, item in enumerate(value):
                    if not isinstance(item, item_type):
                        raise ValidationError(
                            f"Field '{field_name}[{i}]' must be of type {item_type.__name__}, got {type(item).__name__}",
                            field=f"{field_name}[{i}]",
                            value=item
                        )


# Define schemas for different configuration types
LINTER_SCHEMA = SchemaDefinition({
    "linters": {
        "type": dict,
        "required": False,
        "description": "Mapping of programming languages to linter tools"
    }
})

AGENT_SCHEMA = SchemaDefinition({
    "agents": {
        "type": dict,
        "required": False,
        "description": "Agent configurations for code review"
    }
})

COVERAGE_SCHEMA = SchemaDefinition({
    "coverage": {
        "type": dict,
        "required": False,
        "description": "Test coverage configuration"
    }
})

BOT_SCHEMA = SchemaDefinition({
    "github": {
        "type": dict,
        "required": False,
        "description": "GitHub integration configuration"
    },
    "review_criteria": {
        "type": dict,
        "required": False,
        "description": "Code review criteria settings"
    }
})

# Define nested schemas for complex structures
AGENT_DETAIL_SCHEMA = SchemaDefinition({
    "model": {
        "type": str,
        "required": True,
        "description": "Language model to use (e.g., 'gpt-4', 'claude-3')"
    },
    "temperature": {
        "type": (int, float),
        "required": False,
        "min": 0.0,
        "max": 1.0,
        "description": "Model temperature for response randomness (0.0-1.0)"
    },
    "focus_areas": {
        "type": list,
        "required": False,
        "item_type": str,
        "description": "List of areas for the agent to focus on"
    }
})

COVERAGE_DETAIL_SCHEMA = SchemaDefinition({
    "target_threshold": {
        "type": (int, float),
        "required": False,
        "min": 0.0,
        "max": 100.0,
        "description": "Target coverage percentage (0-100)"
    },
    "report_format": {
        "type": str,
        "required": False,
        "allowed": ["html", "json", "text", "xml"],
        "description": "Format for coverage reports"
    },
    "exclude_patterns": {
        "type": list,
        "required": False,
        "item_type": str,
        "description": "Glob patterns for files to exclude from coverage"
    }
})

GITHUB_DETAIL_SCHEMA = SchemaDefinition({
    "webhook_secret": {
        "type": str,
        "required": False,
        "min_length": 1,
        "description": "Secret for validating GitHub webhook requests"
    },
    "bot_token": {
        "type": str,
        "required": False,
        "min_length": 1,
        "description": "GitHub personal access token for API access"
    }
})

# Known valid linter tools
ALLOWED_LINTER_TOOLS: Set[str] = {
    "ruff", "flake8", "pylint", "black", "isort",  # Python
    "eslint", "prettier", "jshint",  # JavaScript/TypeScript
    "golangci-lint", "gofmt", "staticcheck",  # Go
    "rubocop", "reek",  # Ruby
    "rustfmt", "clippy",  # Rust
    "clang-format", "cpplint",  # C/C++
    "shellcheck",  # Shell
    "hadolint",  # Dockerfile
}


def validate_linter_config(config: Dict[str, Any]) -> None:
    """Validate linter configuration structure and content."""
    LINTER_SCHEMA.validate(config)
    
    linters = config.get("linters", {})
    if not isinstance(linters, dict):
        raise ValidationError(
            "Field 'linters' must be a dictionary mapping languages to tools",
            field="linters",
            value=linters
        )
    
    for language, tool in linters.items():
        if not isinstance(language, str):
            raise ValidationError(
                f"Language name must be a string, got {type(language).__name__}",
                field=f"linters.{language}",
                value=language
            )
        
        if not isinstance(tool, str):
            raise ValidationError(
                f"Linter tool for '{language}' must be a string, got {type(tool).__name__}",
                field=f"linters.{language}",
                value=tool
            )
        
        if tool not in ALLOWED_LINTER_TOOLS:
            raise ValidationError(
                f"Unknown linter tool '{tool}' for language '{language}'. "
                f"Allowed tools: {', '.join(sorted(ALLOWED_LINTER_TOOLS))}",
                field=f"linters.{language}",
                value=tool
            )


def validate_agent_config(config: Dict[str, Any]) -> None:
    """Validate agent configuration structure and content."""
    AGENT_SCHEMA.validate(config)
    
    agents = config.get("agents", {})
    if not isinstance(agents, dict):
        raise ValidationError(
            "Field 'agents' must be a dictionary of agent configurations",
            field="agents",
            value=agents
        )
    
    for agent_name, agent_config in agents.items():
        if not isinstance(agent_config, dict):
            raise ValidationError(
                f"Agent '{agent_name}' configuration must be a dictionary",
                field=f"agents.{agent_name}",
                value=agent_config
            )
        
        try:
            AGENT_DETAIL_SCHEMA.validate(agent_config)
        except ValidationError as e:
            # Re-raise with more specific field path
            raise ValidationError(
                f"Agent '{agent_name}': {e}",
                field=f"agents.{agent_name}.{e.field}" if e.field else f"agents.{agent_name}",
                value=e.value
            )


def validate_coverage_config(config: Dict[str, Any]) -> None:
    """Validate coverage configuration structure and content."""
    COVERAGE_SCHEMA.validate(config)
    
    coverage = config.get("coverage", {})
    if not isinstance(coverage, dict):
        raise ValidationError(
            "Field 'coverage' must be a dictionary of coverage settings",
            field="coverage",
            value=coverage
        )
    
    if coverage:  # Only validate if coverage section exists
        try:
            COVERAGE_DETAIL_SCHEMA.validate(coverage)
        except ValidationError as e:
            # Re-raise with more specific field path
            raise ValidationError(
                f"Coverage configuration: {e}",
                field=f"coverage.{e.field}" if e.field else "coverage",
                value=e.value
            )


def validate_bot_config(config: Dict[str, Any]) -> None:
    """Validate main bot configuration structure and content."""
    BOT_SCHEMA.validate(config)
    
    # Validate GitHub section if present
    github = config.get("github", {})
    if github:
        if not isinstance(github, dict):
            raise ValidationError(
                "Field 'github' must be a dictionary of GitHub settings",
                field="github",
                value=github
            )
        
        try:
            GITHUB_DETAIL_SCHEMA.validate(github)
        except ValidationError as e:
            raise ValidationError(
                f"GitHub configuration: {e}",
                field=f"github.{e.field}" if e.field else "github",
                value=e.value
            )
        
        # Additional validation for GitHub fields
        webhook_secret = github.get("webhook_secret", "")
        if webhook_secret == "":
            raise ValidationError(
                "GitHub webhook_secret cannot be empty",
                field="github.webhook_secret",
                value=webhook_secret
            )
    
    # Validate review criteria if present
    review_criteria = config.get("review_criteria", {})
    if review_criteria:
        if not isinstance(review_criteria, dict):
            raise ValidationError(
                "Field 'review_criteria' must be a dictionary",
                field="review_criteria",
                value=review_criteria
            )
        
        # Validate boolean flags
        for key, value in review_criteria.items():
            if not isinstance(value, bool):
                raise ValidationError(
                    f"Review criteria '{key}' must be true or false, got {type(value).__name__}",
                    field=f"review_criteria.{key}",
                    value=value
                )


class ConfigValidator:
    """Main configuration validator that coordinates validation of different config types."""
    
    def __init__(self):
        self.validators = {
            "linter": validate_linter_config,
            "agent": validate_agent_config,
            "coverage": validate_coverage_config,
            "bot": validate_bot_config
        }
    
    def validate_file(self, file_path: str, config_type: str) -> Dict[str, Any]:
        """Validate a configuration file."""
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Malformed YAML in configuration file {file_path}: {e}")
        except OSError as e:
            raise ConfigError(f"Unable to read configuration file {file_path}: {e}")
        
        return self.validate_dict(config, config_type)
    
    def validate_dict(self, config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """Validate a configuration dictionary."""
        if config_type not in self.validators:
            raise ConfigError(
                f"Unknown configuration type '{config_type}'. "
                f"Supported types: {', '.join(self.validators.keys())}"
            )
        
        try:
            self.validators[config_type](config)
            logger.debug(f"Successfully validated {config_type} configuration")
            return config
        except ValidationError as e:
            # Enhance error message with helpful context
            error_msg = str(e)
            if e.field:
                error_msg = f"Configuration error in field '{e.field}': {error_msg}"
            
            # Add suggestions based on error type
            if "unknown" in error_msg.lower() and "linter" in error_msg.lower():
                error_msg += f"\n\nHint: Run 'python -c \"from autogen_code_review_bot.config_validation import ALLOWED_LINTER_TOOLS; print(sorted(ALLOWED_LINTER_TOOLS))\"' to see allowed linter tools."
            elif "temperature" in error_msg.lower():
                error_msg += "\n\nHint: Temperature values should be between 0.0 (deterministic) and 1.0 (creative)."
            elif "required" in error_msg.lower():
                error_msg += f"\n\nHint: Check the configuration documentation for required fields in {config_type} configs."
            
            raise ConfigError(error_msg)
    
    def get_validation_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of supported configuration types and their requirements."""
        return {
            "linter": {
                "description": "Maps programming languages to linter tools",
                "example": {
                    "linters": {
                        "python": "ruff",
                        "javascript": "eslint",
                        "typescript": "eslint"
                    }
                },
                "allowed_tools": sorted(ALLOWED_LINTER_TOOLS)
            },
            "agent": {
                "description": "Configures AI agents for code review",
                "example": {
                    "agents": {
                        "coder": {
                            "model": "gpt-4",
                            "temperature": 0.3,
                            "focus_areas": ["functionality", "bugs"]
                        },
                        "reviewer": {
                            "model": "gpt-4",
                            "temperature": 0.1,
                            "focus_areas": ["security", "performance"]
                        }
                    }
                }
            },
            "coverage": {
                "description": "Configures test coverage requirements and reporting",
                "example": {
                    "coverage": {
                        "target_threshold": 85.0,
                        "report_format": "html",
                        "exclude_patterns": ["*/tests/*", "*/migrations/*"]
                    }
                }
            },
            "bot": {
                "description": "Main bot configuration including GitHub integration",
                "example": {
                    "github": {
                        "webhook_secret": "your_webhook_secret",
                        "bot_token": "your_github_token"
                    },
                    "review_criteria": {
                        "security_scan": True,
                        "performance_check": True,
                        "test_coverage": True
                    }
                }
            }
        }


# Global validator instance
_validator = ConfigValidator()


def validate_config_file(file_path: str, config_type: str) -> Dict[str, Any]:
    """Convenience function to validate a configuration file."""
    return _validator.validate_file(file_path, config_type)


def validate_config_dict(config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
    """Convenience function to validate a configuration dictionary."""
    return _validator.validate_dict(config, config_type)


def get_config_help(config_type: Optional[str] = None) -> str:
    """Get help text for configuration validation."""
    summary = _validator.get_validation_summary()
    
    if config_type:
        if config_type not in summary:
            return f"Unknown configuration type: {config_type}"
        
        info = summary[config_type]
        help_text = f"Configuration type: {config_type}\n"
        help_text += f"Description: {info['description']}\n\n"
        help_text += f"Example:\n{yaml.dump(info['example'], default_flow_style=False)}"
        
        if "allowed_tools" in info:
            help_text += f"\nAllowed tools: {', '.join(info['allowed_tools'])}"
        
        return help_text
    else:
        help_text = "Available configuration types:\n\n"
        for name, info in summary.items():
            help_text += f"- {name}: {info['description']}\n"
        help_text += f"\nUse get_config_help('<type>') for detailed help on a specific type."
        return help_text