"""Comprehensive tests for configuration validation."""

import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from autogen_code_review_bot.config_validation import (
    ConfigValidator,
    ConfigError,
    ValidationError,
    SchemaDefinition,
    validate_linter_config,
    validate_agent_config,
    validate_coverage_config,
    validate_bot_config
)


class TestConfigSchema:
    """Test configuration schema definitions."""
    
    def test_schema_definition_creation(self):
        """Test creating schema definitions with fields and constraints."""
        schema = SchemaDefinition({
            "name": {"type": str, "required": True},
            "age": {"type": int, "min": 0, "max": 150},
            "email": {"type": str, "pattern": r"^[^@]+@[^@]+\.[^@]+$"}
        })
        
        assert "name" in schema.fields
        assert schema.fields["name"]["type"] == str
        assert schema.fields["name"]["required"] == True
        assert schema.fields["age"]["min"] == 0
        assert schema.fields["age"]["max"] == 150

    def test_schema_validation_success(self):
        """Test successful schema validation."""
        schema = SchemaDefinition({
            "name": {"type": str, "required": True},
            "count": {"type": int, "min": 0}
        })
        
        config = {"name": "test", "count": 5}
        
        # Should not raise exception
        schema.validate(config)

    def test_schema_validation_missing_required_field(self):
        """Test validation failure for missing required fields."""
        schema = SchemaDefinition({
            "name": {"type": str, "required": True},
            "count": {"type": int, "required": False}
        })
        
        config = {"count": 5}  # Missing required 'name'
        
        with pytest.raises(ValidationError) as exc_info:
            schema.validate(config)
        
        assert "required field 'name' is missing" in str(exc_info.value).lower()

    def test_schema_validation_wrong_type(self):
        """Test validation failure for incorrect field types."""
        schema = SchemaDefinition({
            "count": {"type": int}
        })
        
        config = {"count": "not_a_number"}
        
        with pytest.raises(ValidationError) as exc_info:
            schema.validate(config)
        
        assert "type" in str(exc_info.value).lower()
        assert "count" in str(exc_info.value)

    def test_schema_validation_constraint_violation(self):
        """Test validation failure for constraint violations."""
        schema = SchemaDefinition({
            "age": {"type": int, "min": 0, "max": 150}
        })
        
        config = {"age": -5}
        
        with pytest.raises(ValidationError) as exc_info:
            schema.validate(config)
        
        assert ">=" in str(exc_info.value) or "minimum" in str(exc_info.value).lower() or "min" in str(exc_info.value).lower()

    def test_schema_validation_pattern_matching(self):
        """Test pattern validation for string fields."""
        schema = SchemaDefinition({
            "email": {"type": str, "pattern": r"^[^@]+@[^@]+\.[^@]+$"}
        })
        
        valid_config = {"email": "test@example.com"}
        invalid_config = {"email": "invalid-email"}
        
        # Valid email should pass
        schema.validate(valid_config)
        
        # Invalid email should fail
        with pytest.raises(ValidationError):
            schema.validate(invalid_config)


class TestLinterConfigValidation:
    """Test linter configuration validation."""
    
    def test_valid_linter_config(self):
        """Test validation of valid linter configuration."""
        config = {
            "linters": {
                "python": "ruff",
                "javascript": "eslint",
                "typescript": "eslint"
            }
        }
        
        # Should not raise exception
        validate_linter_config(config)

    def test_linter_config_invalid_structure(self):
        """Test validation failure for invalid linter config structure."""
        config = {
            "linters": "not_a_dict"  # Should be dict
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_linter_config(config)
        
        assert "linters" in str(exc_info.value)
        assert "dict" in str(exc_info.value).lower() or "mapping" in str(exc_info.value).lower()

    def test_linter_config_invalid_tool(self):
        """Test validation failure for invalid linter tools."""
        config = {
            "linters": {
                "python": "unknown_linter"
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_linter_config(config)
        
        assert "unknown_linter" in str(exc_info.value)
        assert "allowed" in str(exc_info.value).lower()

    def test_linter_config_empty_is_valid(self):
        """Test that empty linter config is valid (uses defaults)."""
        config = {}
        
        # Should not raise exception
        validate_linter_config(config)


class TestAgentConfigValidation:
    """Test agent configuration validation."""
    
    def test_valid_agent_config(self):
        """Test validation of valid agent configuration."""
        config = {
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
        
        # Should not raise exception
        validate_agent_config(config)

    def test_agent_config_invalid_temperature(self):
        """Test validation failure for invalid temperature values."""
        config = {
            "agents": {
                "coder": {
                    "model": "gpt-4",
                    "temperature": 2.0  # Should be between 0 and 1
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_agent_config(config)
        
        assert "temperature" in str(exc_info.value)
        assert "0" in str(exc_info.value) and "1" in str(exc_info.value)

    def test_agent_config_missing_model(self):
        """Test validation failure for missing required model field."""
        config = {
            "agents": {
                "coder": {
                    "temperature": 0.3
                    # Missing required 'model' field
                }
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_agent_config(config)
        
        assert "model" in str(exc_info.value)
        assert "required" in str(exc_info.value).lower()


class TestCoverageConfigValidation:
    """Test coverage configuration validation."""
    
    def test_valid_coverage_config(self):
        """Test validation of valid coverage configuration."""
        config = {
            "coverage": {
                "target_threshold": 85.0,
                "report_format": "html",
                "exclude_patterns": ["*/tests/*", "*/migrations/*"]
            }
        }
        
        # Should not raise exception
        validate_coverage_config(config)

    def test_coverage_config_invalid_threshold(self):
        """Test validation failure for invalid coverage threshold."""
        config = {
            "coverage": {
                "target_threshold": 150.0  # Should be between 0 and 100
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_coverage_config(config)
        
        assert "threshold" in str(exc_info.value)
        assert "100" in str(exc_info.value)

    def test_coverage_config_invalid_format(self):
        """Test validation failure for invalid report format."""
        config = {
            "coverage": {
                "report_format": "invalid_format"
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_coverage_config(config)
        
        assert "format" in str(exc_info.value)
        assert "invalid_format" in str(exc_info.value)


class TestBotConfigValidation:
    """Test main bot configuration validation."""
    
    def test_valid_bot_config(self):
        """Test validation of valid bot configuration."""
        config = {
            "github": {
                "webhook_secret": "secret123",
                "bot_token": "token123"
            },
            "review_criteria": {
                "security_scan": True,
                "performance_check": True,
                "test_coverage": True
            }
        }
        
        # Should not raise exception
        validate_bot_config(config)

    def test_bot_config_invalid_github_section(self):
        """Test validation failure for invalid GitHub configuration."""
        config = {
            "github": {
                "webhook_secret": "",  # Empty secret should fail
                "bot_token": "token123"
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_bot_config(config)
        
        assert "webhook_secret" in str(exc_info.value)
        assert "empty" in str(exc_info.value).lower() or "characters long" in str(exc_info.value).lower()


class TestConfigValidator:
    """Test the main configuration validator."""
    
    def test_validate_file_success(self):
        """Test successful validation of configuration file."""
        valid_config = {
            "linters": {
                "python": "ruff"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_config, f)
            temp_path = f.name
        
        try:
            validator = ConfigValidator()
            result = validator.validate_file(temp_path, "linter")
            
            assert result is not None
            assert "linters" in result
        finally:
            Path(temp_path).unlink()

    def test_validate_file_yaml_error(self):
        """Test validation failure for malformed YAML."""
        malformed_yaml = "invalid: yaml: content: [\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml)
            temp_path = f.name
        
        try:
            validator = ConfigValidator()
            
            with pytest.raises(ConfigError) as exc_info:
                validator.validate_file(temp_path, "linter")
            
            assert "yaml" in str(exc_info.value).lower()
            assert "malformed" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
        finally:
            Path(temp_path).unlink()

    def test_validate_file_not_found(self):
        """Test validation failure for non-existent file."""
        validator = ConfigValidator()
        
        with pytest.raises(ConfigError) as exc_info:
            validator.validate_file("/non/existent/file.yaml", "linter")
        
        assert "not found" in str(exc_info.value).lower() or "does not exist" in str(exc_info.value).lower()

    def test_validate_dict_success(self):
        """Test successful validation of configuration dictionary."""
        config = {
            "linters": {
                "python": "ruff"
            }
        }
        
        validator = ConfigValidator()
        result = validator.validate_dict(config, "linter")
        
        assert result == config

    def test_validate_unknown_config_type(self):
        """Test validation failure for unknown configuration type."""
        validator = ConfigValidator()
        
        with pytest.raises(ConfigError) as exc_info:
            validator.validate_dict({}, "unknown_type")
        
        assert "unknown" in str(exc_info.value).lower()
        assert "type" in str(exc_info.value).lower()

    def test_get_validation_summary(self):
        """Test getting validation summary with helpful information."""
        validator = ConfigValidator()
        summary = validator.get_validation_summary()
        
        assert "linter" in summary
        assert "agent" in summary
        assert "coverage" in summary
        assert "bot" in summary
        
        # Each config type should have examples or descriptions
        for config_type, info in summary.items():
            assert "description" in info or "example" in info


class TestConfigValidationIntegration:
    """Test integration with existing configuration loading."""
    
    def test_validate_with_pr_analysis_config(self):
        """Test validation integration with PR analysis configuration loading."""
        # This would test that our validation integrates properly with existing code
        config = {
            "linters": {
                "python": "ruff",
                "javascript": "eslint"
            }
        }
        
        # Validate using our new system
        validate_linter_config(config)
        
        # Should be compatible with existing pr_analysis.load_linter_config
        # This is an integration test to ensure compatibility

    def test_helpful_error_messages(self):
        """Test that error messages are helpful and actionable."""
        config = {
            "linters": {
                "python": "invalid_tool",
                "javascript": 123  # Wrong type
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_linter_config(config)
        
        error_message = str(exc_info.value)
        
        # Error message should be helpful
        assert len(error_message) > 20  # Not too brief
        assert "invalid_tool" in error_message  # Include the problematic value
        assert any(word in error_message.lower() for word in ["allowed", "valid", "expected"])  # Suggest solution


if __name__ == "__main__":
    pytest.main([__file__])