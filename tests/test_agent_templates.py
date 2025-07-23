"""Tests for agent response template system."""

import pytest
from unittest.mock import patch

from autogen_code_review_bot.agent_templates import AgentResponseTemplates, default_templates
from autogen_code_review_bot.agents import load_agents_from_yaml


class TestAgentResponseTemplates:
    """Test the agent response template system."""

    def test_default_template_initialization(self):
        """Test that default templates are properly initialized."""
        templates = AgentResponseTemplates()
        
        # Check that both agent types are available
        available = templates.get_available_templates()
        assert "coder" in available
        assert "reviewer" in available
        
        # Check that each agent type has expected categories
        assert "improvement_focused" in available["coder"]
        assert "assessment" in available["coder"]
        assert "agreement" in available["coder"]
        
        assert "concern_focused" in available["reviewer"]
        assert "findings" in available["reviewer"]
        assert "opinion" in available["reviewer"]

    def test_get_response_basic(self):
        """Test basic response generation."""
        templates = AgentResponseTemplates()
        
        # Test coder responses
        coder_response = templates.get_response("coder", "improvement_focused")
        assert isinstance(coder_response, str)
        assert len(coder_response) > 0
        
        # Test reviewer responses
        reviewer_response = templates.get_response("reviewer", "concern_focused")
        assert isinstance(reviewer_response, str)
        assert len(reviewer_response) > 0

    def test_get_response_with_substitutions(self):
        """Test response generation with custom substitutions."""
        templates = AgentResponseTemplates()
        
        response = templates.get_response(
            "coder", 
            "improvement_focused",
            focus_area="security"
        )
        
        assert "security" in response

    def test_get_response_invalid_agent_type(self):
        """Test that invalid agent types raise ValueError."""
        templates = AgentResponseTemplates()
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            templates.get_response("invalid_agent", "some_category")

    def test_get_response_invalid_template_category(self):
        """Test that invalid template categories raise ValueError."""
        templates = AgentResponseTemplates()
        
        with pytest.raises(ValueError, match="Unknown template category"):
            templates.get_response("coder", "invalid_category")

    def test_add_template(self):
        """Test adding new templates."""
        templates = AgentResponseTemplates()
        
        # Add a new template
        templates.add_template("coder", "custom_category", "This is a custom template with {placeholder}")
        
        # Verify it was added
        available = templates.get_available_templates()
        assert "custom_category" in available["coder"]
        
        # Test using the new template
        response = templates.get_response("coder", "custom_category", placeholder="test_value")
        assert "test_value" in response

    def test_add_substitution_option(self):
        """Test adding new substitution options."""
        templates = AgentResponseTemplates()
        
        # Add a custom substitution option
        templates.add_substitution_option("custom_placeholder", "custom_value")
        
        # Verify it was added
        options = templates.get_substitution_options()
        assert "custom_placeholder" in options
        assert "custom_value" in options["custom_placeholder"]

    def test_load_from_config(self):
        """Test loading templates from configuration."""
        templates = AgentResponseTemplates()
        
        config = {
            "templates": {
                "tester": {
                    "test_category": [
                        "Test template with {test_placeholder}"
                    ]
                }
            },
            "substitution_options": {
                "test_placeholder": ["test_value1", "test_value2"]
            }
        }
        
        templates.load_from_config(config)
        
        # Verify new agent type was loaded
        available = templates.get_available_templates()
        assert "tester" in available
        assert "test_category" in available["tester"]
        
        # Verify substitution options were loaded
        options = templates.get_substitution_options()
        assert "test_placeholder" in options
        assert "test_value1" in options["test_placeholder"]

    def test_template_substitution_fallback(self):
        """Test that missing substitutions are handled gracefully."""
        templates = AgentResponseTemplates()
        
        # Add a template with a placeholder that doesn't have substitution options
        templates.add_template("coder", "test_fallback", "Template with {missing_placeholder}")
        
        # Should not raise an error, should use fallback
        response = templates.get_response("coder", "test_fallback")
        assert "[missing_placeholder]" in response

    def test_substitution_options_no_duplicates(self):
        """Test that substitution options don't create duplicates."""
        templates = AgentResponseTemplates()
        
        # Add the same option twice
        templates.add_substitution_option("test_placeholder", "duplicate_value")
        templates.add_substitution_option("test_placeholder", "duplicate_value")
        
        options = templates.get_substitution_options()
        assert options["test_placeholder"].count("duplicate_value") == 1

    def test_default_templates_global_instance(self):
        """Test that the default global instance works correctly."""
        # Should be able to use the global default_templates instance
        response = default_templates.get_response("coder", "improvement_focused")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_template_variability(self):
        """Test that templates produce varied responses."""
        templates = AgentResponseTemplates()
        
        # Generate multiple responses from the same template category
        responses = set()
        for _ in range(20):  # Try 20 times to get variety
            response = templates.get_response("coder", "assessment")
            responses.add(response)
        
        # Should have multiple different responses (with randomization)
        assert len(responses) > 1, "Templates should produce varied responses"

    def test_template_integration_with_agents(self):
        """Test that templates integrate properly with agent loading."""
        import tempfile
        import yaml
        import os
        
        config_data = {
            "agents": {
                "coder": {"model": "test", "temperature": 0.1},
                "reviewer": {"model": "test", "temperature": 0.2}
            },
            "response_templates": {
                "templates": {
                    "coder": {
                        "custom_test": ["Test template: {focus_area}"]
                    }
                },
                "substitution_options": {
                    "focus_area": ["testing", "integration"]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # This should load the templates into the global instance
            with patch('autogen_code_review_bot.agents.validate_config_file') as mock_validate:
                mock_validate.return_value = config_data
                agents = load_agents_from_yaml(temp_path)
            
            # Test that the custom template was loaded
            available = default_templates.get_available_templates()
            assert "custom_test" in available["coder"]
            
            # Test that the template works
            response = default_templates.get_response("coder", "custom_test")
            assert "Test template:" in response
            assert any(focus in response for focus in ["testing", "integration"])
            
        finally:
            os.unlink(temp_path)

    def test_template_error_handling(self):
        """Test error handling in template operations."""
        templates = AgentResponseTemplates()
        
        # Test with malformed template that would cause format errors
        templates.add_template("coder", "malformed", "Template with {unclosed")
        
        # Should handle gracefully and return something
        response = templates.get_response("coder", "malformed")
        assert isinstance(response, str)
        assert len(response) > 0