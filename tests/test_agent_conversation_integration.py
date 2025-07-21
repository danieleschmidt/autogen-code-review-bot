"""Integration tests for agent conversation system with bot functionality."""

from __future__ import annotations

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from autogen_code_review_bot.pr_analysis import format_analysis_with_agents
from autogen_code_review_bot.models import PRAnalysisResult, AnalysisSection


@pytest.fixture
def mock_analysis_result():
    """Create a mock analysis result with various issues."""
    return PRAnalysisResult(
        security=AnalysisSection(
            tool="bandit",
            output="Found 2 security issues: SQL injection vulnerability on line 15, hardcoded password on line 23"
        ),
        style=AnalysisSection(
            tool="ruff", 
            output="Found 3 style issues: Line too long (line 10), unused import (line 5), inconsistent indentation (line 18)"
        ),
        performance=AnalysisSection(
            tool="radon",
            output="Found 1 performance issue: High complexity function on line 45"
        )
    )


@pytest.fixture 
def temp_agent_config():
    """Create a temporary agent configuration file."""
    config_content = """
agents:
  coder:
    model: "test-model"
    temperature: 0.1
    focus_areas:
      - "implementation details"
      - "bug detection" 
      - "performance"
  reviewer:
    model: "test-model"
    temperature: 0.2
    focus_areas:
      - "security"
      - "code quality"
      - "maintainability"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        yield f.name
    os.unlink(f.name)


class TestAgentConversationIntegration:
    """Integration tests for agent conversation system."""

    def test_format_analysis_with_agents_success(self, mock_analysis_result, temp_agent_config):
        """Test successful formatting with agent conversation."""
        result = format_analysis_with_agents(mock_analysis_result, temp_agent_config)
        
        # Should contain both original analysis and agent discussion
        assert "Code Review Analysis" in result
        assert "Security (bandit)" in result
        assert "Style (ruff)" in result  
        assert "Performance (radon)" in result
        assert "Agent Discussion" in result
        
        # Should contain conversation elements
        assert any(word in result.lower() for word in ["conversation", "summary", "discussion"])

    def test_format_analysis_with_agents_no_config(self, mock_analysis_result):
        """Test formatting falls back to basic format when no config provided."""
        result = format_analysis_with_agents(mock_analysis_result, None)
        
        # Should contain basic formatting
        assert "Security (bandit)" in result
        assert "Style (ruff)" in result
        assert "Performance (radon)" in result
        
        # Should NOT contain agent conversation elements
        assert "Agent Discussion" not in result
        assert "Conversation Summary" not in result

    def test_format_analysis_with_agents_config_error(self, mock_analysis_result):
        """Test formatting falls back gracefully when config file is invalid."""
        # Use non-existent config file
        result = format_analysis_with_agents(mock_analysis_result, "/nonexistent/config.yaml")
        
        # Should fall back to basic formatting
        assert "Security (bandit)" in result
        assert "Style (ruff)" in result
        assert "Performance (radon)" in result
        
        # Should NOT contain agent conversation elements due to error
        assert "Agent Discussion" not in result

    @patch('autogen_code_review_bot.agents.run_agent_conversation')
    def test_format_analysis_with_conversation_mock(self, mock_conversation, mock_analysis_result, temp_agent_config):
        """Test formatting with mocked agent conversation."""
        mock_conversation.return_value = "## Conversation Summary\n\n**Resolution**: Agents agreed on security fixes\n\n**Discussion Highlights**:\n1. **Coder**: Found SQL injection issue\n2. **Reviewer**: Confirmed security risk"
        
        result = format_analysis_with_agents(mock_analysis_result, temp_agent_config)
        
        # Verify conversation was called
        mock_conversation.assert_called_once()
        
        # Verify output contains both analysis and conversation
        assert "Code Review Analysis" in result
        assert "Agent Discussion" in result
        assert "Resolution" in result and "Agents agreed" in result
        assert "SQL injection" in result

    def test_extract_code_context_with_issues(self, mock_analysis_result):
        """Test code context extraction from analysis results."""
        from autogen_code_review_bot.pr_analysis import _extract_code_context
        
        context = _extract_code_context(mock_analysis_result)
        
        assert "Security findings:" in context
        assert "SQL injection vulnerability" in context
        assert "Style findings:" in context
        assert "Line too long" in context
        assert "Performance findings:" in context
        assert "High complexity function" in context

    def test_extract_code_context_no_issues(self):
        """Test code context extraction when no issues found."""
        from autogen_code_review_bot.pr_analysis import _extract_code_context
        
        clean_result = PRAnalysisResult(
            security=AnalysisSection(tool="bandit", output=""),
            style=AnalysisSection(tool="ruff", output=""),
            performance=AnalysisSection(tool="radon", output="")
        )
        
        context = _extract_code_context(clean_result)
        assert "No issues detected" in context

    def test_basic_format_analysis_result(self, mock_analysis_result):
        """Test basic analysis result formatting."""
        from autogen_code_review_bot.pr_analysis import format_analysis_result
        
        result = format_analysis_result(mock_analysis_result)
        
        assert "**Security (bandit)**:" in result
        assert "**Style (ruff)**:" in result
        assert "**Performance (radon)**:" in result
        assert "SQL injection" in result
        assert "Line too long" in result
        assert "High complexity" in result

    def test_basic_format_analysis_result_no_issues(self):
        """Test basic formatting when no issues found."""
        from autogen_code_review_bot.pr_analysis import format_analysis_result
        
        clean_result = PRAnalysisResult(
            security=AnalysisSection(tool="bandit", output=""),
            style=AnalysisSection(tool="ruff", output=""),
            performance=AnalysisSection(tool="radon", output="")
        )
        
        result = format_analysis_result(clean_result)
        assert "✅ No issues found" in result


class TestBotCLIIntegration:
    """Tests for bot CLI integration with agent conversations."""

    @patch('autogen_code_review_bot.pr_analysis.analyze_pr')
    @patch('autogen_code_review_bot.pr_analysis.format_analysis_with_agents')
    def test_manual_analysis_with_agent_config(self, mock_format, mock_analyze):
        """Test manual analysis CLI with agent configuration."""
        # Mock the analysis result
        mock_result = Mock()
        mock_analyze.return_value = mock_result
        mock_format.return_value = "Enhanced analysis with agent conversation"
        
        # Import and test the manual_analysis function
        from bot import manual_analysis
        
        # Create temp directory for repo path
        with tempfile.TemporaryDirectory() as temp_repo:
            with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as agent_config:
                try:
                    # Test with agent config
                    manual_analysis(temp_repo, None, agent_config.name)
                    
                    # Verify analysis was called
                    mock_analyze.assert_called_once_with(temp_repo, None)
                    
                    # Verify agent formatting was used
                    mock_format.assert_called_once_with(mock_result, agent_config.name)
                    
                finally:
                    os.unlink(agent_config.name)

    @patch('autogen_code_review_bot.pr_analysis.analyze_pr')
    @patch('builtins.print')  # Mock print to capture output
    def test_manual_analysis_without_agent_config(self, mock_print, mock_analyze):
        """Test manual analysis CLI without agent configuration."""
        # Mock the analysis result  
        mock_result = Mock()
        mock_result.security.tool = "bandit"
        mock_result.security.output = "No issues"
        mock_result.style.tool = "ruff"
        mock_result.style.output = "No issues" 
        mock_result.performance.tool = "radon"
        mock_result.performance.output = "No issues"
        
        mock_analyze.return_value = mock_result
        
        # Import and test the manual_analysis function
        from bot import manual_analysis
        
        # Create temp directory for repo path
        with tempfile.TemporaryDirectory() as temp_repo:
            # Test without agent config
            manual_analysis(temp_repo, None, None)
            
            # Verify analysis was called
            mock_analyze.assert_called_once_with(temp_repo, None)
            
            # Verify traditional output format was used (should contain tip about agent config)
            print_calls = [call[0][0] for call in mock_print.call_args_list if call[0]]
            tip_found = any("agent-config" in str(call).lower() for call in print_calls)
            assert tip_found, "Should display tip about --agent-config option"