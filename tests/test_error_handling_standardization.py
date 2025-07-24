"""Tests for standardized error handling patterns."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from autogen_code_review_bot.pr_analysis import (
    analyze_pr, 
    _run_command, 
    _run_single_linter, 
    _run_security_checks, 
    _run_performance_checks
)
from autogen_code_review_bot.exceptions import (
    AnalysisError, 
    ValidationError, 
    ToolError
)


class TestErrorHandlingStandardization:
    """Test standardized error handling patterns."""

    def test_validate_path_safety_raises_validation_error(self):
        """Test that unsafe paths raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsafe repository path rejected"):
            analyze_pr("../../../etc/passwd")

    def test_invalid_repo_path_raises_validation_error(self):
        """Test that invalid repo paths raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid repository path"):
            analyze_pr("")

        with pytest.raises(ValidationError, match="Invalid repository path"):
            analyze_pr(None)

    def test_nonexistent_repo_path_raises_validation_error(self):
        """Test that nonexistent paths raise ValidationError."""
        with pytest.raises(ValidationError, match="Repository path does not exist"):
            analyze_pr("/nonexistent/path/12345")

    def test_run_command_unsafe_command_raises_validation_error(self):
        """Test that unsafe commands raise ValidationError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="Unsafe command rejected"):
                _run_command(["rm", "-rf", "/"], temp_dir)

    def test_run_command_unsafe_path_raises_validation_error(self):
        """Test that unsafe working directories raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsafe working directory path rejected"):
            _run_command(["ls"], "../../../etc")

    def test_run_command_nonexistent_directory_raises_validation_error(self):
        """Test that nonexistent working directories raise ValidationError."""
        with pytest.raises(ValidationError, match="Working directory does not exist"):
            _run_command(["ls"], "/nonexistent/directory/12345")

    def test_run_command_timeout_raises_tool_error(self):
        """Test that command timeouts raise ToolError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ToolError, match="timed out"):
                # Use a command that will timeout
                _run_command(["sleep", "10"], temp_dir, timeout=0.1)

    def test_run_command_nonexistent_tool_raises_tool_error(self):
        """Test that nonexistent but allowed tools raise ToolError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use an allowed tool that's very unlikely to exist
            # but if it does exist, we won't run this test  
            with patch('shutil.which', return_value=None):
                with pytest.raises(ToolError, match="Failed to execute"):
                    _run_command(["eslint"], temp_dir)  # eslint is in ALLOWED_EXECUTABLES

    def test_run_single_linter_tool_error_propagation(self):
        """Test that linter tool errors are properly propagated with context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run, \
                 patch('autogen_code_review_bot.pr_analysis.which', return_value="/usr/bin/ruff"):
                mock_run.side_effect = ToolError("Command failed")
                
                with pytest.raises(ToolError, match="Command failed"):
                    _run_single_linter("python", "ruff", temp_dir)

    def test_security_checks_tool_error_propagation(self):
        """Test that security check errors are properly propagated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            def mock_ensure(tool):
                return ""  # Tool is available
            
            with patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                mock_run.side_effect = ToolError("Bandit failed")
                
                with pytest.raises(ToolError, match="Security analysis failed"):
                    _run_security_checks(temp_dir, mock_ensure)

    def test_performance_checks_tool_error_propagation(self):
        """Test that performance check errors are properly propagated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            def mock_ensure(tool):
                return ""  # Tool is available
            
            with patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                mock_run.side_effect = ValidationError("Invalid path")
                
                with pytest.raises(ToolError, match="Performance analysis failed"):
                    _run_performance_checks(temp_dir, mock_ensure)

    def test_error_logging_with_context(self):
        """Test that errors are logged with appropriate context."""
        with patch('autogen_code_review_bot.pr_analysis.logger') as mock_logger:
            with pytest.raises(ValidationError):
                analyze_pr("")
            
            # Verify error was logged with context
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert "Invalid repository path provided" in call_args[0][0]
            assert "repo_path" in call_args[1]["extra"]

    def test_metrics_recording_on_errors(self):
        """Test that error metrics are properly recorded."""
        with patch('autogen_code_review_bot.pr_analysis.metrics') as mock_metrics:
            with pytest.raises(ValidationError):
                analyze_pr("")
            
            # Verify error metrics were recorded
            mock_metrics.record_counter.assert_called_with(
                "pr_analysis_errors_total", 1, tags={"error_type": "invalid_path"}
            )

    def test_exception_chaining_preserves_context(self):
        """Test that exception chaining preserves original error context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run, \
                 patch('autogen_code_review_bot.pr_analysis.which', return_value="/usr/bin/ruff"):
                original_error = OSError("Permission denied")
                def side_effect(*args, **kwargs):
                    raise ToolError("Command failed") from original_error
                mock_run.side_effect = side_effect
                
                try:
                    _run_single_linter("python", "ruff", temp_dir)
                    assert False, "Should have raised exception"
                except ToolError as e:
                    # Check exception chaining - look for original error in the chain
                    assert e.__cause__ is not None
                    # The original error should be accessible through the cause chain
                    cause = e.__cause__
                    while cause is not None:
                        if "Permission denied" in str(cause):
                            break
                        cause = cause.__cause__
                    assert cause is not None, "Original Permission denied error should be in exception chain"

    def test_error_handling_consistency_across_modules(self):
        """Test that error handling patterns are consistent."""
        # This test verifies that similar error conditions produce similar exceptions
        
        # Path validation errors should all be ValidationError
        with pytest.raises(ValidationError):
            analyze_pr("")
        
        with pytest.raises(ValidationError):
            analyze_pr("../../../etc/passwd")
        
        with pytest.raises(ValidationError):
            analyze_pr("/nonexistent/path")

    def test_specific_exception_types_are_used(self):
        """Test that specific exception types are used appropriately."""
        # ValidationError for input validation
        with pytest.raises(ValidationError, match="Invalid repository path"):
            analyze_pr(None)
        
        # ValidationError for disallowed commands
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValidationError, match="Unsafe command rejected"):
                _run_command(["nonexistent_command_12345"], temp_dir)

    def test_error_messages_are_descriptive(self):
        """Test that error messages provide useful information."""
        try:
            analyze_pr("../../../etc/passwd")
            assert False, "Should have raised exception"
        except ValidationError as e:
            # Error message should include the problematic path
            assert "../../../etc/passwd" in str(e)
            assert "Unsafe repository path" in str(e)

    def test_tool_availability_logging(self):
        """Test that tool availability issues are properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            def mock_ensure(tool):
                return "not installed"
            
            with patch('autogen_code_review_bot.pr_analysis.logger') as mock_logger:
                result = _run_security_checks(temp_dir, mock_ensure)
                
                # Should log warning for unavailable tool
                mock_logger.warning.assert_called_once()
                call_args = mock_logger.warning.call_args
                assert "Security tool not available" in call_args[0][0]
                assert call_args[1]["extra"]["tool"] == "bandit"
                
                # Should return the availability message
                assert result == "not installed"