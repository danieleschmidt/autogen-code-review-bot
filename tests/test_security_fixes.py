"""Tests for security vulnerability fixes in pr_analysis module."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from autogen_code_review_bot.pr_analysis import _run_command, analyze_pr


class TestCommandInjectionPrevention:
    """Test suite for command injection vulnerability fixes."""
    
    def test_run_command_prevents_shell_injection(self):
        """Test that _run_command prevents shell injection attacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test case 1: Basic command injection attempt
            malicious_cmd = ["echo", "safe", "&&", "rm", "-rf", "/"]
            result = _run_command(malicious_cmd, temp_dir)
            # Should not execute the dangerous part
            assert "safe" in result
            assert "rm" not in result or "command not found" in result.lower()
            
    def test_run_command_validates_executable_exists(self):
        """Test that _run_command validates executable exists before running."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with non-existent command
            result = _run_command(["nonexistent_command_12345"], temp_dir)
            assert "not found" in result.lower() or "timed out" in result.lower()
            
    def test_run_command_sanitizes_repo_path(self):
        """Test that repo paths are properly sanitized."""
        # Test with path traversal attempt
        malicious_path = "../../../etc/passwd"
        with pytest.raises((FileNotFoundError, OSError)):
            _run_command(["ls"], malicious_path)
            
    def test_analyze_pr_validates_repo_path(self):
        """Test that analyze_pr validates repository path input."""
        # Test with non-existent repository
        result = analyze_pr("/nonexistent/path/12345")
        
        # Should handle gracefully, not crash
        assert hasattr(result, 'security')
        assert hasattr(result, 'style') 
        assert hasattr(result, 'performance')
        
    def test_analyze_pr_handles_malicious_config_path(self):
        """Test that analyze_pr handles malicious config file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid repo structure
            os.makedirs(f"{temp_dir}/src")
            Path(f"{temp_dir}/src/test.py").write_text("print('hello')")
            
            # Test with path traversal in config
            malicious_config = "../../../etc/passwd"
            
            # Should handle gracefully, not crash or expose system files
            try:
                result = analyze_pr(temp_dir, malicious_config)
                # Should use defaults if config file doesn't exist or is invalid
                assert result is not None
            except (FileNotFoundError, PermissionError):
                # Expected behavior - should not be able to read system files
                pass
                
    def test_linter_command_construction_is_safe(self):
        """Test that linter commands are constructed safely."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('test')")
            
            # Mock the which function to return a safe executable
            with patch('autogen_code_review_bot.pr_analysis.which') as mock_which:
                mock_which.return_value = "/usr/bin/echo"  # Safe command
                
                with patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                    mock_run.return_value = "mocked output"
                    
                    result = analyze_pr(temp_dir)
                    
                    # Verify that commands are constructed properly
                    assert mock_run.called
                    for call in mock_run.call_args_list:
                        cmd = call[0][0]  # First argument is the command list
                        # Ensure command is a list (not a string that could be shell-injected)
                        assert isinstance(cmd, list)
                        # Ensure no shell metacharacters in individual arguments
                        for arg in cmd:
                            assert not any(char in str(arg) for char in ['&', '|', ';', '$', '`'])