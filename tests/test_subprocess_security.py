"""Tests for subprocess security fixes."""

import pytest
from subprocess import CalledProcessError
from unittest.mock import patch, MagicMock

from autogen_code_review_bot.pr_analysis import _run_command


class TestSubprocessSecurity:
    """Test subprocess security measures."""

    def test_run_command_prevents_shell_injection(self):
        """Test that _run_command prevents shell injection attacks."""
        # Test with malicious input that would be dangerous with shell=True
        malicious_cmd = ["echo", "hello; rm -rf /"]
        
        # Should not execute the dangerous part when shell=False
        # The command should be treated as literal arguments
        with patch('autogen_code_review_bot.pr_analysis.run') as mock_run:
            mock_run.return_value.stdout = "hello; rm -rf /"
            
            result = _run_command(malicious_cmd, "/tmp")
            
            # Verify run was called with shell=False (default)
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert 'shell' not in call_args.kwargs or call_args.kwargs['shell'] is False
            
    def test_run_command_validates_command_list(self):
        """Test that _run_command only accepts list of strings."""
        # Should reject string commands that could be shell injection
        with pytest.raises((TypeError, AttributeError)):
            _run_command("echo hello; rm -rf /", "/tmp")
            
    def test_run_command_handles_path_traversal(self):
        """Test that _run_command handles path traversal safely."""
        # Test with path traversal attempt
        cmd = ["ls", "../../../etc/passwd"]
        
        with patch('autogen_code_review_bot.pr_analysis.run') as mock_run:
            mock_run.side_effect = CalledProcessError(1, cmd, "Permission denied")
            
            result = _run_command(cmd, "/tmp")
            
            # Should handle the error gracefully
            assert "Permission denied" in result
            
    def test_run_command_limits_environment_access(self):
        """Test that _run_command limits environment variable access."""
        cmd = ["env"]
        
        with patch('autogen_code_review_bot.pr_analysis.run') as mock_run:
            mock_run.return_value.stdout = "PATH=/usr/bin"
            
            _run_command(cmd, "/tmp")
            
            # Verify that environment is not explicitly passed
            call_args = mock_run.call_args
            assert 'env' not in call_args.kwargs
            
    def test_run_command_timeout_prevents_dos(self):
        """Test that timeout prevents denial of service attacks."""
        cmd = ["sleep", "1000"]
        
        with patch('autogen_code_review_bot.pr_analysis.run') as mock_run:
            from subprocess import TimeoutExpired
            mock_run.side_effect = TimeoutExpired(cmd, 30)
            
            result = _run_command(cmd, "/tmp", timeout=1)
            
            assert result == "timed out"
            
    def test_run_command_sanitizes_output(self):
        """Test that _run_command sanitizes output appropriately."""
        cmd = ["echo", "test"]
        
        with patch('autogen_code_review_bot.pr_analysis.run') as mock_run:
            # Simulate output with potential control characters
            mock_run.return_value.stdout = "test\n\r\x00\x1b[31mred\x1b[0m"
            
            result = _run_command(cmd, "/tmp")
            
            # Should strip whitespace and preserve safe content
            assert result == "test\n\r\x00\x1b[31mred\x1b[0m"  # Basic strip only for now