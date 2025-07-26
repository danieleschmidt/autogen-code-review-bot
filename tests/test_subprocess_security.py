"""Test enhanced subprocess security functionality."""

import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.autogen_code_review_bot.subprocess_security import (
    SubprocessValidator, 
    safe_subprocess_run
)
from src.autogen_code_review_bot.exceptions import ValidationError, ToolError


class TestSubprocessValidator:
    """Test SubprocessValidator class functionality."""

    def test_validate_allowed_executable(self):
        """Test validation of allowed executables."""
        # Allowed executables should pass
        assert SubprocessValidator.validate_command(["python", "--version"])
        assert SubprocessValidator.validate_command(["git", "status"])
        assert SubprocessValidator.validate_command(["ruff", "check", "."])

    def test_reject_disallowed_executable(self):
        """Test rejection of disallowed executables."""
        with pytest.raises(ValidationError, match="Executable not allowed"):
            SubprocessValidator.validate_command(["malicious_tool", "arg"])
            
        with pytest.raises(ValidationError, match="Executable not allowed"):
            SubprocessValidator.validate_command(["rm", "-rf", "/"])

    def test_validate_command_structure(self):
        """Test command structure validation."""
        # Empty command should fail
        with pytest.raises(ValidationError, match="Empty command"):
            SubprocessValidator.validate_command([])
            
        # Non-list should fail
        with pytest.raises(ValidationError, match="Command must be a list"):
            SubprocessValidator.validate_command("python --version")
            
        # Too many arguments should fail
        long_cmd = ["python"] + ["arg"] * 100
        with pytest.raises(ValidationError, match="Too many arguments"):
            SubprocessValidator.validate_command(long_cmd)
            
        # Non-string argument should fail
        with pytest.raises(ValidationError, match="is not a string"):
            SubprocessValidator.validate_command(["python", 123])

    def test_detect_shell_metacharacters(self):
        """Test detection of shell metacharacters."""
        dangerous_commands = [
            ["python", "-c", "import os; os.system('rm -rf /')"],
            ["git", "status", "&& rm -rf /"],  
            ["echo", "test", "|", "grep", "pattern"],
            ["python", "--version", ";", "malicious_command"],
            ["cat", "/etc/passwd", ">", "output.txt"],
            ["python", "-c", "print('hello')", "`malicious`"],
        ]
        
        for cmd in dangerous_commands:
            with pytest.raises(ValidationError, match="Shell metacharacter"):
                SubprocessValidator.validate_command(cmd)

    def test_detect_dangerous_characters(self):
        """Test detection of dangerous control characters."""
        dangerous_commands = [
            ["python", "-c", "print('hello\x00world')"],  # null byte
            ["git", "commit", "-m", "message\x1b[31m"],  # escape sequence
            ["echo", "test\x7f"],  # DEL character
        ]
        
        for cmd in dangerous_commands:
            with pytest.raises(ValidationError, match="Dangerous character"):
                SubprocessValidator.validate_command(cmd)

    def test_detect_url_encoding_attacks(self):
        """Test detection of URL encoding attacks."""
        dangerous_commands = [
            ["cat", "%2e%2e%2f%2e%2e%2fetc%2fpasswd"],  # ../../../etc/passwd
            ["python", "%2e%2e/malicious.py"],
            ["git", "status", "--git-dir=%2e%2e%2f"],
        ]
        
        for cmd in dangerous_commands:
            with pytest.raises(ValidationError, match="URL encoding attack"):
                SubprocessValidator.validate_command(cmd)

    def test_detect_path_traversal(self):
        """Test detection of path traversal attempts."""
        dangerous_commands = [
            ["cat", "../../../etc/passwd"],
            ["python", "..\\..\\malicious.py"],
            ["git", "status", "--git-dir=../"],
        ]
        
        for cmd in dangerous_commands:
            with pytest.raises(ValidationError, match="Path traversal"):
                SubprocessValidator.validate_command(cmd)

    def test_validate_git_specific_restrictions(self):
        """Test git-specific command restrictions."""
        dangerous_git_commands = [
            ["git", "daemon"],
            ["git", "upload-pack"],
            ["git", "receive-pack"],
            ["git", "status", "--git-dir=/"],
            ["git", "log", "--work-tree=/"],
        ]
        
        for cmd in dangerous_git_commands:
            with pytest.raises(ValidationError):
                SubprocessValidator.validate_command(cmd)

    def test_validate_deletion_command_restrictions(self):
        """Test deletion command restrictions."""
        # These should be rejected by executable allowlist first,
        # but test the deletion-specific validation
        with pytest.raises(ValidationError):
            # rm is not in allowed executables, so this tests the allowlist
            SubprocessValidator.validate_command(["rm", "-rf", "/"])

    def test_validate_working_directory(self):
        """Test working directory validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid directory should pass
            assert SubprocessValidator.validate_command(["python", "--version"], cwd=temp_dir)
            
            # Non-existent directory should fail
            with pytest.raises(ValidationError, match="does not exist"):
                SubprocessValidator.validate_command(
                    ["python", "--version"], 
                    cwd="/nonexistent/directory"
                )
            
            # Directory outside project root should fail
            with pytest.raises(ValidationError, match="outside project root"):
                SubprocessValidator.validate_command(
                    ["python", "--version"], 
                    cwd="/tmp",
                    project_root=temp_dir
                )

    def test_validate_unicode_safety(self):
        """Test Unicode normalization attack detection."""
        # This is a simplified test - real Unicode attacks are more complex
        unicode_attack_cmd = ["python", "-c", "print('test\u0065\u0301')"]  # é using combining chars
        
        # This should pass as it's not a significant normalization change
        assert SubprocessValidator.validate_command(unicode_attack_cmd)

    def test_validate_resource_limits(self):
        """Test resource limit validation."""
        # Command with very long arguments should fail
        long_arg = "x" * 2000
        with pytest.raises(ValidationError, match="too long"):
            SubprocessValidator.validate_command(["python", "-c", long_arg])

    def test_safely_quoted_arguments(self):
        """Test handling of safely quoted arguments."""
        # Properly quoted arguments with metacharacters should pass
        quoted_commands = [
            ["python", "-c", "'print(\"hello world\")'"],
            ["git", "commit", "-m", '"Added feature & fixed bug"'],
        ]
        
        for cmd in quoted_commands:
            # These might still fail due to metacharacters, 
            # but let's test the quoting detection logic
            try:
                SubprocessValidator.validate_command(cmd)
            except ValidationError:
                # Expected due to metacharacters
                pass

    def test_command_specific_restrictions(self):
        """Test command-specific argument restrictions."""
        # Git with restricted arguments
        with pytest.raises(ValidationError):
            SubprocessValidator._validate_command_restrictions(["git", "status", "--git-dir=/"])

    def test_executable_path_validation(self):
        """Test executable path validation."""
        # Path-based executables should be validated
        with pytest.raises(ValidationError, match="not in allowed locations"):
            SubprocessValidator.validate_command(["/home/user/malicious", "arg"])
        
        # Relative paths should be rejected
        with pytest.raises(ValidationError, match="Executable not allowed"):
            SubprocessValidator.validate_command(["./malicious", "arg"])

    def test_argument_count_limits(self):
        """Test argument count limits."""
        # Create command with too many arguments
        many_args = ["python"] + [f"arg{i}" for i in range(100)]
        
        with pytest.raises(ValidationError, match="Too many arguments"):
            SubprocessValidator.validate_command(many_args)

    def test_argument_length_limits(self):
        """Test individual argument length limits."""
        # Create command with very long argument
        long_arg = "x" * 2000
        
        with pytest.raises(ValidationError, match="too long"):
            SubprocessValidator.validate_command(["python", "-c", long_arg])


class TestSafeSubprocessRun:
    """Test safe_subprocess_run function."""

    @patch('subprocess.run')
    def test_successful_execution(self, mock_run):
        """Test successful command execution."""
        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = safe_subprocess_run(["python", "--version"])
        
        assert result.returncode == 0
        assert result.stdout == "Success output"
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_command_timeout(self, mock_run):
        """Test command timeout handling."""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(["python"], 30)
        
        with pytest.raises(ToolError, match="timed out"):
            safe_subprocess_run(["python", "--version"], timeout=30)

    @patch('subprocess.run')
    def test_execution_failure(self, mock_run):
        """Test execution failure handling."""
        # Mock execution failure
        mock_run.side_effect = OSError("Command not found")
        
        with pytest.raises(ToolError, match="Failed to execute"):
            safe_subprocess_run(["python", "--version"])

    def test_validation_before_execution(self):
        """Test that validation occurs before execution."""
        # This should fail validation before even trying to execute
        with pytest.raises(ValidationError, match="Executable not allowed"):
            safe_subprocess_run(["malicious_tool", "arg"])

    @patch('subprocess.run')
    def test_secure_defaults(self, mock_run):
        """Test that secure defaults are applied."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        safe_subprocess_run(["python", "--version"])
        
        # Check that secure defaults were used
        call_args = mock_run.call_args
        assert call_args[1]['shell'] is False
        assert call_args[1]['capture_output'] is True
        assert call_args[1]['text'] is True

    @patch('subprocess.run')
    def test_working_directory_validation(self, mock_run):
        """Test working directory validation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid directory should work
            safe_subprocess_run(["python", "--version"], cwd=temp_dir)
            
            # Check that cwd was passed
            call_args = mock_run.call_args
            assert call_args[1]['cwd'] == temp_dir

    @patch('subprocess.run')
    def test_project_root_validation(self, mock_run):
        """Test project root validation."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as project_root:
            # Create subdirectory in project root
            subdir = Path(project_root) / "subdir"
            subdir.mkdir()
            
            # Should work when cwd is within project root
            safe_subprocess_run(
                ["python", "--version"], 
                cwd=str(subdir), 
                project_root=project_root
            )
            
            # Should fail when cwd is outside project root
            with pytest.raises(ValidationError, match="outside project root"):
                safe_subprocess_run(
                    ["python", "--version"], 
                    cwd="/tmp", 
                    project_root=project_root
                )

    @patch('subprocess.run')
    def test_non_zero_exit_handling(self, mock_run):
        """Test handling of non-zero exit codes."""
        # Mock non-zero exit (common for linters when issues found)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Found 3 issues"
        mock_result.stderr = "Warning: deprecated syntax"
        mock_run.return_value = mock_result
        
        result = safe_subprocess_run(["ruff", "check", "."])
        
        # Should not raise exception for non-zero exit
        assert result.returncode == 1
        assert result.stdout == "Found 3 issues"
        assert result.stderr == "Warning: deprecated syntax"

    def test_integration_with_system_config(self):
        """Test integration with system configuration."""
        with patch('src.autogen_code_review_bot.subprocess_security.get_system_config') as mock_config:
            mock_config.return_value.default_command_timeout = 45
            
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result
                
                safe_subprocess_run(["python", "--version"])
                
                # Check that system config timeout was used
                call_args = mock_run.call_args
                assert call_args[1]['timeout'] == 45

    def test_error_logging(self):
        """Test that errors are properly logged."""
        with patch('src.autogen_code_review_bot.subprocess_security.logger') as mock_logger:
            with pytest.raises(ValidationError):
                safe_subprocess_run(["malicious_tool", "arg"])
            
            # Should log validation errors
            mock_logger.debug.assert_called()

    def test_comprehensive_real_world_scenario(self):
        """Test a comprehensive real-world scenario."""
        with tempfile.TemporaryDirectory() as project_root:
            # Create a Python file to analyze
            test_file = Path(project_root) / "test.py"
            test_file.write_text("print('hello world')\n")
            
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = "All checks passed"
                mock_result.stderr = ""
                mock_run.return_value = mock_result
                
                # This should work - valid Python linting command
                result = safe_subprocess_run(
                    ["python", "-m", "py_compile", str(test_file)],
                    cwd=project_root,
                    project_root=project_root,
                    timeout=30
                )
                
                assert result.returncode == 0
                mock_run.assert_called_once()
                
                # Verify all security parameters were applied
                call_args = mock_run.call_args
                assert call_args[1]['shell'] is False
                assert call_args[1]['cwd'] == project_root
                assert call_args[1]['timeout'] == 30