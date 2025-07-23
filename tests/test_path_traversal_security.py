"""Test security fixes for path traversal vulnerabilities."""

import os
import tempfile
from pathlib import Path
import pytest

from autogen_code_review_bot.pr_analysis import _validate_command_safety, _validate_path_safety


class TestPathTraversalSecurity:
    """Test path traversal security fixes."""
    
    def test_validate_command_safety_basic_allowed(self):
        """Test that allowed commands pass validation."""
        assert _validate_command_safety(["ruff", "check", "file.py"])
        assert _validate_command_safety(["eslint", "--format", "json", "file.js"])
        assert _validate_command_safety(["bandit", "-r", "src/"])
    
    def test_validate_command_safety_rejects_path_traversal(self):
        """Test that path traversal in executable paths is rejected."""
        # These should be rejected due to path traversal attempts
        assert not _validate_command_safety(["../../../bin/ruff", "check"])
        assert not _validate_command_safety(["/tmp/../../../usr/bin/ruff", "check"])
        assert not _validate_command_safety(["./../../malicious/ruff", "check"])
        
    def test_validate_command_safety_rejects_absolute_paths(self):
        """Test that absolute paths to executables are rejected unless in allowlist."""
        assert not _validate_command_safety(["/usr/bin/ruff", "check"])
        assert not _validate_command_safety(["/home/user/custom/ruff", "check"])
        
    def test_validate_command_safety_with_symlinks(self):
        """Test path traversal through symlinks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a malicious executable
            malicious_path = Path(temp_dir) / "malicious"
            malicious_path.write_text("#!/bin/bash\necho 'pwned'")
            malicious_path.chmod(0o755)
            
            # Create symlink with allowed name that points to malicious executable
            link_path = Path(temp_dir) / "ruff"
            link_path.symlink_to(malicious_path)
            
            # This should be rejected
            assert not _validate_command_safety([str(link_path), "check"])
    
    def test_validate_path_safety_basic_safe_paths(self):
        """Test that normal safe paths pass validation."""
        assert _validate_path_safety("src/main.py")
        assert _validate_path_safety("./config.yaml")
        assert _validate_path_safety("tests/test_file.py")
        
    def test_validate_path_safety_rejects_traversal_patterns(self):
        """Test that various path traversal patterns are rejected."""
        # Classic traversal patterns
        assert not _validate_path_safety("../../../etc/passwd")
        assert not _validate_path_safety("..\\..\\..\\windows\\system32")
        assert not _validate_path_safety("./../../etc/shadow")
        
        # URL encoded traversal
        assert not _validate_path_safety("%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd")
        
        # Double encoded
        assert not _validate_path_safety("%252e%252e%252fetc%252fpasswd")
        
        # Unicode normalization attacks
        assert not _validate_path_safety("..%c0%af..%c0%afetc%c0%afpasswd")
        
    def test_validate_path_safety_rejects_sensitive_directories(self):
        """Test that paths to sensitive system directories are rejected."""
        sensitive_paths = [
            "/etc/passwd",
            "/etc/shadow", 
            "/proc/self/environ",
            "/home/user/.ssh/id_rsa",
            "/root/.bashrc",
            "C:\\Windows\\System32\\config\\SAM",
            "C:\\Users\\Administrator\\.ssh\\id_rsa"
        ]
        
        for path in sensitive_paths:
            assert not _validate_path_safety(path)
    
    def test_validate_path_safety_with_symlinks(self):
        """Test path validation with symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file outside the temp directory
            outside_file = "/tmp/sensitive_file"
            with open(outside_file, "w") as f:
                f.write("sensitive data")
            
            try:
                # Create symlink inside temp_dir pointing outside
                link_path = Path(temp_dir) / "innocent_name"
                link_path.symlink_to(outside_file)
                
                # This should be detected and rejected
                assert not _validate_path_safety(str(link_path))
                
            finally:
                # Cleanup
                if os.path.exists(outside_file):
                    os.unlink(outside_file)
    
    def test_validate_path_safety_within_project_boundary(self):
        """Test that paths are validated against project boundaries."""
        with tempfile.TemporaryDirectory() as project_root:
            # Valid paths within project
            valid_file = Path(project_root) / "src" / "main.py"
            valid_file.parent.mkdir(parents=True)
            valid_file.write_text("# valid content")
            
            # Test with project root - should pass for paths within project
            assert _validate_path_safety(str(valid_file), project_root)
            
            # Test path outside project - should fail
            outside_file = "/tmp/outside_project.py"
            assert not _validate_path_safety(outside_file, project_root)
    
    def test_validate_path_safety_edge_cases(self):
        """Test edge cases for path validation."""
        # Empty or None paths
        assert not _validate_path_safety("")
        assert not _validate_path_safety(None)
        
        # Non-string types
        assert not _validate_path_safety(123)
        assert not _validate_path_safety(["path", "list"])
        assert not _validate_path_safety({"path": "dict"})
        
        # Very long paths (potential buffer overflow)
        long_path = "a" * 10000
        assert not _validate_path_safety(long_path)
        
        # Paths with null bytes
        assert not _validate_path_safety("valid/path\x00../../../etc/passwd")
        
        # Paths with special characters that might bypass filters
        assert not _validate_path_safety("valid/path\r\n../../../etc/passwd")
        assert not _validate_path_safety("valid/path\t../../../etc/passwd")


if __name__ == "__main__":
    pytest.main([__file__])