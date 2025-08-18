"""
Foundational tests for AutoGen Code Review Bot core functionality.

These tests ensure basic functionality works correctly.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.models import PRAnalysisResult, AnalysisSection
from autogen_code_review_bot.language_detection import detect_language
from autogen_code_review_bot.exceptions import AnalysisError


class TestBasicFunctionality:
    """Test basic system functionality."""

    def test_language_detection_python(self):
        """Test language detection for Python files."""
        files = ["main.py", "utils.py", "test_file.py"]
        detected = detect_language(files)
        assert "python" in detected

    def test_language_detection_javascript(self):
        """Test language detection for JavaScript files.""" 
        files = ["app.js", "script.js", "index.js"]
        detected = detect_language(files)
        assert "javascript" in detected

    def test_language_detection_mixed(self):
        """Test language detection for mixed file types."""
        files = ["main.py", "app.js", "style.css", "README.md"]
        detected = detect_language(files)
        assert "python" in detected
        assert "javascript" in detected

    def test_pr_analysis_result_model(self):
        """Test PRAnalysisResult model creation."""
        security = AnalysisSection(tool="bandit", output="No issues found")
        style = AnalysisSection(tool="ruff", output="Code style looks good")
        performance = AnalysisSection(tool="custom", output="No performance issues")
        
        result = PRAnalysisResult(
            security=security,
            style=style,
            performance=performance
        )
        
        assert result.security.tool == "bandit"
        assert result.style.tool == "ruff"
        assert result.performance.tool == "custom"

    def test_analysis_section_model(self):
        """Test AnalysisSection model."""
        section = AnalysisSection(
            tool="test_tool",
            output="test output",
            return_code=0
        )
        
        assert section.tool == "test_tool"
        assert section.output == "test output"
        assert section.return_code == 0

    @pytest.mark.integration
    def test_analyze_pr_with_temp_repo(self, temp_dir):
        """Test PR analysis with a temporary repository."""
        # Create a simple Python file
        repo_dir = temp_dir / "test_repo"
        repo_dir.mkdir()
        
        python_file = repo_dir / "test.py"
        python_file.write_text("""
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
""")
        
        # Mock subprocess calls for linters
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="No issues found",
                stderr=""
            )
            
            result = analyze_pr(str(repo_dir))
            
            assert isinstance(result, PRAnalysisResult)
            assert result.security is not None
            assert result.style is not None
            assert result.performance is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_analyze_pr_nonexistent_path(self):
        """Test analyze_pr with non-existent path."""
        # Should raise an exception for non-existent paths
        with pytest.raises(AnalysisError):
            analyze_pr("/nonexistent/path")

    def test_analyze_pr_empty_directory(self, temp_dir):
        """Test analyze_pr with empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        # Should handle empty directories gracefully
        result = analyze_pr(str(empty_dir))
        assert isinstance(result, PRAnalysisResult)

    def test_language_detection_empty_files(self):
        """Test language detection with empty file list."""
        detected = detect_language([])
        assert isinstance(detected, (list, set, dict))

    def test_language_detection_unknown_extensions(self):
        """Test language detection with unknown file extensions."""
        files = ["unknown.xyz", "weird.abc"]
        detected = detect_language(files)
        # Should handle gracefully without crashing
        assert isinstance(detected, (list, set, dict))


class TestConfigurationHandling:
    """Test configuration and settings handling."""

    def test_analyze_pr_with_config_path(self, temp_dir):
        """Test analyze_pr with custom config path."""
        repo_dir = temp_dir / "test_repo"
        repo_dir.mkdir()
        
        # Create a simple config file
        config_file = temp_dir / "linters.yaml"
        config_file.write_text("""
linters:
  python: ruff
  javascript: eslint
""")
        
        python_file = repo_dir / "test.py"
        python_file.write_text("print('hello')")
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="No issues found",
                stderr=""
            )
            
            result = analyze_pr(str(repo_dir), str(config_file))
            assert isinstance(result, PRAnalysisResult)

    def test_analyze_pr_with_invalid_config(self, temp_dir):
        """Test analyze_pr with invalid config path."""
        repo_dir = temp_dir / "test_repo"
        repo_dir.mkdir()
        
        python_file = repo_dir / "test.py"
        python_file.write_text("print('hello')")
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="No issues found",
                stderr=""
            )
            
            # Should handle invalid config gracefully
            result = analyze_pr(str(repo_dir), "/nonexistent/config.yaml")
            assert isinstance(result, PRAnalysisResult)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)