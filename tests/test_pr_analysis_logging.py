"""Tests for PR analysis logging integration."""

import json
import logging
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.logging_config import configure_logging


class TestPRAnalysisLogging:
    """Test logging integration in PR analysis."""
    
    def setup_method(self):
        """Set up test logging capture."""
        # Create a string buffer to capture log output
        self.log_buffer = StringIO()
        
        # Configure logging to output to our buffer
        configure_logging(level="DEBUG", service_name="test-pr-analysis")
        
        # Add a handler to capture logs
        self.handler = logging.StreamHandler(self.log_buffer)
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Get the PR analysis logger and add our handler
        logger = logging.getLogger('autogen_code_review_bot.pr_analysis')
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)
    
    def teardown_method(self):
        """Clean up test logging."""
        logger = logging.getLogger('autogen_code_review_bot.pr_analysis')
        logger.removeHandler(self.handler)
    
    def get_log_messages(self):
        """Extract log messages from the buffer."""
        self.log_buffer.seek(0)
        messages = []
        for line in self.log_buffer.readlines():
            line = line.strip()
            if line:
                try:
                    # Try to parse as JSON first
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    # If not JSON, keep as string
                    messages.append(line)
        return messages
    
    def test_successful_analysis_logging(self):
        """Test that successful analysis operations are logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple Python file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('hello world')")
            
            # Mock the linter commands to avoid needing actual tools
            with patch('autogen_code_review_bot.pr_analysis.which') as mock_which, \
                 patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                
                mock_which.return_value = "/usr/bin/ruff"  # Pretend tools are installed
                mock_run.return_value = "No issues found"
                
                # Run analysis
                result = analyze_pr(temp_dir, use_cache=False, use_parallel=True)
                
                # Check that we got a result
                assert result is not None
                
                # Get log messages
                messages = self.get_log_messages()
                
                # Verify key log messages are present
                log_text = ' '.join(str(msg) for msg in messages)
                assert "Starting PR analysis" in log_text
                assert "Running analysis with parallel execution" in log_text
                assert "Analysis completed successfully" in log_text
    
    def test_invalid_path_logging(self):
        """Test that invalid path errors are logged."""
        # Try to analyze a non-existent path
        result = analyze_pr("/nonexistent/path", use_cache=False)
        
        # Get log messages
        messages = self.get_log_messages()
        log_text = ' '.join(str(msg) for msg in messages)
        
        # Verify error logging
        assert "Repository path validation failed" in log_text
        assert result.security.tool == "error"
    
    def test_cache_hit_logging(self):
        """Test that cache operations are logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple Python file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('hello world')")
            
            # Mock git to return a commit hash
            with patch('autogen_code_review_bot.caching.run') as mock_git, \
                 patch('autogen_code_review_bot.pr_analysis.which') as mock_which, \
                 patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                
                # Mock git command to return a commit hash
                mock_git.return_value = Mock(stdout="abc123def456", returncode=0)
                mock_which.return_value = "/usr/bin/ruff"
                mock_run.return_value = "No issues found"
                
                # Run analysis twice - second should hit cache
                analyze_pr(temp_dir, use_cache=True, use_parallel=False)
                self.log_buffer.seek(0)
                self.log_buffer.truncate(0)  # Clear buffer
                
                analyze_pr(temp_dir, use_cache=True, use_parallel=False)
                
                # Get log messages from second run
                messages = self.get_log_messages()
                log_text = ' '.join(str(msg) for msg in messages)
                
                # Should see cache-related logging
                assert "Checking cache for existing results" in log_text
    
    def test_parallel_execution_logging(self):
        """Test that parallel execution steps are logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('hello world')")
            
            with patch('autogen_code_review_bot.pr_analysis.which') as mock_which, \
                 patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                
                mock_which.return_value = "/usr/bin/ruff"
                mock_run.return_value = "No issues found"
                
                # Run with parallel execution
                analyze_pr(temp_dir, use_cache=False, use_parallel=True)
                
                messages = self.get_log_messages()
                log_text = ' '.join(str(msg) for msg in messages)
                
                # Verify parallel execution logging
                assert "Starting parallel analysis checks" in log_text
                assert "Starting security analysis" in log_text
                assert "Starting style analysis" in log_text
                assert "Starting performance analysis" in log_text
                assert "All parallel analysis tasks completed successfully" in log_text
    
    def test_language_detection_logging(self):
        """Test that language detection is logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files in different languages
            (Path(temp_dir) / "test.py").write_text("print('hello')")
            (Path(temp_dir) / "test.js").write_text("console.log('hello');")
            
            with patch('autogen_code_review_bot.pr_analysis.which') as mock_which, \
                 patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                
                mock_which.return_value = "/usr/bin/tool"
                mock_run.return_value = "No issues"
                
                analyze_pr(temp_dir, use_cache=False, use_parallel=False)
                
                messages = self.get_log_messages()
                log_text = ' '.join(str(msg) for msg in messages)
                
                # Should log language detection results
                assert "Language detection completed" in log_text
    
    def test_configuration_loading_logging(self):
        """Test that configuration loading is logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('hello')")
            
            # Create a config file
            config_file = Path(temp_dir) / "linters.yaml"
            config_file.write_text("linters:\n  python: ruff\n")
            
            with patch('autogen_code_review_bot.pr_analysis.which') as mock_which, \
                 patch('autogen_code_review_bot.pr_analysis._run_command') as mock_run:
                
                mock_which.return_value = "/usr/bin/ruff"
                mock_run.return_value = "No issues"
                
                analyze_pr(temp_dir, config_path=str(config_file), use_cache=False)
                
                messages = self.get_log_messages()
                log_text = ' '.join(str(msg) for msg in messages)
                
                assert "Loading linter configuration" in log_text
                assert "Linter configuration loaded" in log_text