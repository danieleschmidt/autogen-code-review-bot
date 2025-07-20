"""Tests for parallel linter execution."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest

from autogen_code_review_bot.pr_analysis import (
    _run_style_checks_parallel,
    _run_all_checks_parallel,
    analyze_pr
)


class TestParallelStyleChecks:
    """Test cases for parallel style checking."""

    def test_parallel_execution_faster_than_sequential(self):
        """Test that parallel execution is faster than sequential for multiple tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files for different languages
            py_file = Path(tmpdir) / "test.py"
            js_file = Path(tmpdir) / "test.js"
            py_file.write_text("print('hello')")
            js_file.write_text("console.log('hello');")
            
            linters = {"python": "echo", "javascript": "echo"}  # Use echo as mock linter
            
            # Mock the _run_command to simulate slow execution
            def slow_mock_command(cmd, cwd, timeout=30):
                time.sleep(0.1)  # Simulate 100ms per linter
                return f"output from {cmd[0]}"
            
            with patch('autogen_code_review_bot.pr_analysis._run_command', slow_mock_command), \
                 patch('autogen_code_review_bot.pr_analysis.which', return_value='/usr/bin/echo'):
                
                # Time sequential execution
                start_time = time.time()
                _run_style_checks_parallel(tmpdir, linters, max_workers=1)
                sequential_time = time.time() - start_time
                
                # Time parallel execution
                start_time = time.time()
                _run_style_checks_parallel(tmpdir, linters, max_workers=2)
                parallel_time = time.time() - start_time
                
                # Parallel should be significantly faster
                assert parallel_time < sequential_time * 0.8

    def test_parallel_execution_preserves_results(self):
        """Test that parallel execution produces same results as sequential."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")
            
            linters = {"python": "echo"}
            
            with patch('autogen_code_review_bot.pr_analysis.which', return_value='/usr/bin/echo'):
                # Get results from both approaches
                tool1, output1 = _run_style_checks_parallel(tmpdir, linters, max_workers=1)
                tool2, output2 = _run_style_checks_parallel(tmpdir, linters, max_workers=2)
                
                # Results should be identical
                assert tool1 == tool2
                assert output1 == output2

    def test_parallel_execution_handles_errors_gracefully(self):
        """Test that parallel execution handles individual tool failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")
            
            linters = {"python": "nonexistent-tool", "javascript": "echo"}
            
            def mock_ensure(tool):
                return "not installed" if tool == "nonexistent-tool" else ""
            
            with patch('autogen_code_review_bot.pr_analysis.which') as mock_which:
                mock_which.side_effect = lambda tool: '/usr/bin/echo' if tool == 'echo' else None
                
                tool, output = _run_style_checks_parallel(tmpdir, linters, max_workers=2)
                
                # Should handle the error gracefully
                assert "not installed" in output or "echo" in tool

    def test_parallel_execution_respects_max_workers(self):
        """Test that parallel execution respects the max_workers parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(5):
                (Path(tmpdir) / f"test{i}.py").write_text("print('hello')")
            
            linters = {f"lang{i}": "echo" for i in range(5)}
            
            with patch('autogen_code_review_bot.pr_analysis._detect_repo_languages') as mock_detect:
                mock_detect.return_value = set(linters.keys())
                
                with patch('autogen_code_review_bot.pr_analysis.ThreadPoolExecutor') as mock_executor:
                    mock_executor.return_value.__enter__.return_value.map.return_value = []
                    
                    _run_style_checks_parallel(tmpdir, linters, max_workers=3)
                    
                    # Should be called with max_workers=3
                    mock_executor.assert_called_once_with(max_workers=3)


class TestParallelAllChecks:
    """Test cases for parallel execution of all check types."""

    def test_all_checks_run_in_parallel(self):
        """Test that security, style, and performance checks run in parallel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")
            
            linters = {"python": "echo"}
            
            def slow_mock_command(cmd, cwd, timeout=30):
                time.sleep(0.1)  # Simulate slow execution
                return f"output from {cmd[0]}"
            
            with patch('autogen_code_review_bot.pr_analysis._run_command', slow_mock_command), \
                 patch('autogen_code_review_bot.pr_analysis.which', return_value='/usr/bin/echo'):
                
                start_time = time.time()
                result = _run_all_checks_parallel(tmpdir, linters)
                execution_time = time.time() - start_time
                
                # Should complete in less time than sequential (3 * 0.1 = 0.3s)
                assert execution_time < 0.25
                assert result.security.tool == "bandit"
                assert result.style.tool is not None
                assert result.performance.tool == "radon"

    def test_analyze_pr_with_parallel_option(self):
        """Test that analyze_pr supports parallel execution option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")
            
            with patch('autogen_code_review_bot.pr_analysis.which', return_value='/usr/bin/echo'):
                # Test with parallel enabled
                result = analyze_pr(tmpdir, use_cache=False, use_parallel=True)
                assert result.security.tool == "bandit"
                
                # Test with parallel disabled
                result = analyze_pr(tmpdir, use_cache=False, use_parallel=False)
                assert result.security.tool == "bandit"


class TestConcurrencySafety:
    """Test cases for thread safety and concurrency issues."""

    def test_concurrent_cache_access(self):
        """Test that cache access is thread-safe during parallel execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = Path(tmpdir) / "test.py"
            py_file.write_text("print('hello')")
            
            def run_analysis():
                return analyze_pr(tmpdir, use_cache=True, use_parallel=True)
            
            with patch('autogen_code_review_bot.pr_analysis.which', return_value='/usr/bin/echo'):
                # Run multiple analyses concurrently
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(run_analysis) for _ in range(5)]
                    results = [f.result() for f in futures]
                
                # All results should be successful
                assert all(r.security.tool == "bandit" for r in results)

    def test_file_system_safety(self):
        """Test that concurrent file system access doesn't cause issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(10):
                (Path(tmpdir) / f"test{i}.py").write_text("print('hello')")
            
            linters = {"python": "echo"}
            
            with patch('autogen_code_review_bot.pr_analysis.which', return_value='/usr/bin/echo'):
                # Should handle concurrent file access safely
                tool, output = _run_style_checks_parallel(tmpdir, linters, max_workers=4)
                assert tool is not None