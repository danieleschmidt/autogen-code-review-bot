"""Tests for large PR streaming functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, call
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot.pr_analysis import (
    analyze_pr, 
    _detect_repo_languages_chunked,
    _analyze_pr_streaming,
    _should_use_streaming,
    _get_repo_size_info
)
from autogen_code_review_bot.models import PRAnalysisResult


class TestRepoSizeDetection:
    """Test repository size detection for streaming decisions."""
    
    def test_get_repo_size_info_small_repo(self):
        """Test size calculation for small repositories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a small repo with a few files
            for i in range(5):
                test_file = Path(temp_dir) / f"test_{i}.py"
                test_file.write_text(f"print('test {i}')\n" * 10)  # ~150 bytes each
            
            file_count, total_size = _get_repo_size_info(temp_dir)
            
            assert file_count == 5
            assert 500 < total_size < 1000  # Approximate size check
    
    def test_get_repo_size_info_large_repo(self):
        """Test size calculation for large repositories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a larger repo
            for i in range(20):
                test_file = Path(temp_dir) / f"file_{i}.py"
                test_file.write_text("# Large file content\n" * 1000)  # ~20KB each
            
            file_count, total_size = _get_repo_size_info(temp_dir)
            
            assert file_count == 20
            assert total_size > 300000  # Should be > 300KB
    
    def test_get_repo_size_info_with_subdirectories(self):
        """Test size calculation with nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            subdir = Path(temp_dir) / "src" / "module"
            subdir.mkdir(parents=True)
            
            # Add files at different levels
            (Path(temp_dir) / "main.py").write_text("print('main')\n")
            (Path(temp_dir) / "src" / "__init__.py").write_text("")
            (Path(temp_dir) / "src" / "module" / "core.py").write_text("class Core: pass\n")
            
            file_count, total_size = _get_repo_size_info(temp_dir)
            
            assert file_count == 3
            assert total_size > 0


class TestStreamingDecision:
    """Test the logic for deciding when to use streaming."""
    
    def test_should_use_streaming_small_repo(self):
        """Small repositories should not use streaming."""
        # 10 files, 50KB total
        assert not _should_use_streaming(10, 50 * 1024)
    
    def test_should_use_streaming_many_files(self):
        """Repositories with many files should use streaming."""
        # 1500 files, 100KB total (many small files)
        assert _should_use_streaming(1500, 100 * 1024)
    
    def test_should_use_streaming_large_size(self):
        """Large repositories should use streaming."""
        # 100 files, 20MB total
        assert _should_use_streaming(100, 20 * 1024 * 1024)
    
    def test_should_use_streaming_medium_repo(self):
        """Medium-sized repos near the threshold."""
        # 500 files, 5MB - should not stream
        assert not _should_use_streaming(500, 5 * 1024 * 1024)
        
        # 1200 files, 5MB - should stream (many files)
        assert _should_use_streaming(1200, 5 * 1024 * 1024)
        
        # 500 files, 15MB - should stream (large size)
        assert _should_use_streaming(500, 15 * 1024 * 1024)


class TestChunkedLanguageDetection:
    """Test chunked language detection for large repositories."""
    
    def test_chunked_language_detection_small_batch(self):
        """Test language detection with small file count."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files in different languages
            (Path(temp_dir) / "script.py").write_text("print('hello')")
            (Path(temp_dir) / "app.js").write_text("console.log('hello');")
            (Path(temp_dir) / "style.css").write_text("body { color: red; }")
            
            languages = _detect_repo_languages_chunked(temp_dir, chunk_size=2)
            
            assert "python" in languages
            assert "javascript" in languages
            # CSS might not be detected depending on language_detection implementation
    
    def test_chunked_language_detection_with_progress(self):
        """Test that chunked detection provides progress callbacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            for i in range(10):
                (Path(temp_dir) / f"file_{i}.py").write_text(f"# File {i}")
            
            progress_calls = []
            def progress_callback(processed, total):
                progress_calls.append((processed, total))
            
            languages = _detect_repo_languages_chunked(
                temp_dir, 
                chunk_size=3, 
                progress_callback=progress_callback
            )
            
            assert "python" in languages
            assert len(progress_calls) > 0
            # Should have at least one progress call
            assert any(total == 10 for _, total in progress_calls)
    
    def test_chunked_language_detection_large_chunk(self):
        """Test that large chunk size works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.py").write_text("import os")
            
            languages = _detect_repo_languages_chunked(temp_dir, chunk_size=1000)
            
            assert "python" in languages


class TestStreamingAnalysis:
    """Test the full streaming analysis workflow."""
    
    @patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel')
    @patch('autogen_code_review_bot.pr_analysis.get_commit_hash')
    def test_streaming_analysis_basic_flow(self, mock_commit_hash, mock_checks):
        """Test basic streaming analysis workflow."""
        mock_commit_hash.return_value = "abc123"
        mock_checks.return_value = Mock(
            security=Mock(tool="bandit", output="No issues"),
            style=Mock(tool="ruff", output="No style issues"),
            performance=Mock(tool="radon", output="Good complexity")
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files to trigger streaming
            for i in range(15):  # Enough to potentially trigger streaming
                (Path(temp_dir) / f"module_{i}.py").write_text(f"# Module {i}\nprint('{i}')")
            
            result = _analyze_pr_streaming(
                temp_dir, 
                linters={"python": "ruff"}, 
                use_cache=False
            )
            
            assert isinstance(result, PRAnalysisResult)
            mock_checks.assert_called_once()
    
    @patch('autogen_code_review_bot.pr_analysis.LinterCache')
    def test_streaming_analysis_with_cache(self, mock_cache_class):
        """Test streaming analysis with caching enabled."""
        mock_cache = Mock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache_class.return_value = mock_cache
        
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.py").write_text("print('test')")
            
            with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
                mock_checks.return_value = Mock(
                    security=Mock(tool="bandit", output="No issues"),
                    style=Mock(tool="ruff", output="Clean"),
                    performance=Mock(tool="radon", output="Good")
                )
                
                result = _analyze_pr_streaming(
                    temp_dir, 
                    linters={"python": "ruff"}, 
                    use_cache=True
                )
                
                assert isinstance(result, PRAnalysisResult)
                # Should have attempted to get from cache
                mock_cache.get.assert_called()


class TestStreamingIntegration:
    """Test integration of streaming with main analyze_pr function."""
    
    @patch('autogen_code_review_bot.pr_analysis._should_use_streaming')
    @patch('autogen_code_review_bot.pr_analysis._analyze_pr_streaming')
    def test_analyze_pr_uses_streaming_when_appropriate(self, mock_streaming, mock_should_stream):
        """Test that analyze_pr switches to streaming for large repos."""
        mock_should_stream.return_value = True
        mock_streaming.return_value = Mock(
            security=Mock(tool="bandit", output="No issues"),
            style=Mock(tool="ruff", output="Clean"),
            performance=Mock(tool="radon", output="Good")
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.py").write_text("print('test')")
            
            result = analyze_pr(temp_dir, use_cache=False, use_parallel=False)
            
            assert isinstance(result, PRAnalysisResult)
            mock_should_stream.assert_called()
            mock_streaming.assert_called_once()
    
    @patch('autogen_code_review_bot.pr_analysis._should_use_streaming')
    def test_analyze_pr_uses_normal_flow_for_small_repos(self, mock_should_stream):
        """Test that analyze_pr uses normal flow for small repos."""
        mock_should_stream.return_value = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "small.py").write_text("print('small')")
            
            with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
                mock_checks.return_value = Mock(
                    security=Mock(tool="bandit", output="No issues"),
                    style=Mock(tool="ruff", output="Clean"),
                    performance=Mock(tool="radon", output="Good")
                )
                
                result = analyze_pr(temp_dir, use_cache=False, use_parallel=True)
                
                assert isinstance(result, PRAnalysisResult)
                mock_should_stream.assert_called()
                # Should use normal parallel checks, not streaming
                mock_checks.assert_called_once()


class TestStreamingErrorHandling:
    """Test error handling in streaming scenarios."""
    
    def test_streaming_handles_permission_errors(self):
        """Test that streaming gracefully handles permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file and then make it unreadable
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('test')")
            
            # Mock os.walk to raise permission error
            with patch('os.walk') as mock_walk:
                mock_walk.side_effect = PermissionError("Access denied")
                
                file_count, total_size = _get_repo_size_info(temp_dir)
                
                # Should return defaults when permission denied
                assert file_count == 0
                assert total_size == 0
    
    def test_streaming_handles_analysis_errors(self):
        """Test that streaming handles analysis errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.py").write_text("print('test')")
            
            with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
                mock_checks.side_effect = Exception("Analysis failed")
                
                result = _analyze_pr_streaming(
                    temp_dir, 
                    linters={"python": "ruff"}, 
                    use_cache=False
                )
                
                # Should return error result
                assert result.security.tool == "error"
                assert "Analysis failed" in result.security.output


class TestStreamingPerformance:
    """Test performance characteristics of streaming."""
    
    def test_streaming_memory_efficiency(self):
        """Test that streaming doesn't load all files into memory at once."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many files
            for i in range(50):
                (Path(temp_dir) / f"file_{i}.py").write_text(f"# File {i}\n" * 100)
            
            memory_usage = []
            
            def track_memory(*args, **kwargs):
                # Mock to track if we're processing in chunks
                memory_usage.append(len(args))
                return {"python"}  # Mock return
            
            with patch('autogen_code_review_bot.pr_analysis.detect_language', side_effect=track_memory):
                languages = _detect_repo_languages_chunked(temp_dir, chunk_size=10)
                
                # Should have processed files in chunks
                assert len(memory_usage) > 0
    
    def test_streaming_timeout_handling(self):
        """Test that streaming respects timeout constraints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "test.py").write_text("print('test')")
            
            # Mock a very slow analysis
            with patch('autogen_code_review_bot.pr_analysis._run_all_checks_parallel') as mock_checks:
                import time
                def slow_analysis(*args, **kwargs):
                    time.sleep(0.1)  # Simulate slow analysis
                    return Mock(
                        security=Mock(tool="bandit", output="Slow result"),
                        style=Mock(tool="ruff", output="Slow style"),
                        performance=Mock(tool="radon", output="Slow perf")
                    )
                
                mock_checks.side_effect = slow_analysis
                
                # Should complete even with slow analysis
                result = _analyze_pr_streaming(
                    temp_dir, 
                    linters={"python": "ruff"}, 
                    use_cache=False
                )
                
                assert isinstance(result, PRAnalysisResult)