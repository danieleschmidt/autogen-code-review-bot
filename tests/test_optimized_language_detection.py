"""Test optimized language detection functionality."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.autogen_code_review_bot.pr_analysis import _detect_repo_languages
from src.autogen_code_review_bot.system_config import SystemConfig, reset_system_config


class TestOptimizedLanguageDetection:
    """Test optimized language detection performance improvements."""

    def setup_method(self):
        """Reset system config before each test."""
        reset_system_config()

    def teardown_method(self):
        """Clean up after each test."""
        reset_system_config()

    def test_extension_caching(self):
        """Test that extension caching reduces language detection calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create multiple files with same extensions
            (repo_path / "file1.py").write_text("print('hello')")
            (repo_path / "file2.py").write_text("print('world')")
            (repo_path / "file3.py").write_text("print('test')")
            (repo_path / "file1.js").write_text("console.log('hello')")
            (repo_path / "file2.js").write_text("console.log('world')")
            
            with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                mock_detect.side_effect = lambda path: "python" if path.suffix == ".py" else "javascript" if path.suffix == ".js" else "unknown"
                
                languages = _detect_repo_languages(repo_path)
                
                # Should have called detect_language only twice (once per unique extension)
                # instead of 5 times (once per file)
                assert mock_detect.call_count <= 2  # At most once per unique extension
                assert "python" in languages
                assert "javascript" in languages

    def test_skip_directories_optimization(self):
        """Test that common directories are skipped for performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create files in directories that should be skipped
            skip_dirs = ['__pycache__', 'node_modules', '.git', '.venv', 'build']
            for skip_dir in skip_dirs:
                skip_path = repo_path / skip_dir
                skip_path.mkdir()
                (skip_path / "should_be_skipped.py").write_text("# Should not be processed")
            
            # Create files in normal directories
            (repo_path / "normal.py").write_text("print('hello')")
            
            with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                mock_detect.return_value = "python"
                
                languages = _detect_repo_languages(repo_path)
                
                # Should only process the normal file, not the ones in skip directories
                assert mock_detect.call_count == 1
                assert "python" in languages

    def test_early_termination_on_language_diversity(self):
        """Test early termination when sufficient language diversity is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create many files with different extensions to trigger early termination
            extensions = ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.ts', '.jsx']
            for i, ext in enumerate(extensions):
                for j in range(20):  # Create 20 files per extension
                    (repo_path / f"file_{i}_{j}{ext}").write_text(f"content {i} {j}")
            
            with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                # Return different languages for different extensions
                def mock_language_detect(path):
                    ext_map = {'.py': 'python', '.js': 'javascript', '.java': 'java', 
                              '.cpp': 'cpp', '.c': 'c', '.go': 'go', '.rs': 'rust',
                              '.rb': 'ruby', '.php': 'php', '.ts': 'typescript', '.jsx': 'jsx'}
                    return ext_map.get(path.suffix, 'unknown')
                
                mock_detect.side_effect = mock_language_detect
                
                languages = _detect_repo_languages(repo_path)
                
                # Should have terminated early due to language diversity
                # We expect fewer calls than total files due to early termination
                total_files = len(extensions) * 20
                assert mock_detect.call_count < total_files
                assert len(languages) >= 10  # Should have found at least 10 languages

    def test_file_limit_enforcement(self):
        """Test that file limits are properly enforced."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create more files than the limit
            max_files = 50
            for i in range(100):  # Create 100 files
                (repo_path / f"file_{i}.py").write_text(f"print('file {i}')")
            
            with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                mock_detect.return_value = "python"
                
                languages = _detect_repo_languages(repo_path, max_files=max_files)
                
                # Should not have processed more than max_files
                assert mock_detect.call_count <= max_files
                assert "python" in languages

    def test_rglob_fallback_to_oswalk(self):
        """Test fallback to os.walk when rglob fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            (repo_path / "test.py").write_text("print('hello')")
            
            # Mock rglob to raise an exception
            with patch.object(Path, 'rglob', side_effect=OSError("Mock error")):
                with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                    mock_detect.return_value = "python"
                    
                    languages = _detect_repo_languages(repo_path)
                    
                    # Should still work via os.walk fallback
                    assert "python" in languages
                    assert mock_detect.call_count == 1

    def test_permission_error_handling(self):
        """Test handling of permission errors during directory traversal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            (repo_path / "accessible.py").write_text("print('hello')")
            
            # Mock rglob to raise PermissionError
            with patch.object(Path, 'rglob', side_effect=PermissionError("Access denied")):
                with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                    mock_detect.return_value = "python"
                    
                    # Should not raise exception, should fallback gracefully
                    languages = _detect_repo_languages(repo_path)
                    
                    # Should work via os.walk fallback
                    assert "python" in languages

    def test_performance_logging(self):
        """Test that performance metrics are logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            (repo_path / "test.py").write_text("print('hello')")
            
            with patch('src.autogen_code_review_bot.pr_analysis.logger') as mock_logger:
                with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                    mock_detect.return_value = "python"
                    
                    _detect_repo_languages(repo_path)
                    
                    # Should log completion with cache performance metrics
                    mock_logger.debug.assert_called()
                    debug_call = mock_logger.debug.call_args
                    assert "Language detection completed" in debug_call[0][0]
                    assert "cache_hits" in debug_call[1]["extra"]

    def test_memory_efficiency_with_islice(self):
        """Test that islice is used for memory efficiency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create files
            for i in range(10):
                (repo_path / f"file_{i}.py").write_text(f"print('file {i}')")
            
            with patch('src.autogen_code_review_bot.pr_analysis.islice') as mock_islice:
                # Configure islice to pass through but track usage
                from itertools import islice as real_islice
                mock_islice.side_effect = real_islice
                
                with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                    mock_detect.return_value = "python"
                    
                    _detect_repo_languages(repo_path, max_files=5)
                    
                    # Should have used islice for memory efficiency
                    mock_islice.assert_called_once()

    def test_directory_filtering_during_walk(self):
        """Test that directories are filtered during os.walk fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create nested structure with skip directories
            (repo_path / "src").mkdir()
            (repo_path / "src" / "main.py").write_text("print('main')")
            (repo_path / "__pycache__").mkdir()
            (repo_path / "__pycache__" / "cache.pyc").write_text("cached")
            
            # Force fallback to os.walk
            with patch.object(Path, 'rglob', side_effect=OSError("Force fallback")):
                with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                    mock_detect.return_value = "python"
                    
                    languages = _detect_repo_languages(repo_path)
                    
                    # Should only process main.py, not the cached file
                    assert mock_detect.call_count == 1
                    assert "python" in languages

    def test_system_config_integration(self):
        """Test integration with system configuration for max files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            
            # Create files
            for i in range(20):
                (repo_path / f"file_{i}.py").write_text(f"print('file {i}')")
            
            # Set system config with low max files
            config = SystemConfig()
            config.language_detection_max_files = 5
            
            with patch('src.autogen_code_review_bot.pr_analysis.get_system_config', return_value=config):
                with patch('src.autogen_code_review_bot.pr_analysis.detect_language') as mock_detect:
                    mock_detect.return_value = "python"
                    
                    languages = _detect_repo_languages(repo_path)
                    
                    # Should respect system config limit
                    assert mock_detect.call_count <= 5
                    assert "python" in languages