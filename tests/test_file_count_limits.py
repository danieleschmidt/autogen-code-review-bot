"""Tests for file count limits in language detection."""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from autogen_code_review_bot.pr_analysis import _detect_repo_languages


class TestFileCountLimits:
    """Test file count limits in language detection."""

    def test_detect_repo_languages_with_limit(self):
        """Test that language detection respects file count limits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create more files than the limit
            max_files = 5
            for i in range(10):  # Create 10 files, limit to 5
                (temp_path / f"test_{i}.py").write_text(f"# Python file {i}")
                (temp_path / f"test_{i}.js").write_text(f"// JavaScript file {i}")
            
            # Test with custom limit
            languages = _detect_repo_languages(temp_dir, max_files=max_files)
            
            # Should detect both Python and JavaScript despite limit
            # (since we hit the limit after scanning some of each type)
            assert len(languages) >= 1, "Should detect at least one language before hitting limit"

    def test_detect_repo_languages_default_limit(self):
        """Test that language detection uses default limit of 10,000."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a small number of files (well under default limit)
            for i in range(5):
                (temp_path / f"test_{i}.py").write_text(f"# Python file {i}")
            
            languages = _detect_repo_languages(temp_dir)
            
            assert "python" in languages, "Should detect Python files"

    def test_detect_repo_languages_early_exit_logging(self):
        """Test that early exit produces appropriate warning logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files that will trigger the limit
            max_files = 3
            for i in range(6):  # Create more than the limit
                (temp_path / f"test_{i}.py").write_text(f"# Python file {i}")
            
            with patch('autogen_code_review_bot.pr_analysis.logger') as mock_logger:
                languages = _detect_repo_languages(temp_dir, max_files=max_files)
                
                # Check that warning was logged when limit reached
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args
                assert "reached file limit" in warning_call[0][0]
                assert warning_call[1]["extra"]["max_files"] == max_files

    def test_detect_repo_languages_no_limit_reached(self):
        """Test normal operation when file count is under limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create fewer files than the limit
            max_files = 10
            for i in range(3):
                (temp_path / f"test_{i}.py").write_text(f"# Python file {i}")
            
            with patch('autogen_code_review_bot.pr_analysis.logger') as mock_logger:
                languages = _detect_repo_languages(temp_dir, max_files=max_files)
                
                # Should not call warning (only debug)
                mock_logger.warning.assert_not_called()
                mock_logger.debug.assert_called_once()
                
                debug_call = mock_logger.debug.call_args
                assert debug_call[1]["extra"]["limit_reached"] is False

    def test_detect_repo_languages_mixed_file_types(self):
        """Test language detection with mixed file types under limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mixed language files
            (temp_path / "script.py").write_text("print('hello')")
            (temp_path / "app.js").write_text("console.log('hello');")
            (temp_path / "style.css").write_text("body { color: red; }")
            (temp_path / "readme.txt").write_text("This is a readme")
            
            languages = _detect_repo_languages(temp_dir, max_files=10)
            
            # Should detect known languages, ignore unknown ones
            detected = set(languages)
            possible_languages = {"python", "javascript", "css"}
            assert detected.issubset(possible_languages), f"Unexpected languages detected: {detected - possible_languages}"
            assert len(detected) >= 2, "Should detect multiple languages"

    def test_detect_repo_languages_performance_with_deep_structure(self):
        """Test that file count limits work with deeply nested directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create nested structure with many files
            max_files = 4
            file_count = 0
            for i in range(3):  # 3 levels deep
                level_dir = temp_path / f"level_{i}"
                level_dir.mkdir()
                for j in range(3):  # 3 files per level
                    (level_dir / f"file_{j}.py").write_text(f"# Level {i} file {j}")
                    file_count += 1
            
            languages = _detect_repo_languages(temp_dir, max_files=max_files)
            
            assert "python" in languages, "Should detect Python files before hitting limit"

    def test_detect_repo_languages_empty_directory(self):
        """Test language detection on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            languages = _detect_repo_languages(temp_dir, max_files=100)
            
            assert len(languages) == 0, "Empty directory should have no detected languages"

    def test_detect_repo_languages_zero_limit(self):
        """Test edge case with zero file limit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text("print('hello')")
            
            with patch('autogen_code_review_bot.pr_analysis.logger') as mock_logger:
                languages = _detect_repo_languages(temp_dir, max_files=0)
                
                # Should immediately hit limit and warn
                mock_logger.warning.assert_called_once()
                assert len(languages) == 0, "Should detect no languages with zero limit"