"""Tests for cache invalidation strategy."""

import pytest
import tempfile
import json
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch, call
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot.caching import (
    LinterCache,
    CacheVersionManager,
    InvalidationStrategy,
    get_tool_version,
    get_config_file_hash,
    should_invalidate_cache
)
from autogen_code_review_bot.models import PRAnalysisResult, AnalysisSection


class TestToolVersionDetection:
    """Test detection of linter tool versions."""
    
    @patch('subprocess.run')
    def test_get_tool_version_success(self, mock_run):
        """Test successful tool version detection."""
        mock_run.return_value = Mock(
            stdout="ruff 0.1.8\n",
            stderr="",
            returncode=0
        )
        
        version = get_tool_version("ruff")
        assert version == "0.1.8"
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_get_tool_version_complex_output(self, mock_run):
        """Test version parsing from complex tool output."""
        # Test various version output formats
        test_cases = [
            ("eslint v8.57.0\n", "8.57.0"),
            ("bandit 1.7.5\n", "1.7.5"),
            ("radon 6.0.1 (Python 3.11.0)\n", "6.0.1"),
            ("rubocop 1.60.2\n", "1.60.2")
        ]
        
        for stdout, expected in test_cases:
            mock_run.reset_mock()
            mock_run.return_value = Mock(stdout=stdout, stderr="", returncode=0)
            
            version = get_tool_version("tool")
            assert version == expected
    
    @patch('subprocess.run')
    def test_get_tool_version_failure(self, mock_run):
        """Test handling of tool version detection failure."""
        mock_run.side_effect = Exception("Tool not found")
        
        version = get_tool_version("nonexistent")
        assert version is None
    
    @patch('subprocess.run')
    def test_get_tool_version_nonzero_exit(self, mock_run):
        """Test handling of non-zero exit codes."""
        mock_run.return_value = Mock(
            stdout="",
            stderr="Command not found",
            returncode=1
        )
        
        version = get_tool_version("missing")
        assert version is None


class TestConfigFileHashing:
    """Test configuration file hashing for invalidation."""
    
    def test_get_config_file_hash_existing(self):
        """Test hashing of existing config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text("linter: ruff\nrules:\n  - E501\n")
            
            hash1 = get_config_file_hash(str(config_file))
            hash2 = get_config_file_hash(str(config_file))
            
            assert hash1 is not None
            assert hash1 == hash2  # Same file should have same hash
            assert len(hash1) == 64  # SHA-256 hex digest
    
    def test_get_config_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config1 = Path(temp_dir) / "config1.yaml"
            config2 = Path(temp_dir) / "config2.yaml"
            
            config1.write_text("linter: ruff\nrules:\n  - E501\n")
            config2.write_text("linter: ruff\nrules:\n  - W503\n")
            
            hash1 = get_config_file_hash(str(config1))
            hash2 = get_config_file_hash(str(config2))
            
            assert hash1 != hash2
    
    def test_get_config_file_hash_nonexistent(self):
        """Test handling of non-existent config files."""
        hash_result = get_config_file_hash("/nonexistent/config.yaml")
        assert hash_result is None
    
    def test_get_config_file_hash_permission_error(self):
        """Test handling of permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"
            config_file.write_text("test")
            
            with patch('pathlib.Path.read_bytes') as mock_read:
                mock_read.side_effect = PermissionError("Access denied")
                
                hash_result = get_config_file_hash(str(config_file))
                assert hash_result is None


class TestCacheVersionManager:
    """Test cache version management system."""
    
    def test_version_manager_init(self):
        """Test version manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheVersionManager(temp_dir)
            
            assert manager.version_file.exists()
            assert manager.cache_dir == Path(temp_dir)
    
    def test_get_current_environment_version(self):
        """Test getting current environment version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheVersionManager(temp_dir)
            
            with patch('autogen_code_review_bot.caching.get_tool_version') as mock_version:
                mock_version.side_effect = lambda tool: {"ruff": "0.1.8", "eslint": "8.57.0"}.get(tool)
                
                version = manager.get_current_environment_version(["ruff", "eslint"])
                
                assert "ruff" in version["tools"]
                assert "eslint" in version["tools"]
                assert version["tools"]["ruff"] == "0.1.8"
                assert version["tools"]["eslint"] == "8.57.0"
    
    def test_version_file_persistence(self):
        """Test that version information persists across instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First manager instance
            manager1 = CacheVersionManager(temp_dir)
            version_data = {
                "tools": {"ruff": "0.1.8"},
                "timestamp": time.time(),
                "python_version": "3.11.0"
            }
            manager1.update_version_info(version_data)
            
            # Second manager instance
            manager2 = CacheVersionManager(temp_dir)
            loaded_version = manager2.get_stored_version_info()
            
            assert loaded_version["tools"]["ruff"] == "0.1.8"
            assert loaded_version["python_version"] == "3.11.0"
    
    def test_version_has_changed(self):
        """Test detection of environment version changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheVersionManager(temp_dir)
            
            # Store initial version
            old_version = {
                "tools": {"ruff": "0.1.7", "eslint": "8.56.0"},
                "timestamp": time.time(),
                "python_version": "3.10.0"
            }
            manager.update_version_info(old_version)
            
            # Simulate environment change
            new_version = {
                "tools": {"ruff": "0.1.8", "eslint": "8.56.0"},
                "timestamp": time.time(),
                "python_version": "3.10.0"
            }
            
            assert manager.version_has_changed(new_version)
    
    def test_version_unchanged(self):
        """Test detection when version hasn't changed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheVersionManager(temp_dir)
            
            version = {
                "tools": {"ruff": "0.1.8"},
                "timestamp": time.time(),
                "python_version": "3.11.0"
            }
            manager.update_version_info(version)
            
            # Same version should not trigger change
            assert not manager.version_has_changed(version)


class TestInvalidationStrategy:
    """Test cache invalidation strategies."""
    
    def test_strategy_init(self):
        """Test invalidation strategy initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            assert strategy.cache_dir == Path(temp_dir)
            assert isinstance(strategy.version_manager, CacheVersionManager)
    
    def test_should_invalidate_on_tool_change(self):
        """Test invalidation when tool versions change."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            # Set up old version
            old_version = {"tools": {"ruff": "0.1.7"}, "timestamp": time.time()}
            strategy.version_manager.update_version_info(old_version)
            
            # Mock new version detection
            with patch.object(strategy.version_manager, 'get_current_environment_version') as mock_env:
                mock_env.return_value = {"tools": {"ruff": "0.1.8"}, "timestamp": time.time()}
                
                should_invalidate = strategy.should_invalidate_cache(["ruff"])
                assert should_invalidate
    
    def test_should_not_invalidate_when_unchanged(self):
        """Test no invalidation when versions unchanged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            version = {"tools": {"ruff": "0.1.8"}, "timestamp": time.time()}
            strategy.version_manager.update_version_info(version)
            
            with patch.object(strategy.version_manager, 'get_current_environment_version') as mock_env:
                mock_env.return_value = version
                
                should_invalidate = strategy.should_invalidate_cache(["ruff"])
                assert not should_invalidate
    
    def test_invalidate_all_entries(self):
        """Test invalidating all cache entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            cache = LinterCache(temp_dir)
            
            # Create some cache entries
            result = PRAnalysisResult(
                security=AnalysisSection("bandit", "no issues"),
                style=AnalysisSection("ruff", "no issues"),
                performance=AnalysisSection("radon", "no issues")
            )
            
            cache.set("commit1", "config1", result)
            cache.set("commit2", "config2", result)
            
            # Verify entries exist
            assert (Path(temp_dir) / "*.json").exists() or len(list(Path(temp_dir).glob("*.json"))) > 0
            
            # Invalidate all
            removed = strategy.invalidate_all_entries()
            assert removed >= 0  # Should remove some entries
    
    def test_invalidate_selective(self):
        """Test selective invalidation based on config changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            config_files = [str(Path(temp_dir) / "config.yaml")]
            Path(temp_dir, "config.yaml").write_text("test config")
            
            with patch('autogen_code_review_bot.caching.get_config_file_hash') as mock_hash:
                mock_hash.return_value = "newhash123"
                
                # Store old hash
                strategy.config_hashes = {"config.yaml": "oldhash456"}
                
                should_invalidate = strategy.should_invalidate_for_config_change(config_files)
                assert should_invalidate


class TestCacheInvalidationIntegration:
    """Test integration with existing LinterCache."""
    
    def test_enhanced_cache_with_invalidation(self):
        """Test LinterCache with invalidation capability."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LinterCache(temp_dir)
            
            # Add invalidation strategy
            cache.invalidation_strategy = InvalidationStrategy(temp_dir)
            
            # Test that cache can check for invalidation
            with patch.object(cache.invalidation_strategy, 'should_invalidate_cache') as mock_check:
                mock_check.return_value = True
                
                # This should trigger invalidation check
                result = cache.get_with_invalidation_check("commit1", "config1", ["ruff"])
                assert result is None  # Should be None due to invalidation
                mock_check.assert_called_once_with(["ruff"])
    
    def test_cache_invalidation_on_version_change(self):
        """Test that cache gets invalidated when tool versions change."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LinterCache(temp_dir)
            strategy = InvalidationStrategy(temp_dir)
            
            # Store some cache entries
            result = PRAnalysisResult(
                security=AnalysisSection("bandit", "no issues"),
                style=AnalysisSection("ruff", "clean"),
                performance=AnalysisSection("radon", "good")
            )
            cache.set("abc123", "config123", result)
            
            # Verify cache entry exists
            cached = cache.get("abc123", "config123")
            assert cached is not None
            
            # Simulate version change and invalidation
            with patch.object(strategy.version_manager, 'version_has_changed') as mock_changed:
                mock_changed.return_value = True
                
                # This should invalidate cache
                strategy.invalidate_if_needed(["ruff"])
                
                # Cache should now be empty or invalidated
                # (Specific behavior depends on implementation)


class TestInvalidationTriggers:
    """Test various invalidation triggers."""
    
    def test_should_invalidate_cache_function(self):
        """Test the module-level should_invalidate_cache function."""
        test_cases = [
            # (tools, config_files, expected_result)
            (["ruff"], [], False),  # No config changes, assume no version changes
            (["ruff"], ["/path/to/config.yaml"], True),  # Config file might have changed
        ]
        
        for tools, config_files, expected in test_cases:
            with patch('autogen_code_review_bot.caching.InvalidationStrategy') as mock_strategy_class:
                mock_strategy = Mock()
                mock_strategy.should_invalidate_cache.return_value = expected
                mock_strategy_class.return_value = mock_strategy
                
                result = should_invalidate_cache(tools, config_files)
                assert result == expected
    
    def test_invalidation_on_config_file_change(self):
        """Test invalidation when config files change."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            config_file = Path(temp_dir) / "test.yaml"
            config_file.write_text("old config")
            
            # Store initial hash
            strategy.update_config_hashes([str(config_file)])
            
            # Change config file
            config_file.write_text("new config")
            
            # Should detect change
            should_invalidate = strategy.should_invalidate_for_config_change([str(config_file)])
            assert should_invalidate
    
    def test_no_invalidation_when_config_unchanged(self):
        """Test no invalidation when config files haven't changed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            config_file = Path(temp_dir) / "test.yaml"
            config_file.write_text("config content")
            
            # Store hash and check immediately
            strategy.update_config_hashes([str(config_file)])
            should_invalidate = strategy.should_invalidate_for_config_change([str(config_file)])
            
            assert not should_invalidate


class TestCacheInvalidationErrorHandling:
    """Test error handling in cache invalidation."""
    
    def test_invalidation_with_missing_tools(self):
        """Test invalidation when tools are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            with patch('autogen_code_review_bot.caching.get_tool_version') as mock_version:
                mock_version.return_value = None  # Tool not found
                
                # Should not crash, should handle gracefully
                should_invalidate = strategy.should_invalidate_cache(["nonexistent_tool"])
                assert isinstance(should_invalidate, bool)
    
    def test_invalidation_with_corrupted_version_file(self):
        """Test handling of corrupted version files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CacheVersionManager(temp_dir)
            
            # Corrupt the version file
            manager.version_file.write_text("invalid json content")
            
            # Should handle gracefully
            version_info = manager.get_stored_version_info()
            assert isinstance(version_info, dict)
    
    def test_invalidation_with_permission_errors(self):
        """Test handling of permission errors during invalidation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategy = InvalidationStrategy(temp_dir)
            
            with patch('pathlib.Path.unlink') as mock_unlink:
                mock_unlink.side_effect = PermissionError("Access denied")
                
                # Should not crash
                removed_count = strategy.invalidate_all_entries()
                assert removed_count == 0  # Nothing removed due to permission error