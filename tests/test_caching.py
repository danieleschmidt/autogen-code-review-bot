"""Tests for the caching system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autogen_code_review_bot.caching import LinterCache, get_commit_hash
from autogen_code_review_bot.pr_analysis import AnalysisSection, PRAnalysisResult


class TestLinterCache:
    """Test cases for the LinterCache class."""

    def test_cache_initialization(self):
        """Test cache can be initialized with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LinterCache(cache_dir=tmpdir)
            assert cache.cache_dir == Path(tmpdir)
            assert cache.cache_dir.exists()

    def test_cache_key_generation(self):
        """Test cache key generation includes commit hash and config."""
        cache = LinterCache()
        key = cache._get_cache_key("abc123", "config_hash")
        assert "abc123" in key
        assert "config_hash" in key

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LinterCache(cache_dir=tmpdir)
            result = cache.get("nonexistent_commit", "config_hash")
            assert result is None

    def test_cache_hit(self):
        """Test cache hit returns stored result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LinterCache(cache_dir=tmpdir)
            
            # Store a result
            test_result = PRAnalysisResult(
                security=AnalysisSection(tool="bandit", output="No issues"),
                style=AnalysisSection(tool="ruff", output="All clean"),
                performance=AnalysisSection(tool="radon", output="Good complexity")
            )
            cache.set("test_commit", "config_hash", test_result)
            
            # Retrieve the result
            retrieved = cache.get("test_commit", "config_hash")
            assert retrieved is not None
            assert retrieved.security.output == "No issues"
            assert retrieved.style.output == "All clean"

    def test_cache_expiration(self):
        """Test cache entries expire after TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LinterCache(cache_dir=tmpdir, ttl_hours=0)  # Immediate expiration
            
            test_result = PRAnalysisResult(
                security=AnalysisSection(tool="bandit", output="Test"),
                style=AnalysisSection(tool="ruff", output="Test"),
                performance=AnalysisSection(tool="radon", output="Test")
            )
            cache.set("test_commit", "config_hash", test_result)
            
            # Should be expired immediately
            retrieved = cache.get("test_commit", "config_hash")
            assert retrieved is None

    def test_cache_cleanup(self):
        """Test cache cleanup removes expired entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = LinterCache(cache_dir=tmpdir, ttl_hours=0)
            
            test_result = PRAnalysisResult(
                security=AnalysisSection(tool="bandit", output="Test"),
                style=AnalysisSection(tool="ruff", output="Test"),
                performance=AnalysisSection(tool="radon", output="Test")
            )
            cache.set("test_commit", "config_hash", test_result)
            
            # Verify file exists
            cache_files = list(Path(tmpdir).glob("*.json"))
            assert len(cache_files) == 1
            
            # Run cleanup
            cache.cleanup()
            
            # Verify file is removed
            cache_files = list(Path(tmpdir).glob("*.json"))
            assert len(cache_files) == 0


class TestCommitHashDetection:
    """Test cases for commit hash detection."""

    def test_get_commit_hash_success(self):
        """Test successful commit hash retrieval."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "abc123def456\n"
            mock_run.return_value.returncode = 0
            
            result = get_commit_hash("/test/repo")
            assert result == "abc123def456"

    def test_get_commit_hash_failure(self):
        """Test commit hash retrieval failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Git not found")
            
            result = get_commit_hash("/test/repo")
            assert result is None

    def test_get_commit_hash_invalid_repo(self):
        """Test commit hash retrieval from invalid repository."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            
            result = get_commit_hash("/invalid/repo")
            assert result is None