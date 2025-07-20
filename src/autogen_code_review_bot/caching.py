"""Performance caching system for linter results."""

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Optional

from .models import AnalysisSection, PRAnalysisResult


def get_commit_hash(repo_path: str) -> Optional[str]:
    """Get the current commit hash for the repository.
    
    Args:
        repo_path: Path to the git repository
        
    Returns:
        Commit hash string or None if unable to determine
    """
    try:
        result = run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (CalledProcessError, OSError, FileNotFoundError):
        return None


class LinterCache:
    """Cache for storing linter results by commit hash and configuration."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_hours: int = 24):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store cache files (defaults to ~/.cache/autogen-review)
            ttl_hours: Time-to-live for cache entries in hours
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/autogen-review")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
    
    def _get_cache_key(self, commit_hash: str, config_hash: str) -> str:
        """Generate a cache key from commit hash and configuration.
        
        Args:
            commit_hash: Git commit hash
            config_hash: Hash of the linter configuration
            
        Returns:
            Unique cache key string
        """
        combined = f"{commit_hash}:{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _hash_config(self, config: dict) -> str:
        """Create a hash of the linter configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get(self, commit_hash: str, config_hash: str) -> Optional[PRAnalysisResult]:
        """Retrieve cached analysis result.
        
        Args:
            commit_hash: Git commit hash
            config_hash: Hash of the linter configuration
            
        Returns:
            Cached PRAnalysisResult or None if not found/expired
        """
        cache_key = self._get_cache_key(commit_hash, config_hash)
        cache_file = self._get_cache_file(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache entry has expired
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.ttl_seconds:
                cache_file.unlink()  # Remove expired entry
                return None
            
            # Load and deserialize the cached result
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return PRAnalysisResult(
                security=AnalysisSection(**data['security']),
                style=AnalysisSection(**data['style']),
                performance=AnalysisSection(**data['performance'])
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            # Remove corrupted cache entry
            if cache_file.exists():
                cache_file.unlink()
            return None
    
    def set(self, commit_hash: str, config_hash: str, result: PRAnalysisResult) -> None:
        """Store analysis result in cache.
        
        Args:
            commit_hash: Git commit hash
            config_hash: Hash of the linter configuration
            result: Analysis result to cache
        """
        cache_key = self._get_cache_key(commit_hash, config_hash)
        cache_file = self._get_cache_file(cache_key)
        
        try:
            # Serialize the result
            data = {
                'security': asdict(result.security),
                'style': asdict(result.style),
                'performance': asdict(result.performance),
                'cached_at': time.time()
            }
            
            # Write to cache file atomically
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            temp_file.rename(cache_file)
        except OSError:
            # Fail silently if unable to write cache
            pass
    
    def cleanup(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                file_age = current_time - cache_file.stat().st_mtime
                if file_age > self.ttl_seconds:
                    cache_file.unlink()
                    removed_count += 1
            except OSError:
                continue
        
        return removed_count
    
    def clear(self) -> int:
        """Remove all cache entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                removed_count += 1
            except OSError:
                continue
        
        return removed_count
    
    def get_config_hash(self, linter_config: dict) -> str:
        """Get hash for linter configuration.
        
        Args:
            linter_config: Dictionary of linter configuration
            
        Returns:
            Configuration hash string
        """
        return self._hash_config(linter_config)