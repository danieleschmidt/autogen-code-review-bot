"""Performance caching system for linter results."""

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Optional

from .logging_config import get_logger
from .models import AnalysisSection, PRAnalysisResult

logger = get_logger(__name__)


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
            with open(cache_file, encoding='utf-8') as f:
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

    def get_with_invalidation_check(self, commit_hash: str, config_hash: str, tools: list) -> Optional[PRAnalysisResult]:
        """Retrieve cached result with invalidation check.
        
        Args:
            commit_hash: Git commit hash
            config_hash: Hash of the linter configuration
            tools: List of tools to check for version changes
            
        Returns:
            Cached result or None if invalidated or not found
        """
        # Check if cache should be invalidated
        if hasattr(self, 'invalidation_strategy') and self.invalidation_strategy:
            if self.invalidation_strategy.should_invalidate_cache(tools):
                logger.info("Cache invalidated due to environment changes",
                           extra={"tools": tools})
                return None

        return self.get(commit_hash, config_hash)


def get_tool_version(tool_name: str) -> Optional[str]:
    """Get version of a linting tool.
    
    Args:
        tool_name: Name of the tool (e.g., 'ruff', 'eslint')
        
    Returns:
        Version string or None if unable to determine
    """
    try:
        # Try common version flags
        version_flags = ['--version', '-v', '-V', 'version']

        for flag in version_flags:
            try:
                result = run(
                    [tool_name, flag],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0 and result.stdout.strip():
                    output = result.stdout.strip()
                    # Extract version number using regex
                    import re
                    version_match = re.search(r'(\d+\.\d+\.\d+)', output)
                    if version_match:
                        return version_match.group(1)

                    # Some tools might have different format
                    version_match = re.search(r'v?(\d+\.\d+)', output)
                    if version_match:
                        return version_match.group(1)

            except (CalledProcessError, OSError):
                continue

        logger.debug("Could not determine version for tool", extra={"tool": tool_name})
        return None

    except Exception as e:
        logger.debug("Tool version detection failed",
                    extra={"tool": tool_name, "error": str(e)})
        return None


def get_config_file_hash(config_path: str) -> Optional[str]:
    """Get hash of a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SHA-256 hash of file content or None if unable to read
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return None

        content = config_file.read_bytes()
        return hashlib.sha256(content).hexdigest()

    except (OSError, PermissionError) as e:
        logger.debug("Could not hash config file",
                    extra={"config_path": config_path, "error": str(e)})
        return None


class CacheVersionManager:
    """Manages version information for cache invalidation."""

    def __init__(self, cache_dir: str):
        """Initialize version manager.
        
        Args:
            cache_dir: Directory where cache and version info are stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.cache_dir / "version_info.json"

        # Ensure version file exists
        if not self.version_file.exists():
            self.update_version_info({})

    def get_current_environment_version(self, tools: list) -> dict:
        """Get current version information for tools and environment.
        
        Args:
            tools: List of tools to check
            
        Returns:
            Dictionary with version information
        """
        import sys

        version_info = {
            "tools": {},
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "timestamp": time.time()
        }

        for tool in tools:
            version = get_tool_version(tool)
            if version:
                version_info["tools"][tool] = version

        return version_info

    def get_stored_version_info(self) -> dict:
        """Get stored version information.
        
        Returns:
            Dictionary with stored version info
        """
        try:
            with open(self.version_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.debug("Could not load version info, returning empty")
            return {}

    def update_version_info(self, version_info: dict) -> None:
        """Update stored version information.
        
        Args:
            version_info: Version information to store
        """
        try:
            with open(self.version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
        except OSError as e:
            logger.warning("Could not save version info", extra={"error": str(e)})

    def version_has_changed(self, current_version: dict) -> bool:
        """Check if version has changed since last stored.
        
        Args:
            current_version: Current version information
            
        Returns:
            True if version has changed
        """
        stored_version = self.get_stored_version_info()

        # Compare tool versions
        stored_tools = stored_version.get("tools", {})
        current_tools = current_version.get("tools", {})

        if set(stored_tools.keys()) != set(current_tools.keys()):
            return True

        for tool, version in current_tools.items():
            if stored_tools.get(tool) != version:
                return True

        # Compare Python version
        if stored_version.get("python_version") != current_version.get("python_version"):
            return True

        return False


class InvalidationStrategy:
    """Strategy for cache invalidation based on environment changes."""

    def __init__(self, cache_dir: str):
        """Initialize invalidation strategy.
        
        Args:
            cache_dir: Directory where cache is stored
        """
        self.cache_dir = Path(cache_dir)
        self.version_manager = CacheVersionManager(cache_dir)
        self.config_hashes = {}

    def should_invalidate_cache(self, tools: list) -> bool:
        """Check if cache should be invalidated based on tool versions.
        
        Args:
            tools: List of tools to check
            
        Returns:
            True if cache should be invalidated
        """
        current_version = self.version_manager.get_current_environment_version(tools)

        if self.version_manager.version_has_changed(current_version):
            logger.info("Tool versions changed, invalidating cache",
                       extra={"current_tools": current_version.get("tools", {})})

            # Update stored version info
            self.version_manager.update_version_info(current_version)

            # Invalidate all cache entries
            self.invalidate_all_entries()
            return True

        return False

    def should_invalidate_for_config_change(self, config_files: list) -> bool:
        """Check if cache should be invalidated due to config file changes.
        
        Args:
            config_files: List of configuration file paths
            
        Returns:
            True if any config file has changed
        """
        for config_file in config_files:
            current_hash = get_config_file_hash(config_file)
            stored_hash = self.config_hashes.get(Path(config_file).name)

            if current_hash != stored_hash:
                logger.info("Configuration file changed",
                           extra={"config_file": config_file})
                return True

        return False

    def update_config_hashes(self, config_files: list) -> None:
        """Update stored hashes for configuration files.
        
        Args:
            config_files: List of configuration file paths
        """
        for config_file in config_files:
            file_hash = get_config_file_hash(config_file)
            if file_hash:
                self.config_hashes[Path(config_file).name] = file_hash

    def invalidate_all_entries(self) -> int:
        """Remove all cache entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0

        try:
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "version_info.json":  # Don't remove version info
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except OSError:
                        continue

            logger.info("Cache invalidation completed",
                       extra={"entries_removed": removed_count})

        except Exception as e:
            logger.error("Cache invalidation failed", extra={"error": str(e)})

        return removed_count

    def invalidate_if_needed(self, tools: list, config_files: list = None) -> bool:
        """Invalidate cache if needed based on tools or config changes.
        
        Args:
            tools: List of tools to check
            config_files: List of config files to check (optional)
            
        Returns:
            True if cache was invalidated
        """
        should_invalidate = False

        # Check tool version changes
        if self.should_invalidate_cache(tools):
            should_invalidate = True

        # Check config file changes
        if config_files and self.should_invalidate_for_config_change(config_files):
            should_invalidate = True
            # Update config hashes for future checks
            self.update_config_hashes(config_files)
            # Invalidate entries
            self.invalidate_all_entries()

        return should_invalidate


def should_invalidate_cache(tools: list, config_files: list = None, cache_dir: str = None) -> bool:
    """Module-level function to check if cache should be invalidated.
    
    Args:
        tools: List of tools to check for version changes
        config_files: List of config files to check for changes
        cache_dir: Cache directory (uses default if not specified)
        
    Returns:
        True if cache should be invalidated
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/autogen-review")

    strategy = InvalidationStrategy(cache_dir)
    return strategy.invalidate_if_needed(tools, config_files or [])
