#!/usr/bin/env python3
"""
Intelligent caching system for AutoGen Code Review Bot.

This module provides advanced caching capabilities with automatic invalidation,
compression, and smart eviction policies.
"""

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import threading

from .logging_config import get_logger
from .robust_error_handling import robust_operation, ErrorSeverity, safe_execute

logger = get_logger(__name__)


class CacheEntry:
    """Represents a cache entry with metadata."""
    
    def __init__(self, data: Any, ttl_seconds: int = 3600):
        self.data = data
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.access_count = 1
        self.last_accessed = self.created_at
        self.data_size = len(pickle.dumps(data))
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def access(self) -> Any:
        """Record access and return data."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.data
    
    def get_age(self) -> float:
        """Get the age of the entry in seconds."""
        return time.time() - self.created_at


class IntelligentCache:
    """Intelligent caching system with advanced features."""
    
    def __init__(self, 
                 cache_dir: str = ".cache/autogen-review",
                 max_memory_mb: int = 100,
                 default_ttl: int = 3600,
                 cleanup_interval: int = 300):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # In-memory cache for fast access
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_reads': 0,
            'disk_writes': 0,
            'cleanup_runs': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Initialized intelligent cache at {self.cache_dir}")
    
    def get_cache_key(self, 
                     repo_path: str, 
                     analysis_type: str, 
                     config_hash: Optional[str] = None,
                     repo_hash: Optional[str] = None) -> str:
        """Generate a unique cache key for analysis results."""
        key_components = [
            repo_path,
            analysis_type,
            config_hash or "default",
            repo_hash or self._get_repo_hash(repo_path)
        ]
        
        key_string = "|".join(str(c) for c in key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_repo_hash(self, repo_path: str) -> str:
        """Get a hash representing the current state of the repository."""
        try:
            repo_path_obj = Path(repo_path)
            if not repo_path_obj.exists():
                return "nonexistent"
            
            # Get modification times of all relevant files
            file_mtimes = []
            for file_path in repo_path_obj.rglob("*"):
                if (file_path.is_file() and 
                    not any(ignored in str(file_path) for ignored in 
                           ['.git', '__pycache__', '.pytest_cache', 'node_modules'])):
                    try:
                        file_mtimes.append((str(file_path), file_path.stat().st_mtime))
                    except (OSError, PermissionError):
                        continue
            
            # Sort for consistent hash
            file_mtimes.sort()
            hash_input = json.dumps(file_mtimes, sort_keys=True)
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to compute repo hash for {repo_path}: {e}")
            return f"error-{int(time.time())}"
    
    @robust_operation(
        component="cache_system",
        operation="get_cached_result", 
        severity=ErrorSeverity.LOW,
        retry_count=1,
        fallback_value=None
    )
    def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data."""
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                
                if entry.is_expired():
                    del self.memory_cache[cache_key]
                    self.stats['misses'] += 1
                    logger.debug(f"Cache expired for key {cache_key[:12]}...")
                    return None
                
                self.stats['hits'] += 1
                logger.debug(f"Memory cache hit for key {cache_key[:12]}...")
                return entry.access()
            
            # Check disk cache
            disk_file = self.cache_dir / f"{cache_key}.cache"
            if disk_file.exists():
                try:
                    with open(disk_file, 'rb') as f:
                        entry_data = pickle.load(f)
                    
                    entry = CacheEntry(
                        data=entry_data['data'],
                        ttl_seconds=entry_data.get('ttl', self.default_ttl)
                    )
                    entry.created_at = entry_data['created_at']
                    
                    if entry.is_expired():
                        disk_file.unlink()
                        self.stats['misses'] += 1
                        logger.debug(f"Disk cache expired for key {cache_key[:12]}...")
                        return None
                    
                    # Move to memory cache
                    self._ensure_memory_space()
                    self.memory_cache[cache_key] = entry
                    
                    self.stats['hits'] += 1
                    self.stats['disk_reads'] += 1
                    logger.debug(f"Disk cache hit for key {cache_key[:12]}...")
                    return entry.access()
                    
                except Exception as e:
                    logger.warning(f"Failed to read disk cache {cache_key[:12]}: {e}")
                    safe_execute(disk_file.unlink, component="cache_system", operation="cleanup")
            
            self.stats['misses'] += 1
            return None
    
    @robust_operation(
        component="cache_system", 
        operation="store_cached_result",
        severity=ErrorSeverity.LOW,
        retry_count=1
    )
    def put(self, cache_key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store data in cache."""
        with self.cache_lock:
            ttl = ttl_seconds or self.default_ttl
            entry = CacheEntry(data, ttl)
            
            # Store in memory cache
            self._ensure_memory_space()
            self.memory_cache[cache_key] = entry
            
            # Store on disk asynchronously
            threading.Thread(
                target=self._write_to_disk,
                args=(cache_key, data, ttl, entry.created_at),
                daemon=True
            ).start()
            
            logger.debug(f"Cached data for key {cache_key[:12]}... (size: {entry.data_size} bytes)")
    
    def _write_to_disk(self, cache_key: str, data: Any, ttl: int, created_at: float) -> None:
        """Write cache entry to disk."""
        try:
            disk_file = self.cache_dir / f"{cache_key}.cache"
            entry_data = {
                'data': data,
                'ttl': ttl,
                'created_at': created_at
            }
            
            with open(disk_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
            self.stats['disk_writes'] += 1
            
        except Exception as e:
            logger.warning(f"Failed to write disk cache {cache_key[:12]}: {e}")
    
    def _ensure_memory_space(self) -> None:
        """Ensure memory cache doesn't exceed limits."""
        current_size = sum(entry.data_size for entry in self.memory_cache.values())
        
        if current_size > self.max_memory_bytes:
            # Evict least recently used entries
            entries_by_access = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            evicted_count = 0
            for cache_key, entry in entries_by_access:
                del self.memory_cache[cache_key]
                current_size -= entry.data_size
                evicted_count += 1
                
                if current_size <= self.max_memory_bytes * 0.8:  # Leave 20% headroom
                    break
            
            self.stats['evictions'] += evicted_count
            logger.debug(f"Evicted {evicted_count} entries from memory cache")
    
    def _periodic_cleanup(self) -> None:
        """Periodically clean up expired cache entries."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup()
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def cleanup(self) -> None:
        """Clean up expired cache entries."""
        try:
            with self.cache_lock:
                # Clean memory cache
                expired_keys = [
                    key for key, entry in self.memory_cache.items() 
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    del self.memory_cache[key]
                
                logger.debug(f"Cleaned {len(expired_keys)} expired memory cache entries")
            
            # Clean disk cache
            disk_cleaned = 0
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        # Check if file is old enough to warrant checking
                        if (time.time() - cache_file.stat().st_mtime) > self.default_ttl:
                            with open(cache_file, 'rb') as f:
                                entry_data = pickle.load(f)
                            
                            created_at = entry_data.get('created_at', 0)
                            ttl = entry_data.get('ttl', self.default_ttl)
                            
                            if (time.time() - created_at) > ttl:
                                cache_file.unlink()
                                disk_cleaned += 1
                                
                    except Exception as e:
                        logger.debug(f"Error checking cache file {cache_file}: {e}")
                        safe_execute(cache_file.unlink, component="cache_system", operation="cleanup")
            
            self.stats['cleanup_runs'] += 1
            logger.debug(f"Cleaned {disk_cleaned} expired disk cache entries")
            
        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests) if total_requests > 0 else 0
            
            memory_size = sum(entry.data_size for entry in self.memory_cache.values())
            memory_entries = len(self.memory_cache)
            
            disk_entries = len(list(self.cache_dir.glob("*.cache"))) if self.cache_dir.exists() else 0
            
            return {
                'hit_rate': round(hit_rate * 100, 2),
                'total_requests': total_requests,
                'memory_entries': memory_entries,
                'memory_size_mb': round(memory_size / (1024 * 1024), 2),
                'disk_entries': disk_entries,
                'stats': self.stats.copy()
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.cache_lock:
            self.memory_cache.clear()
            
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    safe_execute(cache_file.unlink, component="cache_system", operation="clear")
            
            logger.info("Cache cleared")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        invalidated = 0
        
        with self.cache_lock:
            # Invalidate memory cache
            keys_to_remove = [key for key in self.memory_cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated += 1
            
            # Invalidate disk cache
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    if pattern in cache_file.name:
                        safe_execute(cache_file.unlink, component="cache_system", operation="invalidate")
                        invalidated += 1
        
        logger.info(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")
        return invalidated


# Global cache instance
intelligent_cache = IntelligentCache()


def with_cache(analysis_type: str, ttl_seconds: int = 3600):
    """Decorator to add caching to analysis functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key based on function arguments
            repo_path = args[0] if args else kwargs.get('repo_path', 'unknown')
            
            # Create a hash of all arguments for cache key
            arg_hash = hashlib.md5(
                json.dumps([str(arg) for arg in args] + [f"{k}={v}" for k, v in kwargs.items()], 
                          sort_keys=True).encode()
            ).hexdigest()
            
            cache_key = intelligent_cache.get_cache_key(
                repo_path=repo_path,
                analysis_type=f"{analysis_type}_{func.__name__}",
                config_hash=arg_hash
            )
            
            # Try to get from cache
            cached_result = intelligent_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached result for {analysis_type}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            intelligent_cache.put(cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator