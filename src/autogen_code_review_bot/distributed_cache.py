"""Advanced distributed caching system with intelligent eviction and preloading."""

import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

import redis
import redis.exceptions

from .logging_config import get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    tags: Set[str]

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


class IntelligentCache:
    """Intelligent cache with LRU eviction, preloading, and analytics."""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512,
                 default_ttl: float = 3600, cleanup_interval: float = 300):
        """Initialize intelligent cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RWLock() if hasattr(threading, 'RWLock') else threading.Lock()

        # Analytics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        # Tag index for efficient invalidation
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)

        # Background cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("Intelligent cache initialized",
                   max_size=max_size,
                   max_memory_mb=max_memory_mb)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._miss_count += 1
                metrics.record_counter("cache_misses", 1)
                return default

            if entry.is_expired:
                del self._cache[key]
                self._remove_from_access_order(key)
                self._remove_from_tag_index(key, entry.tags)
                self._miss_count += 1
                metrics.record_counter("cache_misses", 1)
                metrics.record_counter("cache_expirations", 1)
                return default

            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_access_order(key)

            self._hit_count += 1
            metrics.record_counter("cache_hits", 1)

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            tags: Optional[Set[str]] = None) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Optional tags for grouping
            
        Returns:
            True if successfully cached
        """
        if ttl is None:
            ttl = self.default_ttl

        tags = tags or set()
        size_bytes = self._estimate_size(value)

        with self._lock:
            # Check if we need to evict
            self._ensure_capacity(size_bytes)

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes,
                tags=tags
            )

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._remove_from_tag_index(key, old_entry.tags)

            # Add new entry
            self._cache[key] = entry
            self._update_access_order(key)
            self._update_tag_index(key, tags)

            metrics.record_counter("cache_sets", 1)
            metrics.record_gauge("cache_size", len(self._cache))

            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                del self._cache[key]
                self._remove_from_access_order(key)
                self._remove_from_tag_index(key, entry.tags)
                metrics.record_counter("cache_deletes", 1)
                return True
            return False

    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with specified tags.
        
        Args:
            tags: Tags to invalidate
            
        Returns:
            Number of entries invalidated
        """
        keys_to_delete = set()

        with self._lock:
            for tag in tags:
                keys_to_delete.update(self._tag_index.get(tag, set()))

            count = 0
            for key in keys_to_delete:
                if key in self._cache:
                    entry = self._cache[key]
                    del self._cache[key]
                    self._remove_from_access_order(key)
                    self._remove_from_tag_index(key, entry.tags)
                    count += 1

            metrics.record_counter("cache_tag_invalidations", count)
            return count

    def preload(self, keys_and_loaders: Dict[str, Callable[[], Any]],
                ttl: Optional[float] = None, background: bool = True) -> None:
        """Preload cache entries.
        
        Args:
            keys_and_loaders: Dictionary mapping keys to loader functions
            ttl: TTL for preloaded entries
            background: Whether to load in background
        """
        if background:
            threading.Thread(target=self._preload_entries,
                           args=(keys_and_loaders, ttl), daemon=True).start()
        else:
            self._preload_entries(keys_and_loaders, ttl)

    def _preload_entries(self, keys_and_loaders: Dict[str, Callable[[], Any]],
                        ttl: Optional[float]):
        """Preload cache entries (internal).
        
        Args:
            keys_and_loaders: Dictionary mapping keys to loader functions
            ttl: TTL for preloaded entries
        """
        for key, loader in keys_and_loaders.items():
            try:
                if self.get(key) is None:  # Only load if not already cached
                    value = loader()
                    self.set(key, value, ttl=ttl, tags={"preloaded"})
                    logger.debug("Preloaded cache entry", key=key)
            except Exception as e:
                logger.warning("Failed to preload cache entry", key=key, error=str(e))

        metrics.record_counter("cache_preload_batches", 1,
                             tags={"size": str(len(keys_and_loaders))})

    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry.
        
        Args:
            new_entry_size: Size of new entry in bytes
        """
        # Check memory limit
        current_memory = sum(entry.size_bytes for entry in self._cache.values())

        while (current_memory + new_entry_size > self.max_memory_bytes or
               len(self._cache) >= self.max_size):

            if not self._access_order:
                break  # No entries to evict

            # Evict LRU entry
            lru_key = self._access_order[0]
            if lru_key in self._cache:
                entry = self._cache[lru_key]
                current_memory -= entry.size_bytes
                del self._cache[lru_key]
                self._remove_from_tag_index(lru_key, entry.tags)
                self._eviction_count += 1
                metrics.record_counter("cache_evictions", 1, tags={"reason": "lru"})

            self._access_order.pop(0)

    def _update_access_order(self, key: str):
        """Update access order for LRU tracking.
        
        Args:
            key: Cache key that was accessed
        """
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove_from_access_order(self, key: str):
        """Remove key from access order tracking.
        
        Args:
            key: Cache key to remove
        """
        if key in self._access_order:
            self._access_order.remove(key)

    def _update_tag_index(self, key: str, tags: Set[str]):
        """Update tag index for key.
        
        Args:
            key: Cache key
            tags: Tags associated with key
        """
        for tag in tags:
            self._tag_index[tag].add(key)

    def _remove_from_tag_index(self, key: str, tags: Set[str]):
        """Remove key from tag index.
        
        Args:
            key: Cache key to remove
            tags: Tags to remove from
        """
        for tag in tags:
            self._tag_index[tag].discard(key)
            if not self._tag_index[tag]:
                del self._tag_index[tag]

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value.
        
        Args:
            value: Value to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v)
                          for k, v in list(value.items())[:10])  # Sample first 10
            else:
                return 1000  # Default estimate

    def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error("Error in cache cleanup", error=str(e))

    def _cleanup_expired(self):
        """Remove expired entries."""
        expired_keys = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self._cache[key]
                del self._cache[key]
                self._remove_from_access_order(key)
                self._remove_from_tag_index(key, entry.tags)

        if expired_keys:
            metrics.record_counter("cache_cleanup_expired", len(expired_keys))
            logger.debug("Cleaned up expired cache entries", count=len(expired_keys))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

            total_memory = sum(entry.size_bytes for entry in self._cache.values())

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_bytes": total_memory,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_utilization": total_memory / self.max_memory_bytes,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self._eviction_count,
                "tag_count": len(self._tag_index)
            }


class DistributedCache:
    """Redis-based distributed cache with local fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379",
                 key_prefix: str = "autogen:", fallback_cache: Optional[IntelligentCache] = None):
        """Initialize distributed cache.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            fallback_cache: Local cache for fallback
        """
        self.key_prefix = key_prefix
        self.fallback_cache = fallback_cache or IntelligentCache()

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            self.redis_client.ping()  # Test connection
            self._redis_available = True
            logger.info("Distributed cache connected to Redis", url=redis_url)
        except Exception as e:
            logger.warning("Redis unavailable, using fallback cache only", error=str(e))
            self.redis_client = None
            self._redis_available = False

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key.
        
        Args:
            key: Original key
            
        Returns:
            Prefixed key
        """
        return f"{self.key_prefix}{key}"

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from distributed cache.
        
        Args:
            key: Cache key
            default: Default value
            
        Returns:
            Cached value or default
        """
        redis_key = self._make_key(key)

        # Try Redis first
        if self._redis_available:
            try:
                value = self.redis_client.get(redis_key)
                if value is not None:
                    deserialized = pickle.loads(value)
                    metrics.record_counter("distributed_cache_hits", 1, tags={"source": "redis"})
                    return deserialized
            except Exception as e:
                logger.warning("Redis get failed, trying fallback", key=key, error=str(e))
                self._redis_available = False

        # Fallback to local cache
        result = self.fallback_cache.get(key, default)
        if result != default:
            metrics.record_counter("distributed_cache_hits", 1, tags={"source": "local"})
        else:
            metrics.record_counter("distributed_cache_misses", 1)

        return result

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 tags: Optional[Set[str]] = None) -> bool:
        """Set value in distributed cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds
            tags: Optional tags
            
        Returns:
            True if successfully cached
        """
        redis_key = self._make_key(key)

        # Set in Redis if available
        redis_success = False
        if self._redis_available:
            try:
                serialized = pickle.dumps(value)
                if ttl:
                    redis_success = self.redis_client.setex(redis_key, ttl, serialized)
                else:
                    redis_success = self.redis_client.set(redis_key, serialized)

                # Set tags in Redis using sets
                if tags and redis_success:
                    for tag in tags:
                        tag_key = f"{self.key_prefix}tag:{tag}"
                        self.redis_client.sadd(tag_key, key)
                        if ttl:
                            self.redis_client.expire(tag_key, ttl)

                metrics.record_counter("distributed_cache_sets", 1, tags={"target": "redis"})

            except Exception as e:
                logger.warning("Redis set failed, using fallback only", key=key, error=str(e))
                self._redis_available = False

        # Always set in local cache as well
        local_success = self.fallback_cache.set(key, value, ttl=ttl, tags=tags)

        if local_success:
            metrics.record_counter("distributed_cache_sets", 1, tags={"target": "local"})

        return redis_success or local_success

    async def delete(self, key: str) -> bool:
        """Delete key from distributed cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        redis_key = self._make_key(key)
        redis_deleted = False

        # Delete from Redis
        if self._redis_available:
            try:
                redis_deleted = bool(self.redis_client.delete(redis_key))
                metrics.record_counter("distributed_cache_deletes", 1, tags={"target": "redis"})
            except Exception as e:
                logger.warning("Redis delete failed", key=key, error=str(e))
                self._redis_available = False

        # Delete from local cache
        local_deleted = self.fallback_cache.delete(key)
        if local_deleted:
            metrics.record_counter("distributed_cache_deletes", 1, tags={"target": "local"})

        return redis_deleted or local_deleted

    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries by tags.
        
        Args:
            tags: Tags to invalidate
            
        Returns:
            Number of entries invalidated
        """
        total_invalidated = 0

        # Invalidate in Redis
        if self._redis_available:
            try:
                pipeline = self.redis_client.pipeline()
                keys_to_delete = set()

                for tag in tags:
                    tag_key = f"{self.key_prefix}tag:{tag}"
                    tagged_keys = self.redis_client.smembers(tag_key)
                    for tagged_key in tagged_keys:
                        if isinstance(tagged_key, bytes):
                            tagged_key = tagged_key.decode('utf-8')
                        keys_to_delete.add(self._make_key(tagged_key))
                    pipeline.delete(tag_key)

                # Delete tagged keys
                if keys_to_delete:
                    pipeline.delete(*keys_to_delete)
                    pipeline.execute()
                    total_invalidated += len(keys_to_delete)

                metrics.record_counter("distributed_cache_tag_invalidations",
                                     len(keys_to_delete), tags={"target": "redis"})

            except Exception as e:
                logger.warning("Redis tag invalidation failed", tags=list(tags), error=str(e))
                self._redis_available = False

        # Invalidate in local cache
        local_invalidated = self.fallback_cache.invalidate_by_tags(tags)
        total_invalidated += local_invalidated

        return total_invalidated

    def get_stats(self) -> Dict[str, Any]:
        """Get distributed cache statistics.
        
        Returns:
            Cache statistics
        """
        stats = {
            "redis_available": self._redis_available,
            "fallback_cache": self.fallback_cache.get_stats()
        }

        if self._redis_available:
            try:
                redis_info = self.redis_client.info()
                stats["redis"] = {
                    "used_memory": redis_info.get("used_memory", 0),
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0)
                }
            except Exception as e:
                logger.warning("Failed to get Redis stats", error=str(e))
                stats["redis"] = {"error": str(e)}

        return stats


# Global cache instance
_global_cache: Optional[Union[IntelligentCache, DistributedCache]] = None


def get_cache() -> Union[IntelligentCache, DistributedCache]:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        # Try to create distributed cache, fallback to local
        try:
            _global_cache = DistributedCache()
        except Exception:
            _global_cache = IntelligentCache()
    return _global_cache


def configure_cache(cache_type: str = "intelligent", **kwargs) -> Union[IntelligentCache, DistributedCache]:
    """Configure global cache instance.
    
    Args:
        cache_type: Type of cache ("intelligent" or "distributed")
        **kwargs: Cache configuration parameters
        
    Returns:
        Configured cache instance
    """
    global _global_cache

    if cache_type == "distributed":
        _global_cache = DistributedCache(**kwargs)
    else:
        _global_cache = IntelligentCache(**kwargs)

    return _global_cache
