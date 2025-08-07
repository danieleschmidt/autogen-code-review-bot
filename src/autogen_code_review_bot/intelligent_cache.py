#!/usr/bin/env python3
"""
Intelligent Caching and Optimization System for AutoGen Code Review Bot.

Implements multi-level caching, predictive pre-loading, cache warming,
intelligent invalidation, and performance optimization strategies.
"""

import asyncio
import json
import time
import hashlib
import pickle
import gzip
import threading
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import weakref

from redis.asyncio import Redis
import aiohttp

from .logging_utils import get_logger
from .metrics import get_metrics_registry
from .models import PRAnalysisResult

logger = get_logger(__name__)
metrics = get_metrics_registry()


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    compression: bool = False
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    
    def calculate_hit_rate(self):
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        self.hit_rate = (self.hits / total * 100) if total > 0 else 0.0


class AdaptiveLRU:
    """Adaptive LRU cache with intelligent eviction."""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 1024):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self._frequency_scores: Dict[str, float] = {}
        
        self.logger = get_logger(__name__ + ".AdaptiveLRU")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired():
                self._stats.misses += 1
                if entry:
                    del self._cache[key]
                return None
            
            # Update access statistics
            entry.update_access()
            self._update_access_pattern(key)
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.calculate_hit_rate()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            tags: Set[str] = None, compress: bool = False) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Compress if requested and beneficial
            stored_value = value
            is_compressed = False
            
            if compress and size_bytes > 1024:  # Only compress if > 1KB
                try:
                    serialized = pickle.dumps(value)
                    compressed = gzip.compress(serialized)
                    if len(compressed) < len(serialized) * 0.8:  # 20% savings threshold
                        stored_value = compressed
                        is_compressed = True
                        size_bytes = len(compressed)
                except Exception as e:
                    self.logger.warning(f"Compression failed for key {key}: {e}")
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=stored_value,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                tags=tags or set(),
                compression=is_compressed
            )
            
            # Check if we need to evict
            while (len(self._cache) >= self.max_size or 
                   self._stats.size_bytes + size_bytes > self.max_memory_bytes):
                if not self._evict_entry():
                    # Can't evict any more entries
                    return False
            
            # Add to cache
            if key in self._cache:
                # Update existing entry
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._stats.size_bytes += size_bytes
            self._stats.entry_count = len(self._cache)
            
            self._update_access_pattern(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.size_bytes -= entry.size_bytes
                del self._cache[key]
                self._stats.entry_count = len(self._cache)
                return True
            return False
    
    def clear_by_tags(self, tags: Set[str]) -> int:
        """Clear entries matching any of the provided tags."""
        with self._lock:
            keys_to_delete = []
            
            for key, entry in self._cache.items():
                if entry.tags.intersection(tags):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self.delete(key)
            
            return len(keys_to_delete)
    
    def _evict_entry(self) -> bool:
        """Evict least valuable entry using adaptive algorithm."""
        if not self._cache:
            return False
        
        # Calculate eviction scores for all entries
        scores = {}
        current_time = datetime.now(timezone.utc)
        
        for key, entry in self._cache.items():
            # Base score: inverse of access frequency
            frequency_score = self._frequency_scores.get(key, 1.0)
            
            # Recency score: how recently accessed
            time_since_access = (current_time - entry.last_accessed).total_seconds()
            recency_score = 1.0 / (1.0 + time_since_access / 3600)  # Decay over hours
            
            # Size score: prefer evicting larger items
            size_score = entry.size_bytes / 1024  # KB
            
            # TTL score: prefer evicting items closer to expiration
            ttl_score = 1.0
            if entry.ttl_seconds:
                age = (current_time - entry.created_at).total_seconds()
                remaining_ttl = entry.ttl_seconds - age
                if remaining_ttl > 0:
                    ttl_score = entry.ttl_seconds / remaining_ttl
            
            # Combined score (lower is more likely to be evicted)
            scores[key] = frequency_score * recency_score - size_score * 0.1 - ttl_score * 0.1
        
        # Find entry with lowest score
        victim_key = min(scores, key=scores.get)
        victim_entry = self._cache[victim_key]
        
        self.logger.debug(f"Evicting cache entry: {victim_key}", extra={
            'size_bytes': victim_entry.size_bytes,
            'access_count': victim_entry.access_count,
            'age_seconds': (current_time - victim_entry.created_at).total_seconds()
        })
        
        # Remove entry
        self._stats.size_bytes -= victim_entry.size_bytes
        self._stats.evictions += 1
        del self._cache[victim_key]
        
        # Clean up frequency tracking
        if victim_key in self._frequency_scores:
            del self._frequency_scores[victim_key]
        if victim_key in self._access_patterns:
            del self._access_patterns[victim_key]
        
        return True
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for predictive analysis."""
        current_time = datetime.now(timezone.utc)
        
        # Record access time
        self._access_patterns[key].append(current_time)
        
        # Keep only recent access times (last 24 hours)
        cutoff = current_time - timedelta(hours=24)
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff
        ]
        
        # Update frequency score
        access_count = len(self._access_patterns[key])
        if access_count > 0:
            # Calculate frequency based on accesses in last hour vs last 24 hours
            hour_ago = current_time - timedelta(hours=1)
            recent_accesses = len([t for t in self._access_patterns[key] if t > hour_ago])
            
            self._frequency_scores[key] = recent_accesses * 10 + access_count
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in value.items())
            else:
                # Fallback: serialize to estimate size
                return len(pickle.dumps(value))
        except:
            return 1024  # Default 1KB estimate
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.entry_count = len(self._cache)
            return self._stats
    
    def get_key_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a cache key."""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            
            return {
                'key': key,
                'size_bytes': entry.size_bytes,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count,
                'ttl_seconds': entry.ttl_seconds,
                'is_expired': entry.is_expired(),
                'tags': list(entry.tags),
                'compressed': entry.compression,
                'frequency_score': self._frequency_scores.get(key, 0.0)
            }


class DistributedCache:
    """Distributed cache with Redis backend and local L1 cache."""
    
    def __init__(self, redis: Redis, local_cache_size: int = 5000, 
                 default_ttl: int = 3600, namespace: str = "autogen"):
        self.redis = redis
        self.namespace = namespace
        self.default_ttl = default_ttl
        
        # Local L1 cache
        self.local_cache = AdaptiveLRU(max_size=local_cache_size, max_memory_mb=512)
        
        self.logger = get_logger(__name__ + ".DistributedCache")
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str, use_local: bool = True) -> Optional[Any]:
        """Get item from cache (L1 local, then L2 Redis)."""
        redis_key = self._make_key(key)
        
        # Try L1 cache first
        if use_local:
            local_value = self.local_cache.get(key)
            if local_value is not None:
                metrics.record_counter("cache_hits_total", 1, tags={"level": "L1"})
                return local_value
        
        # Try L2 Redis cache
        try:
            redis_value = await self.redis.get(redis_key)
            if redis_value:
                # Deserialize
                value = json.loads(redis_value)
                
                # Store in L1 cache
                if use_local:
                    self.local_cache.put(key, value, ttl_seconds=self.default_ttl)
                
                metrics.record_counter("cache_hits_total", 1, tags={"level": "L2"})
                return value
            
        except Exception as e:
            self.logger.error(f"Redis cache get error for key {key}: {e}")
        
        metrics.record_counter("cache_misses_total", 1, tags={"key": key[:50]})
        return None
    
    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 tags: Set[str] = None, use_local: bool = True) -> bool:
        """Put item in cache (both L1 and L2)."""
        redis_key = self._make_key(key)
        ttl = ttl_seconds or self.default_ttl
        
        try:
            # Store in Redis (L2)
            serialized = json.dumps(value, default=str)
            await self.redis.setex(redis_key, ttl, serialized)
            
            # Store in local cache (L1)
            if use_local:
                self.local_cache.put(key, value, ttl_seconds=ttl, tags=tags)
            
            # Store tags for invalidation
            if tags:
                for tag in tags:
                    await self.redis.sadd(f"{self.namespace}:tag:{tag}", key)
                    await self.redis.expire(f"{self.namespace}:tag:{tag}", ttl)
            
            metrics.record_counter("cache_writes_total", 1, tags={"key": key[:50]})
            return True
            
        except Exception as e:
            self.logger.error(f"Cache put error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from both caches."""
        redis_key = self._make_key(key)
        
        try:
            # Delete from Redis
            await self.redis.delete(redis_key)
            
            # Delete from local cache
            self.local_cache.delete(key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries by tags."""
        total_invalidated = 0
        
        try:
            # Get keys for each tag from Redis
            all_keys = set()
            for tag in tags:
                tag_key = f"{self.namespace}:tag:{tag}"
                keys = await self.redis.smembers(tag_key)
                all_keys.update(keys)
                
                # Clean up tag set
                await self.redis.delete(tag_key)
            
            # Delete keys from both caches
            for key in all_keys:
                await self.delete(key)
                total_invalidated += 1
            
            # Also clear from local cache by tags
            local_cleared = self.local_cache.clear_by_tags(tags)
            
            self.logger.info(f"Cache invalidation completed", extra={
                'tags': list(tags),
                'redis_keys_invalidated': total_invalidated,
                'local_keys_invalidated': local_cleared
            })
            
            return total_invalidated
            
        except Exception as e:
            self.logger.error(f"Cache invalidation error: {e}")
            return 0
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        try:
            # Redis info
            redis_info = await self.redis.info("memory")
            redis_keyspace = await self.redis.info("keyspace")
            
            # Local cache stats
            local_stats = self.local_cache.get_stats()
            
            return {
                'redis': {
                    'used_memory_mb': redis_info.get('used_memory', 0) / 1024 / 1024,
                    'keyspace': redis_keyspace
                },
                'local': {
                    'hit_rate': local_stats.hit_rate,
                    'entry_count': local_stats.entry_count,
                    'size_mb': local_stats.size_bytes / 1024 / 1024,
                    'hits': local_stats.hits,
                    'misses': local_stats.misses,
                    'evictions': local_stats.evictions
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache info: {e}")
            return {'error': str(e)}


class PredictiveCache:
    """Predictive cache with machine learning-based pre-loading."""
    
    def __init__(self, distributed_cache: DistributedCache):
        self.cache = distributed_cache
        self.access_log: List[Tuple[str, datetime, str]] = []  # key, time, context
        self.prediction_models: Dict[str, Any] = {}
        self.preload_executor = ThreadPoolExecutor(max_workers=2)
        
        self.logger = get_logger(__name__ + ".PredictiveCache")
        
        # Start background analysis
        self._analysis_task: Optional[asyncio.Task] = None
        self._running = False
    
    def log_access(self, key: str, context: str = ""):
        """Log cache access for pattern learning."""
        self.access_log.append((key, datetime.now(timezone.utc), context))
        
        # Keep only recent accesses (last 7 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        self.access_log = [
            (k, t, c) for k, t, c in self.access_log if t > cutoff
        ]
    
    async def get_with_prediction(self, key: str, loader_func: Callable = None, 
                                context: str = "") -> Optional[Any]:
        """Get item from cache with predictive pre-loading."""
        # Log the access
        self.log_access(key, context)
        
        # Try to get from cache
        value = await self.cache.get(key)
        
        if value is None and loader_func:
            # Load and cache
            try:
                value = await self._load_and_cache(key, loader_func)
            except Exception as e:
                self.logger.error(f"Failed to load key {key}: {e}")
        
        # Trigger predictive preloading
        asyncio.create_task(self._predict_and_preload(key, context))
        
        return value
    
    async def _load_and_cache(self, key: str, loader_func: Callable) -> Any:
        """Load data and put in cache."""
        if asyncio.iscoroutinefunction(loader_func):
            value = await loader_func()
        else:
            # Run in executor
            value = await asyncio.get_event_loop().run_in_executor(
                self.preload_executor, loader_func
            )
        
        # Cache the loaded value
        await self.cache.put(key, value)
        return value
    
    async def _predict_and_preload(self, accessed_key: str, context: str):
        """Predict and preload likely next accesses."""
        try:
            # Simple pattern-based prediction
            predictions = self._predict_next_keys(accessed_key, context)
            
            # Preload predicted keys (limit to top 3)
            for predicted_key in predictions[:3]:
                # Check if already in cache
                cached_value = await self.cache.get(predicted_key, use_local=True)
                if cached_value is None:
                    # Would need a loader function registry for preloading
                    self.logger.debug(f"Would preload key: {predicted_key}")
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
    
    def _predict_next_keys(self, current_key: str, context: str) -> List[str]:
        """Predict next likely cache keys based on patterns."""
        predictions = []
        
        # Analyze recent access patterns
        recent_accesses = [
            (k, c) for k, t, c in self.access_log[-100:]  # Last 100 accesses
        ]
        
        # Find keys that often follow the current key
        following_keys = defaultdict(int)
        
        for i, (key, ctx) in enumerate(recent_accesses[:-1]):
            if key == current_key and ctx == context:
                next_key, next_ctx = recent_accesses[i + 1]
                following_keys[next_key] += 1
        
        # Sort by frequency
        predictions = sorted(following_keys.items(), key=lambda x: x[1], reverse=True)
        return [key for key, count in predictions]
    
    def get_access_patterns(self) -> Dict[str, Any]:
        """Get analysis of access patterns."""
        if not self.access_log:
            return {}
        
        # Key frequency
        key_counts = defaultdict(int)
        context_counts = defaultdict(int)
        
        for key, timestamp, context in self.access_log:
            key_counts[key] += 1
            if context:
                context_counts[context] += 1
        
        return {
            'total_accesses': len(self.access_log),
            'unique_keys': len(key_counts),
            'most_accessed_keys': sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'contexts': sorted(context_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'time_range': {
                'start': min(t for _, t, _ in self.access_log).isoformat(),
                'end': max(t for _, t, _ in self.access_log).isoformat()
            }
        }


class CacheWarmer:
    """Cache warming service for preloading frequently accessed data."""
    
    def __init__(self, cache: DistributedCache):
        self.cache = cache
        self.warming_strategies: Dict[str, Callable] = {}
        self.warming_schedule: Dict[str, Dict[str, Any]] = {}
        
        self.logger = get_logger(__name__ + ".CacheWarmer")
        self._warming_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_warming_strategy(self, name: str, loader_func: Callable, 
                                schedule: Dict[str, Any]):
        """Register a cache warming strategy."""
        self.warming_strategies[name] = loader_func
        self.warming_schedule[name] = {
            'interval_seconds': schedule.get('interval_seconds', 3600),
            'keys': schedule.get('keys', []),
            'tags': schedule.get('tags', set()),
            'ttl_seconds': schedule.get('ttl_seconds', 3600),
            'last_run': None
        }
        
        self.logger.info(f"Registered cache warming strategy: {name}")
    
    async def start_warming(self):
        """Start cache warming service."""
        if self._running:
            return
        
        self._running = True
        self._warming_task = asyncio.create_task(self._warming_loop())
        self.logger.info("Cache warmer started")
    
    async def stop_warming(self):
        """Stop cache warming service."""
        self._running = False
        if self._warming_task:
            self._warming_task.cancel()
            try:
                await self._warming_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Cache warmer stopped")
    
    async def _warming_loop(self):
        """Main cache warming loop."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check each warming strategy
                for name, schedule in self.warming_schedule.items():
                    last_run = schedule.get('last_run')
                    interval = schedule['interval_seconds']
                    
                    if (last_run is None or 
                        (current_time - last_run).total_seconds() >= interval):
                        
                        await self._execute_warming_strategy(name)
                        schedule['last_run'] = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(300)  # Back off on errors
    
    async def _execute_warming_strategy(self, strategy_name: str):
        """Execute a specific warming strategy."""
        try:
            strategy = self.warming_strategies[strategy_name]
            schedule = self.warming_schedule[strategy_name]
            
            self.logger.info(f"Executing cache warming strategy: {strategy_name}")
            
            # Load data using strategy
            if asyncio.iscoroutinefunction(strategy):
                data = await strategy()
            else:
                data = strategy()
            
            # Cache the data
            for key, value in data.items():
                await self.cache.put(
                    key, value, 
                    ttl_seconds=schedule['ttl_seconds'],
                    tags=schedule['tags']
                )
            
            self.logger.info(f"Cache warming completed for {strategy_name}", extra={
                'keys_warmed': len(data)
            })
            
            metrics.record_counter("cache_warming_executions_total", 1, tags={
                "strategy": strategy_name,
                "keys_count": len(data)
            })
            
        except Exception as e:
            self.logger.error(f"Cache warming failed for {strategy_name}: {e}")


# Global cache instances
_distributed_cache: Optional[DistributedCache] = None
_predictive_cache: Optional[PredictiveCache] = None
_cache_warmer: Optional[CacheWarmer] = None


def get_distributed_cache(redis: Redis) -> DistributedCache:
    """Get global distributed cache instance."""
    global _distributed_cache
    if _distributed_cache is None:
        _distributed_cache = DistributedCache(redis)
    return _distributed_cache


def get_predictive_cache(redis: Redis) -> PredictiveCache:
    """Get global predictive cache instance."""
    global _predictive_cache
    if _predictive_cache is None:
        cache = get_distributed_cache(redis)
        _predictive_cache = PredictiveCache(cache)
    return _predictive_cache


def get_cache_warmer(redis: Redis) -> CacheWarmer:
    """Get global cache warmer instance."""
    global _cache_warmer
    if _cache_warmer is None:
        cache = get_distributed_cache(redis)
        _cache_warmer = CacheWarmer(cache)
    return _cache_warmer


# Convenience decorators
def cached(key_func: Callable = None, ttl_seconds: int = 3600, tags: Set[str] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__] + [str(arg) for arg in args]
                key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            redis = Redis.from_url("redis://localhost:6379", decode_responses=True)
            cache = get_distributed_cache(redis)
            
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.put(cache_key, result, ttl_seconds=ttl_seconds, tags=tags)
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            # Sync wrapper would need different implementation
            return func
    
    return decorator