#!/usr/bin/env python3
"""
Intelligent caching system for AutoGen Code Review Bot.

This module provides advanced caching capabilities with automatic invalidation,
compression, and smart eviction policies.
"""

import hashlib
import json
import pickle
import threading
import time
import asyncio
import redis
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import concurrent.futures

from .logging_config import get_logger
from .robust_error_handling import ErrorSeverity, robust_operation, safe_execute
from .quantum_scale_optimizer import QuantumScaleOptimizer, OptimizationLevel
from .metrics import get_metrics_registry, record_operation_metrics
from .quantum_cache_support import (
    PredictiveLoader, CacheAnalytics, AnomalyDetector,
    QuantumCacheOptimizer, CacheCompressionEngine
)

logger = get_logger(__name__)


@dataclass
class CacheMetrics:
    """Advanced cache performance metrics"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    distributed_hit_rate: float = 0.0
    compression_ratio: float = 0.0
    eviction_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass 
class CachePattern:
    """Cache access pattern analysis"""
    pattern_id: str
    access_frequency: float
    time_locality: float
    size_characteristics: Dict[str, float]
    prediction_accuracy: float
    optimization_potential: float


@dataclass
class QuantumCacheState:
    """Quantum-inspired cache state"""
    coherence_score: float
    entanglement_level: float
    superposition_states: List[str]
    measurement_confidence: float
    quantum_advantage: float


class CacheEntry:
    """Enhanced cache entry with quantum-inspired optimization."""

    def __init__(self, data: Any, ttl_seconds: int = 3600, priority: float = 1.0):
        self.data = data
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
        self.access_count = 1
        self.last_accessed = self.created_at
        self.data_size = len(pickle.dumps(data))
        self.priority = priority
        self.access_pattern = deque(maxlen=100)
        self.compression_ratio = 1.0
        self.quantum_state = QuantumCacheState(
            coherence_score=1.0,
            entanglement_level=0.0,
            superposition_states=[],
            measurement_confidence=1.0,
            quantum_advantage=1.0
        )

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    def access(self) -> Any:
        """Record access and return data with quantum optimization."""
        self.access_count += 1
        self.last_accessed = time.time()
        self.access_pattern.append(time.time())
        
        # Update quantum state based on access patterns
        self._update_quantum_state()
        
        return self.data

    def get_age(self) -> float:
        """Get the age of the entry in seconds."""
        return time.time() - self.created_at
    
    def get_access_frequency(self) -> float:
        """Calculate recent access frequency."""
        if len(self.access_pattern) < 2:
            return 0.0
        
        recent_accesses = [t for t in self.access_pattern if time.time() - t < 3600]
        return len(recent_accesses) / 3600.0  # accesses per second
    
    def get_temporal_locality(self) -> float:
        """Calculate temporal locality score."""
        if len(self.access_pattern) < 3:
            return 1.0
        
        intervals = []
        for i in range(1, len(self.access_pattern)):
            intervals.append(self.access_pattern[i] - self.access_pattern[i-1])
        
        if not intervals:
            return 1.0
        
        # Lower variance in access intervals = higher temporal locality
        variance = np.var(intervals)
        return 1.0 / (1.0 + variance)
    
    def _update_quantum_state(self):
        """Update quantum state based on access patterns."""
        frequency = self.get_access_frequency()
        locality = self.get_temporal_locality()
        
        # Quantum coherence based on access predictability
        self.quantum_state.coherence_score = locality * 0.8 + (frequency / 10.0) * 0.2
        
        # Entanglement with other cache entries (simplified)
        self.quantum_state.entanglement_level = min(0.9, frequency * 0.1)
        
        # Quantum advantage based on optimization potential
        age_factor = max(0.1, 1.0 - (self.get_age() / self.ttl_seconds))
        self.quantum_state.quantum_advantage = (
            self.quantum_state.coherence_score * age_factor * 
            (1.0 + self.quantum_state.entanglement_level)
        )


class QuantumIntelligentCache:
    """Quantum-enhanced intelligent caching system with breakthrough optimization."""

    def __init__(
        self,
        cache_dir: str = ".cache/autogen-review",
        max_memory_mb: int = 500,
        default_ttl: int = 3600,
        cleanup_interval: int = 300,
        redis_url: Optional[str] = None,
        enable_quantum_optimization: bool = True,
        enable_distributed_cache: bool = True,
        enable_predictive_loading: bool = True,
    ):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_quantum_optimization = enable_quantum_optimization
        self.enable_distributed_cache = enable_distributed_cache
        self.enable_predictive_loading = enable_predictive_loading

        # Core components
        self.metrics = get_metrics_registry()
        self.quantum_optimizer = QuantumScaleOptimizer(OptimizationLevel.TRANSCENDENT) if enable_quantum_optimization else None

        # Multi-tier cache architecture
        self.memory_cache: Dict[str, CacheEntry] = {}  # L1 Cache
        self.ssd_cache: Dict[str, CacheEntry] = {}     # L2 Cache 
        self.distributed_cache = None                   # L3 Cache (Redis)
        self.cache_lock = threading.RLock()

        # Advanced features
        self.access_patterns: Dict[str, CachePattern] = {}
        self.predictive_loader = PredictiveLoader() if enable_predictive_loading else None
        self.cache_analytics = CacheAnalytics()
        self.quantum_optimizer = QuantumCacheOptimizer() if enable_quantum_optimization else None
        self.compression_engine = CacheCompressionEngine()
        
        # Performance tracking
        self.latency_tracker = deque(maxlen=10000)
        self.hit_rate_tracker = deque(maxlen=1000)
        self.quantum_metrics = defaultdict(list)

        # Enhanced statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_reads": 0,
            "disk_writes": 0,
            "cleanup_runs": 0,
            "distributed_hits": 0,
            "distributed_misses": 0,
            "quantum_optimizations": 0,
            "predictive_loads": 0,
            "compression_saves": 0,
        }

        # Initialize distributed cache
        if enable_distributed_cache and redis_url:
            try:
                self.distributed_cache = redis.from_url(redis_url, decode_responses=False)
                self.distributed_cache.ping()
                logger.info("Connected to distributed cache (Redis)")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.distributed_cache = None

        # Start background threads
        self.cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True
        )
        self.cleanup_thread.start()
        
        if enable_predictive_loading:
            self.prediction_thread = threading.Thread(
                target=self._predictive_loading_worker, daemon=True
            )
            self.prediction_thread.start()
        
        if enable_quantum_optimization:
            self.quantum_thread = threading.Thread(
                target=self._quantum_optimization_worker, daemon=True
            )
            self.quantum_thread.start()

        logger.info(
            f"Initialized quantum intelligent cache at {self.cache_dir} "
            f"(quantum: {enable_quantum_optimization}, distributed: {bool(self.distributed_cache)}, "
            f"predictive: {enable_predictive_loading})"
        )

    def get_cache_key(
        self,
        repo_path: str,
        analysis_type: str,
        config_hash: Optional[str] = None,
        repo_hash: Optional[str] = None,
    ) -> str:
        """Generate a unique cache key for analysis results."""
        key_components = [
            repo_path,
            analysis_type,
            config_hash or "default",
            repo_hash or self._get_repo_hash(repo_path),
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
                if file_path.is_file() and not any(
                    ignored in str(file_path)
                    for ignored in [
                        ".git",
                        "__pycache__",
                        ".pytest_cache",
                        "node_modules",
                    ]
                ):
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
        component="quantum_cache_system",
        operation="get_cached_result",
        severity=ErrorSeverity.LOW,
        retry_count=1,
        fallback_value=None,
    )
    @record_operation_metrics("quantum_cache_get")
    async def get(self, cache_key: str) -> Optional[Any]:
        """Retrieve cached data with quantum optimization."""
        start_time = time.time()
        
        try:
            with self.cache_lock:
                # L1 Cache: Check memory cache first
                result = await self._check_memory_cache(cache_key)
                if result is not None:
                    self._record_cache_hit("memory", time.time() - start_time)
                    return result

                # L2 Cache: Check SSD cache
                result = await self._check_ssd_cache(cache_key)
                if result is not None:
                    self._record_cache_hit("ssd", time.time() - start_time)
                    return result

                # L3 Cache: Check distributed cache
                if self.distributed_cache:
                    result = await self._check_distributed_cache(cache_key)
                    if result is not None:
                        self._record_cache_hit("distributed", time.time() - start_time)
                        return result

                # Cache miss - trigger predictive loading
                if self.predictive_loader:
                    await self._trigger_predictive_loading(cache_key)

                self._record_cache_miss(time.time() - start_time)
                return None

        except Exception as e:
            logger.error(f"Error in quantum cache get: {e}")
            self._record_cache_miss(time.time() - start_time)
            return None

    async def _check_memory_cache(self, cache_key: str) -> Optional[Any]:
        """Check L1 memory cache."""
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]

            if entry.is_expired():
                del self.memory_cache[cache_key]
                return None

            self.stats["hits"] += 1
            logger.debug(f"L1 cache hit for key {cache_key[:12]}...")
            
            # Record access for pattern learning
            if self.predictive_loader:
                self.predictive_loader.record_access(cache_key)
                
            return entry.access()
        
        return None

    async def _check_ssd_cache(self, cache_key: str) -> Optional[Any]:
        """Check L2 SSD cache."""

            # Check disk cache
            disk_file = self.cache_dir / f"{cache_key}.cache"
            if disk_file.exists():
                try:
                    with open(disk_file, "rb") as f:
                        entry_data = pickle.load(f)

                    entry = CacheEntry(
                        data=entry_data["data"],
                        ttl_seconds=entry_data.get("ttl", self.default_ttl),
                    )
                    entry.created_at = entry_data["created_at"]

                    if entry.is_expired():
                        disk_file.unlink()
                        self.stats["misses"] += 1
                        logger.debug(f"Disk cache expired for key {cache_key[:12]}...")
                        return None

                    # Move to memory cache
                    self._ensure_memory_space()
                    self.memory_cache[cache_key] = entry

                    self.stats["hits"] += 1
                    self.stats["disk_reads"] += 1
                    logger.debug(f"Disk cache hit for key {cache_key[:12]}...")
                    return entry.access()

                except Exception as e:
                    logger.warning(f"Failed to read disk cache {cache_key[:12]}: {e}")
                    safe_execute(
                        disk_file.unlink, component="cache_system", operation="cleanup"
                    )

            self.stats["misses"] += 1
            return None

    @robust_operation(
        component="cache_system",
        operation="store_cached_result",
        severity=ErrorSeverity.LOW,
        retry_count=1,
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
                daemon=True,
            ).start()

            logger.debug(
                f"Cached data for key {cache_key[:12]}... (size: {entry.data_size} bytes)"
            )

    def _write_to_disk(
        self, cache_key: str, data: Any, ttl: int, created_at: float
    ) -> None:
        """Write cache entry to disk."""
        try:
            disk_file = self.cache_dir / f"{cache_key}.cache"
            entry_data = {"data": data, "ttl": ttl, "created_at": created_at}

            with open(disk_file, "wb") as f:
                pickle.dump(entry_data, f)

            self.stats["disk_writes"] += 1

        except Exception as e:
            logger.warning(f"Failed to write disk cache {cache_key[:12]}: {e}")

    def _ensure_memory_space(self) -> None:
        """Ensure memory cache doesn't exceed limits."""
        current_size = sum(entry.data_size for entry in self.memory_cache.values())

        if current_size > self.max_memory_bytes:
            # Evict least recently used entries
            entries_by_access = sorted(
                self.memory_cache.items(), key=lambda x: x[1].last_accessed
            )

            evicted_count = 0
            for cache_key, entry in entries_by_access:
                del self.memory_cache[cache_key]
                current_size -= entry.data_size
                evicted_count += 1

                if current_size <= self.max_memory_bytes * 0.8:  # Leave 20% headroom
                    break

            self.stats["evictions"] += evicted_count
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
                    key
                    for key, entry in self.memory_cache.items()
                    if entry.is_expired()
                ]

                for key in expired_keys:
                    del self.memory_cache[key]

                logger.debug(
                    f"Cleaned {len(expired_keys)} expired memory cache entries"
                )

            # Clean disk cache
            disk_cleaned = 0
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        # Check if file is old enough to warrant checking
                        if (
                            time.time() - cache_file.stat().st_mtime
                        ) > self.default_ttl:
                            with open(cache_file, "rb") as f:
                                entry_data = pickle.load(f)

                            created_at = entry_data.get("created_at", 0)
                            ttl = entry_data.get("ttl", self.default_ttl)

                            if (time.time() - created_at) > ttl:
                                cache_file.unlink()
                                disk_cleaned += 1

                    except Exception as e:
                        logger.debug(f"Error checking cache file {cache_file}: {e}")
                        safe_execute(
                            cache_file.unlink,
                            component="cache_system",
                            operation="cleanup",
                        )

            self.stats["cleanup_runs"] += 1
            logger.debug(f"Cleaned {disk_cleaned} expired disk cache entries")

        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                (self.stats["hits"] / total_requests) if total_requests > 0 else 0
            )

            memory_size = sum(entry.data_size for entry in self.memory_cache.values())
            memory_entries = len(self.memory_cache)

            disk_entries = (
                len(list(self.cache_dir.glob("*.cache")))
                if self.cache_dir.exists()
                else 0
            )

            return {
                "hit_rate": round(hit_rate * 100, 2),
                "total_requests": total_requests,
                "memory_entries": memory_entries,
                "memory_size_mb": round(memory_size / (1024 * 1024), 2),
                "disk_entries": disk_entries,
                "stats": self.stats.copy(),
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.cache_lock:
            self.memory_cache.clear()

            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.cache"):
                    safe_execute(
                        cache_file.unlink, component="cache_system", operation="clear"
                    )

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
                        safe_execute(
                            cache_file.unlink,
                            component="cache_system",
                            operation="invalidate",
                        )
                        invalidated += 1

        logger.info(
            f"Invalidated {invalidated} cache entries matching pattern: {pattern}"
        )
        return invalidated


# Global cache instance
intelligent_cache = IntelligentCache()


def with_cache(analysis_type: str, ttl_seconds: int = 3600):
    """Decorator to add caching to analysis functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key based on function arguments
            repo_path = args[0] if args else kwargs.get("repo_path", "unknown")

            # Create a hash of all arguments for cache key
            arg_hash = hashlib.md5(
                json.dumps(
                    [str(arg) for arg in args]
                    + [f"{k}={v}" for k, v in kwargs.items()],
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            cache_key = intelligent_cache.get_cache_key(
                repo_path=repo_path,
                analysis_type=f"{analysis_type}_{func.__name__}",
                config_hash=arg_hash,
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
