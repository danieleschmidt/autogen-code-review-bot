"""
Quantum Task Planner Optimization and Scaling

High-performance optimizations including caching, parallel processing,
load balancing, and auto-scaling for quantum-inspired task planning.
"""

import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .quantum_planner import QuantumTask, QuantumTaskPlanner
from .quantum_validator import RobustQuantumPlanner, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for planning results."""
    data: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > ttl_seconds


class IntelligentCache:
    """Intelligent caching system with LRU eviction and performance metrics."""

    def __init__(self, max_size_mb: float = 100.0, ttl_hours: float = 24.0):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.ttl_seconds = ttl_hours * 3600
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.RLock()

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            return len(str(data)) * 2

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        while (self.current_size_bytes > self.max_size_bytes * 0.8 and
               len(self.access_order) > 1):
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                entry = self.cache.pop(lru_key)
                self.current_size_bytes -= entry.size_bytes
                self.evictions += 1
                logger.debug(f"Evicted cache entry: {lru_key}")

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired(self.ttl_seconds)
        ]

        for key in expired_keys:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes
            if key in self.access_order:
                self.access_order.remove(key)
            logger.debug(f"Expired cache entry: {key}")

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        with self._lock:
            self._cleanup_expired()

            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]
            if entry.is_expired(self.ttl_seconds):
                self.cache.pop(key)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.current_size_bytes -= entry.size_bytes
                self.misses += 1
                return None

            # Update access pattern
            entry.access_count += 1
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            self.hits += 1
            return entry.data

    def put(self, key: str, data: Any) -> None:
        """Cache a value."""
        with self._lock:
            size_bytes = self._calculate_size(data)

            # Don't cache if too large
            if size_bytes > self.max_size_bytes * 0.5:
                logger.warning(f"Data too large to cache: {size_bytes} bytes")
                return

            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size_bytes -= old_entry.size_bytes
                if key in self.access_order:
                    self.access_order.remove(key)

            # Create new entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                size_bytes=size_bytes
            )

            self.cache[key] = entry
            self.access_order.append(key)
            self.current_size_bytes += size_bytes

            # Evict if necessary
            self._evict_lru()

    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self._lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_order.clear()
                self.current_size_bytes = 0
                return count

            # Pattern matching
            keys_to_remove = [
                key for key in self.cache.keys()
                if pattern in key
            ]

            for key in keys_to_remove:
                entry = self.cache.pop(key)
                self.current_size_bytes -= entry.size_bytes
                if key in self.access_order:
                    self.access_order.remove(key)

            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'entries': len(self.cache),
                'size_mb': self.current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'ttl_hours': self.ttl_seconds / 3600
            }


class ParallelQuantumProcessor:
    """Parallel processing engine for quantum task operations."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=min(4, multiprocessing.cpu_count() or 1))

    def parallel_task_validation(self, tasks: List[QuantumTask]) -> List[ValidationResult]:
        """Validate tasks in parallel."""
        from .quantum_validator import TaskValidator

        def validate_single_task(task: QuantumTask) -> ValidationResult:
            validator = TaskValidator()
            return validator.validate_task(task)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(validate_single_task, task): task
                for task in tasks
            }

            results = []
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task validation failed: {e}")
                    # Create error result
                    error_result = ValidationResult(False, [str(e)], [])
                    results.append(error_result)

        return results

    def parallel_interference_calculation(self, tasks: List[QuantumTask]) -> Dict[Tuple[str, str], float]:
        """Calculate quantum interference coupling in parallel."""
        def calculate_coupling_batch(task_pairs: List[Tuple[QuantumTask, QuantumTask]]) -> Dict[Tuple[str, str], float]:
            """Calculate coupling for a batch of task pairs."""
            results = {}
            for task1, task2 in task_pairs:
                coupling = self._calculate_coupling_strength(task1, task2)
                results[(task1.id, task2.id)] = coupling
            return results

        # Create task pairs
        task_pairs = []
        for i, task1 in enumerate(tasks):
            for task2 in tasks[i+1:]:
                task_pairs.append((task1, task2))

        # Batch pairs for parallel processing
        batch_size = max(1, len(task_pairs) // self.max_workers)
        batches = [
            task_pairs[i:i + batch_size]
            for i in range(0, len(task_pairs), batch_size)
        ]

        # Process batches in parallel
        coupling_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(calculate_coupling_batch, batch): batch
                for batch in batches
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    coupling_results.update(batch_results)
                except Exception as e:
                    logger.error(f"Parallel coupling calculation failed: {e}")

        return coupling_results

    def _calculate_coupling_strength(self, task1: QuantumTask, task2: QuantumTask) -> float:
        """Calculate quantum coupling strength between tasks."""
        coupling = 0.0

        # Dependency coupling
        if task2.id in task1.dependencies or task1.id in task2.dependencies:
            coupling += 0.3

        # Entanglement coupling
        if task2.id in task1.entangled_tasks:
            coupling += 0.5

        # Similarity coupling (based on effort similarity)
        effort_diff = abs(task1.estimated_effort - task2.estimated_effort)
        similarity_coupling = max(0, 0.2 - effort_diff * 0.1)
        coupling += similarity_coupling

        return min(coupling, 1.0)

    def async_plan_generation(self, planner: QuantumTaskPlanner) -> concurrent.futures.Future:
        """Generate execution plan asynchronously."""
        return self.thread_pool.submit(planner.generate_execution_plan)

    def shutdown(self) -> None:
        """Shutdown parallel processing pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class LoadBalancer:
    """Load balancer for distributing quantum planning workload."""

    def __init__(self):
        self.planners: List[QuantumTaskPlanner] = []
        self.workload_distribution: Dict[int, int] = defaultdict(int)
        self.response_times: Dict[int, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_planner(self, planner: QuantumTaskPlanner) -> int:
        """Add a planner instance to the pool."""
        with self._lock:
            planner_id = len(self.planners)
            self.planners.append(planner)
            logger.info(f"Added planner {planner_id} to load balancer")
            return planner_id

    def get_optimal_planner(self) -> Tuple[int, QuantumTaskPlanner]:
        """Get the optimal planner based on current load."""
        with self._lock:
            if not self.planners:
                raise RuntimeError("No planners available")

            # Calculate load scores
            scores = []
            for i, planner in enumerate(self.planners):
                current_load = self.workload_distribution[i]
                avg_response_time = (
                    sum(self.response_times[i][-10:]) / len(self.response_times[i][-10:])
                    if self.response_times[i] else 0.1
                )

                # Lower score is better
                score = current_load * 0.7 + avg_response_time * 0.3
                scores.append((score, i))

            # Select planner with lowest score
            best_score, planner_id = min(scores)
            self.workload_distribution[planner_id] += 1

            return planner_id, self.planners[planner_id]

    def record_completion(self, planner_id: int, response_time: float) -> None:
        """Record completion of a planning task."""
        with self._lock:
            self.workload_distribution[planner_id] = max(0, self.workload_distribution[planner_id] - 1)
            self.response_times[planner_id].append(response_time)

            # Keep only recent response times
            if len(self.response_times[planner_id]) > 100:
                self.response_times[planner_id] = self.response_times[planner_id][-100:]

    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self._lock:
            total_load = sum(self.workload_distribution.values())
            avg_response_times = {
                planner_id: (
                    sum(times[-10:]) / len(times[-10:])
                    if times else 0
                )
                for planner_id, times in self.response_times.items()
            }

            return {
                'total_planners': len(self.planners),
                'total_current_load': total_load,
                'load_distribution': dict(self.workload_distribution),
                'average_response_times': avg_response_times,
                'requests_processed': sum(len(times) for times in self.response_times.values())
            }


class AutoScaler:
    """Auto-scaling system for quantum planners based on load."""

    def __init__(self, load_balancer: LoadBalancer,
                 min_planners: int = 1, max_planners: int = 8,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3):
        self.load_balancer = load_balancer
        self.min_planners = min_planners
        self.max_planners = max_planners
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_history: List[Dict] = []
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self, check_interval: float = 30.0) -> None:
        """Start auto-scaling monitoring."""
        if self._monitoring:
            return

        self._monitoring = True

        def monitor_loop():
            while self._monitoring:
                try:
                    self._check_and_scale()
                    time.sleep(check_interval)
                except Exception as e:
                    logger.error(f"Auto-scaling error: {e}")

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started auto-scaling monitor")

    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _check_and_scale(self) -> None:
        """Check load and scale if necessary."""
        load_stats = self.load_balancer.get_load_stats()
        total_planners = load_stats['total_planners']
        total_load = load_stats['total_current_load']

        if total_planners == 0:
            return

        # Calculate average load per planner
        avg_load = total_load / total_planners

        # Calculate average response time
        response_times = list(load_stats['average_response_times'].values())
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        should_scale_up = (
            avg_load > self.scale_up_threshold and
            total_planners < self.max_planners and
            avg_response_time > 0.5
        )

        should_scale_down = (
            avg_load < self.scale_down_threshold and
            total_planners > self.min_planners and
            avg_response_time < 0.2
        )

        if should_scale_up:
            self._scale_up()
        elif should_scale_down:
            self._scale_down()

    def _scale_up(self) -> None:
        """Add a new planner instance."""
        try:
            new_planner = RobustQuantumPlanner()
            planner_id = self.load_balancer.add_planner(new_planner)

            self.scaling_history.append({
                'timestamp': time.time(),
                'action': 'scale_up',
                'planner_id': planner_id,
                'total_planners': len(self.load_balancer.planners)
            })

            logger.info(f"Scaled up: added planner {planner_id}")
        except Exception as e:
            logger.error(f"Scale up failed: {e}")

    def _scale_down(self) -> None:
        """Remove a planner instance (if load allows)."""
        load_stats = self.load_balancer.get_load_stats()
        if load_stats['total_planners'] <= self.min_planners:
            return

        # Find planner with lowest load
        load_distribution = load_stats['load_distribution']
        min_load_planner = min(load_distribution.items(), key=lambda x: x[1])
        planner_id, current_load = min_load_planner

        # Only scale down if planner has no current load
        if current_load == 0:
            try:
                # Note: In a real implementation, you'd want to properly remove
                # the planner from the load balancer
                self.scaling_history.append({
                    'timestamp': time.time(),
                    'action': 'scale_down',
                    'planner_id': planner_id,
                    'total_planners': len(self.load_balancer.planners) - 1
                })

                logger.info(f"Would scale down planner {planner_id} (simulated)")
            except Exception as e:
                logger.error(f"Scale down failed: {e}")

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        scale_ups = sum(1 for event in self.scaling_history if event['action'] == 'scale_up')
        scale_downs = sum(1 for event in self.scaling_history if event['action'] == 'scale_down')

        return {
            'is_monitoring': self._monitoring,
            'min_planners': self.min_planners,
            'max_planners': self.max_planners,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'total_scaling_events': len(self.scaling_history),
            'scale_ups': scale_ups,
            'scale_downs': scale_downs,
            'recent_events': self.scaling_history[-10:]
        }


class OptimizedQuantumPlanner(RobustQuantumPlanner):
    """Optimized quantum planner with caching, parallel processing, and auto-scaling."""

    def __init__(self, cache_size_mb: float = 100.0, max_workers: Optional[int] = None):
        super().__init__()
        self.cache = IntelligentCache(max_size_mb=cache_size_mb)
        self.parallel_processor = ParallelQuantumProcessor(max_workers=max_workers)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self._optimization_enabled = True

    def _generate_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation."""
        # Create deterministic key from operation and parameters
        key_data = {'op': operation, **kwargs}
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def create_task(self, task_id: str, title: str, description: str,
                   estimated_effort: float = 1.0, dependencies: Optional[List[str]] = None) -> QuantumTask:
        """Create task with cache invalidation."""
        task = super().create_task(task_id, title, description, estimated_effort, dependencies)

        # Invalidate relevant cache entries
        self.cache.invalidate('plan_')
        self.cache.invalidate('validation_')

        return task

    def generate_execution_plan(self) -> Dict:
        """Generate execution plan with intelligent caching."""
        start_time = time.time()

        # Generate cache key based on current system state
        system_hash = self.integrity_checker.calculate_system_hash(self.scheduler)
        cache_key = self._generate_cache_key('execution_plan', system_hash=system_hash)

        # Check cache first
        if self._optimization_enabled:
            cached_plan = self.cache.get(cache_key)
            if cached_plan:
                logger.info(f"Retrieved execution plan from cache: {cache_key[:8]}")
                self.performance_metrics['cache_hits'].append(time.time() - start_time)
                return cached_plan

        # Generate new plan
        try:
            # Use parallel processing for validation if many tasks
            if len(self.scheduler.tasks) > 10:
                tasks = list(self.scheduler.tasks.values())
                validation_results = self.parallel_processor.parallel_task_validation(tasks)

                # Check if any validations failed
                failed_validations = [r for r in validation_results if not r.is_valid]
                if failed_validations:
                    errors = []
                    for result in failed_validations:
                        errors.extend(result.errors)
                    raise ValidationError(f"Parallel validation failed: {', '.join(errors)}")

            plan = super().generate_execution_plan()

            # Cache the result
            if self._optimization_enabled:
                self.cache.put(cache_key, plan)

            generation_time = time.time() - start_time
            self.performance_metrics['plan_generation'].append(generation_time)

            logger.info(f"Generated optimized execution plan in {generation_time:.3f}s")
            return plan

        except Exception:
            self.performance_metrics['plan_errors'].append(time.time() - start_time)
            raise

    def bulk_create_tasks(self, task_definitions: List[Dict]) -> List[QuantumTask]:
        """Create multiple tasks efficiently."""
        start_time = time.time()

        created_tasks = []
        try:
            for task_def in task_definitions:
                task = self.create_task(
                    task_id=task_def['id'],
                    title=task_def['title'],
                    description=task_def.get('description', ''),
                    estimated_effort=task_def.get('estimated_effort', 1.0),
                    dependencies=task_def.get('dependencies', [])
                )
                created_tasks.append(task)

            # Apply optimizations after bulk creation
            if self._optimization_enabled and len(created_tasks) > 5:
                self._optimize_bulk_operations(created_tasks)

            bulk_time = time.time() - start_time
            self.performance_metrics['bulk_operations'].append(bulk_time)

            logger.info(f"Bulk created {len(created_tasks)} tasks in {bulk_time:.3f}s")
            return created_tasks

        except Exception as e:
            logger.error(f"Bulk task creation failed: {e}")
            # Clean up partially created tasks
            for task in created_tasks:
                if task.id in self.scheduler.tasks:
                    del self.scheduler.tasks[task.id]
            raise

    def _optimize_bulk_operations(self, tasks: List[QuantumTask]) -> None:
        """Apply optimizations after bulk operations."""
        # Parallel interference calculation for large task sets
        if len(tasks) > 10:
            coupling_results = self.parallel_processor.parallel_interference_calculation(tasks)

            # Apply coupling optimizations
            for (task1_id, task2_id), coupling in coupling_results.items():
                if coupling > 0.7:  # High coupling threshold
                    # Consider creating entanglement for highly coupled tasks
                    try:
                        self.create_task_entanglement(task1_id, task2_id)
                        logger.debug(f"Auto-entangled high-coupling tasks: {task1_id} <-> {task2_id}")
                    except Exception as e:
                        logger.warning(f"Auto-entanglement failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = self.cache.get_stats()

        metrics = {
            'cache_performance': cache_stats,
            'operation_times': {
                metric: {
                    'count': len(times),
                    'avg_seconds': sum(times) / len(times) if times else 0,
                    'min_seconds': min(times) if times else 0,
                    'max_seconds': max(times) if times else 0
                }
                for metric, times in self.performance_metrics.items()
            },
            'system_health': self.get_system_health(),
            'optimization_enabled': self._optimization_enabled,
            'parallel_workers': self.parallel_processor.max_workers
        }

        return metrics

    def optimize_system(self) -> Dict[str, Any]:
        """Run comprehensive system optimization."""
        start_time = time.time()
        optimizations_applied = []

        # Clean expired cache entries
        expired_count = self.cache.invalidate()
        if expired_count > 0:
            optimizations_applied.append(f"Cleaned {expired_count} expired cache entries")

        # Auto-repair system issues
        repair_result = self.auto_repair_system()
        if repair_result['repairs_performed'] > 0:
            optimizations_applied.extend(repair_result['repair_log'])

        # Optimize quantum interference patterns
        tasks = list(self.scheduler.tasks.values())
        if len(tasks) > 5:
            self._optimize_bulk_operations(tasks)
            optimizations_applied.append("Optimized quantum interference patterns")

        optimization_time = time.time() - start_time

        return {
            'optimization_time_seconds': optimization_time,
            'optimizations_applied': optimizations_applied,
            'performance_metrics': self.get_performance_metrics()
        }

    def enable_optimization(self, enable: bool = True) -> None:
        """Enable or disable performance optimizations."""
        self._optimization_enabled = enable
        logger.info(f"Performance optimizations {'enabled' if enable else 'disabled'}")

    def shutdown(self) -> None:
        """Shutdown optimization resources."""
        self.parallel_processor.shutdown()
        logger.info("Optimized quantum planner shutdown complete")
