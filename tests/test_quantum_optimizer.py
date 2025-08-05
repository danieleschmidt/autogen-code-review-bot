"""
Comprehensive tests for quantum planner optimization and scaling.
"""

import pytest
import time
import threading
import concurrent.futures
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from src.autogen_code_review_bot.quantum_planner import (
    QuantumTask, TaskPriority, QuantumTaskPlanner
)
from src.autogen_code_review_bot.quantum_validator import RobustQuantumPlanner, ValidationResult
from src.autogen_code_review_bot.quantum_optimizer import (
    CacheEntry, IntelligentCache, ParallelQuantumProcessor,
    LoadBalancer, AutoScaler, OptimizedQuantumPlanner
)


class TestCacheEntry:
    """Test CacheEntry class."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        data = {"test": "data"}
        timestamp = time.time()
        entry = CacheEntry(data=data, timestamp=timestamp, size_bytes=100)
        
        assert entry.data == data
        assert entry.timestamp == timestamp
        assert entry.access_count == 0
        assert entry.size_bytes == 100
    
    def test_is_expired(self):
        """Test expiration checking."""
        old_timestamp = time.time() - 3600  # 1 hour ago
        recent_timestamp = time.time() - 60  # 1 minute ago
        
        old_entry = CacheEntry(data={}, timestamp=old_timestamp)
        recent_entry = CacheEntry(data={}, timestamp=recent_timestamp)
        
        # Test with 30 minute TTL
        ttl_seconds = 30 * 60
        
        assert old_entry.is_expired(ttl_seconds) is True
        assert recent_entry.is_expired(ttl_seconds) is False


class TestIntelligentCache:
    """Test IntelligentCache class."""
    
    def test_cache_creation(self):
        """Test cache initialization."""
        cache = IntelligentCache(max_size_mb=50.0, ttl_hours=12.0)
        
        assert cache.max_size_bytes == 50 * 1024 * 1024
        assert cache.ttl_seconds == 12 * 3600
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        cache = IntelligentCache()
        
        test_data = {"key": "value", "number": 42}
        cache.put("test_key", test_data)
        
        retrieved_data = cache.get("test_key")
        assert retrieved_data == test_data
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = IntelligentCache()
        
        result = cache.get("nonexistent_key")
        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    def test_cache_update_access_order(self):
        """Test LRU access order updating."""
        cache = IntelligentCache()
        
        cache.put("key1", "data1")
        cache.put("key2", "data2")
        
        assert cache.access_order == ["key1", "key2"]
        
        # Access key1 again
        cache.get("key1")
        assert cache.access_order == ["key2", "key1"]  # key1 moved to end
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        # Create small cache
        cache = IntelligentCache(max_size_mb=0.001)  # Very small cache
        
        # Fill cache beyond capacity
        large_data = "x" * 1000  # 1KB data
        for i in range(10):
            cache.put(f"key{i}", large_data)
        
        # Some entries should have been evicted
        assert len(cache.cache) < 10
        assert cache.evictions > 0
    
    def test_cache_too_large_item(self):
        """Test handling of items too large to cache."""
        cache = IntelligentCache(max_size_mb=0.001)  # Very small cache
        
        huge_data = "x" * (1024 * 1024)  # 1MB data
        cache.put("huge_key", huge_data)
        
        # Should not be cached
        assert "huge_key" not in cache.cache
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        cache = IntelligentCache(ttl_hours=0.001)  # Very short TTL (3.6 seconds)
        
        cache.put("test_key", "test_data")
        
        # Should be accessible immediately
        assert cache.get("test_key") == "test_data"
        
        # Wait for expiration
        time.sleep(4)
        
        # Should be expired
        assert cache.get("test_key") is None
        assert cache.misses > 0
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = IntelligentCache()
        
        cache.put("key1", "data1")
        cache.put("key2", "data2")
        cache.put("other_key", "other_data")
        
        # Invalidate by pattern
        invalidated = cache.invalidate("key")
        assert invalidated == 2  # key1 and key2
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("other_key") == "other_data"
        
        # Invalidate all
        cache.put("new_key", "new_data")
        total_invalidated = cache.invalidate()
        assert total_invalidated == 2  # other_key and new_key
        assert len(cache.cache) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = IntelligentCache(max_size_mb=10.0, ttl_hours=24.0)
        
        cache.put("key1", "data1")
        cache.put("key2", "data2")
        
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        
        assert stats['entries'] == 2
        assert stats['max_size_mb'] == 10.0
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['ttl_hours'] == 24.0
    
    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = IntelligentCache()
        
        def worker(thread_id):
            for i in range(100):
                cache.put(f"thread{thread_id}_key{i}", f"data{i}")
                cache.get(f"thread{thread_id}_key{i}")
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have high hit rate and no crashes
        stats = cache.get_stats()
        assert stats['hits'] >= 400  # Most gets should be hits


class TestParallelQuantumProcessor:
    """Test ParallelQuantumProcessor class."""
    
    def test_processor_creation(self):
        """Test processor initialization."""
        processor = ParallelQuantumProcessor(max_workers=4)
        
        assert processor.max_workers == 4
        assert hasattr(processor, 'thread_pool')
        assert hasattr(processor, 'process_pool')
    
    def test_parallel_task_validation(self):
        """Test parallel task validation."""
        processor = ParallelQuantumProcessor(max_workers=2)
        
        # Create test tasks
        tasks = []
        for i in range(5):
            task = QuantumTask(id=f"task{i}", title=f"Task {i}", description="Test task")
            tasks.append(task)
        
        # Add one invalid task
        invalid_task = QuantumTask(id="", title="Invalid", description="")  # Empty ID
        tasks.append(invalid_task)
        
        # Run parallel validation
        results = processor.parallel_task_validation(tasks)
        
        assert len(results) == 6
        assert all(isinstance(result, ValidationResult) for result in results)
        
        # Should have some valid and some invalid results
        valid_results = [r for r in results if r.is_valid]
        invalid_results = [r for r in results if not r.is_valid]
        
        assert len(valid_results) == 5
        assert len(invalid_results) == 1
    
    def test_parallel_interference_calculation(self):
        """Test parallel interference calculation."""
        processor = ParallelQuantumProcessor(max_workers=2)
        
        # Create test tasks
        tasks = []
        for i in range(4):
            task = QuantumTask(id=f"task{i}", title=f"Task {i}", description="", estimated_effort=float(i+1))
            tasks.append(task)
        
        # Add some dependencies and entanglements
        tasks[1].add_dependency("task0")
        tasks[0].entangle_with("task2")
        
        coupling_results = processor.parallel_interference_calculation(tasks)
        
        # Should have coupling values for all pairs
        expected_pairs = 6  # 4 choose 2
        assert len(coupling_results) == expected_pairs
        
        # All coupling values should be between 0 and 1
        for coupling in coupling_results.values():
            assert 0 <= coupling <= 1
        
        # Task with dependency should have higher coupling
        dep_coupling = coupling_results.get(("task1", "task0"))
        if dep_coupling is None:
            dep_coupling = coupling_results.get(("task0", "task1"))
        
        assert dep_coupling is not None
        assert dep_coupling >= 0.3  # Dependency coupling
    
    def test_async_plan_generation(self):
        """Test asynchronous plan generation."""
        processor = ParallelQuantumProcessor(max_workers=2)
        
        # Create a simple planner
        planner = QuantumTaskPlanner()
        planner.create_task("task1", "Task 1", "Test task")
        
        # Submit async plan generation
        future = processor.async_plan_generation(planner)
        
        # Should return a Future
        assert isinstance(future, concurrent.futures.Future)
        
        # Get result
        plan = future.result(timeout=5.0)
        assert 'plan_id' in plan
        assert plan['total_tasks'] == 1
    
    def test_processor_shutdown(self):
        """Test processor shutdown."""
        processor = ParallelQuantumProcessor(max_workers=2)
        
        # Shutdown should not raise exceptions
        processor.shutdown()


class TestLoadBalancer:
    """Test LoadBalancer class."""
    
    def test_load_balancer_creation(self):
        """Test load balancer initialization."""
        balancer = LoadBalancer()
        
        assert len(balancer.planners) == 0
        assert len(balancer.workload_distribution) == 0
        assert len(balancer.response_times) == 0
    
    def test_add_planner(self):
        """Test adding planners to load balancer."""
        balancer = LoadBalancer()
        planner1 = QuantumTaskPlanner()
        planner2 = QuantumTaskPlanner()
        
        id1 = balancer.add_planner(planner1)
        id2 = balancer.add_planner(planner2)
        
        assert id1 == 0
        assert id2 == 1
        assert len(balancer.planners) == 2
        assert balancer.planners[0] == planner1
        assert balancer.planners[1] == planner2
    
    def test_get_optimal_planner_single(self):
        """Test getting optimal planner with single planner."""
        balancer = LoadBalancer()
        planner = QuantumTaskPlanner()
        
        planner_id = balancer.add_planner(planner)
        
        selected_id, selected_planner = balancer.get_optimal_planner()
        
        assert selected_id == planner_id
        assert selected_planner == planner
        assert balancer.workload_distribution[selected_id] == 1
    
    def test_get_optimal_planner_multiple(self):
        """Test optimal planner selection with multiple planners."""
        balancer = LoadBalancer()
        
        planners = []
        for i in range(3):
            planner = QuantumTaskPlanner()
            balancer.add_planner(planner)
            planners.append(planner)
        
        # Simulate different loads
        balancer.workload_distribution[0] = 5  # High load
        balancer.workload_distribution[1] = 2  # Medium load
        balancer.workload_distribution[2] = 0  # No load
        
        # Should select planner with lowest load (planner 2)
        selected_id, selected_planner = balancer.get_optimal_planner()
        assert selected_id == 2
        assert balancer.workload_distribution[2] == 1  # Load increased
    
    def test_get_optimal_planner_no_planners(self):
        """Test getting optimal planner with no planners."""
        balancer = LoadBalancer()
        
        with pytest.raises(RuntimeError) as exc_info:
            balancer.get_optimal_planner()
        
        assert "No planners available" in str(exc_info.value)
    
    def test_record_completion(self):
        """Test recording task completion."""
        balancer = LoadBalancer()
        planner = QuantumTaskPlanner()
        
        planner_id = balancer.add_planner(planner)
        
        # Get planner (increases load)
        balancer.get_optimal_planner()
        assert balancer.workload_distribution[planner_id] == 1
        
        # Record completion
        balancer.record_completion(planner_id, 0.5)
        
        assert balancer.workload_distribution[planner_id] == 0
        assert len(balancer.response_times[planner_id]) == 1
        assert balancer.response_times[planner_id][0] == 0.5
    
    def test_get_load_stats(self):
        """Test load balancing statistics."""
        balancer = LoadBalancer()
        
        # Add planners
        for i in range(2):
            balancer.add_planner(QuantumTaskPlanner())
        
        # Simulate some work
        balancer.get_optimal_planner()
        balancer.record_completion(0, 0.3)
        balancer.record_completion(0, 0.4)
        
        stats = balancer.get_load_stats()
        
        assert stats['total_planners'] == 2
        assert stats['total_current_load'] == 0  # All completed
        assert stats['requests_processed'] == 2
        assert 0 in stats['average_response_times']
        assert stats['average_response_times'][0] == 0.35  # Average of 0.3 and 0.4


class TestAutoScaler:
    """Test AutoScaler class."""
    
    def test_auto_scaler_creation(self):
        """Test auto scaler initialization."""
        balancer = LoadBalancer()
        scaler = AutoScaler(balancer, min_planners=1, max_planners=4)
        
        assert scaler.load_balancer == balancer
        assert scaler.min_planners == 1
        assert scaler.max_planners == 4
        assert scaler.scale_up_threshold == 0.8
        assert scaler.scale_down_threshold == 0.3
        assert len(scaler.scaling_history) == 0
        assert scaler._monitoring is False
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        balancer = LoadBalancer()
        scaler = AutoScaler(balancer)
        
        # Start monitoring
        scaler.start_monitoring(check_interval=0.1)
        assert scaler._monitoring is True
        assert scaler._monitor_thread is not None
        
        # Stop monitoring
        scaler.stop_monitoring()
        assert scaler._monitoring is False
    
    def test_scale_up_conditions(self):
        """Test scale up conditions."""
        balancer = LoadBalancer()
        scaler = AutoScaler(balancer, min_planners=1, max_planners=4)
        
        # Add initial planner
        balancer.add_planner(RobustQuantumPlanner())
        
        # Simulate high load
        balancer.workload_distribution[0] = 5  # High load
        balancer.response_times[0] = [1.0, 1.2, 0.8]  # High response times
        
        # Mock the scale up method to track calls
        original_scale_up = scaler._scale_up
        scale_up_called = False
        
        def mock_scale_up():
            nonlocal scale_up_called
            scale_up_called = True
            original_scale_up()
        
        scaler._scale_up = mock_scale_up
        
        # Check and scale
        scaler._check_and_scale()
        
        # Should have scaled up
        assert scale_up_called
        assert len(scaler.scaling_history) > 0
        assert scaler.scaling_history[-1]['action'] == 'scale_up'
    
    def test_scale_down_conditions(self):
        """Test scale down conditions."""
        balancer = LoadBalancer()
        scaler = AutoScaler(balancer, min_planners=1, max_planners=4)
        
        # Add multiple planners
        for i in range(3):
            balancer.add_planner(RobustQuantumPlanner())
        
        # Simulate low load
        for i in range(3):
            balancer.workload_distribution[i] = 0  # No load
            balancer.response_times[i] = [0.1, 0.05, 0.08]  # Low response times
        
        # Mock the scale down method
        original_scale_down = scaler._scale_down
        scale_down_called = False
        
        def mock_scale_down():
            nonlocal scale_down_called
            scale_down_called = True
            original_scale_down()
        
        scaler._scale_down = mock_scale_down
        
        # Check and scale
        scaler._check_and_scale()
        
        # Should have scaled down
        assert scale_down_called
        assert len(scaler.scaling_history) > 0
        assert scaler.scaling_history[-1]['action'] == 'scale_down'
    
    def test_get_scaling_stats(self):
        """Test scaling statistics."""
        balancer = LoadBalancer()
        scaler = AutoScaler(balancer)
        
        # Add some scaling history
        scaler.scaling_history = [
            {'action': 'scale_up', 'timestamp': time.time()},
            {'action': 'scale_up', 'timestamp': time.time()},
            {'action': 'scale_down', 'timestamp': time.time()},
        ]
        
        stats = scaler.get_scaling_stats()
        
        assert stats['is_monitoring'] is False
        assert stats['total_scaling_events'] == 3
        assert stats['scale_ups'] == 2
        assert stats['scale_downs'] == 1
        assert len(stats['recent_events']) == 3


class TestOptimizedQuantumPlanner:
    """Test OptimizedQuantumPlanner class."""
    
    def test_optimized_planner_creation(self):
        """Test optimized planner initialization."""
        planner = OptimizedQuantumPlanner(cache_size_mb=50.0, max_workers=4)
        
        assert hasattr(planner, 'cache')
        assert hasattr(planner, 'parallel_processor')
        assert hasattr(planner, 'performance_metrics')
        assert planner._optimization_enabled is True
        assert planner.parallel_processor.max_workers == 4
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        planner = OptimizedQuantumPlanner()
        
        key1 = planner._generate_cache_key("test_op", param1="value1", param2="value2")
        key2 = planner._generate_cache_key("test_op", param1="value1", param2="value2")
        key3 = planner._generate_cache_key("test_op", param1="different", param2="value2")
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        assert key1 != key3
        
        # Keys should be valid MD5 hashes
        assert len(key1) == 32
        assert all(c in '0123456789abcdef' for c in key1)
    
    def test_create_task_with_cache_invalidation(self):
        """Test task creation with cache invalidation."""
        planner = OptimizedQuantumPlanner()
        
        # Put some data in cache
        planner.cache.put("plan_test", {"test": "data"})
        planner.cache.put("validation_test", {"validation": "data"})
        planner.cache.put("other_test", {"other": "data"})
        
        # Create task (should invalidate plan_ and validation_ entries)
        task = planner.create_task("task1", "Task 1", "Test task")
        
        assert task.id == "task1"
        assert planner.cache.get("plan_test") is None
        assert planner.cache.get("validation_test") is None
        assert planner.cache.get("other_test") == {"other": "data"}  # Should remain
    
    def test_generate_execution_plan_with_caching(self):
        """Test execution plan generation with caching."""
        planner = OptimizedQuantumPlanner()
        
        # Create task
        planner.create_task("task1", "Task 1", "Test task")
        
        # Generate plan first time
        plan1 = planner.generate_execution_plan()
        assert 'plan_id' in plan1
        assert len(planner.performance_metrics['plan_generation']) == 1
        
        # Generate plan second time (should be cached)
        plan2 = planner.generate_execution_plan()
        assert plan1['plan_id'] == plan2['plan_id']  # Same plan
        assert len(planner.performance_metrics['cache_hits']) == 1
    
    def test_generate_execution_plan_without_optimization(self):
        """Test execution plan generation with optimization disabled."""
        planner = OptimizedQuantumPlanner()
        planner.enable_optimization(False)
        
        # Create task
        planner.create_task("task1", "Task 1", "Test task")
        
        # Generate plan twice
        plan1 = planner.generate_execution_plan()
        plan2 = planner.generate_execution_plan()
        
        # Should not use cache
        assert len(planner.performance_metrics['cache_hits']) == 0
        assert len(planner.performance_metrics['plan_generation']) == 2
    
    def test_bulk_create_tasks(self):
        """Test bulk task creation."""
        planner = OptimizedQuantumPlanner()
        
        task_definitions = [
            {
                'id': 'task1',
                'title': 'Task 1',
                'description': 'First task',
                'estimated_effort': 2.0,
                'dependencies': []
            },
            {
                'id': 'task2',
                'title': 'Task 2',
                'description': 'Second task',
                'estimated_effort': 3.0,
                'dependencies': ['task1']
            },
            {
                'id': 'task3',
                'title': 'Task 3',
                'description': 'Third task',
                'estimated_effort': 1.5,
                'dependencies': []
            }
        ]
        
        created_tasks = planner.bulk_create_tasks(task_definitions)
        
        assert len(created_tasks) == 3
        assert all(task.id in ['task1', 'task2', 'task3'] for task in created_tasks)
        assert 'task1' in planner.scheduler.tasks['task2'].dependencies
        assert len(planner.performance_metrics['bulk_operations']) == 1
    
    def test_bulk_create_tasks_with_optimization(self):
        """Test bulk task creation with optimization."""
        planner = OptimizedQuantumPlanner()
        
        # Create many tasks to trigger optimization
        task_definitions = []
        for i in range(12):  # More than optimization threshold
            task_definitions.append({
                'id': f'task{i}',
                'title': f'Task {i}',
                'description': f'Task number {i}',
                'estimated_effort': 1.0 + (i % 3),
            })
        
        created_tasks = planner.bulk_create_tasks(task_definitions)
        
        assert len(created_tasks) == 12
        # Some tasks might be auto-entangled due to high coupling
        total_entanglements = sum(len(task.entangled_tasks) for task in created_tasks)
        # Entanglements might be created, but not guaranteed
    
    def test_bulk_create_tasks_partial_failure(self):
        """Test bulk task creation with partial failure."""
        planner = OptimizedQuantumPlanner()
        
        # Create first task
        planner.create_task("existing", "Existing Task", "Already exists")
        
        task_definitions = [
            {
                'id': 'new_task',
                'title': 'New Task',
                'description': 'This should succeed',
                'estimated_effort': 1.0,
            },
            {
                'id': 'existing',  # This should fail due to duplicate ID
                'title': 'Duplicate Task',
                'description': 'This should fail',
                'estimated_effort': 2.0,
            }
        ]
        
        with pytest.raises(Exception):  # Should raise ValidationError
            planner.bulk_create_tasks(task_definitions)
        
        # Should clean up partially created tasks
        assert 'new_task' not in planner.scheduler.tasks
    
    def test_get_performance_metrics(self):
        """Test performance metrics collection."""
        planner = OptimizedQuantumPlanner()
        
        # Create some tasks and generate plans
        planner.create_task("task1", "Task 1", "Test task")
        planner.generate_execution_plan()
        planner.generate_execution_plan()  # Second one should be cached
        
        metrics = planner.get_performance_metrics()
        
        assert 'cache_performance' in metrics
        assert 'operation_times' in metrics
        assert 'system_health' in metrics
        assert 'optimization_enabled' in metrics
        assert 'parallel_workers' in metrics
        
        assert metrics['optimization_enabled'] is True
        assert 'plan_generation' in metrics['operation_times']
        assert 'cache_hits' in metrics['operation_times']
    
    def test_optimize_system(self):
        """Test system optimization."""
        planner = OptimizedQuantumPlanner()
        
        # Create tasks with issues
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        
        # Add some cache entries
        planner.cache.put("old_entry", "old_data")
        
        # Introduce system issues
        task1.priority_amplitudes[TaskPriority.HIGH] = 1.5  # Non-normalized
        
        optimization_result = planner.optimize_system()
        
        assert 'optimization_time_seconds' in optimization_result
        assert 'optimizations_applied' in optimization_result
        assert 'performance_metrics' in optimization_result
        
        assert len(optimization_result['optimizations_applied']) > 0
    
    def test_enable_disable_optimization(self):
        """Test enabling and disabling optimization."""
        planner = OptimizedQuantumPlanner()
        
        assert planner._optimization_enabled is True
        
        planner.enable_optimization(False)
        assert planner._optimization_enabled is False
        
        planner.enable_optimization(True)
        assert planner._optimization_enabled is True
    
    def test_shutdown(self):
        """Test planner shutdown."""
        planner = OptimizedQuantumPlanner()
        
        # Shutdown should not raise exceptions
        planner.shutdown()


class TestOptimizationIntegration:
    """Integration tests for optimization system."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization flow."""
        planner = OptimizedQuantumPlanner(cache_size_mb=10.0, max_workers=2)
        
        # Create a realistic project
        tasks_data = [
            {'id': 'requirements', 'title': 'Requirements Analysis', 'estimated_effort': 4.0},
            {'id': 'design', 'title': 'System Design', 'estimated_effort': 6.0, 'dependencies': ['requirements']},
            {'id': 'backend', 'title': 'Backend Development', 'estimated_effort': 16.0, 'dependencies': ['design']},
            {'id': 'frontend', 'title': 'Frontend Development', 'estimated_effort': 12.0, 'dependencies': ['design']},
            {'id': 'testing', 'title': 'Testing', 'estimated_effort': 8.0, 'dependencies': ['backend', 'frontend']},
            {'id': 'deployment', 'title': 'Deployment', 'estimated_effort': 4.0, 'dependencies': ['testing']}
        ]
        
        # Bulk create tasks
        created_tasks = planner.bulk_create_tasks(tasks_data)
        assert len(created_tasks) == 6
        
        # Create entanglements
        planner.create_task_entanglement('backend', 'frontend')
        
        # Generate execution plan multiple times
        plan1 = planner.generate_execution_plan()
        plan2 = planner.generate_execution_plan()  # Should be cached
        
        assert plan1['plan_id'] == plan2['plan_id']
        
        # Check performance metrics
        metrics = planner.get_performance_metrics()
        assert metrics['cache_performance']['hits'] > 0
        assert len(metrics['operation_times']['plan_generation']) >= 1
        assert len(metrics['operation_times']['cache_hits']) >= 1
        
        # Optimize system
        optimization_result = planner.optimize_system()
        assert len(optimization_result['optimizations_applied']) >= 0
        
        # System should still be healthy
        health = planner.get_system_health()
        assert health['system_valid'] is True
        
        # Shutdown
        planner.shutdown()
    
    def test_load_balancer_with_auto_scaling(self):
        """Test load balancer with auto-scaling."""
        # Create load balancer and auto-scaler
        balancer = LoadBalancer()
        scaler = AutoScaler(balancer, min_planners=1, max_planners=3)
        
        # Add initial planner
        initial_planner = OptimizedQuantumPlanner()
        balancer.add_planner(initial_planner)
        
        # Simulate high load to trigger scaling
        for i in range(10):
            planner_id, planner = balancer.get_optimal_planner()
            # Simulate work
            time.sleep(0.001)
            balancer.record_completion(planner_id, 0.1)
        
        # Check load stats
        stats = balancer.get_load_stats()
        assert stats['total_planners'] >= 1
        assert stats['requests_processed'] == 10
        
        # Check scaling stats
        scaling_stats = scaler.get_scaling_stats()
        assert 'total_scaling_events' in scaling_stats
    
    def test_performance_under_load(self):
        """Test system performance under load."""
        planner = OptimizedQuantumPlanner(cache_size_mb=50.0, max_workers=4)
        
        # Create many tasks
        start_time = time.time()
        
        tasks_data = []
        for i in range(100):
            tasks_data.append({
                'id': f'task{i}',
                'title': f'Task {i}',
                'description': f'Task number {i}',
                'estimated_effort': 1.0 + (i % 5),
                'dependencies': [f'task{j}' for j in range(max(0, i-2), i) if j % 10 == 0]
            })
        
        created_tasks = planner.bulk_create_tasks(tasks_data)
        creation_time = time.time() - start_time
        
        assert len(created_tasks) == 100
        assert creation_time < 5.0  # Should be reasonably fast
        
        # Generate execution plans
        plan_start_time = time.time()
        plan1 = planner.generate_execution_plan()
        first_plan_time = time.time() - plan_start_time
        
        plan2 = planner.generate_execution_plan()  # Should be cached
        second_plan_time = time.time() - plan_start_time - first_plan_time
        
        assert plan1['total_tasks'] == 100
        assert plan1['plan_id'] == plan2['plan_id']
        assert second_plan_time < first_plan_time  # Cached should be faster
        
        # Check metrics
        metrics = planner.get_performance_metrics()
        assert metrics['cache_performance']['hit_rate'] > 0
        
        planner.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])