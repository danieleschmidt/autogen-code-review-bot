"""Performance optimization and scaling utilities."""

import asyncio
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import PriorityQueue, Queue
from typing import Any, Callable, Dict, List, Optional

import psutil

from .exceptions import AnalysisError
from .logging_config import get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


@dataclass
class TaskPriority:
    """Task priority levels for queue management."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ProcessingTask:
    """Represents a processing task with metadata."""

    id: str
    priority: int
    created_at: float
    function: Callable
    args: tuple
    kwargs: dict
    timeout: Optional[float] = None

    def __lt__(self, other):
        """Enable priority queue ordering."""
        return self.priority < other.priority


class AdaptiveThreadPool:
    """Thread pool that adapts size based on system load and queue depth."""

    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = None,
        scale_factor: float = 1.5,
        load_threshold: float = 0.8,
    ):
        """Initialize adaptive thread pool.

        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads (defaults to CPU count * 2)
            scale_factor: Factor for scaling workers based on queue depth
            load_threshold: System load threshold for scaling decisions
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or (mp.cpu_count() * 2)
        self.scale_factor = scale_factor
        self.load_threshold = load_threshold

        self.current_workers = min_workers
        self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        self.task_queue = PriorityQueue()
        self.active_tasks = 0
        self.last_scale_time = 0
        self.scale_cooldown = 30  # seconds

        self._lock = threading.Lock()
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_and_scale, daemon=True
        )
        self._monitor_thread.start()

        logger.info(
            "Adaptive thread pool initialized",
            min_workers=min_workers,
            max_workers=self.max_workers,
        )

    def submit_task(self, task: ProcessingTask) -> asyncio.Future:
        """Submit a task for processing.

        Args:
            task: Processing task to execute

        Returns:
            Future representing the task execution
        """
        if not self._running:
            raise AnalysisError("Thread pool is shutting down")

        self.task_queue.put(task)

        # Record queue metrics
        metrics.record_gauge("thread_pool_queue_size", self.task_queue.qsize())
        metrics.record_counter(
            "thread_pool_tasks_submitted", 1, tags={"priority": str(task.priority)}
        )

        return self._process_next_task()

    def _process_next_task(self) -> asyncio.Future:
        """Process the next task from the queue."""
        try:
            task = self.task_queue.get_nowait()
        except:
            # Queue empty, return completed future
            future = asyncio.Future()
            future.set_result(None)
            return future

        with self._lock:
            self.active_tasks += 1

        # Submit to thread pool
        future = self.executor.submit(self._execute_task, task)

        # Wrap in async future
        async_future = asyncio.Future()

        def on_done(thread_future):
            try:
                result = thread_future.result()
                async_future.set_result(result)
            except Exception as e:
                async_future.set_exception(e)
            finally:
                with self._lock:
                    self.active_tasks -= 1

        future.add_done_callback(on_done)
        return async_future

    def _execute_task(self, task: ProcessingTask) -> Any:
        """Execute a single task with timeout and error handling.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        start_time = time.time()

        try:
            # Apply timeout if specified
            if task.timeout:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(
                        f"Task {task.id} timed out after {task.timeout}s"
                    )

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(task.timeout))

            # Execute task
            result = task.function(*task.args, **task.kwargs)

            # Record success metrics
            duration = time.time() - start_time
            metrics.record_histogram("thread_pool_task_duration", duration)
            metrics.record_counter(
                "thread_pool_tasks_completed",
                1,
                tags={"status": "success", "priority": str(task.priority)},
            )

            logger.debug(
                "Task completed successfully",
                task_id=task.id,
                duration=duration,
                priority=task.priority,
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            metrics.record_counter(
                "thread_pool_tasks_completed",
                1,
                tags={"status": "error", "priority": str(task.priority)},
            )

            logger.error(
                "Task execution failed",
                task_id=task.id,
                error=str(e),
                duration=duration,
            )
            raise

        finally:
            if task.timeout:
                signal.alarm(0)  # Cancel timeout

    def _monitor_and_scale(self):
        """Monitor system load and scale workers accordingly."""
        while self._running:
            try:
                time.sleep(10)  # Monitor every 10 seconds

                current_time = time.time()
                if current_time - self.last_scale_time < self.scale_cooldown:
                    continue  # In cooldown period

                queue_size = self.task_queue.qsize()
                cpu_percent = psutil.cpu_percent(interval=1.0) / 100.0
                memory_percent = psutil.virtual_memory().percent / 100.0

                # Determine if scaling is needed
                should_scale_up = (
                    queue_size > self.current_workers * 2  # Queue backing up
                    and cpu_percent < self.load_threshold  # CPU has capacity
                    and memory_percent < 0.8  # Memory available
                    and self.current_workers < self.max_workers  # Can scale up
                )

                should_scale_down = (
                    queue_size < self.current_workers / 2  # Low queue depth
                    and self.active_tasks < self.current_workers / 2  # Low active tasks
                    and self.current_workers > self.min_workers  # Can scale down
                )

                if should_scale_up:
                    new_workers = min(
                        self.max_workers, int(self.current_workers * self.scale_factor)
                    )
                    self._scale_workers(new_workers)

                elif should_scale_down:
                    new_workers = max(
                        self.min_workers, int(self.current_workers / self.scale_factor)
                    )
                    self._scale_workers(new_workers)

            except Exception as e:
                logger.error("Error in thread pool monitor", error=str(e))

    def _scale_workers(self, new_worker_count: int):
        """Scale the thread pool to new worker count.

        Args:
            new_worker_count: Target number of workers
        """
        if new_worker_count == self.current_workers:
            return

        old_count = self.current_workers

        # Create new executor with new worker count
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=new_worker_count)
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()

        # Shutdown old executor gracefully
        threading.Thread(
            target=lambda: old_executor.shutdown(wait=True), daemon=True
        ).start()

        # Record scaling metrics
        metrics.record_gauge("thread_pool_workers", new_worker_count)
        metrics.record_counter(
            "thread_pool_scaling_events",
            1,
            tags={"direction": "up" if new_worker_count > old_count else "down"},
        )

        logger.info(
            "Thread pool scaled",
            old_workers=old_count,
            new_workers=new_worker_count,
            queue_size=self.task_queue.qsize(),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current thread pool statistics.

        Returns:
            Dictionary with current stats
        """
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "active_tasks": self.active_tasks,
            "queue_size": self.task_queue.qsize(),
            "last_scale_time": self.last_scale_time,
        }

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool.

        Args:
            wait: Whether to wait for running tasks to complete
        """
        self._running = False
        self.executor.shutdown(wait=wait)
        logger.info("Thread pool shutdown completed")


class ConcurrentProcessor:
    """High-performance concurrent processor for code analysis tasks."""

    def __init__(self, max_workers: int = None, enable_batching: bool = True):
        """Initialize concurrent processor.

        Args:
            max_workers: Maximum number of worker processes
            enable_batching: Whether to batch similar requests
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.enable_batching = enable_batching

        self.thread_pool = AdaptiveThreadPool(
            min_workers=2, max_workers=self.max_workers
        )

        self.batch_queue = Queue()
        self.batch_size = 10
        self.batch_timeout = 2.0  # seconds

        if enable_batching:
            self._batch_thread = threading.Thread(
                target=self._process_batches, daemon=True
            )
            self._batch_thread.start()

    async def process_analysis_request(
        self,
        repo_path: str,
        config: Optional[Dict] = None,
        priority: int = TaskPriority.NORMAL,
    ) -> Any:
        """Process a code analysis request with optimal concurrency.

        Args:
            repo_path: Path to repository to analyze
            config: Optional configuration
            priority: Task priority level

        Returns:
            Analysis result
        """
        task_id = f"analysis_{int(time.time())}_{id(repo_path)}"

        if self.enable_batching and priority >= TaskPriority.NORMAL:
            # Add to batch queue for non-critical requests
            return await self._queue_for_batch(task_id, repo_path, config)
        else:
            # Process immediately for critical requests
            return await self._process_immediately(task_id, repo_path, config, priority)

    async def _process_immediately(
        self, task_id: str, repo_path: str, config: Optional[Dict], priority: int
    ) -> Any:
        """Process request immediately without batching.

        Args:
            task_id: Unique task identifier
            repo_path: Repository path
            config: Analysis configuration
            priority: Task priority

        Returns:
            Analysis result
        """
        from .pr_analysis import analyze_pr

        task = ProcessingTask(
            id=task_id,
            priority=priority,
            created_at=time.time(),
            function=analyze_pr,
            args=(repo_path,),
            kwargs={"config_path": config} if config else {},
            timeout=300,  # 5 minute timeout
        )

        return await self.thread_pool.submit_task(task)

    async def _queue_for_batch(
        self, task_id: str, repo_path: str, config: Optional[Dict]
    ) -> Any:
        """Queue request for batch processing.

        Args:
            task_id: Unique task identifier
            repo_path: Repository path
            config: Analysis configuration

        Returns:
            Analysis result future
        """
        future = asyncio.Future()

        batch_item = {
            "id": task_id,
            "repo_path": repo_path,
            "config": config,
            "future": future,
            "created_at": time.time(),
        }

        self.batch_queue.put(batch_item)
        return await future

    def _process_batches(self):
        """Background thread for processing batched requests."""
        batch = []
        last_process_time = time.time()

        while True:
            try:
                # Collect batch items
                while len(batch) < self.batch_size:
                    try:
                        item = self.batch_queue.get(timeout=0.5)
                        batch.append(item)
                    except:
                        break  # Timeout or queue empty

                # Process batch if we have items and conditions are met
                current_time = time.time()
                should_process = len(batch) >= self.batch_size or (
                    batch and current_time - last_process_time > self.batch_timeout
                )

                if should_process and batch:
                    self._execute_batch(batch)
                    batch = []
                    last_process_time = current_time

            except Exception as e:
                logger.error("Error in batch processor", error=str(e))
                # Complete any pending futures with error
                for item in batch:
                    if not item["future"].done():
                        item["future"].set_exception(e)
                batch = []

    def _execute_batch(self, batch: List[Dict]):
        """Execute a batch of analysis requests concurrently.

        Args:
            batch: List of batch items to process
        """
        logger.info("Processing analysis batch", batch_size=len(batch))

        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(
            max_workers=min(len(batch), self.max_workers)
        ) as executor:
            # Submit all tasks
            future_to_item = {}
            for item in batch:
                from .pr_analysis import analyze_pr

                future = executor.submit(analyze_pr, item["repo_path"])
                future_to_item[future] = item

            # Collect results
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    item["future"].set_result(result)

                    # Record batch metrics
                    duration = time.time() - item["created_at"]
                    metrics.record_histogram("batch_processing_duration", duration)

                except Exception as e:
                    item["future"].set_exception(e)
                    logger.error("Batch item failed", item_id=item["id"], error=str(e))

        metrics.record_counter(
            "batches_processed", 1, tags={"batch_size": str(len(batch))}
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics.

        Returns:
            Performance statistics dictionary
        """
        thread_stats = self.thread_pool.get_stats()

        return {
            "thread_pool": thread_stats,
            "batch_queue_size": self.batch_queue.qsize(),
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_load": (
                psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            ),
        }

    def shutdown(self):
        """Shutdown the concurrent processor."""
        self.thread_pool.shutdown()
        logger.info("Concurrent processor shutdown completed")


class PerformanceProfiler:
    """Performance profiler for identifying bottlenecks."""

    def __init__(self):
        self.profiles = {}
        self._lock = threading.Lock()

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance.

        Args:
            func: Function to profile

        Returns:
            Wrapped function with profiling
        """

        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss

                duration = end_time - start_time
                memory_delta = end_memory - start_memory

                # Record profile data
                with self._lock:
                    if func.__name__ not in self.profiles:
                        self.profiles[func.__name__] = []

                    self.profiles[func.__name__].append(
                        {
                            "duration": duration,
                            "memory_delta": memory_delta,
                            "timestamp": start_time,
                            "success": True,
                        }
                    )

                # Record metrics
                metrics.record_histogram(f"function_duration_{func.__name__}", duration)
                metrics.record_histogram(
                    f"function_memory_delta_{func.__name__}", memory_delta
                )

                return result

            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time

                with self._lock:
                    if func.__name__ not in self.profiles:
                        self.profiles[func.__name__] = []

                    self.profiles[func.__name__].append(
                        {
                            "duration": duration,
                            "memory_delta": 0,
                            "timestamp": start_time,
                            "success": False,
                            "error": str(e),
                        }
                    )

                raise

        return wrapper

    def get_profile_summary(
        self, function_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get performance profile summary.

        Args:
            function_name: Optional specific function to analyze

        Returns:
            Profile summary statistics
        """
        with self._lock:
            if function_name:
                if function_name not in self.profiles:
                    return {}
                data = self.profiles[function_name]
                return self._calculate_stats(function_name, data)
            else:
                summary = {}
                for func_name, data in self.profiles.items():
                    summary[func_name] = self._calculate_stats(func_name, data)
                return summary

    def _calculate_stats(self, function_name: str, data: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for function profile data.

        Args:
            function_name: Name of the function
            data: List of profile data points

        Returns:
            Statistical summary
        """
        if not data:
            return {}

        durations = [d["duration"] for d in data]
        memory_deltas = [d["memory_delta"] for d in data]
        successes = sum(1 for d in data if d["success"])

        return {
            "call_count": len(data),
            "success_rate": successes / len(data),
            "avg_duration": sum(durations) / len(durations),
            "max_duration": max(durations),
            "min_duration": min(durations),
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
            "max_memory_delta": max(memory_deltas),
            "total_time": sum(durations),
        }


# Global instances
_concurrent_processor = None
_performance_profiler = PerformanceProfiler()


def get_concurrent_processor() -> ConcurrentProcessor:
    """Get global concurrent processor instance."""
    global _concurrent_processor
    if _concurrent_processor is None:
        _concurrent_processor = ConcurrentProcessor()
    return _concurrent_processor


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    return _performance_profiler


def profile_performance(func: Callable) -> Callable:
    """Decorator for profiling function performance."""
    return _performance_profiler.profile_function(func)
