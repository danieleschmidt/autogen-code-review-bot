#!/usr/bin/env python3
"""
Distributed Processing Engine for AutoGen Code Review Bot.

Implements horizontal scaling, distributed task execution, load balancing,
auto-scaling, and multi-region deployment capabilities.
"""

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing as mp

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
import aiohttp

from .logging_utils import get_logger
from .metrics import get_metrics_registry, record_operation_metrics
from .resilience import with_retry, with_circuit_breaker, RetryConfig
from .models import PRAnalysisResult

logger = get_logger(__name__)
metrics = get_metrics_registry()


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class DistributedTask:
    """Represents a distributed task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'payload': self.payload,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'assigned_to': self.assigned_to,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """Create from dictionary."""
        return cls(
            task_id=data['task_id'],
            task_type=data['task_type'],
            payload=data['payload'],
            priority=TaskPriority(data['priority']),
            status=TaskStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            assigned_to=data.get('assigned_to'),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            result=data.get('result'),
            error=data.get('error'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            timeout_seconds=data.get('timeout_seconds', 300),
            tags=data.get('tags', {})
        )


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    node_id: str
    hostname: str
    region: str
    capabilities: List[str]
    max_concurrent_tasks: int = 10
    current_load: int = 0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"  # active, draining, offline
    total_processed: int = 0
    total_errors: int = 0
    average_processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'region': self.region,
            'capabilities': self.capabilities,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'current_load': self.current_load,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'status': self.status,
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'average_processing_time': self.average_processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerNode':
        """Create from dictionary."""
        return cls(
            node_id=data['node_id'],
            hostname=data['hostname'],
            region=data['region'],
            capabilities=data['capabilities'],
            max_concurrent_tasks=data.get('max_concurrent_tasks', 10),
            current_load=data.get('current_load', 0),
            last_heartbeat=datetime.fromisoformat(data['last_heartbeat']),
            status=data.get('status', 'active'),
            total_processed=data.get('total_processed', 0),
            total_errors=data.get('total_errors', 0),
            average_processing_time=data.get('average_processing_time', 0.0)
        )
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage."""
        return (self.current_load / self.max_concurrent_tasks) * 100 if self.max_concurrent_tasks > 0 else 0
    
    def can_accept_task(self) -> bool:
        """Check if node can accept new tasks."""
        return (
            self.status == "active" and 
            self.current_load < self.max_concurrent_tasks and
            (datetime.now(timezone.utc) - self.last_heartbeat).seconds < 60
        )


class TaskScheduler:
    """Intelligent task scheduler with load balancing."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
        self.logger = get_logger(__name__ + ".TaskScheduler")
        self._scheduling_strategies = {
            'round_robin': self._round_robin_strategy,
            'least_loaded': self._least_loaded_strategy,
            'geographic': self._geographic_strategy,
            'capability_based': self._capability_based_strategy
        }
        self._current_strategy = 'least_loaded'
    
    async def schedule_task(self, task: DistributedTask, workers: List[WorkerNode]) -> Optional[str]:
        """Schedule task to best available worker."""
        strategy = self._scheduling_strategies[self._current_strategy]
        selected_worker = await strategy(task, workers)
        
        if selected_worker:
            # Assign task to worker
            task.assigned_to = selected_worker.node_id
            task.status = TaskStatus.ASSIGNED
            
            # Update worker load
            selected_worker.current_load += 1
            
            self.logger.info(f"Task {task.task_id} scheduled to worker {selected_worker.node_id}", extra={
                'task_type': task.task_type,
                'worker_load': selected_worker.current_load,
                'strategy': self._current_strategy
            })
            
            return selected_worker.node_id
        
        return None
    
    async def _round_robin_strategy(self, task: DistributedTask, workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Round-robin scheduling strategy."""
        available_workers = [w for w in workers if w.can_accept_task()]
        if not available_workers:
            return None
        
        # Simple round-robin based on task creation time
        index = hash(task.task_id) % len(available_workers)
        return available_workers[index]
    
    async def _least_loaded_strategy(self, task: DistributedTask, workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Least loaded scheduling strategy."""
        available_workers = [w for w in workers if w.can_accept_task()]
        if not available_workers:
            return None
        
        # Sort by load percentage
        return min(available_workers, key=lambda w: w.get_load_percentage())
    
    async def _geographic_strategy(self, task: DistributedTask, workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Geographic affinity scheduling strategy."""
        available_workers = [w for w in workers if w.can_accept_task()]
        if not available_workers:
            return None
        
        # Prefer workers in same region if specified
        preferred_region = task.tags.get('region')
        if preferred_region:
            regional_workers = [w for w in available_workers if w.region == preferred_region]
            if regional_workers:
                return min(regional_workers, key=lambda w: w.get_load_percentage())
        
        # Fallback to least loaded
        return min(available_workers, key=lambda w: w.get_load_percentage())
    
    async def _capability_based_strategy(self, task: DistributedTask, workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Capability-based scheduling strategy."""
        available_workers = [w for w in workers if w.can_accept_task()]
        if not available_workers:
            return None
        
        # Filter by required capabilities
        required_capabilities = task.tags.get('required_capabilities', [])
        if required_capabilities:
            capable_workers = [
                w for w in available_workers 
                if all(cap in w.capabilities for cap in required_capabilities)
            ]
            if capable_workers:
                return min(capable_workers, key=lambda w: w.get_load_percentage())
        
        # Fallback to least loaded
        return min(available_workers, key=lambda w: w.get_load_percentage())


class LoadBalancer:
    """Advanced load balancer with health checking and auto-scaling."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
        self.logger = get_logger(__name__ + ".LoadBalancer")
        self.health_check_interval = 30  # seconds
        self.scale_check_interval = 60   # seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start load balancer monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Load balancer started")
    
    async def stop(self):
        """Stop load balancer monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Load balancer stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        health_check_last = 0
        scale_check_last = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Health checks
                if current_time - health_check_last >= self.health_check_interval:
                    await self._perform_health_checks()
                    health_check_last = current_time
                
                # Auto-scaling checks
                if current_time - scale_check_last >= self.scale_check_interval:
                    await self._check_scaling_needs()
                    scale_check_last = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Load balancer monitoring error: {e}")
                await asyncio.sleep(30)  # Back off on errors
    
    async def _perform_health_checks(self):
        """Perform health checks on all workers."""
        try:
            # Get all workers
            workers = await self.get_all_workers()
            
            for worker in workers:
                # Check if worker is responsive
                if await self._check_worker_health(worker):
                    if worker.status == "offline":
                        worker.status = "active"
                        await self._update_worker(worker)
                        self.logger.info(f"Worker {worker.node_id} back online")
                else:
                    if worker.status == "active":
                        worker.status = "offline"
                        await self._update_worker(worker)
                        self.logger.warning(f"Worker {worker.node_id} marked offline")
                        
                        # Reschedule its tasks
                        await self._reschedule_worker_tasks(worker.node_id)
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    async def _check_worker_health(self, worker: WorkerNode) -> bool:
        """Check individual worker health."""
        try:
            # Check heartbeat timestamp
            time_since_heartbeat = datetime.now(timezone.utc) - worker.last_heartbeat
            if time_since_heartbeat.seconds > 120:  # 2 minutes
                return False
            
            # Additional health checks could go here
            # e.g., HTTP health endpoint, resource usage, etc.
            
            return True
            
        except Exception as e:
            self.logger.error(f"Worker health check failed for {worker.node_id}: {e}")
            return False
    
    async def _reschedule_worker_tasks(self, node_id: str):
        """Reschedule tasks from offline worker."""
        try:
            # Get tasks assigned to this worker
            task_keys = await self.redis.keys(f"task:*")
            
            for task_key in task_keys:
                task_data = await self.redis.get(task_key)
                if task_data:
                    task = DistributedTask.from_dict(json.loads(task_data))
                    
                    if (task.assigned_to == node_id and 
                        task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]):
                        
                        # Reset task for rescheduling
                        task.assigned_to = None
                        task.status = TaskStatus.PENDING
                        task.retry_count += 1
                        
                        if task.retry_count <= task.max_retries:
                            await self.redis.set(task_key, json.dumps(task.to_dict()))
                            self.logger.info(f"Rescheduled task {task.task_id} from offline worker {node_id}")
                        else:
                            task.status = TaskStatus.FAILED
                            task.error = f"Max retries exceeded after worker {node_id} went offline"
                            await self.redis.set(task_key, json.dumps(task.to_dict()))
                            self.logger.error(f"Task {task.task_id} failed after max retries")
            
        except Exception as e:
            self.logger.error(f"Error rescheduling tasks from worker {node_id}: {e}")
    
    async def _check_scaling_needs(self):
        """Check if auto-scaling is needed."""
        try:
            workers = await self.get_all_workers()
            active_workers = [w for w in workers if w.status == "active"]
            
            if not active_workers:
                return
            
            # Calculate average load
            total_load = sum(w.current_load for w in active_workers)
            total_capacity = sum(w.max_concurrent_tasks for w in active_workers)
            average_load_percentage = (total_load / total_capacity) * 100 if total_capacity > 0 else 0
            
            # Get pending tasks count
            pending_tasks = await self._get_pending_tasks_count()
            
            self.logger.debug(f"System load: {average_load_percentage:.1f}%, Pending tasks: {pending_tasks}")
            
            # Scale up if load > 80% or many pending tasks
            if average_load_percentage > 80 or pending_tasks > 50:
                await self._scale_up()
            
            # Scale down if load < 30% and sufficient workers
            elif average_load_percentage < 30 and len(active_workers) > 2:
                await self._scale_down()
            
        except Exception as e:
            self.logger.error(f"Auto-scaling check error: {e}")
    
    async def _scale_up(self):
        """Scale up worker instances."""
        self.logger.info("Scaling up workers due to high load")
        
        # Record scaling event
        metrics.record_counter("autoscaling_events_total", 1, tags={"direction": "up"})
        
        # In a real implementation, this would trigger container orchestration
        # or cloud provider APIs to spin up new instances
        # For now, we'll just log the intent
        
    async def _scale_down(self):
        """Scale down worker instances."""
        self.logger.info("Scaling down workers due to low load")
        
        # Record scaling event  
        metrics.record_counter("autoscaling_events_total", 1, tags={"direction": "down"})
        
        # In a real implementation, this would gracefully drain and terminate workers
    
    async def get_all_workers(self) -> List[WorkerNode]:
        """Get all registered workers."""
        try:
            worker_keys = await self.redis.keys("worker:*")
            workers = []
            
            for key in worker_keys:
                worker_data = await self.redis.get(key)
                if worker_data:
                    workers.append(WorkerNode.from_dict(json.loads(worker_data)))
            
            return workers
            
        except Exception as e:
            self.logger.error(f"Error getting workers: {e}")
            return []
    
    async def _update_worker(self, worker: WorkerNode):
        """Update worker information in Redis."""
        try:
            await self.redis.set(f"worker:{worker.node_id}", json.dumps(worker.to_dict()))
        except Exception as e:
            self.logger.error(f"Error updating worker {worker.node_id}: {e}")
    
    async def _get_pending_tasks_count(self) -> int:
        """Get count of pending tasks."""
        try:
            task_keys = await self.redis.keys("task:*")
            pending_count = 0
            
            for key in task_keys:
                task_data = await self.redis.get(key)
                if task_data:
                    task = DistributedTask.from_dict(json.loads(task_data))
                    if task.status == TaskStatus.PENDING:
                        pending_count += 1
            
            return pending_count
            
        except Exception as e:
            self.logger.error(f"Error getting pending tasks count: {e}")
            return 0


class DistributedTaskManager:
    """Main coordinator for distributed task processing."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 node_id: str = None, region: str = "default"):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.node_id = node_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.region = region
        
        self.scheduler = TaskScheduler(self.redis)
        self.load_balancer = LoadBalancer(self.redis)
        
        self.local_executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.process_executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        self.task_handlers: Dict[str, Callable] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        self.worker_node: Optional[WorkerNode] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._task_processor_task: Optional[asyncio.Task] = None
        self._running = False
        
        self.logger = get_logger(__name__ + ".DistributedTaskManager")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register handler for specific task type."""
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    async def start_worker(self, capabilities: List[str] = None, 
                          max_concurrent_tasks: int = 10):
        """Start worker node."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize worker node
        self.worker_node = WorkerNode(
            node_id=self.node_id,
            hostname=f"host-{self.node_id}",
            region=self.region,
            capabilities=capabilities or ['analysis', 'security', 'style'],
            max_concurrent_tasks=max_concurrent_tasks
        )
        
        # Register worker
        await self._register_worker()
        
        # Start services
        await self.load_balancer.start()
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._task_processor_task = asyncio.create_task(self._task_processor_loop())
        
        self.logger.info(f"Worker node {self.node_id} started", extra={
            'region': self.region,
            'capabilities': self.worker_node.capabilities,
            'max_concurrent_tasks': max_concurrent_tasks
        })
    
    async def stop_worker(self):
        """Stop worker node gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Update worker status
        if self.worker_node:
            self.worker_node.status = "draining"
            await self._update_worker()
        
        # Cancel background tasks
        for task in [self._heartbeat_task, self._task_processor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Wait for running tasks to complete
        if self.running_tasks:
            self.logger.info(f"Waiting for {len(self.running_tasks)} tasks to complete")
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Stop services
        await self.load_balancer.stop()
        
        # Shutdown executors
        self.local_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        # Unregister worker
        await self._unregister_worker()
        
        self.logger.info(f"Worker node {self.node_id} stopped")
    
    async def submit_task(self, task_type: str, payload: Dict[str, Any], 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         tags: Dict[str, str] = None) -> str:
        """Submit task for distributed processing."""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            tags=tags or {}
        )
        
        # Store task
        await self.redis.set(f"task:{task.task_id}", json.dumps(task.to_dict()), ex=3600)
        
        # Add to priority queue
        await self.redis.zadd("task_queue", {task.task_id: priority.value})
        
        self.logger.info(f"Task {task.task_id} submitted", extra={
            'task_type': task_type,
            'priority': priority.name
        })
        
        return task.task_id
    
    async def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get task status and result."""
        task_data = await self.redis.get(f"task:{task_id}")
        if task_data:
            return DistributedTask.from_dict(json.loads(task_data))
        return None
    
    async def _register_worker(self):
        """Register worker in Redis."""
        await self.redis.set(
            f"worker:{self.node_id}", 
            json.dumps(self.worker_node.to_dict()),
            ex=120  # Auto-expire in 2 minutes
        )
    
    async def _unregister_worker(self):
        """Unregister worker from Redis."""
        await self.redis.delete(f"worker:{self.node_id}")
    
    async def _update_worker(self):
        """Update worker information."""
        if self.worker_node:
            self.worker_node.last_heartbeat = datetime.now(timezone.utc)
            await self.redis.set(
                f"worker:{self.node_id}",
                json.dumps(self.worker_node.to_dict()),
                ex=120
            )
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat."""
        while self._running:
            try:
                await self._update_worker()
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    async def _task_processor_loop(self):
        """Main task processing loop."""
        while self._running:
            try:
                # Get available task
                task = await self._get_next_task()
                if task and len(self.running_tasks) < self.worker_node.max_concurrent_tasks:
                    # Process task asynchronously
                    task_coroutine = self._process_task(task)
                    self.running_tasks[task.task_id] = asyncio.create_task(task_coroutine)
                else:
                    await asyncio.sleep(1)  # No tasks or at capacity
                
                # Clean up completed tasks
                completed_tasks = [
                    task_id for task_id, task_future in self.running_tasks.items()
                    if task_future.done()
                ]
                
                for task_id in completed_tasks:
                    del self.running_tasks[task_id]
                
            except Exception as e:
                self.logger.error(f"Task processor error: {e}")
                await asyncio.sleep(10)
    
    async def _get_next_task(self) -> Optional[DistributedTask]:
        """Get next task from queue."""
        try:
            # Get highest priority task
            task_data = await self.redis.zpopmax("task_queue", 1)
            if not task_data:
                return None
            
            task_id, _ = task_data[0]
            
            # Get task details
            task_json = await self.redis.get(f"task:{task_id}")
            if not task_json:
                return None
            
            task = DistributedTask.from_dict(json.loads(task_json))
            
            # Check if we can handle this task type
            if task.task_type not in self.task_handlers:
                # Put back in queue for other workers
                await self.redis.zadd("task_queue", {task_id: task.priority.value})
                return None
            
            # Assign to this worker
            task.assigned_to = self.node_id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            
            # Update worker load
            if self.worker_node:
                self.worker_node.current_load += 1
            
            # Save updated task
            await self.redis.set(f"task:{task_id}", json.dumps(task.to_dict()))
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error getting next task: {e}")
            return None
    
    async def _process_task(self, task: DistributedTask):
        """Process individual task."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing task {task.task_id}", extra={
                'task_type': task.task_type,
                'worker': self.node_id
            })
            
            # Get task handler
            handler = self.task_handlers[task.task_type]
            
            # Execute task with timeout
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await asyncio.wait_for(
                        handler(task.payload),
                        timeout=task.timeout_seconds
                    )
                else:
                    # Run in thread pool
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.local_executor, handler, task.payload
                        ),
                        timeout=task.timeout_seconds
                    )
                
                # Update task with result
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = datetime.now(timezone.utc)
                
                # Record success metrics
                processing_time = time.time() - start_time
                metrics.record_histogram("task_processing_time_seconds", processing_time, tags={
                    "task_type": task.task_type,
                    "worker": self.node_id,
                    "status": "success"
                })
                
                # Update worker stats
                if self.worker_node:
                    self.worker_node.total_processed += 1
                    self.worker_node.average_processing_time = (
                        (self.worker_node.average_processing_time * (self.worker_node.total_processed - 1) + 
                         processing_time) / self.worker_node.total_processed
                    )
                
                self.logger.info(f"Task {task.task_id} completed successfully", extra={
                    'processing_time_seconds': processing_time
                })
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Task timed out after {task.timeout_seconds} seconds"
                self.logger.error(f"Task {task.task_id} timed out")
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                
                if self.worker_node:
                    self.worker_node.total_errors += 1
                
                self.logger.error(f"Task {task.task_id} failed: {e}")
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"Processing error: {str(e)}"
            self.logger.error(f"Task processing error for {task.task_id}: {e}")
        
        finally:
            # Update worker load
            if self.worker_node:
                self.worker_node.current_load = max(0, self.worker_node.current_load - 1)
            
            # Save final task state
            await self.redis.set(f"task:{task.task_id}", json.dumps(task.to_dict()))
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            workers = await self.load_balancer.get_all_workers()
            pending_count = await self.load_balancer._get_pending_tasks_count()
            
            return {
                'workers': {
                    'total': len(workers),
                    'active': len([w for w in workers if w.status == "active"]),
                    'offline': len([w for w in workers if w.status == "offline"]),
                    'details': [w.to_dict() for w in workers]
                },
                'tasks': {
                    'pending': pending_count,
                    'running': len(self.running_tasks),
                    'queue_size': await self.redis.zcard("task_queue")
                },
                'system': {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'node_id': self.node_id,
                    'region': self.region
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


# Default task handlers
async def analyze_repository_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Task handler for repository analysis."""
    from .pr_analysis import analyze_pr
    
    repo_path = payload['repo_path']
    config_path = payload.get('config_path')
    
    # Run analysis
    result = analyze_pr(repo_path, config_path)
    
    return {
        'analysis_result': {
            'security': asdict(result.security),
            'style': asdict(result.style),
            'performance': asdict(result.performance),
            'metadata': result.metadata
        }
    }


def create_distributed_manager(redis_url: str = "redis://localhost:6379",
                              region: str = "default") -> DistributedTaskManager:
    """Create and configure distributed task manager."""
    manager = DistributedTaskManager(redis_url=redis_url, region=region)
    
    # Register default task handlers
    manager.register_task_handler('analyze_repository', analyze_repository_task)
    
    return manager