"""Load balancing and auto-scaling for distributed processing."""

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_config import get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CONSISTENT_HASH = "consistent_hash"


@dataclass
class WorkerNode:
    """Represents a worker node in the load balancer."""
    id: str
    endpoint: str
    weight: float
    max_connections: int
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    is_healthy: bool = True
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)."""
        if not self.is_healthy:
            return float('inf')

        # Combine multiple factors into load score
        connection_load = self.current_connections / self.max_connections if self.max_connections > 0 else 0
        resource_load = (self.cpu_usage + self.memory_usage) / 2
        response_time_factor = min(self.avg_response_time / 1000, 2.0)  # Cap at 2 seconds

        return (connection_load * 0.4 + resource_load * 0.3 + response_time_factor * 0.3) / self.weight


class HealthChecker:
    """Health checker for worker nodes."""

    def __init__(self, check_interval: float = 30.0, timeout: float = 5.0):
        """Initialize health checker.
        
        Args:
            check_interval: Interval between health checks in seconds
            timeout: Timeout for health check requests
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nodes: List[WorkerNode] = []
        self._lock = threading.Lock()

    def start(self, nodes: List[WorkerNode]):
        """Start health checking.
        
        Args:
            nodes: List of worker nodes to monitor
        """
        self._nodes = nodes
        self._running = True
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        logger.info("Health checker started", node_count=len(nodes))

    def stop(self):
        """Stop health checking."""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Health checker stopped")

    def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                self._check_all_nodes()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))

    def _check_all_nodes(self):
        """Check health of all nodes."""
        for node in self._nodes:
            try:
                is_healthy = self._check_node_health(node)

                with self._lock:
                    old_status = node.is_healthy
                    node.is_healthy = is_healthy
                    node.last_health_check = time.time()

                # Log status changes
                if old_status != is_healthy:
                    status = "healthy" if is_healthy else "unhealthy"
                    logger.warning("Node health status changed",
                                 node_id=node.id,
                                 status=status)
                    metrics.record_counter("load_balancer_health_changes", 1,
                                         tags={"node_id": node.id, "status": status})

            except Exception as e:
                logger.error("Failed to check node health", node_id=node.id, error=str(e))
                node.is_healthy = False

    def _check_node_health(self, node: WorkerNode) -> bool:
        """Check health of a single node.
        
        Args:
            node: Worker node to check
            
        Returns:
            True if node is healthy
        """
        try:
            # Simple HTTP health check (can be extended)
            import requests

            health_url = f"{node.endpoint}/health"
            response = requests.get(health_url, timeout=self.timeout)

            if response.status_code == 200:
                # Parse response for additional metrics
                try:
                    data = response.json()
                    node.cpu_usage = data.get("cpu_usage", 0.0)
                    node.memory_usage = data.get("memory_usage", 0.0)
                except:
                    pass  # Use defaults if parsing fails

                return True
            else:
                return False

        except Exception:
            return False


class LoadBalancer:
    """Advanced load balancer with multiple strategies and health checking."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
                 health_check_interval: float = 30.0):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Health check interval in seconds
        """
        self.strategy = strategy
        self._nodes: List[WorkerNode] = []
        self._current_index = 0
        self._lock = threading.RLock()

        # Health checking
        self._health_checker = HealthChecker(check_interval=health_check_interval)

        # Response time tracking
        self._response_times: Dict[str, deque] = {}

        # Consistent hashing ring (for consistent hash strategy)
        self._hash_ring: List[Tuple[int, str]] = []

        logger.info("Load balancer initialized", strategy=strategy.value)

    def add_node(self, node: WorkerNode):
        """Add a worker node.
        
        Args:
            node: Worker node to add
        """
        with self._lock:
            self._nodes.append(node)
            self._response_times[node.id] = deque(maxlen=100)  # Keep last 100 response times

            # Update hash ring for consistent hashing
            if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._update_hash_ring()

        logger.info("Added worker node", node_id=node.id, endpoint=node.endpoint)
        metrics.record_gauge("load_balancer_nodes", len(self._nodes))

    def remove_node(self, node_id: str):
        """Remove a worker node.
        
        Args:
            node_id: ID of node to remove
        """
        with self._lock:
            self._nodes = [n for n in self._nodes if n.id != node_id]
            if node_id in self._response_times:
                del self._response_times[node_id]

            # Update hash ring for consistent hashing
            if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._update_hash_ring()

        logger.info("Removed worker node", node_id=node_id)
        metrics.record_gauge("load_balancer_nodes", len(self._nodes))

    def start(self):
        """Start the load balancer and health checking."""
        if self._nodes:
            self._health_checker.start(self._nodes)
        logger.info("Load balancer started")

    def stop(self):
        """Stop the load balancer."""
        self._health_checker.stop()
        logger.info("Load balancer stopped")

    def get_next_node(self, request_key: Optional[str] = None) -> Optional[WorkerNode]:
        """Get next worker node based on load balancing strategy.
        
        Args:
            request_key: Optional key for consistent hashing
            
        Returns:
            Selected worker node or None if no healthy nodes available
        """
        with self._lock:
            healthy_nodes = [node for node in self._nodes if node.is_healthy]

            if not healthy_nodes:
                logger.warning("No healthy nodes available")
                metrics.record_counter("load_balancer_no_healthy_nodes", 1)
                return None

            # Select node based on strategy
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                node = self._round_robin_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                node = self._least_connections_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                node = self._weighted_round_robin_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                node = self._least_response_time_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                node = self._resource_based_selection(healthy_nodes)
            elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                node = self._consistent_hash_selection(healthy_nodes, request_key)
            else:
                node = random.choice(healthy_nodes)  # Fallback

            # Update node statistics
            if node:
                node.current_connections += 1
                node.total_requests += 1
                metrics.record_counter("load_balancer_requests", 1,
                                     tags={"node_id": node.id, "strategy": self.strategy.value})

            return node

    def record_response(self, node_id: str, response_time: float, success: bool):
        """Record response metrics for a node.
        
        Args:
            node_id: ID of the node
            response_time: Response time in seconds
            success: Whether the request was successful
        """
        with self._lock:
            # Find node and update statistics
            for node in self._nodes:
                if node.id == node_id:
                    node.current_connections = max(0, node.current_connections - 1)

                    if not success:
                        node.failed_requests += 1

                    # Update average response time
                    if node_id in self._response_times:
                        self._response_times[node_id].append(response_time)
                        node.avg_response_time = sum(self._response_times[node_id]) / len(self._response_times[node_id])

                    break

        # Record metrics
        status = "success" if success else "error"
        metrics.record_histogram("load_balancer_response_time", response_time,
                               tags={"node_id": node_id, "status": status})
        metrics.record_counter("load_balancer_responses", 1,
                             tags={"node_id": node_id, "status": status})

    def _round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round robin selection.
        
        Args:
            nodes: List of healthy nodes
            
        Returns:
            Selected node
        """
        node = nodes[self._current_index % len(nodes)]
        self._current_index = (self._current_index + 1) % len(nodes)
        return node

    def _least_connections_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Least connections selection.
        
        Args:
            nodes: List of healthy nodes
            
        Returns:
            Node with least connections
        """
        return min(nodes, key=lambda n: n.current_connections)

    def _weighted_round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin selection.
        
        Args:
            nodes: List of healthy nodes
            
        Returns:
            Selected node based on weights
        """
        # Create weighted list
        weighted_nodes = []
        for node in nodes:
            weight_count = max(1, int(node.weight * 10))  # Scale weight
            weighted_nodes.extend([node] * weight_count)

        if weighted_nodes:
            return weighted_nodes[self._current_index % len(weighted_nodes)]
        return nodes[0]

    def _least_response_time_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Least response time selection.
        
        Args:
            nodes: List of healthy nodes
            
        Returns:
            Node with lowest average response time
        """
        return min(nodes, key=lambda n: n.avg_response_time)

    def _resource_based_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Resource-based selection using load scores.
        
        Args:
            nodes: List of healthy nodes
            
        Returns:
            Node with lowest load score
        """
        return min(nodes, key=lambda n: n.load_score)

    def _consistent_hash_selection(self, nodes: List[WorkerNode],
                                 request_key: Optional[str]) -> WorkerNode:
        """Consistent hash selection.
        
        Args:
            nodes: List of healthy nodes
            request_key: Key for consistent hashing
            
        Returns:
            Consistently selected node
        """
        if not request_key:
            return random.choice(nodes)

        # Simple hash function
        key_hash = hash(request_key) % (2**32)

        # Find node in hash ring
        for hash_value, node_id in self._hash_ring:
            if key_hash <= hash_value:
                # Find the actual node object
                for node in nodes:
                    if node.id == node_id:
                        return node

        # Fallback to first node
        return nodes[0] if nodes else None

    def _update_hash_ring(self):
        """Update consistent hash ring."""
        self._hash_ring.clear()

        for node in self._nodes:
            # Create multiple virtual nodes for better distribution
            for i in range(100):  # 100 virtual nodes per physical node
                virtual_key = f"{node.id}:{i}"
                hash_value = hash(virtual_key) % (2**32)
                self._hash_ring.append((hash_value, node.id))

        # Sort by hash value
        self._hash_ring.sort()

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            node_stats = []
            for node in self._nodes:
                node_stats.append({
                    "id": node.id,
                    "endpoint": node.endpoint,
                    "is_healthy": node.is_healthy,
                    "current_connections": node.current_connections,
                    "total_requests": node.total_requests,
                    "success_rate": node.success_rate,
                    "avg_response_time": node.avg_response_time,
                    "load_score": node.load_score,
                    "cpu_usage": node.cpu_usage,
                    "memory_usage": node.memory_usage
                })

            healthy_count = sum(1 for node in self._nodes if node.is_healthy)

            return {
                "strategy": self.strategy.value,
                "total_nodes": len(self._nodes),
                "healthy_nodes": healthy_count,
                "nodes": node_stats
            }


class AutoScaler:
    """Auto-scaling system for dynamic resource management."""

    def __init__(self, min_nodes: int = 2, max_nodes: int = 10,
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3,
                 scale_cooldown: float = 300):
        """Initialize auto-scaler.
        
        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            scale_up_threshold: Threshold for scaling up
            scale_down_threshold: Threshold for scaling down
            scale_cooldown: Cooldown period between scaling actions
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_cooldown = scale_cooldown

        self._last_scale_time = 0
        self._load_history = deque(maxlen=60)  # Keep 60 data points
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._node_factory: Optional[Callable[[], WorkerNode]] = None
        self._load_balancer: Optional[LoadBalancer] = None

        logger.info("Auto-scaler initialized",
                   min_nodes=min_nodes,
                   max_nodes=max_nodes)

    def set_node_factory(self, factory: Callable[[], WorkerNode]):
        """Set factory function for creating new nodes.
        
        Args:
            factory: Function that creates new worker nodes
        """
        self._node_factory = factory

    def set_load_balancer(self, load_balancer: LoadBalancer):
        """Set load balancer to manage.
        
        Args:
            load_balancer: Load balancer instance
        """
        self._load_balancer = load_balancer

    def start(self):
        """Start auto-scaling."""
        if not self._node_factory or not self._load_balancer:
            raise ValueError("Node factory and load balancer must be set")

        self._running = True
        self._thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._thread.start()
        logger.info("Auto-scaler started")

    def stop(self):
        """Stop auto-scaling."""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Auto-scaler stopped")

    def _scaling_loop(self):
        """Main auto-scaling loop."""
        while self._running:
            try:
                time.sleep(30)  # Check every 30 seconds
                self._evaluate_scaling()
            except Exception as e:
                logger.error("Error in auto-scaling loop", error=str(e))

    def _evaluate_scaling(self):
        """Evaluate if scaling is needed."""
        current_time = time.time()

        # Check cooldown period
        if current_time - self._last_scale_time < self.scale_cooldown:
            return

        # Calculate current load
        load_stats = self._calculate_load()
        self._load_history.append(load_stats)

        # Need at least some history for scaling decisions
        if len(self._load_history) < 5:
            return

        # Calculate average load over recent history
        avg_load = sum(stats["avg_load"] for stats in self._load_history[-5:]) / 5
        current_nodes = load_stats["healthy_nodes"]

        # Scaling decisions
        if avg_load > self.scale_up_threshold and current_nodes < self.max_nodes:
            self._scale_up()
        elif avg_load < self.scale_down_threshold and current_nodes > self.min_nodes:
            self._scale_down()

    def _calculate_load(self) -> Dict[str, float]:
        """Calculate current system load.
        
        Returns:
            Load statistics
        """
        lb_stats = self._load_balancer.get_stats()

        if lb_stats["healthy_nodes"] == 0:
            return {"avg_load": 0.0, "healthy_nodes": 0, "total_connections": 0}

        total_connections = sum(node["current_connections"] for node in lb_stats["nodes"] if node["is_healthy"])
        total_capacity = sum(50 for node in lb_stats["nodes"] if node["is_healthy"])  # Assume 50 connections per node

        avg_load = total_connections / total_capacity if total_capacity > 0 else 0.0

        return {
            "avg_load": avg_load,
            "healthy_nodes": lb_stats["healthy_nodes"],
            "total_connections": total_connections
        }

    def _scale_up(self):
        """Scale up by adding a new node."""
        try:
            new_node = self._node_factory()
            self._load_balancer.add_node(new_node)
            self._last_scale_time = time.time()

            logger.info("Scaled up", new_node_id=new_node.id)
            metrics.record_counter("autoscaler_scale_events", 1, tags={"direction": "up"})

        except Exception as e:
            logger.error("Failed to scale up", error=str(e))

    def _scale_down(self):
        """Scale down by removing a node."""
        try:
            lb_stats = self._load_balancer.get_stats()

            # Find node with least connections to remove
            nodes_to_consider = [node for node in lb_stats["nodes"] if node["is_healthy"]]
            if len(nodes_to_consider) <= self.min_nodes:
                return

            node_to_remove = min(nodes_to_consider, key=lambda n: n["current_connections"])

            self._load_balancer.remove_node(node_to_remove["id"])
            self._last_scale_time = time.time()

            logger.info("Scaled down", removed_node_id=node_to_remove["id"])
            metrics.record_counter("autoscaler_scale_events", 1, tags={"direction": "down"})

        except Exception as e:
            logger.error("Failed to scale down", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics.
        
        Returns:
            Statistics dictionary
        """
        recent_loads = list(self._load_history)[-10:]  # Last 10 measurements

        return {
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "last_scale_time": self._last_scale_time,
            "recent_loads": recent_loads,
            "running": self._running
        }


# Global instances
_load_balancer: Optional[LoadBalancer] = None
_auto_scaler: Optional[AutoScaler] = None


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler
