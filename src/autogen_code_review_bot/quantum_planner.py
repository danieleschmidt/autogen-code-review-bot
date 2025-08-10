"""
Quantum-Inspired Task Planner

Implements quantum computing principles for intelligent task planning:
- Superposition: Tasks exist in multiple priority states simultaneously
- Entanglement: Task dependencies are quantum-linked
- Interference: Task outcomes influence each other probabilistically
- Measurement: Collapsing quantum states to concrete execution plans
"""

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels with quantum probability amplitudes."""
    CRITICAL = (1.0, "Critical")
    HIGH = (0.8, "High")
    MEDIUM = (0.6, "Medium")
    LOW = (0.4, "Low")
    DEFERRED = (0.2, "Deferred")

    def __init__(self, amplitude: float, label: str):
        self.amplitude = amplitude
        self.label = label


class TaskState(Enum):
    """Quantum task states."""
    SUPERPOSITION = "superposition"  # Multiple states simultaneously
    COLLAPSED = "collapsed"          # Definite state after measurement
    ENTANGLED = "entangled"         # Linked to other tasks
    ISOLATED = "isolated"           # Independent task


@dataclass
class QuantumTask:
    """A task with quantum properties."""
    id: str
    title: str
    description: str
    estimated_effort: float = 1.0
    dependencies: Set[str] = field(default_factory=set)

    # Quantum properties
    priority_amplitudes: Dict[TaskPriority, float] = field(default_factory=dict)
    state: TaskState = TaskState.SUPERPOSITION
    entangled_tasks: Set[str] = field(default_factory=set)
    coherence_time: float = 100.0  # How long task maintains quantum properties

    # Classical properties (after measurement)
    measured_priority: Optional[TaskPriority] = None
    execution_order: Optional[int] = None

    def __post_init__(self):
        """Initialize quantum properties."""
        if not self.priority_amplitudes:
            # Initialize with equal superposition
            base_amplitude = 1.0 / math.sqrt(len(TaskPriority))
            self.priority_amplitudes = dict.fromkeys(TaskPriority, base_amplitude)

    def add_dependency(self, task_id: str) -> None:
        """Add a task dependency."""
        self.dependencies.add(task_id)

    def entangle_with(self, task_id: str) -> None:
        """Create quantum entanglement with another task."""
        self.entangled_tasks.add(task_id)
        self.state = TaskState.ENTANGLED

    def measure_priority(self) -> TaskPriority:
        """Collapse quantum superposition to definite priority."""
        if self.state == TaskState.COLLAPSED and self.measured_priority:
            return self.measured_priority

        # Calculate probability distribution
        total_amplitude = sum(amp ** 2 for amp in self.priority_amplitudes.values())
        probabilities = {
            priority: (amp ** 2) / total_amplitude
            for priority, amp in self.priority_amplitudes.items()
        }

        # Quantum measurement (probabilistic collapse)
        rand_val = random.random()
        cumulative_prob = 0.0

        for priority, prob in probabilities.items():
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                self.measured_priority = priority
                self.state = TaskState.COLLAPSED
                logger.debug(f"Task {self.id} priority collapsed to {priority.label}")
                return priority

        # Fallback (shouldn't reach here)
        self.measured_priority = TaskPriority.MEDIUM
        self.state = TaskState.COLLAPSED
        return self.measured_priority

    def apply_interference(self, other_task: 'QuantumTask', coupling_strength: float = 0.1) -> None:
        """Apply quantum interference with another task."""
        if other_task.id in self.entangled_tasks:
            # Stronger interference for entangled tasks
            coupling_strength *= 2.0

        # Modify priority amplitudes based on interference
        for priority in TaskPriority:
            phase_shift = coupling_strength * other_task.priority_amplitudes.get(priority, 0)
            self.priority_amplitudes[priority] += phase_shift

        # Normalize amplitudes
        total_norm = math.sqrt(sum(amp ** 2 for amp in self.priority_amplitudes.values()))
        if total_norm > 0:
            self.priority_amplitudes = {
                priority: amp / total_norm
                for priority, amp in self.priority_amplitudes.items()
            }


class QuantumScheduler:
    """Quantum-inspired task scheduler using superposition and entanglement."""

    def __init__(self):
        self.tasks: Dict[str, QuantumTask] = {}
        self.quantum_circuits: List[List[str]] = []  # Entanglement patterns
        self.measurement_history: List[Dict] = []

    def add_task(self, task: QuantumTask) -> None:
        """Add a task to the quantum system."""
        self.tasks[task.id] = task
        logger.info(f"Added quantum task: {task.id}")

    def create_entanglement(self, task1_id: str, task2_id: str) -> None:
        """Create quantum entanglement between two tasks."""
        if task1_id in self.tasks and task2_id in self.tasks:
            self.tasks[task1_id].entangle_with(task2_id)
            self.tasks[task2_id].entangle_with(task1_id)
            self.quantum_circuits.append([task1_id, task2_id])
            logger.info(f"Created entanglement: {task1_id} <-> {task2_id}")

    def apply_quantum_interference(self) -> None:
        """Apply interference patterns across all tasks."""
        task_list = list(self.tasks.values())

        for i, task1 in enumerate(task_list):
            for task2 in task_list[i+1:]:
                # Calculate coupling strength based on similarity and dependencies
                coupling = self._calculate_coupling_strength(task1, task2)
                if coupling > 0:
                    task1.apply_interference(task2, coupling)
                    task2.apply_interference(task1, coupling)

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

        return min(coupling, 1.0)  # Cap at 1.0

    def measure_and_schedule(self) -> List[QuantumTask]:
        """Perform quantum measurement and generate execution schedule."""
        # Apply quantum interference
        self.apply_quantum_interference()

        # Measure all tasks to collapse their states
        measured_tasks = []
        for task in self.tasks.values():
            priority = task.measure_priority()
            measured_tasks.append(task)

        # Sort by measured priority and dependencies
        scheduled_tasks = self._topological_sort_with_priority(measured_tasks)

        # Record measurement
        measurement_record = {
            'timestamp': time.time(),
            'tasks_measured': len(measured_tasks),
            'entangled_pairs': len(self.quantum_circuits),
            'schedule_length': len(scheduled_tasks)
        }
        self.measurement_history.append(measurement_record)

        logger.info(f"Quantum measurement completed: {len(scheduled_tasks)} tasks scheduled")
        return scheduled_tasks

    def _topological_sort_with_priority(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Topological sort considering both dependencies and quantum-measured priorities."""
        # Create adjacency list for dependencies
        in_degree = {task.id: 0 for task in tasks}
        adj_list = {task.id: [] for task in tasks}

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in adj_list:
                    adj_list[dep_id].append(task.id)
                    in_degree[task.id] += 1

        # Priority queues for each priority level
        priority_queues = {priority: [] for priority in TaskPriority}

        # Initialize with tasks having no dependencies
        for task in tasks:
            if in_degree[task.id] == 0:
                priority = task.measured_priority or TaskPriority.MEDIUM
                priority_queues[priority].append(task)

        scheduled = []
        execution_order = 0

        while any(priority_queues.values()):
            # Process highest priority tasks first
            for priority in sorted(TaskPriority, key=lambda p: p.amplitude, reverse=True):
                if priority_queues[priority]:
                    task = priority_queues[priority].pop(0)
                    task.execution_order = execution_order
                    scheduled.append(task)
                    execution_order += 1

                    # Update dependent tasks
                    for dependent_id in adj_list[task.id]:
                        in_degree[dependent_id] -= 1
                        if in_degree[dependent_id] == 0:
                            dependent_task = next(t for t in tasks if t.id == dependent_id)
                            dep_priority = dependent_task.measured_priority or TaskPriority.MEDIUM
                            priority_queues[dep_priority].append(dependent_task)
                    break

        return scheduled

    def get_quantum_state_summary(self) -> Dict:
        """Get summary of current quantum system state."""
        total_tasks = len(self.tasks)
        entangled_tasks = sum(1 for task in self.tasks.values() if task.state == TaskState.ENTANGLED)
        collapsed_tasks = sum(1 for task in self.tasks.values() if task.state == TaskState.COLLAPSED)

        return {
            'total_tasks': total_tasks,
            'entangled_tasks': entangled_tasks,
            'collapsed_tasks': collapsed_tasks,
            'quantum_circuits': len(self.quantum_circuits),
            'measurement_history_length': len(self.measurement_history)
        }


class QuantumTaskPlanner:
    """Main quantum-inspired task planning interface."""

    def __init__(self):
        self.scheduler = QuantumScheduler()
        self.planning_history: List[Dict] = []

    def create_task(self, task_id: str, title: str, description: str,
                   estimated_effort: float = 1.0, dependencies: Optional[List[str]] = None) -> QuantumTask:
        """Create a new quantum task."""
        dependencies_set = set(dependencies or [])
        task = QuantumTask(
            id=task_id,
            title=title,
            description=description,
            estimated_effort=estimated_effort,
            dependencies=dependencies_set
        )
        self.scheduler.add_task(task)
        return task

    def set_task_priority_bias(self, task_id: str, priority: TaskPriority, bias_strength: float = 0.3) -> None:
        """Bias a task toward a specific priority (quantum amplitude manipulation)."""
        if task_id in self.scheduler.tasks:
            task = self.scheduler.tasks[task_id]

            # Increase amplitude for target priority
            task.priority_amplitudes[priority] += bias_strength

            # Normalize amplitudes
            total_norm = math.sqrt(sum(amp ** 2 for amp in task.priority_amplitudes.values()))
            if total_norm > 0:
                task.priority_amplitudes = {
                    p: amp / total_norm for p, amp in task.priority_amplitudes.items()
                }

            logger.info(f"Applied priority bias to task {task_id}: {priority.label} (+{bias_strength})")

    def create_task_entanglement(self, task1_id: str, task2_id: str) -> None:
        """Create quantum entanglement between tasks."""
        self.scheduler.create_entanglement(task1_id, task2_id)

    def generate_execution_plan(self) -> Dict:
        """Generate quantum-optimized execution plan."""
        start_time = time.time()

        # Get quantum state before measurement
        initial_state = self.scheduler.get_quantum_state_summary()

        # Perform quantum measurement and scheduling
        scheduled_tasks = self.scheduler.measure_and_schedule()

        # Calculate plan metrics
        total_effort = sum(task.estimated_effort for task in scheduled_tasks)
        critical_path_length = self._calculate_critical_path_length(scheduled_tasks)

        planning_time = time.time() - start_time

        execution_plan = {
            'plan_id': f"qplan_{int(time.time())}",
            'generation_time': planning_time,
            'total_tasks': len(scheduled_tasks),
            'total_estimated_effort': total_effort,
            'critical_path_length': critical_path_length,
            'quantum_state_initial': initial_state,
            'quantum_state_final': self.scheduler.get_quantum_state_summary(),
            'scheduled_tasks': [
                {
                    'id': task.id,
                    'title': task.title,
                    'description': task.description,
                    'estimated_effort': task.estimated_effort,
                    'measured_priority': task.measured_priority.label if task.measured_priority else None,
                    'execution_order': task.execution_order,
                    'dependencies': list(task.dependencies),
                    'entangled_with': list(task.entangled_tasks)
                }
                for task in scheduled_tasks
            ]
        }

        self.planning_history.append(execution_plan)
        logger.info(f"Generated quantum execution plan with {len(scheduled_tasks)} tasks in {planning_time:.3f}s")

        return execution_plan

    def _calculate_critical_path_length(self, tasks: List[QuantumTask]) -> float:
        """Calculate critical path length through task dependencies."""
        task_dict = {task.id: task for task in tasks}
        memo = {}

        def longest_path(task_id: str) -> float:
            if task_id in memo:
                return memo[task_id]

            if task_id not in task_dict:
                return 0.0

            task = task_dict[task_id]
            max_dep_length = 0.0

            for dep_id in task.dependencies:
                dep_length = longest_path(dep_id)
                max_dep_length = max(max_dep_length, dep_length)

            memo[task_id] = task.estimated_effort + max_dep_length
            return memo[task_id]

        return max(longest_path(task.id) for task in tasks) if tasks else 0.0

    def export_plan_to_json(self, plan: Dict, filename: str) -> None:
        """Export execution plan to JSON file."""
        with open(filename, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        logger.info(f"Execution plan exported to {filename}")

    def get_planning_analytics(self) -> Dict:
        """Get analytics on planning performance and quantum behavior."""
        if not self.planning_history:
            return {'message': 'No planning history available'}

        total_plans = len(self.planning_history)
        avg_planning_time = sum(plan['generation_time'] for plan in self.planning_history) / total_plans
        avg_tasks_per_plan = sum(plan['total_tasks'] for plan in self.planning_history) / total_plans

        return {
            'total_plans_generated': total_plans,
            'average_planning_time_seconds': avg_planning_time,
            'average_tasks_per_plan': avg_tasks_per_plan,
            'quantum_measurements_performed': len(self.scheduler.measurement_history),
            'current_quantum_state': self.scheduler.get_quantum_state_summary()
        }
