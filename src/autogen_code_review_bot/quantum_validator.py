"""
Quantum Task Planner Validation and Error Handling

Provides comprehensive validation, error handling, and data integrity
for the quantum-inspired task planning system.
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .quantum_planner import (
    QuantumScheduler,
    QuantumTask,
    QuantumTaskPlanner,
    TaskState,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class QuantumError(Exception):
    """Custom exception for quantum system errors."""

    pass


class CircularDependencyError(ValidationError):
    """Exception for circular dependency detection."""

    pass


@dataclass
class ValidationResult:
    """Result of validation operations."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid


class TaskValidator:
    """Validator for individual tasks."""

    @staticmethod
    def validate_task_id(task_id: str) -> ValidationResult:
        """Validate task ID format and constraints."""
        result = ValidationResult(True, [], [])

        if not task_id:
            result.add_error("Task ID cannot be empty")
            return result

        if not isinstance(task_id, str):
            result.add_error("Task ID must be a string")
            return result

        if len(task_id) > 100:
            result.add_error("Task ID too long (max 100 characters)")

        if not task_id.replace("_", "").replace("-", "").replace(".", "").isalnum():
            result.add_warning(
                "Task ID contains special characters (recommended: alphanumeric, _, -, .)"
            )

        return result

    @staticmethod
    def validate_task_effort(effort: float) -> ValidationResult:
        """Validate estimated effort value."""
        result = ValidationResult(True, [], [])

        if not isinstance(effort, (int, float)):
            result.add_error("Effort must be a number")
            return result

        if effort <= 0:
            result.add_error("Effort must be positive")
        elif effort > 1000:
            result.add_warning("Very large effort value (>1000 hours)")
        elif effort < 0.1:
            result.add_warning("Very small effort value (<0.1 hours)")

        return result

    @staticmethod
    def validate_quantum_properties(task: QuantumTask) -> ValidationResult:
        """Validate quantum properties of a task."""
        result = ValidationResult(True, [], [])

        # Validate priority amplitudes
        if not task.priority_amplitudes:
            result.add_error("Task must have priority amplitudes")
            return result

        # Check amplitude normalization
        total_amplitude_squared = sum(
            amp**2 for amp in task.priority_amplitudes.values()
        )
        if abs(total_amplitude_squared - 1.0) > 0.001:
            result.add_error(
                f"Priority amplitudes not normalized: {total_amplitude_squared:.6f}"
            )

        # Validate individual amplitudes
        for priority, amplitude in task.priority_amplitudes.items():
            if not isinstance(amplitude, (int, float)):
                result.add_error(f"Amplitude for {priority.label} must be numeric")
            elif amplitude < 0:
                result.add_error(f"Amplitude for {priority.label} must be non-negative")
            elif amplitude > 1:
                result.add_warning(
                    f"Amplitude for {priority.label} is greater than 1.0"
                )

        # Validate coherence time
        if task.coherence_time <= 0:
            result.add_error("Coherence time must be positive")

        return result


class DependencyValidator:
    """Validator for task dependencies."""

    @staticmethod
    def detect_circular_dependencies(tasks: Dict[str, QuantumTask]) -> ValidationResult:
        """Detect circular dependencies using DFS."""
        result = ValidationResult(True, [], [])

        def has_cycle(
            task_id: str, visited: Set[str], rec_stack: Set[str]
        ) -> Optional[List[str]]:
            """DFS to detect cycles and return the cycle path."""
            if task_id in rec_stack:
                return [task_id]  # Found a cycle

            if task_id in visited:
                return None

            visited.add(task_id)
            rec_stack.add(task_id)

            if task_id in tasks:
                for dep_id in tasks[task_id].dependencies:
                    cycle_path = has_cycle(dep_id, visited, rec_stack)
                    if cycle_path is not None:
                        if cycle_path[0] == task_id:
                            # Complete cycle found
                            return cycle_path
                        else:
                            # Part of larger cycle
                            return [task_id] + cycle_path

            rec_stack.remove(task_id)
            return None

        visited = set()
        for task_id in tasks:
            if task_id not in visited:
                cycle = has_cycle(task_id, visited, set())
                if cycle:
                    cycle_str = " -> ".join(cycle + [cycle[0]])
                    result.add_error(f"Circular dependency detected: {cycle_str}")

        return result

    @staticmethod
    def validate_dependencies_exist(tasks: Dict[str, QuantumTask]) -> ValidationResult:
        """Validate that all dependencies reference existing tasks."""
        result = ValidationResult(True, [], [])

        for task_id, task in tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in tasks:
                    result.add_error(
                        f"Task '{task_id}' depends on non-existent task '{dep_id}'"
                    )

        return result

    @staticmethod
    def validate_entanglements(tasks: Dict[str, QuantumTask]) -> ValidationResult:
        """Validate quantum entanglements are symmetric and valid."""
        result = ValidationResult(True, [], [])

        for task_id, task in tasks.items():
            for entangled_id in task.entangled_tasks:
                if entangled_id not in tasks:
                    result.add_error(
                        f"Task '{task_id}' entangled with non-existent task '{entangled_id}'"
                    )
                elif task_id not in tasks[entangled_id].entangled_tasks:
                    result.add_error(
                        f"Asymmetric entanglement: '{task_id}' -> '{entangled_id}' but not vice versa"
                    )

        return result


class QuantumSystemValidator:
    """Validator for the entire quantum system."""

    def __init__(self):
        self.task_validator = TaskValidator()
        self.dependency_validator = DependencyValidator()

    def validate_system(self, scheduler: QuantumScheduler) -> ValidationResult:
        """Comprehensive validation of the quantum system."""
        result = ValidationResult(True, [], [])

        # Validate individual tasks
        for task_id, task in scheduler.tasks.items():
            task_result = self.validate_task(task)
            result.errors.extend(task_result.errors)
            result.warnings.extend(task_result.warnings)
            if not task_result.is_valid:
                result.is_valid = False

        # Validate dependencies
        dep_result = self.dependency_validator.validate_dependencies_exist(
            scheduler.tasks
        )
        result.errors.extend(dep_result.errors)
        result.warnings.extend(dep_result.warnings)
        if not dep_result.is_valid:
            result.is_valid = False

        # Check for circular dependencies
        circular_result = self.dependency_validator.detect_circular_dependencies(
            scheduler.tasks
        )
        result.errors.extend(circular_result.errors)
        result.warnings.extend(circular_result.warnings)
        if not circular_result.is_valid:
            result.is_valid = False

        # Validate entanglements
        entangle_result = self.dependency_validator.validate_entanglements(
            scheduler.tasks
        )
        result.errors.extend(entangle_result.errors)
        result.warnings.extend(entangle_result.warnings)
        if not entangle_result.is_valid:
            result.is_valid = False

        # System-level validations
        if len(scheduler.tasks) == 0:
            result.add_warning("No tasks in system")
        elif len(scheduler.tasks) > 10000:
            result.add_warning(
                "Very large number of tasks (>10000) - performance may be impacted"
            )

        return result

    def validate_task(self, task: QuantumTask) -> ValidationResult:
        """Validate a single task comprehensively."""
        result = ValidationResult(True, [], [])

        # Basic validations
        id_result = self.task_validator.validate_task_id(task.id)
        effort_result = self.task_validator.validate_task_effort(task.estimated_effort)
        quantum_result = self.task_validator.validate_quantum_properties(task)

        # Combine results
        for validation_result in [id_result, effort_result, quantum_result]:
            result.errors.extend(validation_result.errors)
            result.warnings.extend(validation_result.warnings)
            if not validation_result.is_valid:
                result.is_valid = False

        # Task-specific validations
        if not task.title:
            result.add_error("Task title cannot be empty")
        elif len(task.title) > 200:
            result.add_warning("Task title is very long (>200 characters)")

        if len(task.description) > 1000:
            result.add_warning("Task description is very long (>1000 characters)")

        return result


class QuantumDataIntegrity:
    """Data integrity and consistency checker."""

    @staticmethod
    def calculate_system_hash(scheduler: QuantumScheduler) -> str:
        """Calculate hash of entire system state for integrity checking."""
        system_data = {
            "tasks": {
                task_id: {
                    "title": task.title,
                    "description": task.description,
                    "effort": task.estimated_effort,
                    "dependencies": sorted(list(task.dependencies)),
                    "entangled_tasks": sorted(list(task.entangled_tasks)),
                    "priority_amplitudes": {
                        p.label: amp for p, amp in task.priority_amplitudes.items()
                    },
                }
                for task_id, task in scheduler.tasks.items()
            },
            "quantum_circuits": sorted(
                [sorted(circuit) for circuit in scheduler.quantum_circuits]
            ),
        }

        system_json = json.dumps(system_data, sort_keys=True)
        return hashlib.sha256(system_json.encode()).hexdigest()

    @staticmethod
    def verify_quantum_conservation(scheduler: QuantumScheduler) -> ValidationResult:
        """Verify quantum conservation laws are maintained."""
        result = ValidationResult(True, [], [])

        for task_id, task in scheduler.tasks.items():
            # Check probability conservation (amplitudes squared sum to 1)
            total_prob = sum(amp**2 for amp in task.priority_amplitudes.values())
            if abs(total_prob - 1.0) > 0.001:
                result.add_error(
                    f"Task '{task_id}' violates probability conservation: {total_prob:.6f}"
                )

        return result


class RobustQuantumPlanner(QuantumTaskPlanner):
    """Enhanced quantum planner with robust error handling and validation."""

    def __init__(self):
        super().__init__()
        self.validator = QuantumSystemValidator()
        self.integrity_checker = QuantumDataIntegrity()
        self._last_system_hash = None
        self.validation_history: List[Dict] = []

    def create_task(
        self,
        task_id: str,
        title: str,
        description: str,
        estimated_effort: float = 1.0,
        dependencies: Optional[List[str]] = None,
    ) -> QuantumTask:
        """Create task with comprehensive validation."""
        # Validate inputs before creating task
        id_result = TaskValidator.validate_task_id(task_id)
        if not id_result:
            raise ValidationError(f"Invalid task ID: {', '.join(id_result.errors)}")

        effort_result = TaskValidator.validate_task_effort(estimated_effort)
        if not effort_result:
            raise ValidationError(f"Invalid effort: {', '.join(effort_result.errors)}")

        # Check for duplicate task ID
        if task_id in self.scheduler.tasks:
            raise ValidationError(f"Task ID '{task_id}' already exists")

        # Create task
        task = super().create_task(
            task_id, title, description, estimated_effort, dependencies
        )

        # Validate the created task
        task_result = self.validator.validate_task(task)
        if not task_result:
            # Remove the invalid task
            del self.scheduler.tasks[task_id]
            raise ValidationError(
                f"Task creation failed validation: {', '.join(task_result.errors)}"
            )

        logger.info(f"Created validated quantum task: {task_id}")
        return task

    def create_task_entanglement(self, task1_id: str, task2_id: str) -> None:
        """Create entanglement with validation."""
        if task1_id not in self.scheduler.tasks:
            raise ValidationError(f"Task '{task1_id}' does not exist")

        if task2_id not in self.scheduler.tasks:
            raise ValidationError(f"Task '{task2_id}' does not exist")

        if task1_id == task2_id:
            raise ValidationError("Cannot entangle task with itself")

        # Check if already entangled
        if task2_id in self.scheduler.tasks[task1_id].entangled_tasks:
            logger.warning(f"Tasks {task1_id} and {task2_id} are already entangled")
            return

        super().create_task_entanglement(task1_id, task2_id)
        logger.info(f"Created validated entanglement: {task1_id} <-> {task2_id}")

    def generate_execution_plan(self) -> Dict:
        """Generate execution plan with full validation."""
        start_time = time.time()

        # Pre-generation validation
        validation_result = self.validator.validate_system(self.scheduler)

        if not validation_result:
            error_msg = (
                f"System validation failed: {', '.join(validation_result.errors)}"
            )
            logger.error(error_msg)
            raise QuantumError(error_msg)

        if validation_result.warnings:
            logger.warning(f"System warnings: {', '.join(validation_result.warnings)}")

        # Check data integrity
        integrity_result = self.integrity_checker.verify_quantum_conservation(
            self.scheduler
        )
        if not integrity_result:
            error_msg = (
                f"Quantum conservation violated: {', '.join(integrity_result.errors)}"
            )
            logger.error(error_msg)
            raise QuantumError(error_msg)

        try:
            # Generate plan with error handling
            plan = super().generate_execution_plan()

            # Post-generation validation
            current_hash = self.integrity_checker.calculate_system_hash(self.scheduler)
            if self._last_system_hash and self._last_system_hash != current_hash:
                logger.info("System state changed during planning")
            self._last_system_hash = current_hash

            # Record validation history
            validation_record = {
                "timestamp": time.time(),
                "validation_time": time.time() - start_time,
                "errors_count": len(validation_result.errors),
                "warnings_count": len(validation_result.warnings),
                "system_hash": current_hash,
                "plan_id": plan["plan_id"],
            }
            self.validation_history.append(validation_record)

            logger.info(f"Generated validated execution plan: {plan['plan_id']}")
            return plan

        except Exception as e:
            logger.error(f"Plan generation failed: {str(e)}")
            raise QuantumError(f"Failed to generate execution plan: {str(e)}") from e

    def get_system_health(self) -> Dict:
        """Get comprehensive system health report."""
        validation_result = self.validator.validate_system(self.scheduler)
        integrity_result = self.integrity_checker.verify_quantum_conservation(
            self.scheduler
        )

        return {
            "system_valid": validation_result.is_valid,
            "validation_errors": validation_result.errors,
            "validation_warnings": validation_result.warnings,
            "quantum_conservation_valid": integrity_result.is_valid,
            "conservation_errors": integrity_result.errors,
            "total_tasks": len(self.scheduler.tasks),
            "entangled_tasks": sum(
                1
                for task in self.scheduler.tasks.values()
                if task.state == TaskState.ENTANGLED
            ),
            "collapsed_tasks": sum(
                1
                for task in self.scheduler.tasks.values()
                if task.state == TaskState.COLLAPSED
            ),
            "quantum_circuits": len(self.scheduler.quantum_circuits),
            "validation_history_length": len(self.validation_history),
            "system_hash": self.integrity_checker.calculate_system_hash(self.scheduler),
        }

    def auto_repair_system(self) -> Dict:
        """Attempt to automatically repair common system issues."""
        repair_log = []

        # Fix amplitude normalization issues
        for task_id, task in self.scheduler.tasks.items():
            total_amplitude_squared = sum(
                amp**2 for amp in task.priority_amplitudes.values()
            )
            if abs(total_amplitude_squared - 1.0) > 0.001:
                # Renormalize amplitudes
                norm_factor = math.sqrt(total_amplitude_squared)
                task.priority_amplitudes = {
                    priority: amp / norm_factor
                    for priority, amp in task.priority_amplitudes.items()
                }
                repair_log.append(f"Renormalized amplitudes for task '{task_id}'")

        # Fix asymmetric entanglements
        for task_id, task in self.scheduler.tasks.items():
            for entangled_id in list(task.entangled_tasks):
                if entangled_id in self.scheduler.tasks:
                    other_task = self.scheduler.tasks[entangled_id]
                    if task_id not in other_task.entangled_tasks:
                        other_task.entangled_tasks.add(task_id)
                        other_task.state = TaskState.ENTANGLED
                        repair_log.append(
                            f"Fixed asymmetric entanglement: {entangled_id} -> {task_id}"
                        )
                else:
                    # Remove entanglement to non-existent task
                    task.entangled_tasks.remove(entangled_id)
                    repair_log.append(
                        f"Removed entanglement to non-existent task '{entangled_id}' from '{task_id}'"
                    )

        return {
            "repairs_performed": len(repair_log),
            "repair_log": repair_log,
            "system_health": self.get_system_health(),
        }
