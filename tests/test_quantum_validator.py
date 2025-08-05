"""
Comprehensive tests for quantum planner validation and error handling.
"""

import pytest
import math
import hashlib
from unittest.mock import Mock, patch

from src.autogen_code_review_bot.quantum_planner import (
    QuantumTask, TaskPriority, TaskState, QuantumScheduler, QuantumTaskPlanner
)
from src.autogen_code_review_bot.quantum_validator import (
    ValidationError, QuantumError, CircularDependencyError,
    ValidationResult, TaskValidator, DependencyValidator, 
    QuantumSystemValidator, QuantumDataIntegrity, RobustQuantumPlanner
)


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(True, [], [])
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult(True, [], [])
        result.add_error("Test error")
        
        assert not result.is_valid
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult(True, [], [])
        result.add_warning("Test warning")
        
        assert result.is_valid  # Warnings don't invalidate
        assert "Test warning" in result.warnings
    
    def test_boolean_conversion(self):
        """Test using ValidationResult in boolean context."""
        valid_result = ValidationResult(True, [], [])
        invalid_result = ValidationResult(False, ["error"], [])
        
        assert bool(valid_result) is True
        assert bool(invalid_result) is False


class TestTaskValidator:
    """Test TaskValidator class."""
    
    def test_validate_task_id_valid(self):
        """Test valid task ID validation."""
        result = TaskValidator.validate_task_id("valid-task-id_123")
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_task_id_empty(self):
        """Test empty task ID validation."""
        result = TaskValidator.validate_task_id("")
        assert not result.is_valid
        assert "Task ID cannot be empty" in result.errors
    
    def test_validate_task_id_non_string(self):
        """Test non-string task ID validation."""
        result = TaskValidator.validate_task_id(123)
        assert not result.is_valid
        assert "Task ID must be a string" in result.errors
    
    def test_validate_task_id_too_long(self):
        """Test overly long task ID validation."""
        long_id = "a" * 101
        result = TaskValidator.validate_task_id(long_id)
        assert not result.is_valid
        assert "Task ID too long" in result.errors
    
    def test_validate_task_id_special_chars(self):
        """Test task ID with special characters."""
        result = TaskValidator.validate_task_id("task@#$%")
        assert result.is_valid  # Warnings don't invalidate
        assert len(result.warnings) > 0
        assert "special characters" in result.warnings[0]
    
    def test_validate_task_effort_valid(self):
        """Test valid effort validation."""
        result = TaskValidator.validate_task_effort(5.5)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_task_effort_non_numeric(self):
        """Test non-numeric effort validation."""
        result = TaskValidator.validate_task_effort("not a number")
        assert not result.is_valid
        assert "Effort must be a number" in result.errors
    
    def test_validate_task_effort_negative(self):
        """Test negative effort validation."""
        result = TaskValidator.validate_task_effort(-1.0)
        assert not result.is_valid
        assert "Effort must be positive" in result.errors
    
    def test_validate_task_effort_zero(self):
        """Test zero effort validation."""
        result = TaskValidator.validate_task_effort(0.0)
        assert not result.is_valid
        assert "Effort must be positive" in result.errors
    
    def test_validate_task_effort_large(self):
        """Test very large effort validation."""
        result = TaskValidator.validate_task_effort(1500.0)
        assert result.is_valid
        assert "Very large effort value" in result.warnings[0]
    
    def test_validate_task_effort_small(self):
        """Test very small effort validation."""
        result = TaskValidator.validate_task_effort(0.05)
        assert result.is_valid
        assert "Very small effort value" in result.warnings[0]
    
    def test_validate_quantum_properties_valid(self):
        """Test valid quantum properties validation."""
        task = QuantumTask(id="test", title="Test", description="")
        result = TaskValidator.validate_quantum_properties(task)
        assert result.is_valid
    
    def test_validate_quantum_properties_no_amplitudes(self):
        """Test task without priority amplitudes."""
        task = QuantumTask(id="test", title="Test", description="")
        task.priority_amplitudes = {}
        
        result = TaskValidator.validate_quantum_properties(task)
        assert not result.is_valid
        assert "Task must have priority amplitudes" in result.errors
    
    def test_validate_quantum_properties_not_normalized(self):
        """Test task with non-normalized amplitudes."""
        task = QuantumTask(id="test", title="Test", description="")
        # Artificially set non-normalized amplitudes
        task.priority_amplitudes = {
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.8,
            TaskPriority.LOW: 0.8
        }
        
        result = TaskValidator.validate_quantum_properties(task)
        assert not result.is_valid
        assert "Priority amplitudes not normalized" in result.errors[0]
    
    def test_validate_quantum_properties_negative_amplitude(self):
        """Test task with negative amplitude."""
        task = QuantumTask(id="test", title="Test", description="")
        task.priority_amplitudes[TaskPriority.HIGH] = -0.5
        
        result = TaskValidator.validate_quantum_properties(task)
        assert not result.is_valid
        assert "must be non-negative" in result.errors[0]
    
    def test_validate_quantum_properties_negative_coherence(self):
        """Test task with negative coherence time."""
        task = QuantumTask(id="test", title="Test", description="")
        task.coherence_time = -10.0
        
        result = TaskValidator.validate_quantum_properties(task)
        assert not result.is_valid
        assert "Coherence time must be positive" in result.errors


class TestDependencyValidator:
    """Test DependencyValidator class."""
    
    def test_detect_circular_dependencies_none(self):
        """Test detecting no circular dependencies."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        task1.add_dependency("task2")
        
        tasks = {"task1": task1, "task2": task2}
        result = DependencyValidator.detect_circular_dependencies(tasks)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_detect_circular_dependencies_simple(self):
        """Test detecting simple circular dependency."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        
        task1.add_dependency("task2")
        task2.add_dependency("task1")
        
        tasks = {"task1": task1, "task2": task2}
        result = DependencyValidator.detect_circular_dependencies(tasks)
        
        assert not result.is_valid
        assert "Circular dependency detected" in result.errors[0]
    
    def test_detect_circular_dependencies_complex(self):
        """Test detecting complex circular dependency."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        task3 = QuantumTask(id="task3", title="Task 3", description="")
        
        task1.add_dependency("task2")
        task2.add_dependency("task3")
        task3.add_dependency("task1")  # Creates cycle
        
        tasks = {"task1": task1, "task2": task2, "task3": task3}
        result = DependencyValidator.detect_circular_dependencies(tasks)
        
        assert not result.is_valid
        assert "Circular dependency detected" in result.errors[0]
    
    def test_validate_dependencies_exist_valid(self):
        """Test validating existing dependencies."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        task1.add_dependency("task2")
        
        tasks = {"task1": task1, "task2": task2}
        result = DependencyValidator.validate_dependencies_exist(tasks)
        
        assert result.is_valid
    
    def test_validate_dependencies_exist_missing(self):
        """Test validating missing dependencies."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task1.add_dependency("nonexistent")
        
        tasks = {"task1": task1}
        result = DependencyValidator.validate_dependencies_exist(tasks)
        
        assert not result.is_valid
        assert "depends on non-existent task" in result.errors[0]
    
    def test_validate_entanglements_valid(self):
        """Test validating symmetric entanglements."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        
        task1.entangle_with("task2")
        task2.entangle_with("task1")
        
        tasks = {"task1": task1, "task2": task2}
        result = DependencyValidator.validate_entanglements(tasks)
        
        assert result.is_valid
    
    def test_validate_entanglements_asymmetric(self):
        """Test detecting asymmetric entanglements."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        
        task1.entangle_with("task2")
        # task2 not entangled with task1
        
        tasks = {"task1": task1, "task2": task2}
        result = DependencyValidator.validate_entanglements(tasks)
        
        assert not result.is_valid
        assert "Asymmetric entanglement" in result.errors[0]
    
    def test_validate_entanglements_nonexistent(self):
        """Test detecting entanglement with nonexistent task."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task1.entangle_with("nonexistent")
        
        tasks = {"task1": task1}
        result = DependencyValidator.validate_entanglements(tasks)
        
        assert not result.is_valid
        assert "entangled with non-existent task" in result.errors[0]


class TestQuantumSystemValidator:
    """Test QuantumSystemValidator class."""
    
    def test_validate_system_empty(self):
        """Test validating empty system."""
        scheduler = QuantumScheduler()
        validator = QuantumSystemValidator()
        
        result = validator.validate_system(scheduler)
        
        assert result.is_valid
        assert "No tasks in system" in result.warnings
    
    def test_validate_system_valid(self):
        """Test validating valid system."""
        scheduler = QuantumScheduler()
        task1 = QuantumTask(id="task1", title="Task 1", description="Test task")
        task2 = QuantumTask(id="task2", title="Task 2", description="Another task")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        validator = QuantumSystemValidator()
        result = validator.validate_system(scheduler)
        
        assert result.is_valid
    
    def test_validate_system_with_errors(self):
        """Test validating system with errors."""
        scheduler = QuantumScheduler()
        
        # Create task with invalid properties
        task = QuantumTask(id="", title="", description="")  # Empty ID and title
        scheduler.add_task(task)
        
        validator = QuantumSystemValidator()
        result = validator.validate_system(scheduler)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_system_large(self):
        """Test validating very large system."""
        scheduler = QuantumScheduler()
        
        # Create many tasks
        for i in range(15000):  # Over the warning threshold
            task = QuantumTask(id=f"task{i}", title=f"Task {i}", description="")
            scheduler.add_task(task)
        
        validator = QuantumSystemValidator()
        result = validator.validate_system(scheduler)
        
        assert result.is_valid
        assert "Very large number of tasks" in result.warnings[0]
    
    def test_validate_task_comprehensive(self):
        """Test comprehensive task validation."""
        validator = QuantumSystemValidator()
        
        # Valid task
        valid_task = QuantumTask(id="valid", title="Valid Task", description="A valid task")
        result = validator.validate_task(valid_task)
        assert result.is_valid
        
        # Invalid task - no title
        invalid_task = QuantumTask(id="invalid", title="", description="No title")
        result = validator.validate_task(invalid_task)
        assert not result.is_valid
        assert "Task title cannot be empty" in result.errors
        
        # Task with warnings - long title and description
        long_task = QuantumTask(
            id="long", 
            title="A" * 250,  # Very long title
            description="B" * 1500  # Very long description
        )
        result = validator.validate_task(long_task)
        assert result.is_valid
        assert len(result.warnings) == 2  # Title and description warnings


class TestQuantumDataIntegrity:
    """Test QuantumDataIntegrity class."""
    
    def test_calculate_system_hash(self):
        """Test system hash calculation."""
        scheduler = QuantumScheduler()
        task1 = QuantumTask(id="task1", title="Task 1", description="First task")
        task2 = QuantumTask(id="task2", title="Task 2", description="Second task")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        hash1 = QuantumDataIntegrity.calculate_system_hash(scheduler)
        
        # Hash should be consistent
        hash2 = QuantumDataIntegrity.calculate_system_hash(scheduler)
        assert hash1 == hash2
        
        # Hash should change when system changes
        task3 = QuantumTask(id="task3", title="Task 3", description="Third task")
        scheduler.add_task(task3)
        hash3 = QuantumDataIntegrity.calculate_system_hash(scheduler)
        assert hash1 != hash3
        
        # Hash should be valid SHA256
        assert len(hash1) == 64  # SHA256 hex length
        assert all(c in '0123456789abcdef' for c in hash1)
    
    def test_verify_quantum_conservation_valid(self):
        """Test quantum conservation verification for valid system."""
        scheduler = QuantumScheduler()
        task = QuantumTask(id="task1", title="Task 1", description="")
        scheduler.add_task(task)
        
        result = QuantumDataIntegrity.verify_quantum_conservation(scheduler)
        assert result.is_valid
    
    def test_verify_quantum_conservation_invalid(self):
        """Test quantum conservation verification for invalid system."""
        scheduler = QuantumScheduler()
        task = QuantumTask(id="task1", title="Task 1", description="")
        
        # Artificially break conservation by setting non-normalized amplitudes
        task.priority_amplitudes = {
            TaskPriority.HIGH: 0.9,
            TaskPriority.MEDIUM: 0.9,
            TaskPriority.LOW: 0.9
        }
        scheduler.add_task(task)
        
        result = QuantumDataIntegrity.verify_quantum_conservation(scheduler)
        assert not result.is_valid
        assert "violates probability conservation" in result.errors[0]


class TestRobustQuantumPlanner:
    """Test RobustQuantumPlanner class."""
    
    def test_robust_planner_creation(self):
        """Test robust planner initialization."""
        planner = RobustQuantumPlanner()
        
        assert hasattr(planner, 'validator')
        assert hasattr(planner, 'integrity_checker')
        assert hasattr(planner, 'validation_history')
        assert len(planner.validation_history) == 0
    
    def test_create_task_with_validation(self):
        """Test task creation with validation."""
        planner = RobustQuantumPlanner()
        
        # Valid task creation
        task = planner.create_task("valid-task", "Valid Task", "A valid task", 2.5)
        assert task.id == "valid-task"
        assert "valid-task" in planner.scheduler.tasks
    
    def test_create_task_validation_failure(self):
        """Test task creation with validation failure."""
        planner = RobustQuantumPlanner()
        
        # Invalid task ID (empty)
        with pytest.raises(ValidationError) as exc_info:
            planner.create_task("", "Task", "Description")
        
        assert "Invalid task ID" in str(exc_info.value)
        assert "" not in planner.scheduler.tasks
    
    def test_create_task_duplicate_id(self):
        """Test creating task with duplicate ID."""
        planner = RobustQuantumPlanner()
        
        # Create first task
        planner.create_task("task1", "Task 1", "First task")
        
        # Try to create task with same ID
        with pytest.raises(ValidationError) as exc_info:
            planner.create_task("task1", "Task 1 Again", "Duplicate ID")
        
        assert "already exists" in str(exc_info.value)
    
    def test_create_task_entanglement_with_validation(self):
        """Test entanglement creation with validation."""
        planner = RobustQuantumPlanner()
        
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        
        # Valid entanglement
        planner.create_task_entanglement("task1", "task2")
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
    
    def test_create_task_entanglement_nonexistent(self):
        """Test entanglement with nonexistent task."""
        planner = RobustQuantumPlanner()
        
        task1 = planner.create_task("task1", "Task 1", "First task")
        
        with pytest.raises(ValidationError) as exc_info:
            planner.create_task_entanglement("task1", "nonexistent")
        
        assert "does not exist" in str(exc_info.value)
    
    def test_create_task_entanglement_self(self):
        """Test self-entanglement prevention."""
        planner = RobustQuantumPlanner()
        
        task1 = planner.create_task("task1", "Task 1", "First task")
        
        with pytest.raises(ValidationError) as exc_info:
            planner.create_task_entanglement("task1", "task1")
        
        assert "Cannot entangle task with itself" in str(exc_info.value)
    
    def test_create_task_entanglement_already_entangled(self):
        """Test creating entanglement that already exists."""
        planner = RobustQuantumPlanner()
        
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        
        # Create entanglement
        planner.create_task_entanglement("task1", "task2")
        
        # Try to create same entanglement again (should not raise error)
        planner.create_task_entanglement("task1", "task2")
        
        # Should still be entangled
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
    
    def test_generate_execution_plan_with_validation(self):
        """Test execution plan generation with validation."""
        planner = RobustQuantumPlanner()
        
        # Create valid tasks
        task1 = planner.create_task("task1", "Task 1", "First task", 1.0)
        task2 = planner.create_task("task2", "Task 2", "Second task", 2.0, ["task1"])
        
        # Generate plan
        plan = planner.generate_execution_plan()
        
        assert 'plan_id' in plan
        assert plan['total_tasks'] == 2
        assert len(planner.validation_history) == 1
    
    def test_generate_execution_plan_validation_failure(self):
        """Test execution plan generation with validation failure."""
        planner = RobustQuantumPlanner()
        
        # Create task with invalid dependency
        task1 = planner.create_task("task1", "Task 1", "First task")
        task1.dependencies.add("nonexistent")  # Manually add invalid dependency
        
        with pytest.raises(QuantumError) as exc_info:
            planner.generate_execution_plan()
        
        assert "System validation failed" in str(exc_info.value)
    
    def test_generate_execution_plan_conservation_failure(self):
        """Test execution plan generation with conservation failure."""
        planner = RobustQuantumPlanner()
        
        task1 = planner.create_task("task1", "Task 1", "First task")
        
        # Break conservation manually
        task1.priority_amplitudes = {
            TaskPriority.HIGH: 2.0,  # Invalid amplitude
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.LOW: 0.3
        }
        
        with pytest.raises(QuantumError) as exc_info:
            planner.generate_execution_plan()
        
        assert "Quantum conservation violated" in str(exc_info.value)
    
    def test_get_system_health(self):
        """Test system health reporting."""
        planner = RobustQuantumPlanner()
        
        # Create some tasks
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        planner.create_task_entanglement("task1", "task2")
        
        health = planner.get_system_health()
        
        assert 'system_valid' in health
        assert 'validation_errors' in health
        assert 'validation_warnings' in health
        assert 'quantum_conservation_valid' in health
        assert 'total_tasks' in health
        assert 'entangled_tasks' in health
        assert 'system_hash' in health
        
        assert health['system_valid'] is True
        assert health['total_tasks'] == 2
        assert health['entangled_tasks'] == 2
    
    def test_auto_repair_system(self):
        """Test automatic system repair."""
        planner = RobustQuantumPlanner()
        
        # Create tasks
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        
        # Introduce issues
        # 1. Non-normalized amplitudes
        task1.priority_amplitudes = {
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.8,
            TaskPriority.LOW: 0.5
        }
        
        # 2. Asymmetric entanglement
        task1.entangle_with("task2")
        # Don't entangle task2 with task1
        
        repair_result = planner.auto_repair_system()
        
        assert repair_result['repairs_performed'] > 0
        assert len(repair_result['repair_log']) > 0
        
        # Check that amplitudes are now normalized
        total_amplitude_squared = sum(amp ** 2 for amp in task1.priority_amplitudes.values())
        assert abs(total_amplitude_squared - 1.0) < 0.001
        
        # Check that entanglement is now symmetric
        assert "task1" in task2.entangled_tasks


class TestValidationIntegration:
    """Integration tests for validation system."""
    
    def test_end_to_end_validation_success(self):
        """Test complete validation flow with successful outcome."""
        planner = RobustQuantumPlanner()
        
        # Create a complex but valid project
        req_task = planner.create_task("requirements", "Requirements", "Gather requirements", 4.0)
        design_task = planner.create_task("design", "Design", "System design", 6.0, ["requirements"])
        dev_task = planner.create_task("development", "Development", "Implementation", 12.0, ["design"])
        test_task = planner.create_task("testing", "Testing", "QA testing", 8.0, ["development"])
        
        # Add entanglements
        planner.create_task_entanglement("design", "development")
        
        # Set priority biases
        planner.set_task_priority_bias("requirements", TaskPriority.CRITICAL, 0.4)
        planner.set_task_priority_bias("testing", TaskPriority.HIGH, 0.3)
        
        # Generate plan (should succeed with comprehensive validation)
        plan = planner.generate_execution_plan()
        
        # Verify plan
        assert plan['total_tasks'] == 4
        assert len(planner.validation_history) == 1
        
        # Check system health
        health = planner.get_system_health()
        assert health['system_valid'] is True
        assert health['quantum_conservation_valid'] is True
    
    def test_end_to_end_validation_with_repair(self):
        """Test validation with automatic repair."""
        planner = RobustQuantumPlanner()
        
        # Create tasks
        task1 = planner.create_task("task1", "Task 1", "First task")
        task2 = planner.create_task("task2", "Task 2", "Second task")
        
        # Introduce repairable issues
        task1.priority_amplitudes[TaskPriority.HIGH] = 1.5  # Too large
        task1.entangle_with("task2")  # Asymmetric entanglement
        
        # Auto-repair
        repair_result = planner.auto_repair_system()
        assert repair_result['repairs_performed'] > 0
        
        # Should now be able to generate plan
        plan = planner.generate_execution_plan()
        assert plan['total_tasks'] == 2
        
        # System should be healthy
        health = planner.get_system_health()
        assert health['system_valid'] is True
    
    def test_validation_error_messages(self):
        """Test that validation error messages are informative."""
        planner = RobustQuantumPlanner()
        
        # Test various validation errors
        test_cases = [
            ("", "Task 1", "Description", 1.0, None, "Invalid task ID"),
            ("task1", "Task 1", "Description", -1.0, None, "Invalid effort"),
            ("task1", "Task 1", "Description", "not_a_number", None, "Invalid effort"),
        ]
        
        for task_id, title, desc, effort, deps, expected_error in test_cases:
            with pytest.raises(ValidationError) as exc_info:
                planner.create_task(task_id, title, desc, effort, deps)
            
            assert expected_error in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])