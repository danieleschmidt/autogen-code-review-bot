"""
Comprehensive tests for quantum-inspired task planner.
"""

import pytest
import math
import time
from unittest.mock import Mock, patch

from src.autogen_code_review_bot.quantum_planner import (
    QuantumTask, TaskPriority, TaskState, QuantumScheduler, QuantumTaskPlanner
)


class TestTaskPriority:
    """Test TaskPriority enum."""
    
    def test_priority_amplitudes(self):
        """Test priority amplitude values."""
        assert TaskPriority.CRITICAL.amplitude == 1.0
        assert TaskPriority.HIGH.amplitude == 0.8
        assert TaskPriority.MEDIUM.amplitude == 0.6
        assert TaskPriority.LOW.amplitude == 0.4
        assert TaskPriority.DEFERRED.amplitude == 0.2
    
    def test_priority_labels(self):
        """Test priority labels."""
        assert TaskPriority.CRITICAL.label == "Critical"
        assert TaskPriority.HIGH.label == "High"
        assert TaskPriority.MEDIUM.label == "Medium"
        assert TaskPriority.LOW.label == "Low"
        assert TaskPriority.DEFERRED.label == "Deferred"


class TestQuantumTask:
    """Test QuantumTask class."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = QuantumTask(
            id="test-task",
            title="Test Task",
            description="A test task",
            estimated_effort=2.5
        )
        
        assert task.id == "test-task"
        assert task.title == "Test Task"
        assert task.description == "A test task"
        assert task.estimated_effort == 2.5
        assert task.state == TaskState.SUPERPOSITION
        assert len(task.dependencies) == 0
        assert len(task.entangled_tasks) == 0
    
    def test_amplitude_initialization(self):
        """Test quantum amplitude initialization."""
        task = QuantumTask(id="test", title="Test", description="")
        
        # Should have amplitudes for all priorities
        assert len(task.priority_amplitudes) == len(TaskPriority)
        
        # Should be normalized (sum of squares = 1)
        total_amplitude_squared = sum(amp ** 2 for amp in task.priority_amplitudes.values())
        assert abs(total_amplitude_squared - 1.0) < 0.001
    
    def test_add_dependency(self):
        """Test adding dependencies."""
        task = QuantumTask(id="test", title="Test", description="")
        
        task.add_dependency("dep1")
        task.add_dependency("dep2")
        
        assert "dep1" in task.dependencies
        assert "dep2" in task.dependencies
        assert len(task.dependencies) == 2
    
    def test_entangle_with(self):
        """Test quantum entanglement."""
        task = QuantumTask(id="test", title="Test", description="")
        
        task.entangle_with("other-task")
        
        assert "other-task" in task.entangled_tasks
        assert task.state == TaskState.ENTANGLED
    
    def test_measure_priority(self):
        """Test priority measurement (quantum collapse)."""
        task = QuantumTask(id="test", title="Test", description="")
        
        # First measurement should collapse the superposition
        priority1 = task.measure_priority()
        assert isinstance(priority1, TaskPriority)
        assert task.state == TaskState.COLLAPSED
        assert task.measured_priority == priority1
        
        # Second measurement should return the same result
        priority2 = task.measure_priority()
        assert priority2 == priority1
    
    def test_apply_interference(self):
        """Test quantum interference between tasks."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        
        # Store original amplitudes
        original_amplitudes = task1.priority_amplitudes.copy()
        
        # Apply interference
        task1.apply_interference(task2, coupling_strength=0.2)
        
        # Amplitudes should have changed
        assert task1.priority_amplitudes != original_amplitudes
        
        # Should still be normalized
        total_amplitude_squared = sum(amp ** 2 for amp in task1.priority_amplitudes.values())
        assert abs(total_amplitude_squared - 1.0) < 0.001
    
    def test_entangled_interference(self):
        """Test stronger interference for entangled tasks."""
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        
        # Entangle the tasks
        task1.entangle_with("task2")
        
        # Store original amplitudes
        original_amplitudes = task1.priority_amplitudes.copy()
        
        # Apply interference (should be stronger due to entanglement)
        task1.apply_interference(task2, coupling_strength=0.1)
        
        # Check that amplitudes changed significantly
        changes = [
            abs(task1.priority_amplitudes[p] - original_amplitudes[p])
            for p in TaskPriority
        ]
        max_change = max(changes)
        assert max_change > 0.05  # Significant change due to entanglement


class TestQuantumScheduler:
    """Test QuantumScheduler class."""
    
    def test_scheduler_creation(self):
        """Test scheduler initialization."""
        scheduler = QuantumScheduler()
        
        assert len(scheduler.tasks) == 0
        assert len(scheduler.quantum_circuits) == 0
        assert len(scheduler.measurement_history) == 0
    
    def test_add_task(self):
        """Test adding tasks to scheduler."""
        scheduler = QuantumScheduler()
        task = QuantumTask(id="test", title="Test", description="")
        
        scheduler.add_task(task)
        
        assert "test" in scheduler.tasks
        assert scheduler.tasks["test"] == task
    
    def test_create_entanglement(self):
        """Test creating quantum entanglement."""
        scheduler = QuantumScheduler()
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        scheduler.create_entanglement("task1", "task2")
        
        # Check entanglement is bidirectional
        assert "task2" in scheduler.tasks["task1"].entangled_tasks
        assert "task1" in scheduler.tasks["task2"].entangled_tasks
        assert ["task1", "task2"] in scheduler.quantum_circuits
    
    def test_calculate_coupling_strength(self):
        """Test coupling strength calculation."""
        scheduler = QuantumScheduler()
        
        # Create tasks with different relationships
        task1 = QuantumTask(id="task1", title="Task 1", description="", estimated_effort=2.0)
        task2 = QuantumTask(id="task2", title="Task 2", description="", estimated_effort=2.1)
        task3 = QuantumTask(id="task3", title="Task 3", description="", estimated_effort=5.0)
        
        # Add dependency
        task1.add_dependency("task2")
        
        # Test dependency coupling
        coupling_dep = scheduler._calculate_coupling_strength(task1, task2)
        assert coupling_dep >= 0.3  # Should include dependency coupling
        
        # Test similarity coupling
        coupling_sim = scheduler._calculate_coupling_strength(task1, task2)
        coupling_diff = scheduler._calculate_coupling_strength(task1, task3)
        assert coupling_sim > coupling_diff  # More similar effort should have higher coupling
        
        # Test entanglement coupling
        task1.entangle_with("task3")
        coupling_entangled = scheduler._calculate_coupling_strength(task1, task3)
        assert coupling_entangled >= 0.5  # Should include entanglement coupling
    
    def test_measure_and_schedule(self):
        """Test quantum measurement and scheduling."""
        scheduler = QuantumScheduler()
        
        # Create tasks with dependencies
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        task3 = QuantumTask(id="task3", title="Task 3", description="")
        
        task2.add_dependency("task1")
        task3.add_dependency("task2")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        
        # Measure and schedule
        scheduled_tasks = scheduler.measure_and_schedule()
        
        assert len(scheduled_tasks) == 3
        assert all(task.state == TaskState.COLLAPSED for task in scheduled_tasks)
        assert all(task.measured_priority is not None for task in scheduled_tasks)
        assert all(task.execution_order is not None for task in scheduled_tasks)
        
        # Check dependency ordering
        task_order = {task.id: task.execution_order for task in scheduled_tasks}
        assert task_order["task1"] < task_order["task2"]
        assert task_order["task2"] < task_order["task3"]
        
        # Check measurement history
        assert len(scheduler.measurement_history) == 1
        assert scheduler.measurement_history[0]['tasks_measured'] == 3
    
    def test_topological_sort_with_priority(self):
        """Test topological sorting with priority consideration."""
        scheduler = QuantumScheduler()
        
        # Create tasks with known priorities
        task1 = QuantumTask(id="critical", title="Critical Task", description="")
        task2 = QuantumTask(id="low", title="Low Task", description="")
        
        # Force specific priorities
        task1.measured_priority = TaskPriority.CRITICAL
        task1.state = TaskState.COLLAPSED
        task2.measured_priority = TaskPriority.LOW
        task2.state = TaskState.COLLAPSED
        
        scheduled = scheduler._topological_sort_with_priority([task1, task2])
        
        # Critical priority task should come first
        assert scheduled[0].id == "critical"
        assert scheduled[1].id == "low"
    
    def test_get_quantum_state_summary(self):
        """Test quantum state summary."""
        scheduler = QuantumScheduler()
        
        # Add tasks in different states
        task1 = QuantumTask(id="task1", title="Task 1", description="")
        task2 = QuantumTask(id="task2", title="Task 2", description="")
        task3 = QuantumTask(id="task3", title="Task 3", description="")
        
        task2.state = TaskState.ENTANGLED
        task3.state = TaskState.COLLAPSED
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        
        scheduler.create_entanglement("task1", "task2")
        
        summary = scheduler.get_quantum_state_summary()
        
        assert summary['total_tasks'] == 3
        assert summary['entangled_tasks'] == 2  # task1 and task2
        assert summary['collapsed_tasks'] == 1  # task3
        assert summary['quantum_circuits'] == 1


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner class."""
    
    def test_planner_creation(self):
        """Test planner initialization."""
        planner = QuantumTaskPlanner()
        
        assert isinstance(planner.scheduler, QuantumScheduler)
        assert len(planner.planning_history) == 0
    
    def test_create_task(self):
        """Test task creation through planner."""
        planner = QuantumTaskPlanner()
        
        task = planner.create_task(
            task_id="test-task",
            title="Test Task",
            description="A test task",
            estimated_effort=3.0,
            dependencies=["dep1", "dep2"]
        )
        
        assert task.id == "test-task"
        assert task.title == "Test Task"
        assert task.estimated_effort == 3.0
        assert "dep1" in task.dependencies
        assert "dep2" in task.dependencies
        assert "test-task" in planner.scheduler.tasks
    
    def test_set_task_priority_bias(self):
        """Test setting priority bias."""
        planner = QuantumTaskPlanner()
        task = planner.create_task("test", "Test", "")
        
        original_amplitude = task.priority_amplitudes[TaskPriority.HIGH]
        
        planner.set_task_priority_bias("test", TaskPriority.HIGH, 0.3)
        
        # Amplitude for HIGH priority should have increased
        new_amplitude = task.priority_amplitudes[TaskPriority.HIGH]
        assert new_amplitude > original_amplitude
        
        # Should still be normalized
        total_amplitude_squared = sum(amp ** 2 for amp in task.priority_amplitudes.values())
        assert abs(total_amplitude_squared - 1.0) < 0.001
    
    def test_create_task_entanglement(self):
        """Test creating entanglement through planner."""
        planner = QuantumTaskPlanner()
        
        task1 = planner.create_task("task1", "Task 1", "")
        task2 = planner.create_task("task2", "Task 2", "")
        
        planner.create_task_entanglement("task1", "task2")
        
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
        assert ["task1", "task2"] in planner.scheduler.quantum_circuits
    
    def test_generate_execution_plan(self):
        """Test execution plan generation."""
        planner = QuantumTaskPlanner()
        
        # Create a small task set
        task1 = planner.create_task("task1", "Task 1", "First task", 1.0)
        task2 = planner.create_task("task2", "Task 2", "Second task", 2.0, ["task1"])
        task3 = planner.create_task("task3", "Task 3", "Third task", 1.5)
        
        planner.create_task_entanglement("task1", "task3")
        
        plan = planner.generate_execution_plan()
        
        # Check plan structure
        assert 'plan_id' in plan
        assert 'generation_time' in plan
        assert 'total_tasks' in plan
        assert 'total_estimated_effort' in plan
        assert 'critical_path_length' in plan
        assert 'quantum_state_initial' in plan
        assert 'quantum_state_final' in plan
        assert 'scheduled_tasks' in plan
        
        # Check plan content
        assert plan['total_tasks'] == 3
        assert plan['total_estimated_effort'] == 4.5
        assert len(plan['scheduled_tasks']) == 3
        
        # Check task details in plan
        for task_info in plan['scheduled_tasks']:
            assert 'id' in task_info
            assert 'title' in task_info
            assert 'measured_priority' in task_info
            assert 'execution_order' in task_info
            assert 'dependencies' in task_info
            assert 'entangled_with' in task_info
        
        # Check dependency ordering in plan
        task_orders = {t['id']: t['execution_order'] for t in plan['scheduled_tasks']}
        assert task_orders['task1'] < task_orders['task2']
        
        # Check planning history
        assert len(planner.planning_history) == 1
        assert planner.planning_history[0] == plan
    
    def test_calculate_critical_path_length(self):
        """Test critical path calculation."""
        planner = QuantumTaskPlanner()
        
        # Create tasks with known critical path
        task1 = planner.create_task("task1", "Task 1", "", 2.0)
        task2 = planner.create_task("task2", "Task 2", "", 3.0, ["task1"])
        task3 = planner.create_task("task3", "Task 3", "", 1.0, ["task2"])
        task4 = planner.create_task("task4", "Task 4", "", 2.0)  # Parallel path
        
        tasks = list(planner.scheduler.tasks.values())
        critical_path_length = planner._calculate_critical_path_length(tasks)
        
        # Critical path: task1 (2.0) -> task2 (3.0) -> task3 (1.0) = 6.0
        assert critical_path_length == 6.0
    
    def test_export_plan_to_json(self, tmp_path):
        """Test exporting plan to JSON."""
        planner = QuantumTaskPlanner()
        planner.create_task("task1", "Task 1", "Test task")
        
        plan = planner.generate_execution_plan()
        
        json_file = tmp_path / "test_plan.json"
        planner.export_plan_to_json(plan, str(json_file))
        
        assert json_file.exists()
        
        # Verify JSON content
        import json
        with open(json_file) as f:
            loaded_plan = json.load(f)
        
        assert loaded_plan['plan_id'] == plan['plan_id']
        assert loaded_plan['total_tasks'] == plan['total_tasks']
    
    def test_get_planning_analytics(self):
        """Test planning analytics."""
        planner = QuantumTaskPlanner()
        
        # No planning history initially
        analytics = planner.get_planning_analytics()
        assert analytics['message'] == 'No planning history available'
        
        # Generate some plans
        planner.create_task("task1", "Task 1", "")
        plan1 = planner.generate_execution_plan()
        
        time.sleep(0.001)  # Ensure different timestamps
        
        planner.create_task("task2", "Task 2", "")
        plan2 = planner.generate_execution_plan()
        
        analytics = planner.get_planning_analytics()
        
        assert analytics['total_plans_generated'] == 2
        assert analytics['average_planning_time_seconds'] > 0
        assert analytics['average_tasks_per_plan'] >= 1.0
        assert 'quantum_measurements_performed' in analytics
        assert 'current_quantum_state' in analytics


class TestQuantumPlannerIntegration:
    """Integration tests for the quantum planning system."""
    
    def test_complex_project_planning(self):
        """Test planning a complex project with multiple dependencies and entanglements."""
        planner = QuantumTaskPlanner()
        
        # Create a realistic project structure
        # Foundation tasks
        req_analysis = planner.create_task("req_analysis", "Requirements Analysis", 
                                         "Gather and analyze requirements", 8.0)
        design = planner.create_task("design", "System Design", 
                                   "Design system architecture", 12.0, ["req_analysis"])
        
        # Development tasks
        backend = planner.create_task("backend", "Backend Development", 
                                    "Implement backend API", 20.0, ["design"])
        frontend = planner.create_task("frontend", "Frontend Development", 
                                     "Implement user interface", 16.0, ["design"])
        database = planner.create_task("database", "Database Setup", 
                                     "Set up database schema", 6.0, ["design"])
        
        # Integration and testing
        integration = planner.create_task("integration", "System Integration", 
                                        "Integrate all components", 8.0, 
                                        ["backend", "frontend", "database"])
        testing = planner.create_task("testing", "Testing", 
                                    "Comprehensive system testing", 12.0, ["integration"])
        
        # Deployment
        deployment = planner.create_task("deployment", "Deployment", 
                                       "Deploy to production", 4.0, ["testing"])
        
        # Create entanglements for related tasks
        planner.create_task_entanglement("backend", "frontend")
        planner.create_task_entanglement("backend", "database")
        
        # Set priority biases
        planner.set_task_priority_bias("req_analysis", TaskPriority.CRITICAL, 0.4)
        planner.set_task_priority_bias("design", TaskPriority.CRITICAL, 0.4)
        planner.set_task_priority_bias("deployment", TaskPriority.HIGH, 0.3)
        
        # Generate execution plan
        plan = planner.generate_execution_plan()
        
        # Verify plan properties
        assert plan['total_tasks'] == 8
        assert plan['total_estimated_effort'] == 86.0
        
        # Verify dependency constraints are respected
        task_orders = {t['id']: t['execution_order'] for t in plan['scheduled_tasks']}
        
        # Requirements analysis should be first
        assert task_orders['req_analysis'] < task_orders['design']
        
        # Design should come before all development tasks
        assert task_orders['design'] < task_orders['backend']
        assert task_orders['design'] < task_orders['frontend']
        assert task_orders['design'] < task_orders['database']
        
        # Development tasks should come before integration
        assert task_orders['backend'] < task_orders['integration']
        assert task_orders['frontend'] < task_orders['integration']
        assert task_orders['database'] < task_orders['integration']
        
        # Integration should come before testing
        assert task_orders['integration'] < task_orders['testing']
        
        # Testing should come before deployment
        assert task_orders['testing'] < task_orders['deployment']
        
        # Check entanglements are present
        entanglement_info = {t['id']: t['entangled_with'] for t in plan['scheduled_tasks']}
        assert 'frontend' in entanglement_info['backend']
        assert 'database' in entanglement_info['backend']
        assert 'backend' in entanglement_info['frontend']
    
    def test_quantum_measurement_consistency(self):
        """Test that quantum measurements are consistent and repeatable."""
        planner = QuantumTaskPlanner()
        
        # Create tasks
        for i in range(5):
            planner.create_task(f"task{i}", f"Task {i}", f"Task number {i}")
        
        # Generate multiple plans (should be identical due to quantum collapse)
        plan1 = planner.generate_execution_plan()
        plan2 = planner.generate_execution_plan()
        
        # Plans should be identical since quantum states are already collapsed
        schedule1 = {t['id']: t['execution_order'] for t in plan1['scheduled_tasks']}
        schedule2 = {t['id']: t['execution_order'] for t in plan2['scheduled_tasks']}
        
        assert schedule1 == schedule2
    
    def test_large_scale_planning(self):
        """Test planning with a large number of tasks."""
        planner = QuantumTaskPlanner()
        
        # Create a large number of tasks
        num_tasks = 50
        for i in range(num_tasks):
            dependencies = []
            if i > 0:
                # Add some dependencies to previous tasks
                dependencies = [f"task{j}" for j in range(max(0, i-3), i) if j % 3 == 0]
            
            planner.create_task(f"task{i}", f"Task {i}", f"Task number {i}", 
                              estimated_effort=1.0 + (i % 5), dependencies=dependencies)
        
        # Create some entanglements
        for i in range(0, num_tasks - 1, 5):
            try:
                planner.create_task_entanglement(f"task{i}", f"task{i+1}")
            except:
                pass  # Might fail due to dependencies
        
        # Generate plan
        start_time = time.time()
        plan = planner.generate_execution_plan()
        planning_time = time.time() - start_time
        
        # Verify plan
        assert plan['total_tasks'] == num_tasks
        assert len(plan['scheduled_tasks']) == num_tasks
        assert planning_time < 5.0  # Should complete within reasonable time
        
        # Verify all tasks have valid execution orders
        execution_orders = [t['execution_order'] for t in plan['scheduled_tasks']]
        assert len(set(execution_orders)) == num_tasks  # All unique
        assert min(execution_orders) == 0
        assert max(execution_orders) == num_tasks - 1


if __name__ == "__main__":
    pytest.main([__file__])