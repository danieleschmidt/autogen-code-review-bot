"""
Comprehensive tests for quantum task planner CLI interface.
"""

import pytest
import json
import io
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.autogen_code_review_bot.quantum_planner import QuantumTaskPlanner, TaskPriority
from src.autogen_code_review_bot.quantum_cli import (
    create_task_interactive, load_tasks_from_json, display_execution_plan, main
)


class TestLoadTasksFromJson:
    """Test JSON task loading functionality."""
    
    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON task definition."""
        planner = QuantumTaskPlanner()
        
        # Create test JSON file
        test_data = {
            "tasks": [
                {
                    "id": "task1",
                    "title": "Task 1",
                    "description": "First task",
                    "estimated_effort": 2.5,
                    "dependencies": []
                },
                {
                    "id": "task2", 
                    "title": "Task 2",
                    "description": "Second task",
                    "estimated_effort": 3.0,
                    "dependencies": ["task1"],
                    "priority_bias": {
                        "priority": "HIGH",
                        "strength": 0.4
                    }
                }
            ],
            "entanglements": [
                {"task1": "task1", "task2": "task2"}
            ]
        }
        
        json_file = tmp_path / "test_tasks.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        # Capture output
        with patch('builtins.print') as mock_print:
            load_tasks_from_json(planner, str(json_file))
        
        # Verify tasks were created
        assert len(planner.scheduler.tasks) == 2
        assert "task1" in planner.scheduler.tasks
        assert "task2" in planner.scheduler.tasks
        
        # Verify dependency
        task2 = planner.scheduler.tasks["task2"]
        assert "task1" in task2.dependencies
        
        # Verify entanglement
        task1 = planner.scheduler.tasks["task1"]
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
        
        # Verify output message
        mock_print.assert_called_with("✅ Loaded 2 tasks and 1 entanglements from " + str(json_file))
    
    def test_load_missing_file(self):
        """Test loading non-existent file."""
        planner = QuantumTaskPlanner()
        
        with patch('builtins.print') as mock_print:
            load_tasks_from_json(planner, "nonexistent.json")
        
        mock_print.assert_called_with("❌ File not found: nonexistent.json")
        assert len(planner.scheduler.tasks) == 0
    
    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        planner = QuantumTaskPlanner()
        
        # Create invalid JSON file
        json_file = tmp_path / "invalid.json"
        with open(json_file, 'w') as f:
            f.write("{ invalid json }")
        
        with patch('builtins.print') as mock_print:
            load_tasks_from_json(planner, str(json_file))
        
        # Should print error message
        assert any("Invalid JSON" in str(call) for call in mock_print.call_args_list)
        assert len(planner.scheduler.tasks) == 0
    
    def test_load_minimal_json(self, tmp_path):
        """Test loading minimal JSON (tasks only)."""
        planner = QuantumTaskPlanner()
        
        test_data = {
            "tasks": [
                {
                    "id": "simple_task",
                    "title": "Simple Task"
                }
            ]
        }
        
        json_file = tmp_path / "minimal.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('builtins.print'):
            load_tasks_from_json(planner, str(json_file))
        
        assert len(planner.scheduler.tasks) == 1
        task = planner.scheduler.tasks["simple_task"]
        assert task.title == "Simple Task"
        assert task.description == ""  # Default
        assert task.estimated_effort == 1.0  # Default
    
    def test_load_invalid_priority(self, tmp_path):
        """Test loading with invalid priority bias."""
        planner = QuantumTaskPlanner()
        
        test_data = {
            "tasks": [
                {
                    "id": "task1",
                    "title": "Task 1",
                    "priority_bias": {
                        "priority": "INVALID_PRIORITY",
                        "strength": 0.3
                    }
                }
            ]
        }
        
        json_file = tmp_path / "invalid_priority.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('builtins.print') as mock_print:
            load_tasks_from_json(planner, str(json_file))
        
        # Should still create task but warn about invalid priority
        assert len(planner.scheduler.tasks) == 1
        assert any("Unknown priority" in str(call) for call in mock_print.call_args_list)
    
    def test_load_failed_entanglement(self, tmp_path):
        """Test loading with failed entanglement creation."""
        planner = QuantumTaskPlanner()
        
        test_data = {
            "tasks": [
                {
                    "id": "task1",
                    "title": "Task 1"
                }
            ],
            "entanglements": [
                {"task1": "task1", "task2": "nonexistent"}
            ]
        }
        
        json_file = tmp_path / "failed_entanglement.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('builtins.print') as mock_print:
            load_tasks_from_json(planner, str(json_file))
        
        # Should create task but fail entanglement
        assert len(planner.scheduler.tasks) == 1
        assert any("Could not create entanglement" in str(call) for call in mock_print.call_args_list)


class TestDisplayExecutionPlan:
    """Test execution plan display functionality."""
    
    def test_display_basic_plan(self):
        """Test displaying a basic execution plan."""
        plan = {
            'plan_id': 'test_plan_123',
            'generation_time': 0.123,
            'total_tasks': 3,
            'total_estimated_effort': 7.5,
            'critical_path_length': 6.0,
            'quantum_state_initial': {
                'total_tasks': 3,
                'entangled_tasks': 0,
                'collapsed_tasks': 0,
                'quantum_circuits': 0
            },
            'quantum_state_final': {
                'total_tasks': 3,
                'entangled_tasks': 2,
                'collapsed_tasks': 3,
                'quantum_circuits': 1
            },
            'scheduled_tasks': [
                {
                    'id': 'task1',
                    'title': 'First Task',
                    'description': 'The first task to complete',
                    'estimated_effort': 2.5,
                    'measured_priority': 'Critical',
                    'execution_order': 0,
                    'dependencies': [],
                    'entangled_with': ['task2']
                },
                {
                    'id': 'task2',
                    'title': 'Second Task',
                    'description': 'The second task',
                    'estimated_effort': 3.0,
                    'measured_priority': 'High',
                    'execution_order': 1,
                    'dependencies': ['task1'],
                    'entangled_with': ['task1']
                },
                {
                    'id': 'task3',
                    'title': 'Third Task',
                    'description': '',
                    'estimated_effort': 2.0,
                    'measured_priority': 'Medium',
                    'execution_order': 2,
                    'dependencies': [],
                    'entangled_with': []
                }
            ]
        }
        
        with patch('builtins.print') as mock_print:
            display_execution_plan(plan)
        
        # Verify key information is displayed
        printed_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        
        assert 'test_plan_123' in printed_output
        assert '0.123 seconds' in printed_output
        assert 'Total Tasks: 3' in printed_output
        assert '7.5 hours' in printed_output
        assert '6.0 hours' in printed_output
        assert 'First Task' in printed_output
        assert 'Second Task' in printed_output
        assert 'Third Task' in printed_output
        assert '🔴' in printed_output  # Critical priority icon
        assert '🟠' in printed_output  # High priority icon
        assert '🟡' in printed_output  # Medium priority icon
    
    def test_display_plan_with_all_priorities(self):
        """Test displaying plan with all priority levels."""
        plan = {
            'plan_id': 'priority_test',
            'generation_time': 0.1,
            'total_tasks': 5,
            'total_estimated_effort': 10.0,
            'critical_path_length': 8.0,
            'quantum_state_initial': {'total_tasks': 5, 'entangled_tasks': 0, 'collapsed_tasks': 0, 'quantum_circuits': 0},
            'quantum_state_final': {'total_tasks': 5, 'entangled_tasks': 0, 'collapsed_tasks': 5, 'quantum_circuits': 0},
            'scheduled_tasks': [
                {'id': 'critical', 'title': 'Critical Task', 'description': '', 'estimated_effort': 2.0, 
                 'measured_priority': 'Critical', 'execution_order': 0, 'dependencies': [], 'entangled_with': []},
                {'id': 'high', 'title': 'High Task', 'description': '', 'estimated_effort': 2.0,
                 'measured_priority': 'High', 'execution_order': 1, 'dependencies': [], 'entangled_with': []},
                {'id': 'medium', 'title': 'Medium Task', 'description': '', 'estimated_effort': 2.0,
                 'measured_priority': 'Medium', 'execution_order': 2, 'dependencies': [], 'entangled_with': []},
                {'id': 'low', 'title': 'Low Task', 'description': '', 'estimated_effort': 2.0,
                 'measured_priority': 'Low', 'execution_order': 3, 'dependencies': [], 'entangled_with': []},
                {'id': 'deferred', 'title': 'Deferred Task', 'description': '', 'estimated_effort': 2.0,
                 'measured_priority': 'Deferred', 'execution_order': 4, 'dependencies': [], 'entangled_with': []}
            ]
        }
        
        with patch('builtins.print') as mock_print:
            display_execution_plan(plan)
        
        printed_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        
        # Check all priority icons are present
        assert '🔴' in printed_output  # Critical
        assert '🟠' in printed_output  # High
        assert '🟡' in printed_output  # Medium
        assert '🟢' in printed_output  # Low
        assert '⚪' in printed_output  # Deferred
    
    def test_display_plan_with_unknown_priority(self):
        """Test displaying plan with unknown priority."""
        plan = {
            'plan_id': 'unknown_test',
            'generation_time': 0.1,
            'total_tasks': 1,
            'total_estimated_effort': 2.0,
            'critical_path_length': 2.0,
            'quantum_state_initial': {'total_tasks': 1, 'entangled_tasks': 0, 'collapsed_tasks': 0, 'quantum_circuits': 0},
            'quantum_state_final': {'total_tasks': 1, 'entangled_tasks': 0, 'collapsed_tasks': 1, 'quantum_circuits': 0},
            'scheduled_tasks': [
                {
                    'id': 'unknown',
                    'title': 'Unknown Priority Task',
                    'description': '',
                    'estimated_effort': 2.0,
                    'measured_priority': 'UnknownPriority',
                    'execution_order': 0,
                    'dependencies': [],
                    'entangled_with': []
                }
            ]
        }
        
        with patch('builtins.print') as mock_print:
            display_execution_plan(plan)
        
        printed_output = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
        assert '❓' in printed_output  # Unknown priority icon


class TestCreateTaskInteractive:
    """Test interactive task creation functionality."""
    
    def test_create_task_interactive_complete(self):
        """Test complete interactive task creation."""
        planner = QuantumTaskPlanner()
        
        # Mock user inputs
        inputs = [
            "test-task",           # Task ID
            "Test Task",           # Title
            "A test description",  # Description
            "2.5",                # Estimated effort
            "dep1,dep2",          # Dependencies
            "HIGH",               # Priority bias
            "0.4",                # Bias strength
            "other-task"          # Entanglement
        ]
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            # Mock the planner methods to avoid actual creation issues
            with patch.object(planner, 'create_task') as mock_create, \
                 patch.object(planner, 'set_task_priority_bias') as mock_bias, \
                 patch.object(planner, 'create_task_entanglement') as mock_entangle:
                
                mock_task = Mock()
                mock_create.return_value = mock_task
                
                create_task_interactive(planner)
        
        # Verify task creation was called correctly
        mock_create.assert_called_once_with("test-task", "Test Task", "A test description", 2.5, ["dep1", "dep2"])
        mock_bias.assert_called_once_with("test-task", TaskPriority.HIGH, 0.4)
        mock_entangle.assert_called_once_with("test-task", "other-task")
        
        # Verify success messages
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("✅ Applied HIGH priority bias" in msg for msg in printed_messages)
        assert any("✅ Created quantum entanglement" in msg for msg in printed_messages)
        assert any("✅ Quantum task 'test-task' created successfully!" in msg for msg in printed_messages)
    
    def test_create_task_interactive_minimal(self):
        """Test interactive task creation with minimal inputs."""
        planner = QuantumTaskPlanner()
        
        inputs = [
            "minimal-task",  # Task ID
            "Minimal Task",  # Title
            "",             # Description (empty)
            "",             # Effort (default)
            "",             # Dependencies (none)
            "",             # Priority (skip)
            ""              # Entanglement (skip)
        ]
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            with patch.object(planner, 'create_task') as mock_create:
                mock_task = Mock()
                mock_create.return_value = mock_task
                
                create_task_interactive(planner)
        
        # Should use defaults
        mock_create.assert_called_once_with("minimal-task", "Minimal Task", "", 1.0, None)
    
    def test_create_task_interactive_empty_id(self):
        """Test interactive task creation with empty ID."""
        planner = QuantumTaskPlanner()
        
        inputs = [""]  # Empty task ID
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            create_task_interactive(planner)
        
        # Should print error and return early
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("❌ Task ID is required" in msg for msg in printed_messages)
    
    def test_create_task_interactive_empty_title(self):
        """Test interactive task creation with empty title."""
        planner = QuantumTaskPlanner()
        
        inputs = ["test-id", ""]  # Valid ID, empty title
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            create_task_interactive(planner)
        
        # Should print error and return early
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("❌ Task title is required" in msg for msg in printed_messages)
    
    def test_create_task_interactive_invalid_effort(self):
        """Test interactive task creation with invalid effort."""
        planner = QuantumTaskPlanner()
        
        inputs = [
            "test-task",
            "Test Task", 
            "Description",
            "not-a-number",  # Invalid effort
            "",              # Dependencies
            "",              # Priority
            ""               # Entanglement
        ]
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            with patch.object(planner, 'create_task') as mock_create:
                mock_task = Mock()
                mock_create.return_value = mock_task
                
                create_task_interactive(planner)
        
        # Should use default effort and print warning
        mock_create.assert_called_once()
        call_args = mock_create.call_args[0]
        assert call_args[3] == 1.0  # Default effort
        
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("⚠️  Using default effort: 1.0 hours" in msg for msg in printed_messages)
    
    def test_create_task_interactive_invalid_priority(self):
        """Test interactive task creation with invalid priority."""
        planner = QuantumTaskPlanner()
        
        inputs = [
            "test-task",
            "Test Task",
            "Description", 
            "1.0",
            "",
            "INVALID_PRIORITY",  # Invalid priority
            ""
        ]
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            with patch.object(planner, 'create_task') as mock_create:
                mock_task = Mock()
                mock_create.return_value = mock_task
                
                create_task_interactive(planner)
        
        # Should print warning about invalid priority
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("⚠️  Invalid priority" in msg for msg in printed_messages)
    
    def test_create_task_interactive_entanglement_failure(self):
        """Test interactive task creation with entanglement failure."""
        planner = QuantumTaskPlanner()
        
        inputs = [
            "test-task",
            "Test Task",
            "Description",
            "1.0",
            "",
            "",
            "nonexistent-task"  # Non-existent task for entanglement
        ]
        
        with patch('builtins.input', side_effect=inputs), \
             patch('builtins.print') as mock_print:
            
            with patch.object(planner, 'create_task') as mock_create, \
                 patch.object(planner, 'create_task_entanglement') as mock_entangle:
                
                mock_task = Mock()
                mock_create.return_value = mock_task
                mock_entangle.side_effect = Exception("Task does not exist")
                
                create_task_interactive(planner)
        
        # Should print warning about entanglement failure
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("⚠️  Could not create entanglement" in msg for msg in printed_messages)


class TestMainFunction:
    """Test main CLI function."""
    
    def test_main_help(self):
        """Test main function help display."""
        with patch('sys.argv', ['quantum_cli.py']), \
             patch('builtins.print') as mock_print:
            
            # Mock argparse to avoid actual parsing
            with patch('argparse.ArgumentParser.print_help') as mock_help:
                main()
                mock_help.assert_called_once()
    
    def test_main_interactive_mode(self):
        """Test main function in interactive mode."""
        with patch('sys.argv', ['quantum_cli.py', '--interactive']), \
             patch('builtins.input', side_effect=['4']):  # Exit immediately
            
            with patch('builtins.print') as mock_print:
                main()
            
            # Should show interactive menu
            printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any("🌌 Quantum Task Planner - Interactive Mode" in msg for msg in printed_messages)
    
    def test_main_load_and_plan(self, tmp_path):
        """Test main function with load and plan options."""
        # Create test JSON file
        test_data = {
            "tasks": [
                {"id": "task1", "title": "Task 1"}
            ]
        }
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('sys.argv', ['quantum_cli.py', '--load', str(json_file), '--plan']), \
             patch('builtins.print') as mock_print:
            
            main()
            
            # Should load tasks and generate plan
            printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any("✅ Loaded 1 tasks" in msg for msg in printed_messages)
            assert any("🎲 Generating quantum execution plan" in msg for msg in printed_messages)
    
    def test_main_plan_without_tasks(self):
        """Test main function plan without tasks."""
        with patch('sys.argv', ['quantum_cli.py', '--plan']), \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            
            main()
            
            # Should exit with error
            mock_exit.assert_called_once_with(1)
            printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any("❌ No tasks loaded" in msg for msg in printed_messages)
    
    def test_main_export_plan(self, tmp_path):
        """Test main function with plan export."""
        # Create test JSON file
        test_data = {
            "tasks": [
                {"id": "task1", "title": "Task 1"}
            ]
        }
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        export_file = tmp_path / "plan.json"
        
        with patch('sys.argv', ['quantum_cli.py', '--load', str(json_file), '--plan', '--export', str(export_file)]), \
             patch('builtins.print') as mock_print:
            
            main()
            
            # Should export plan
            assert export_file.exists()
            printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any(f"✅ Plan exported to {export_file}" in msg for msg in printed_messages)
    
    def test_main_analytics(self, tmp_path):
        """Test main function with analytics."""
        # Create test JSON file
        test_data = {
            "tasks": [
                {"id": "task1", "title": "Task 1"}
            ]
        }
        
        json_file = tmp_path / "test.json"
        with open(json_file, 'w') as f:
            json.dump(test_data, f)
        
        with patch('sys.argv', ['quantum_cli.py', '--load', str(json_file), '--plan', '--analytics']), \
             patch('builtins.print') as mock_print:
            
            main()
            
            # Should show analytics
            printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
            assert any("📊 Planning Analytics" in msg for msg in printed_messages)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_complete_workflow(self, tmp_path):
        """Test complete CLI workflow from JSON to plan export."""
        # Create comprehensive test JSON
        test_data = {
            "tasks": [
                {
                    "id": "requirements",
                    "title": "Requirements Analysis",
                    "description": "Gather and analyze requirements",
                    "estimated_effort": 4.0,
                    "dependencies": [],
                    "priority_bias": {
                        "priority": "CRITICAL",
                        "strength": 0.5
                    }
                },
                {
                    "id": "design",
                    "title": "System Design",
                    "description": "Design system architecture",
                    "estimated_effort": 6.0,
                    "dependencies": ["requirements"],
                    "priority_bias": {
                        "priority": "HIGH",
                        "strength": 0.3
                    }
                },
                {
                    "id": "implementation",
                    "title": "Implementation",
                    "description": "Implement the system",
                    "estimated_effort": 16.0,
                    "dependencies": ["design"]
                },
                {
                    "id": "testing",
                    "title": "Testing",
                    "description": "Test the system",
                    "estimated_effort": 8.0,
                    "dependencies": ["implementation"]
                }
            ],
            "entanglements": [
                {"task1": "design", "task2": "implementation"}
            ]
        }
        
        # Create input and output files
        input_file = tmp_path / "project.json"
        output_file = tmp_path / "execution_plan.json"
        
        with open(input_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Run CLI command
        with patch('sys.argv', [
            'quantum_cli.py', 
            '--load', str(input_file),
            '--plan',
            '--export', str(output_file),
            '--analytics'
        ]), patch('builtins.print') as mock_print:
            
            main()
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify output file contains valid plan
        with open(output_file) as f:
            exported_plan = json.load(f)
        
        assert 'plan_id' in exported_plan
        assert exported_plan['total_tasks'] == 4
        assert exported_plan['total_estimated_effort'] == 34.0
        assert len(exported_plan['scheduled_tasks']) == 4
        
        # Verify dependency ordering in exported plan
        task_orders = {t['id']: t['execution_order'] for t in exported_plan['scheduled_tasks']}
        assert task_orders['requirements'] < task_orders['design']
        assert task_orders['design'] < task_orders['implementation']
        assert task_orders['implementation'] < task_orders['testing']
        
        # Verify entanglements
        entanglement_info = {t['id']: t['entangled_with'] for t in exported_plan['scheduled_tasks']}
        assert 'implementation' in entanglement_info['design']
        assert 'design' in entanglement_info['implementation']
        
        # Verify console output
        printed_messages = [str(call.args[0]) for call in mock_print.call_args_list]
        assert any("✅ Loaded 4 tasks and 1 entanglements" in msg for msg in printed_messages)
        assert any("🎲 Generating quantum execution plan" in msg for msg in printed_messages)
        assert any("🚀 Quantum Execution Plan" in msg for msg in printed_messages)
        assert any("📊 Planning Analytics" in msg for msg in printed_messages)
        assert any(f"✅ Plan exported to {output_file}" in msg for msg in printed_messages)


if __name__ == "__main__":
    pytest.main([__file__])