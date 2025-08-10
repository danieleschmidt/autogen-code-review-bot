"""
Quantum Task Planner CLI Interface

Command-line interface for the quantum-inspired task planning system.
Provides easy access to quantum task creation, scheduling, and execution planning.
"""

import argparse
import json
import sys
from typing import Dict

from .quantum_planner import QuantumTaskPlanner, TaskPriority


def create_task_interactive(planner: QuantumTaskPlanner) -> None:
    """Interactive task creation wizard."""
    print("\n🎯 Quantum Task Creation Wizard")
    print("=" * 40)

    task_id = input("Task ID (unique identifier): ").strip()
    if not task_id:
        print("❌ Task ID is required")
        return

    title = input("Task Title: ").strip()
    if not title:
        print("❌ Task title is required")
        return

    description = input("Task Description: ").strip()

    try:
        effort = float(input("Estimated Effort (hours, default 1.0): ") or "1.0")
    except ValueError:
        effort = 1.0
        print("⚠️  Using default effort: 1.0 hours")

    dependencies_input = input("Dependencies (comma-separated task IDs, optional): ").strip()
    dependencies = [dep.strip() for dep in dependencies_input.split(",") if dep.strip()] if dependencies_input else None

    # Create the task
    task = planner.create_task(task_id, title, description, effort, dependencies)

    # Ask about priority bias
    print("\n🎲 Quantum Priority Configuration")
    print("Available priorities: CRITICAL, HIGH, MEDIUM, LOW, DEFERRED")
    priority_input = input("Priority bias (optional, press Enter to skip): ").strip().upper()

    if priority_input:
        try:
            priority = TaskPriority[priority_input]
            bias_strength = float(input("Bias strength (0.1-1.0, default 0.3): ") or "0.3")
            planner.set_task_priority_bias(task_id, priority, bias_strength)
            print(f"✅ Applied {priority.label} priority bias ({bias_strength})")
        except (KeyError, ValueError):
            print("⚠️  Invalid priority or bias strength, using default quantum superposition")

    # Ask about entanglement
    entangle_input = input("Entangle with task ID (optional): ").strip()
    if entangle_input:
        try:
            planner.create_task_entanglement(task_id, entangle_input)
            print(f"✅ Created quantum entanglement: {task_id} <-> {entangle_input}")
        except Exception as e:
            print(f"⚠️  Could not create entanglement: {e}")

    print(f"\n✅ Quantum task '{task_id}' created successfully!")


def load_tasks_from_json(planner: QuantumTaskPlanner, filename: str) -> None:
    """Load tasks from JSON file."""
    try:
        with open(filename) as f:
            data = json.load(f)

        tasks_data = data.get('tasks', [])
        entanglements = data.get('entanglements', [])

        # Create tasks
        created_tasks = 0
        for task_data in tasks_data:
            task = planner.create_task(
                task_id=task_data['id'],
                title=task_data['title'],
                description=task_data.get('description', ''),
                estimated_effort=task_data.get('estimated_effort', 1.0),
                dependencies=task_data.get('dependencies', [])
            )

            # Apply priority bias if specified
            if 'priority_bias' in task_data:
                priority_name = task_data['priority_bias']['priority'].upper()
                bias_strength = task_data['priority_bias'].get('strength', 0.3)
                try:
                    priority = TaskPriority[priority_name]
                    planner.set_task_priority_bias(task['id'], priority, bias_strength)
                except KeyError:
                    print(f"⚠️  Unknown priority '{priority_name}' for task {task_data['id']}")

            created_tasks += 1

        # Create entanglements
        created_entanglements = 0
        for entanglement in entanglements:
            try:
                planner.create_task_entanglement(entanglement['task1'], entanglement['task2'])
                created_entanglements += 1
            except Exception as e:
                print(f"⚠️  Could not create entanglement {entanglement}: {e}")

        print(f"✅ Loaded {created_tasks} tasks and {created_entanglements} entanglements from {filename}")

    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filename}: {e}")
    except Exception as e:
        print(f"❌ Error loading tasks: {e}")


def display_execution_plan(plan: Dict) -> None:
    """Display execution plan in formatted output."""
    print("\n🚀 Quantum Execution Plan")
    print("=" * 50)
    print(f"Plan ID: {plan['plan_id']}")
    print(f"Generation Time: {plan['generation_time']:.3f} seconds")
    print(f"Total Tasks: {plan['total_tasks']}")
    print(f"Total Estimated Effort: {plan['total_estimated_effort']:.1f} hours")
    print(f"Critical Path Length: {plan['critical_path_length']:.1f} hours")

    print("\n🔬 Quantum State Analysis")
    print("-" * 30)
    initial = plan['quantum_state_initial']
    final = plan['quantum_state_final']
    print(f"Tasks Measured: {initial['total_tasks']} → {final['collapsed_tasks']}")
    print(f"Entangled Tasks: {final['entangled_tasks']}")
    print(f"Quantum Circuits: {final['quantum_circuits']}")

    print("\n📋 Scheduled Tasks")
    print("-" * 30)
    for i, task in enumerate(plan['scheduled_tasks'], 1):
        priority_icon = {
            'Critical': '🔴',
            'High': '🟠',
            'Medium': '🟡',
            'Low': '🟢',
            'Deferred': '⚪'
        }.get(task['measured_priority'], '❓')

        entangled_info = f" [⚛️  {len(task['entangled_with'])} entangled]" if task['entangled_with'] else ""
        deps_info = f" [📎 {len(task['dependencies'])} deps]" if task['dependencies'] else ""

        print(f"{i:2d}. {priority_icon} {task['title']}")
        print(f"     ID: {task['id']} | Effort: {task['estimated_effort']:.1f}h{deps_info}{entangled_info}")
        if task['description']:
            print(f"     {task['description']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum-Inspired Task Planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive task creation and planning
  python -m autogen_code_review_bot.quantum_cli --interactive
  
  # Load tasks from JSON file and generate plan
  python -m autogen_code_review_bot.quantum_cli --load tasks.json --plan
  
  # Create tasks interactively and export plan
  python -m autogen_code_review_bot.quantum_cli --interactive --plan --export plan.json
  
  # Show planning analytics
  python -m autogen_code_review_bot.quantum_cli --load tasks.json --plan --analytics

JSON Task Format:
{
  "tasks": [
    {
      "id": "task1",
      "title": "Implement feature X",
      "description": "Add new functionality",
      "estimated_effort": 3.5,
      "dependencies": ["task0"],
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
"""
    )

    # Input options
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive task creation mode'
    )

    parser.add_argument(
        '--load', '-l',
        metavar='JSON_FILE',
        help='Load tasks from JSON file'
    )

    # Action options
    parser.add_argument(
        '--plan', '-p',
        action='store_true',
        help='Generate quantum execution plan'
    )

    parser.add_argument(
        '--export', '-e',
        metavar='OUTPUT_FILE',
        help='Export execution plan to JSON file'
    )

    parser.add_argument(
        '--analytics', '-a',
        action='store_true',
        help='Show planning analytics'
    )

    # Configuration options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Initialize planner
    planner = QuantumTaskPlanner()

    # Load tasks if specified
    if args.load:
        load_tasks_from_json(planner, args.load)

    # Interactive task creation
    if args.interactive:
        print("🌌 Quantum Task Planner - Interactive Mode")
        while True:
            print("\nOptions:")
            print("1. Create new task")
            print("2. Generate execution plan")
            print("3. Show analytics")
            print("4. Exit")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                create_task_interactive(planner)
            elif choice == '2':
                if planner.scheduler.tasks:
                    plan = planner.generate_execution_plan()
                    display_execution_plan(plan)

                    if args.export:
                        planner.export_plan_to_json(plan, args.export)
                        print(f"✅ Plan exported to {args.export}")
                else:
                    print("❌ No tasks available. Create some tasks first.")
            elif choice == '3':
                analytics = planner.get_planning_analytics()
                print("\n📊 Planning Analytics:")
                for key, value in analytics.items():
                    print(f"  {key}: {value}")
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")

    # Generate plan if requested
    elif args.plan:
        if not planner.scheduler.tasks:
            print("❌ No tasks loaded. Use --load or --interactive to add tasks.")
            sys.exit(1)

        print("🎲 Generating quantum execution plan...")
        plan = planner.generate_execution_plan()
        display_execution_plan(plan)

        if args.export:
            planner.export_plan_to_json(plan, args.export)
            print(f"✅ Plan exported to {args.export}")

    # Show analytics if requested
    if args.analytics:
        analytics = planner.get_planning_analytics()
        print("\n📊 Planning Analytics:")
        print("=" * 30)
        for key, value in analytics.items():
            print(f"{key}: {value}")

    # If no actions specified, show help
    if not any([args.interactive, args.load, args.plan, args.analytics]):
        parser.print_help()


if __name__ == "__main__":
    main()
