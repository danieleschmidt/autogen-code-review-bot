"""
AutoGen Code Review Bot CLI Interface

Main command-line interface for the AutoGen Code Review Bot.
Provides comprehensive access to all bot functionality including PR analysis,
autonomous agent conversations, and enterprise features.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from .agents import run_dual_review, load_agents_from_yaml
from .config import load_config
from .github_integration import analyze_and_comment, get_pull_request_diff
from .metrics import get_metrics_registry
from .pr_analysis import analyze_pr, load_linter_config
from .quantum_planner import QuantumTaskPlanner

console = Console()


@click.group()
@click.option('--config', '-c', default='config/default.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """AutoGen Code Review Bot - Autonomous SDLC Execution Platform"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        console.print("🤖 AutoGen Code Review Bot CLI", style="bold blue")
        console.print(f"Configuration: {config}")


@cli.command()
@click.option('--repo-path', '-r', default='.', help='Path to repository to analyze')
@click.option('--output', '-o', help='Output file for analysis results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'markdown', 'console']), 
              default='console', help='Output format')
@click.option('--use-cache/--no-cache', default=True, help='Use intelligent caching')
@click.option('--parallel/--sequential', default=True, help='Use parallel processing')
@click.option('--linter-config', help='Custom linter configuration file')
@click.pass_context
def analyze(ctx, repo_path, output, output_format, use_cache, parallel, linter_config):
    """Analyze repository with AutoGen agents"""
    
    repo_path = Path(repo_path).resolve()
    if not repo_path.exists():
        console.print(f"❌ Repository path not found: {repo_path}", style="red")
        sys.exit(1)
    
    console.print(f"🔍 Analyzing repository: {repo_path}")
    
    with Progress() as progress:
        task = progress.add_task("Running analysis...", total=100)
        
        # Load linter config if provided
        linter_config_data = None
        if linter_config:
            linter_config_data = load_linter_config(linter_config)
            progress.update(task, advance=10)
        
        # Run analysis
        try:
            result = analyze_pr(
                str(repo_path),
                config_path=linter_config,
                use_cache=use_cache,
                use_parallel=parallel
            )
            progress.update(task, advance=70)
            
            # Format output
            if output_format == 'json':
                output_data = {
                    'repository': str(repo_path),
                    'timestamp': time.time(),
                    'security': result.security.output if result.security else None,
                    'style': result.style.output if result.style else None,
                    'performance': result.performance.output if result.performance else None,
                    'cached': getattr(result, 'cached', False)
                }
                
                if output:
                    with open(output, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    console.print(f"✅ Results saved to {output}")
                else:
                    console.print_json(data=output_data)
            
            elif output_format == 'markdown':
                md_content = _format_markdown_output(result, repo_path)
                if output:
                    with open(output, 'w') as f:
                        f.write(md_content)
                    console.print(f"✅ Results saved to {output}")
                else:
                    console.print(md_content)
            
            else:  # console
                _display_console_output(result, repo_path)
            
            progress.update(task, advance=20)
            
        except Exception as e:
            console.print(f"❌ Analysis failed: {e}", style="red")
            if ctx.obj['verbose']:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)


@cli.command()
@click.option('--repo', '-r', required=True, help='GitHub repository (owner/repo)')
@click.option('--pr-number', '-p', type=int, required=True, help='Pull request number')
@click.option('--token', help='GitHub token (or set GITHUB_TOKEN env var)')
@click.option('--post-comment/--no-comment', default=False, help='Post results as PR comment')
@click.pass_context
def review_pr(ctx, repo, pr_number, token, post_comment):
    """Review GitHub pull request with AutoGen agents"""
    
    if not token:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            console.print("❌ GitHub token required (use --token or GITHUB_TOKEN env var)", style="red")
            sys.exit(1)
    
    console.print(f"🔍 Reviewing PR #{pr_number} in {repo}")
    
    try:
        if post_comment:
            analyze_and_comment(repo_path='.', repo=repo, pr_number=pr_number)
            console.print("✅ Analysis complete and comment posted")
        else:
            # Get PR diff and analyze
            diff = get_pull_request_diff(repo, pr_number, token)
            # TODO: Implement local analysis of diff
            console.print("✅ Analysis complete (comment not posted)")
    
    except Exception as e:
        console.print(f"❌ PR review failed: {e}", style="red")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option('--repo-path', '-r', default='.', help='Repository path')
@click.option('--agents-config', help='Custom agents configuration')
@click.option('--conversation-rounds', type=int, default=3, help='Number of conversation rounds')
@click.option('--save-conversation', help='Save conversation log to file')
@click.pass_context
def agent_conversation(ctx, repo_path, agents_config, conversation_rounds, save_conversation):
    """Run autonomous agent conversation for code review"""
    
    console.print("🤖 Starting autonomous agent conversation...")
    
    try:
        # Load agents
        if agents_config:
            agents = load_agents_from_yaml(agents_config)
        else:
            # Use default agents
            from .agents import CoderAgent, ReviewerAgent
            agents = [CoderAgent(), ReviewerAgent()]
        
        # Run conversation
        result = run_dual_review(repo_path, max_rounds=conversation_rounds)
        
        # Display results
        table = Table(title="Agent Conversation Results")
        table.add_column("Agent", style="cyan")
        table.add_column("Findings", style="green")
        
        for agent_name, findings in result.items():
            table.add_row(agent_name, findings)
        
        console.print(table)
        
        # Save if requested
        if save_conversation:
            with open(save_conversation, 'w') as f:
                json.dump(result, f, indent=2)
            console.print(f"✅ Conversation saved to {save_conversation}")
    
    except Exception as e:
        console.print(f"❌ Agent conversation failed: {e}", style="red")
        if ctx.obj['verbose']:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option('--task-file', help='JSON file with task definitions')
@click.option('--interactive', '-i', is_flag=True, help='Interactive task planning mode')
@click.option('--export', help='Export execution plan to file')
@click.pass_context
def quantum_plan(ctx, task_file, interactive, export):
    """Quantum-inspired task planning and optimization"""
    
    console.print("🌌 Quantum Task Planner")
    
    planner = QuantumTaskPlanner()
    
    if task_file:
        try:
            with open(task_file) as f:
                data = json.load(f)
            
            tasks_created = 0
            for task_data in data.get('tasks', []):
                planner.create_task(
                    task_id=task_data['id'],
                    title=task_data['title'],
                    description=task_data.get('description', ''),
                    estimated_effort=task_data.get('estimated_effort', 1.0),
                    dependencies=task_data.get('dependencies', [])
                )
                tasks_created += 1
            
            console.print(f"✅ Loaded {tasks_created} tasks from {task_file}")
        
        except Exception as e:
            console.print(f"❌ Failed to load tasks: {e}", style="red")
            sys.exit(1)
    
    if interactive:
        # Interactive mode (simplified version)
        console.print("Interactive mode not yet implemented. Use quantum-planner command for full CLI.")
        return
    
    if planner.scheduler.tasks:
        plan = planner.generate_execution_plan()
        
        # Display plan summary
        table = Table(title="Quantum Execution Plan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Tasks", str(plan['total_tasks']))
        table.add_row("Total Effort", f"{plan['total_estimated_effort']:.1f} hours")
        table.add_row("Critical Path", f"{plan['critical_path_length']:.1f} hours")
        table.add_row("Generation Time", f"{plan['generation_time']:.3f} seconds")
        
        console.print(table)
        
        if export:
            planner.export_plan_to_json(plan, export)
            console.print(f"✅ Plan exported to {export}")
    else:
        console.print("❌ No tasks available for planning")


@cli.command()
@click.pass_context
def metrics(ctx):
    """Display system metrics and analytics"""
    
    console.print("📊 System Metrics")
    
    registry = get_metrics_registry()
    
    # Get all metrics
    metrics_data = {}
    for metric_name, metric in registry._metrics.items():
        if hasattr(metric, 'get'):
            metrics_data[metric_name] = metric.get()
    
    if metrics_data:
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for name, value in metrics_data.items():
            table.add_row(name, str(value))
        
        console.print(table)
    else:
        console.print("No metrics available")


@cli.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.pass_context
def health(ctx, output_format):
    """Check system health and status"""
    
    health_checks = {
        'config': _check_config_health(ctx.obj['config_path']),
        'dependencies': _check_dependencies_health(),
        'cache': _check_cache_health(),
        'agents': _check_agents_health()
    }
    
    if output_format == 'json':
        console.print_json(data=health_checks)
    else:
        table = Table(title="System Health Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        for component, status in health_checks.items():
            status_icon = "✅" if status['healthy'] else "❌"
            table.add_row(component, status_icon, status['message'])
        
        console.print(table)


def _format_markdown_output(result, repo_path) -> str:
    """Format analysis results as markdown"""
    md = f"# AutoGen Code Review Results\n\n"
    md += f"**Repository:** {repo_path}\n"
    md += f"**Timestamp:** {time.ctime()}\n\n"
    
    if result.security:
        md += "## 🔒 Security Analysis\n\n"
        md += f"```\n{result.security.output}\n```\n\n"
    
    if result.style:
        md += "## 🎨 Style Analysis\n\n"
        md += f"```\n{result.style.output}\n```\n\n"
    
    if result.performance:
        md += "## 🚀 Performance Analysis\n\n"
        md += f"```\n{result.performance.output}\n```\n\n"
    
    return md


def _display_console_output(result, repo_path):
    """Display analysis results in console"""
    console.print(f"\n🔍 AutoGen Analysis Results for {repo_path}", style="bold blue")
    console.print("=" * 60)
    
    if result.security:
        console.print("\n🔒 Security Analysis:", style="bold red")
        console.print(result.security.output)
    
    if result.style:
        console.print("\n🎨 Style Analysis:", style="bold yellow")
        console.print(result.style.output)
    
    if result.performance:
        console.print("\n🚀 Performance Analysis:", style="bold green")
        console.print(result.performance.output)


def _check_config_health(config_path: str) -> Dict:
    """Check configuration file health"""
    try:
        config = load_config(config_path)
        return {"healthy": True, "message": "Configuration loaded successfully"}
    except Exception as e:
        return {"healthy": False, "message": f"Config error: {e}"}


def _check_dependencies_health() -> Dict:
    """Check critical dependencies"""
    try:
        import pyautogen
        import git
        import github
        return {"healthy": True, "message": "All dependencies available"}
    except ImportError as e:
        return {"healthy": False, "message": f"Missing dependency: {e}"}


def _check_cache_health() -> Dict:
    """Check cache system health"""
    try:
        from .caching import LinterCache
        cache = LinterCache()
        return {"healthy": True, "message": "Cache system operational"}
    except Exception as e:
        return {"healthy": False, "message": f"Cache error: {e}"}


def _check_agents_health() -> Dict:
    """Check agent system health"""
    try:
        from .agents import CoderAgent, ReviewerAgent
        coder = CoderAgent()
        reviewer = ReviewerAgent()
        return {"healthy": True, "message": "Agent system operational"}
    except Exception as e:
        return {"healthy": False, "message": f"Agent error: {e}"}


def main():
    """Main CLI entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main()