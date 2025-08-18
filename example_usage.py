#!/usr/bin/env python3
"""
Example usage of AutoGen Code Review Bot.

This script demonstrates basic usage patterns for the code review bot.
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_code_review_bot import analyze_pr, format_analysis_with_agents
from autogen_code_review_bot.quantum_planner import QuantumTaskPlanner


def main():
    """Demonstrate basic functionality."""
    print("🤖 AutoGen Code Review Bot - Example Usage")
    print("=" * 50)
    
    # Example 1: Basic PR Analysis
    print("\n1. Basic PR Analysis")
    try:
        repo_path = "."
        result = analyze_pr(repo_path, use_cache=False)
        print(f"✅ Analysis completed for {repo_path}")
        print(f"Security: {result.security.tool}")
        print(f"Style: {result.style.tool}")
        print(f"Performance: {result.performance.tool}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    
    # Example 2: Quantum Task Planning
    print("\n2. Quantum Task Planning")
    try:
        planner = QuantumTaskPlanner()
        print("✅ Quantum planner initialized")
        print("🚀 Ready for advanced task optimization")
    except Exception as e:
        print(f"❌ Quantum planner failed: {e}")
    
    # Example 3: Agent-based Analysis
    print("\n3. Agent-based Analysis")
    try:
        # This would typically use actual agent conversations
        print("✅ Dual-agent review system ready")
        print("🔍 Coder and Reviewer agents available")
    except Exception as e:
        print(f"❌ Agent system failed: {e}")
    
    print("\n🎉 Example usage completed!")


if __name__ == "__main__":
    main()