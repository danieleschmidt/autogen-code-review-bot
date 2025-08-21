#!/usr/bin/env python3
"""
Autonomous Execution Runner
Executes the complete autonomous SDLC pipeline with performance optimizations.
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime

from src.autogen_code_review_bot.enhanced_autonomous_executor import (
    EnhancedAutonomousExecutor,
    ExecutionMode
)


async def main():
    """Execute autonomous SDLC with performance monitoring"""
    print("🚀 Starting Autonomous SDLC Execution")
    
    repo_path = Path(__file__).parent.absolute()
    executor = EnhancedAutonomousExecutor()
    
    # Execute with research opportunities
    research_opportunities = [
        "autonomous_agent_optimization",
        "quantum_inspired_algorithms", 
        "performance_benchmarking"
    ]
    
    try:
        start_time = time.time()
        
        result = await executor.execute_autonomous_sdlc(
            str(repo_path),
            mode=ExecutionMode.STANDARD,
            research_opportunities=research_opportunities
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n✅ Autonomous SDLC Execution Complete!")
        print(f"⏱️  Total Time: {execution_time:.2f}s")
        print(f"🧪 Hypotheses Validated: {result.get('hypotheses_validated', 0)}")
        print(f"🔧 Self-Improvement Score: {result.get('self_improvement', {}).get('total_effectiveness_score', 0):.2f}")
        
        # Save execution report
        report_path = repo_path / "AUTONOMOUS_EXECUTION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"📊 Full report saved to: {report_path}")
        
        # Generate comprehensive report
        comprehensive_report = await executor.generate_execution_report()
        
        summary_path = repo_path / "EXECUTION_SUMMARY.json"
        with open(summary_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"📈 Execution summary saved to: {summary_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())