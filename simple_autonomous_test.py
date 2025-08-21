#!/usr/bin/env python3
"""
Simple Autonomous SDLC Test Runner
Tests the core autonomous functionality without complex dependencies.
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from autogen_code_review_bot.autonomous_sdlc import (
    AutonomousSDLC,
    create_sdlc_config_for_project,
    SDLCGeneration
)


async def run_simple_autonomous_test():
    """Run a simple autonomous SDLC test"""
    print("🚀 Starting Simple Autonomous SDLC Test")
    
    repo_path = Path(__file__).parent.absolute()
    executor = AutonomousSDLC()
    
    try:
        start_time = time.time()
        
        print("📊 Phase 1: Intelligent Analysis")
        analysis_result = await executor.intelligent_analysis(str(repo_path))
        print(f"✅ Analysis complete: {analysis_result['project_info']['type']} project detected")
        
        print("⚙️  Phase 2: SDLC Configuration")
        project_type = analysis_result["project_info"]["type"]
        sdlc_config = create_sdlc_config_for_project(project_type)
        print(f"✅ SDLC Config created: {len(sdlc_config.checkpoints)} checkpoints")
        
        print("🔧 Phase 3: Progressive Enhancement")
        execution_result = await executor.progressive_enhancement_execution(
            str(repo_path), sdlc_config
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n✅ Autonomous SDLC Test Complete!")
        print(f"⏱️  Total Time: {execution_time:.2f}s")
        print(f"📈 Final Generation: {execution_result.get('completed_generation', 'unknown')}")
        
        # Save test results
        test_results = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "execution_time": execution_time,
            "analysis_result": analysis_result,
            "execution_result": execution_result,
            "success": True
        }
        
        report_path = repo_path / "SIMPLE_AUTONOMOUS_TEST_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"📊 Test report saved to: {report_path}")
        return test_results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error report
        error_results = {
            "test_timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "success": False
        }
        
        error_path = repo_path / "AUTONOMOUS_TEST_ERROR.json"
        with open(error_path, 'w') as f:
            json.dump(error_results, f, indent=2, default=str)
        
        return None


if __name__ == "__main__":
    result = asyncio.run(run_simple_autonomous_test())
    if result and result.get("success"):
        print("\n🎉 Simple Autonomous SDLC Test: SUCCESS")
        exit(0)
    else:
        print("\n💥 Simple Autonomous SDLC Test: FAILED")
        exit(1)