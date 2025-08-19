#!/usr/bin/env python3
"""
Autonomous SDLC Execution Engine
Demonstrates the complete autonomous software development lifecycle execution
with progressive enhancement through all three generations.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from autogen_code_review_bot.autonomous_sdlc import (
    AutonomousSDLC,
    SDLCGeneration, 
    create_sdlc_config_for_project,
    QualityGate
)


class AutonomousSDLCDemo:
    """Demonstration of autonomous SDLC execution capabilities"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.sdlc_engine = AutonomousSDLC()
        self.execution_results = {}

    async def run_complete_sdlc_cycle(self) -> Dict[str, Any]:
        """Execute the complete autonomous SDLC cycle"""
        print("🚀 Starting Autonomous SDLC Execution")
        print("=" * 60)
        
        cycle_start = time.time()

        # STEP 1: Intelligent Analysis
        print("\n🧠 Phase 1: INTELLIGENT ANALYSIS")
        print("-" * 40)
        
        analysis_result = await self.sdlc_engine.intelligent_analysis(str(self.repo_path))
        self.execution_results["analysis"] = analysis_result
        
        print(f"📊 Project Type: {analysis_result['project_info']['type']}")
        print(f"📈 Complexity: {analysis_result['project_info']['complexity']}")
        print(f"🔍 Languages: {', '.join(analysis_result['project_info']['languages'])}")
        print(f"📝 Status: {analysis_result['implementation_status']['status']}")
        print(f"⏱️  Analysis Time: {analysis_result['analysis_time']:.3f}s")

        # STEP 2: Configure SDLC Pipeline
        print("\n⚙️  Phase 2: SDLC CONFIGURATION")
        print("-" * 40)
        
        project_type = analysis_result['project_info']['type']
        research_mode = analysis_result['domain_analysis']['purpose'] in ['data_analysis', 'security']
        
        sdlc_config = create_sdlc_config_for_project(project_type, research_mode)
        
        # Add enhanced quality gates
        enhanced_gates = [
            QualityGate(
                name="documentation_quality", 
                description="Documentation completeness and quality",
                enabled=True
            ),
            QualityGate(
                name="dependency_health",
                description="Dependency security and version health",
                enabled=True
            )
        ]
        sdlc_config.quality_gates.extend(enhanced_gates)
        
        print(f"🎯 Target Generation: {sdlc_config.target_generation.value}")
        print(f"🔄 Checkpoints: {[c.value for c in sdlc_config.checkpoints]}")
        print(f"✅ Quality Gates: {len(sdlc_config.quality_gates)} configured")
        print(f"🔬 Research Mode: {'Enabled' if research_mode else 'Disabled'}")

        # STEP 3: Progressive Enhancement Execution
        print("\n🚀 Phase 3: PROGRESSIVE ENHANCEMENT EXECUTION")
        print("-" * 40)
        
        execution_results = await self.sdlc_engine.progressive_enhancement_execution(
            str(self.repo_path), sdlc_config
        )
        self.execution_results["execution"] = execution_results
        
        # Display results for each generation
        for gen_key, gen_result in execution_results.items():
            if gen_key.startswith("generation_"):
                gen_num = gen_key.split("_")[1]
                print(f"\n🔄 Generation {gen_num} Results:")
                print(f"   ⏱️  Time: {gen_result.get('generation_time', 0):.3f}s")
                print(f"   ✅ Status: {gen_result.get('status', 'unknown')}")
                
                # Show quality gate results
                quality_gates = {k: v for k, v in gen_result.items() if k.endswith('_quality_gates')}
                if quality_gates:
                    for gate_group, gates in quality_gates.items():
                        passed = sum(1 for g in gates.values() if g.get('status') == 'passed')
                        total = len(gates)
                        print(f"   📊 Quality Gates: {passed}/{total} passed")

        # STEP 4: Final System Health Check
        print("\n🏥 Phase 4: SYSTEM HEALTH VALIDATION")
        print("-" * 40)
        
        final_health = await self.comprehensive_health_check()
        self.execution_results["health_check"] = final_health
        
        total_time = time.time() - cycle_start
        self.execution_results["total_execution_time"] = total_time
        
        # STEP 5: Final Report
        print("\n📋 EXECUTION SUMMARY")
        print("=" * 60)
        print(f"🎯 Final Generation: {execution_results.get('completed_generation', 'unknown')}")
        print(f"⏱️  Total Time: {total_time:.3f}s")
        print(f"✅ Overall Health: {final_health.get('overall_status', 'unknown')}")
        print(f"📊 Health Score: {final_health.get('health_score', 0)}/100")
        
        if final_health.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in final_health['recommendations'][:5]:
                print(f"   • {rec}")

        return self.execution_results

    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health validation"""
        health_checks = {
            "code_runs": await self.sdlc_engine._validate_code_execution(self.repo_path, "optimized"),
            "tests_pass": await self.sdlc_engine._validate_test_suite(self.repo_path, "optimized"), 
            "security_scan": await self.sdlc_engine._validate_security_posture(self.repo_path, "optimized"),
            "performance_benchmark": await self.sdlc_engine._validate_performance_benchmarks(self.repo_path, "optimized"),
            "documentation_quality": await self.sdlc_engine._validate_documentation(self.repo_path, "optimized"),
            "dependency_health": await self.sdlc_engine._validate_dependencies(self.repo_path, "optimized")
        }
        
        # Calculate overall health score
        total_score = 0
        max_score = 0
        recommendations = []
        
        for check_name, result in health_checks.items():
            if result.get("status") == "passed":
                total_score += 100
            elif result.get("status") == "warning":
                total_score += 70
            elif result.get("status") == "failed":
                total_score += 30
            else:  # error
                total_score += 10
                
            max_score += 100
            
            # Collect recommendations
            if result.get("recommendations"):
                recommendations.extend(result["recommendations"])
        
        overall_score = (total_score / max_score * 100) if max_score > 0 else 0
        
        if overall_score >= 90:
            overall_status = "excellent"
        elif overall_score >= 75:
            overall_status = "good"
        elif overall_score >= 60:
            overall_status = "needs_improvement"
        else:
            overall_status = "critical"
            
        return {
            "overall_status": overall_status,
            "health_score": round(overall_score),
            "individual_checks": health_checks,
            "recommendations": list(set(recommendations)),
            "validation_time": time.time()
        }

    def save_results(self, output_file: str = "autonomous_sdlc_results.json") -> None:
        """Save execution results to file"""
        output_path = self.repo_path / output_file
        
        # Convert any non-serializable objects
        serializable_results = json.loads(json.dumps(self.execution_results, default=str))
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"\n💾 Results saved to: {output_path}")


async def main():
    """Main execution function"""
    # Use current repository as the target
    repo_path = Path(__file__).parent
    
    print("🤖 Autonomous SDLC Execution Engine")
    print("🏢 Powered by Terragon Labs")
    print(f"📂 Target Repository: {repo_path}")
    
    # Create and run the SDLC demo
    sdlc_demo = AutonomousSDLCDemo(str(repo_path))
    
    try:
        results = await sdlc_demo.run_complete_sdlc_cycle()
        
        # Save results
        sdlc_demo.save_results("autonomous_sdlc_execution_results.json")
        
        print("\n🎉 Autonomous SDLC Execution Complete!")
        print("📊 Check the results file for detailed analytics")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Execution failed with error: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # Still try to save partial results
        if hasattr(sdlc_demo, 'execution_results') and sdlc_demo.execution_results:
            try:
                sdlc_demo.save_results("autonomous_sdlc_partial_results.json")
                print("💾 Partial results saved for debugging")
            except:
                pass
                
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)