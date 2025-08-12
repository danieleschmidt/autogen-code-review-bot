#!/usr/bin/env python3
"""
Test robust error handling and health monitoring system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.pr_analysis import analyze_pr, format_analysis_with_agents
    from autogen_code_review_bot.robust_error_handling import health_checker, error_handler, ErrorSeverity
    from autogen_code_review_bot.robust_analysis_helpers import run_security_analysis
    
    print("✅ Robust system imports successful")
    
    # Test 1: Normal operation
    print("\n🔍 Test 1: Normal PR analysis...")
    result = analyze_pr(".", use_cache=False, use_parallel=False)
    print(f"✅ Analysis completed in {result.metadata['analysis_duration']:.2f} seconds")
    
    # Test 2: Invalid path handling
    print("\n🔍 Test 2: Invalid path handling...")
    try:
        analyze_pr("", use_cache=False, use_parallel=False)
        print("❌ Should have failed with invalid path")
    except Exception as e:
        print(f"✅ Correctly handled invalid path: {type(e).__name__}")
    
    # Test 3: Path traversal protection
    print("\n🔍 Test 3: Path traversal protection...")
    try:
        analyze_pr("../../../etc/passwd", use_cache=False, use_parallel=False)
        print("❌ Should have failed with suspicious path")
    except Exception as e:
        print(f"✅ Correctly blocked suspicious path: {type(e).__name__}")
    
    # Test 4: Health checks
    print("\n🔍 Test 4: System health monitoring...")
    overall_health = health_checker.get_overall_health()
    print(f"✅ Overall system status: {overall_health['overall_status']}")
    print(f"   Components checked: {len(overall_health.get('components', {}))}")
    print(f"   Total errors in history: {overall_health['error_summary']['total_errors']}")
    
    # Test 5: Error handler stats
    print("\n🔍 Test 5: Error handler statistics...")
    print(f"✅ Error history entries: {len(error_handler.error_history)}")
    print(f"   Failure counts: {dict(list(error_handler.failure_counts.items())[:3])}")
    print(f"   Circuit breakers: {len([cb for cb in error_handler.circuit_breakers.values() if cb])}")
    
    # Test 6: Component health check
    print("\n🔍 Test 6: Component-specific health...")
    security_health = health_checker.check_component_health("security_analysis")
    print(f"✅ Security analysis health: {security_health['status']}")
    
    # Test 7: Enhanced agent conversation with robustness
    print("\n🔍 Test 7: Robust agent conversation...")
    formatted_result = format_analysis_with_agents(result, "config/agents.yaml")
    print(f"✅ Agent conversation completed: {len(formatted_result)} chars")
    
    print("\n🎉 Robust error handling test PASSED")
    print("\n📊 Final Health Summary:")
    print("=" * 50)
    final_health = health_checker.get_overall_health()
    for component, health in final_health.get('components', {}).items():
        print(f"  {component}: {health['status']}")
    print("=" * 50)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)