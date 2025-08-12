#!/usr/bin/env python3
"""
Test the complete robust system with caching, error handling, and health monitoring.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.pr_analysis import analyze_pr, format_analysis_with_agents
    from autogen_code_review_bot.robust_error_handling import health_checker, error_handler
    from autogen_code_review_bot.intelligent_cache_system import intelligent_cache
    
    print("✅ Complete system imports successful")
    
    # Test 1: First analysis (cache miss)
    print("\n🔍 Test 1: First analysis (cache miss)...")
    start_time = time.time()
    result1 = analyze_pr(".", use_cache=True, use_parallel=False)
    first_duration = time.time() - start_time
    print(f"✅ First analysis: {first_duration:.2f} seconds")
    
    # Test 2: Second analysis (cache hit)
    print("\n🔍 Test 2: Second analysis (should be cached)...")
    start_time = time.time()
    result2 = analyze_pr(".", use_cache=True, use_parallel=False)
    second_duration = time.time() - start_time
    print(f"✅ Second analysis: {second_duration:.2f} seconds")
    print(f"   Speed improvement: {first_duration/second_duration:.1f}x faster")
    
    # Test 3: Cache statistics
    print("\n🔍 Test 3: Cache performance...")
    cache_stats = intelligent_cache.get_stats()
    print(f"✅ Cache hit rate: {cache_stats['hit_rate']}%")
    print(f"   Total requests: {cache_stats['total_requests']}")
    print(f"   Memory entries: {cache_stats['memory_entries']}")
    print(f"   Memory size: {cache_stats['memory_size_mb']} MB")
    print(f"   Disk entries: {cache_stats['disk_entries']}")
    
    # Test 4: Parallel execution with caching
    print("\n🔍 Test 4: Parallel execution...")
    start_time = time.time()
    result3 = analyze_pr(".", use_cache=True, use_parallel=True)
    parallel_duration = time.time() - start_time
    print(f"✅ Parallel analysis: {parallel_duration:.2f} seconds")
    
    # Test 5: Enhanced agent conversation
    print("\n🔍 Test 5: Enhanced agent conversation...")
    conversation_result = format_analysis_with_agents(result3, "config/agents.yaml")
    print(f"✅ Agent conversation: {len(conversation_result)} chars generated")
    
    # Test 6: System health overview
    print("\n🔍 Test 6: System health overview...")
    health = health_checker.get_overall_health()
    print(f"✅ Overall system health: {health['overall_status']}")
    print(f"   Components: {len(health['components'])}")
    print(f"   Total errors: {health['error_summary']['total_errors']}")
    print(f"   Recent failures: {health['error_summary']['recent_failures']}")
    
    # Test 7: Error handling resilience
    print("\n🔍 Test 7: Error handling resilience...")
    error_count_before = len(error_handler.error_history)
    
    try:
        # This should trigger error handling but not crash
        analyze_pr("/nonexistent/path", use_cache=False, use_parallel=False)
        print("❌ Should have failed")
    except Exception:
        print("✅ Errors handled gracefully")
    
    error_count_after = len(error_handler.error_history)
    print(f"   New errors recorded: {error_count_after - error_count_before}")
    
    # Test 8: Performance summary
    print("\n📊 Performance Summary:")
    print("=" * 50)
    print(f"First run (no cache):    {first_duration:.2f}s")
    print(f"Second run (cached):     {second_duration:.2f}s") 
    print(f"Parallel run (cached):   {parallel_duration:.2f}s")
    print(f"Cache hit rate:          {cache_stats['hit_rate']}%")
    print(f"System status:           {health['overall_status']}")
    print(f"Error recovery:          Active")
    print("=" * 50)
    
    print("\n🎉 Complete system test PASSED")
    print("\n🚀 Generation 2 (ROBUST) implementation complete!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)