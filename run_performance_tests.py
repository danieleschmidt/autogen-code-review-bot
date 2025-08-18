#!/usr/bin/env python3
"""
Performance testing and benchmarking script.

Runs comprehensive performance tests including:
- Load testing
- Memory usage analysis
- Response time benchmarks
- Scaling performance tests
"""

import asyncio
import json
import time
import concurrent.futures
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.pr_analysis import analyze_pr
    from autogen_code_review_bot.quantum_scale_optimizer import optimize_system_performance
    from autogen_code_review_bot.advanced_monitoring import (
        metrics_collector, 
        performance_monitor,
        health_checker
    )
    from autogen_code_review_bot.language_detection import detect_language
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def benchmark_language_detection(iterations: int = 1000) -> Dict[str, float]:
    """Benchmark language detection performance."""
    print(f"🔍 Benchmarking language detection ({iterations} iterations)")
    
    test_files = [
        "script.py", "app.js", "component.tsx", "main.go", 
        "service.java", "handler.rs", "module.rb", "utils.php"
    ]
    
    start_time = time.time()
    
    for i in range(iterations):
        for filename in test_files:
            detect_language(filename)
    
    end_time = time.time()
    total_time = end_time - start_time
    ops_per_second = (iterations * len(test_files)) / total_time
    
    return {
        "total_time": total_time,
        "operations": iterations * len(test_files),
        "ops_per_second": ops_per_second,
        "avg_time_per_op": total_time / (iterations * len(test_files))
    }


def benchmark_pr_analysis(iterations: int = 10) -> Dict[str, float]:
    """Benchmark PR analysis performance."""
    print(f"📊 Benchmarking PR analysis ({iterations} iterations)")
    
    # Use current directory for testing
    repo_path = "."
    times = []
    
    for i in range(iterations):
        print(f"  Running iteration {i+1}/{iterations}")
        
        start_time = time.time()
        try:
            result = analyze_pr(repo_path, use_cache=False)
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"    ✅ Completed in {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            continue
    
    if not times:
        return {"error": "All iterations failed"}
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "iterations": len(times),
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "total_time": sum(times)
    }


def benchmark_concurrent_analysis(max_workers: int = 4) -> Dict[str, float]:
    """Benchmark concurrent PR analysis."""
    print(f"⚡ Benchmarking concurrent analysis ({max_workers} workers)")
    
    repo_path = "."
    num_tasks = max_workers * 2
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = []
        for i in range(num_tasks):
            future = executor.submit(analyze_pr, repo_path, use_cache=False)
            futures.append(future)
        
        # Wait for completion
        completed = 0
        failed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(f"    ✅ Task {completed} completed")
            except Exception as e:
                failed += 1
                print(f"    ❌ Task failed: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "workers": max_workers,
        "tasks": num_tasks,
        "completed": completed,
        "failed": failed,
        "total_time": total_time,
        "tasks_per_second": completed / total_time if total_time > 0 else 0,
        "avg_time_per_task": total_time / completed if completed > 0 else 0
    }


def benchmark_quantum_optimization(iterations: int = 50) -> Dict[str, float]:
    """Benchmark quantum optimization performance."""
    print(f"🚀 Benchmarking quantum optimization ({iterations} iterations)")
    
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        try:
            result = optimize_system_performance()
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"    ❌ Iteration {i+1} failed: {e}")
            continue
    
    if not times:
        return {"error": "All iterations failed"}
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "iterations": len(times),
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "ops_per_second": 1 / avg_time if avg_time > 0 else 0
    }


def benchmark_memory_usage() -> Dict[str, float]:
    """Benchmark memory usage during operations."""
    print("💾 Benchmarking memory usage")
    
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Memory during analysis
    result = analyze_pr(".", use_cache=False)
    analysis_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Memory after cleanup
    del result
    gc.collect()
    cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "baseline_mb": baseline_memory,
        "peak_mb": analysis_memory,
        "after_cleanup_mb": cleanup_memory,
        "memory_increase_mb": analysis_memory - baseline_memory,
        "memory_leaked_mb": cleanup_memory - baseline_memory
    }


def test_health_endpoints() -> Dict[str, float]:
    """Test health check endpoints performance."""
    print("🏥 Testing health check performance")
    
    times = []
    
    for i in range(10):
        start_time = time.time()
        try:
            health_status = health_checker.get_overall_health()
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"    ❌ Health check {i+1} failed: {e}")
            continue
    
    if not times:
        return {"error": "All health checks failed"}
    
    avg_time = sum(times) / len(times)
    
    return {
        "health_checks": len(times),
        "avg_response_time": avg_time,
        "min_response_time": min(times),
        "max_response_time": max(times)
    }


def run_performance_tests():
    """Run comprehensive performance tests."""
    print("🚀 AutoGen Code Review Bot - Performance Tests")
    print("=" * 60)
    
    test_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "summary": {}
    }
    
    # 1. Language Detection Benchmark
    print("\n" + "="*60)
    try:
        lang_results = benchmark_language_detection(500)
        test_results["tests"]["language_detection"] = lang_results
        print(f"✅ Language detection: {lang_results['ops_per_second']:.1f} ops/sec")
    except Exception as e:
        print(f"❌ Language detection benchmark failed: {e}")
        test_results["tests"]["language_detection"] = {"error": str(e)}
    
    # 2. PR Analysis Benchmark
    print("\n" + "="*60)
    try:
        pr_results = benchmark_pr_analysis(5)
        test_results["tests"]["pr_analysis"] = pr_results
        if "error" not in pr_results:
            print(f"✅ PR analysis: {pr_results['avg_time']:.2f}s average")
        else:
            print(f"❌ PR analysis benchmark failed: {pr_results['error']}")
    except Exception as e:
        print(f"❌ PR analysis benchmark failed: {e}")
        test_results["tests"]["pr_analysis"] = {"error": str(e)}
    
    # 3. Concurrent Analysis Benchmark
    print("\n" + "="*60)
    try:
        concurrent_results = benchmark_concurrent_analysis(4)
        test_results["tests"]["concurrent_analysis"] = concurrent_results
        print(f"✅ Concurrent analysis: {concurrent_results['tasks_per_second']:.2f} tasks/sec")
    except Exception as e:
        print(f"❌ Concurrent analysis benchmark failed: {e}")
        test_results["tests"]["concurrent_analysis"] = {"error": str(e)}
    
    # 4. Quantum Optimization Benchmark
    print("\n" + "="*60)
    try:
        quantum_results = benchmark_quantum_optimization(20)
        test_results["tests"]["quantum_optimization"] = quantum_results
        if "error" not in quantum_results:
            print(f"✅ Quantum optimization: {quantum_results['ops_per_second']:.2f} ops/sec")
        else:
            print(f"❌ Quantum optimization benchmark failed: {quantum_results['error']}")
    except Exception as e:
        print(f"❌ Quantum optimization benchmark failed: {e}")
        test_results["tests"]["quantum_optimization"] = {"error": str(e)}
    
    # 5. Memory Usage Benchmark
    print("\n" + "="*60)
    try:
        memory_results = benchmark_memory_usage()
        test_results["tests"]["memory_usage"] = memory_results
        print(f"✅ Memory usage: {memory_results['memory_increase_mb']:.1f}MB increase")
        if memory_results['memory_leaked_mb'] > 10:
            print(f"⚠️  Potential memory leak: {memory_results['memory_leaked_mb']:.1f}MB")
    except Exception as e:
        print(f"❌ Memory usage benchmark failed: {e}")
        test_results["tests"]["memory_usage"] = {"error": str(e)}
    
    # 6. Health Endpoint Performance
    print("\n" + "="*60)
    try:
        health_results = test_health_endpoints()
        test_results["tests"]["health_endpoints"] = health_results
        if "error" not in health_results:
            print(f"✅ Health endpoints: {health_results['avg_response_time']*1000:.1f}ms average")
        else:
            print(f"❌ Health endpoint test failed: {health_results['error']}")
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        test_results["tests"]["health_endpoints"] = {"error": str(e)}
    
    # Performance Summary
    print("\n" + "="*60)
    print("📊 Performance Test Summary")
    print("="*60)
    
    successful_tests = sum(1 for test in test_results["tests"].values() 
                          if "error" not in test)
    total_tests = len(test_results["tests"])
    
    test_results["summary"] = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": total_tests - successful_tests,
        "success_rate": successful_tests / total_tests if total_tests > 0 else 0
    }
    
    print(f"Tests passed: {successful_tests}/{total_tests}")
    print(f"Success rate: {test_results['summary']['success_rate']*100:.1f}%")
    
    # Performance benchmarks achieved
    if "language_detection" in test_results["tests"]:
        lang_test = test_results["tests"]["language_detection"]
        if "ops_per_second" in lang_test and lang_test["ops_per_second"] > 1000:
            print("✅ Language detection performance: EXCELLENT (>1000 ops/sec)")
        elif "ops_per_second" in lang_test and lang_test["ops_per_second"] > 500:
            print("✅ Language detection performance: GOOD (>500 ops/sec)")
        else:
            print("⚠️  Language detection performance: NEEDS IMPROVEMENT")
    
    if "pr_analysis" in test_results["tests"]:
        pr_test = test_results["tests"]["pr_analysis"]
        if "avg_time" in pr_test and pr_test["avg_time"] < 5:
            print("✅ PR analysis performance: EXCELLENT (<5s)")
        elif "avg_time" in pr_test and pr_test["avg_time"] < 15:
            print("✅ PR analysis performance: GOOD (<15s)")
        else:
            print("⚠️  PR analysis performance: NEEDS IMPROVEMENT")
    
    # Save results
    with open("performance_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Results saved to performance_test_results.json")
    
    return test_results["summary"]["success_rate"] >= 0.8


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)