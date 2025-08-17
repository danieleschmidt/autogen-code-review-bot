#!/usr/bin/env python3
"""Quick performance validation."""

import time
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

def quick_performance_test():
    print("⚡ Quick Performance Test")
    print("=" * 30)
    
    # Test 1: Language Detection Speed
    print("\n1. Language Detection")
    try:
        from autogen_code_review_bot.language_detection import detect_language
        
        start = time.time()
        files = ["test.py", "app.js", "main.go"] * 100  # 300 operations
        for filename in files:
            detect_language(filename)
        end = time.time()
        
        ops_per_sec = len(files) / (end - start)
        print(f"✅ {ops_per_sec:.0f} operations/second")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Configuration Loading Speed
    print("\n2. Configuration Loading")
    try:
        from autogen_code_review_bot.global_config import get_config
        
        start = time.time()
        for i in range(50):
            config = get_config()
        end = time.time()
        
        ops_per_sec = 50 / (end - start)
        print(f"✅ {ops_per_sec:.0f} operations/second")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Health Check Speed
    print("\n3. Health Check")
    try:
        from autogen_code_review_bot.advanced_monitoring import health_checker
        
        start = time.time()
        health = health_checker.get_overall_health()
        end = time.time()
        
        response_time = (end - start) * 1000  # ms
        print(f"✅ {response_time:.1f}ms response time")
        print(f"Status: {health.status}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Memory Usage
    print("\n4. Memory Usage")
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"✅ {memory_mb:.1f}MB memory usage")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n🎉 Quick performance test completed!")

if __name__ == "__main__":
    quick_performance_test()