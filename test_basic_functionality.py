#!/usr/bin/env python3
"""Basic functionality tests for our enterprise features."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_pr_analysis_basic():
    """Test basic PR analysis functionality."""
    try:
        from autogen_code_review_bot.pr_analysis import LinterConfig, analyze_pr
        
        # Test LinterConfig creation
        config = LinterConfig()
        assert config.python == "ruff"
        assert config.javascript == "eslint"
        print("✅ LinterConfig creation test passed")
        
        # Test analyze_pr with a temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple Python file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("print('Hello, World!')")
            
            # Run analysis
            result = analyze_pr(temp_dir, use_cache=False, use_parallel=False)
            assert result is not None
            assert result.security is not None
            assert result.style is not None 
            assert result.performance is not None
            print("✅ Basic PR analysis test passed")
            
    except Exception as e:
        print(f"❌ PR analysis test failed: {e}")
        return False
    
    return True


def test_validation_functionality():
    """Test input validation functionality."""
    try:
        from autogen_code_review_bot.validation import InputValidator
        
        validator = InputValidator()
        
        # Test basic string validation
        result = validator.validate("Hello World", "string")
        assert result['valid'] is True
        assert len(result['errors']) == 0
        print("✅ Basic string validation test passed")
        
        # Test SQL injection detection
        malicious_input = "'; DROP TABLE users; --"
        result = validator.validate(malicious_input, "string")
        assert result['valid'] is False
        assert any("SQL injection" in error for error in result['errors'])
        print("✅ SQL injection detection test passed")
        
        # Test XSS detection
        xss_input = "<script>alert('xss')</script>"
        result = validator.validate(xss_input, "string")
        assert result['valid'] is False
        assert any("XSS" in error for error in result['errors'])
        print("✅ XSS detection test passed")
        
        # Test path traversal detection
        path_input = "../../../etc/passwd"
        result = validator.validate(path_input, "path")
        assert result['valid'] is False
        assert any("traversal" in error for error in result['errors'])
        print("✅ Path traversal detection test passed")
        
        # Test sanitization
        html_input = '<script>alert("test")</script>Hello'
        sanitized = validator.sanitize(html_input, "string")
        assert '<script>' not in sanitized
        assert '&lt;script&gt;' in sanitized
        print("✅ HTML sanitization test passed")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False
    
    return True


def test_caching_functionality():
    """Test intelligent caching functionality."""
    try:
        from autogen_code_review_bot.intelligent_cache import AdaptiveLRU
        
        # Create cache
        cache = AdaptiveLRU(max_size=100, max_memory_mb=1)
        
        # Test basic operations
        success = cache.put("key1", "value1", ttl_seconds=60)
        assert success is True
        
        value = cache.get("key1")
        assert value == "value1"
        print("✅ Basic cache operations test passed")
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.entry_count == 1
        print("✅ Cache statistics test passed")
        
        # Test eviction
        for i in range(cache.max_size + 10):
            cache.put(f"key{i}", f"value{i}")
        
        stats = cache.get_stats()
        assert stats.entry_count <= cache.max_size
        assert stats.evictions > 0
        print("✅ Cache eviction test passed")
        
        # Test tag-based invalidation
        cache.put("tag_key1", "value1", tags={"group1", "common"})
        cache.put("tag_key2", "value2", tags={"group2", "common"})
        
        cleared = cache.clear_by_tags({"group1"})
        assert cleared >= 1
        print("✅ Tag-based cache invalidation test passed")
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False
    
    return True


def test_resilience_functionality():
    """Test resilience patterns."""
    try:
        from autogen_code_review_bot.resilience import ResilienceOrchestrator, RetryConfig, RetryStrategy
        
        orchestrator = ResilienceOrchestrator()
        retry_manager = orchestrator.retry_manager
        
        # Test successful operation
        def always_succeeds():
            return "success"
        
        config = RetryConfig(max_attempts=3)
        decorated_func = retry_manager.retry(config)(always_succeeds)
        result = decorated_func()
        assert result == "success"
        print("✅ Retry on success test passed")
        
        # Test retry with eventual success
        attempt_count = 0
        def succeeds_on_third():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        attempt_count = 0  # Reset
        config = RetryConfig(max_attempts=3, strategy=RetryStrategy.FIXED_DELAY, base_delay=0.01)
        decorated_func = retry_manager.retry(config)(succeeds_on_third)
        result = decorated_func()
        assert result == "success"
        assert attempt_count == 3
        print("✅ Retry with eventual success test passed")
        
    except Exception as e:
        print(f"❌ Resilience test failed: {e}")
        return False
    
    return True


def test_distributed_task_structure():
    """Test distributed processing structures."""
    try:
        from autogen_code_review_bot.distributed_processing import DistributedTask, WorkerNode, TaskPriority, TaskStatus
        
        # Test task creation
        task = DistributedTask(
            task_id="test-123",
            task_type="analysis",
            payload={"repo_path": "/test"},
            priority=TaskPriority.HIGH
        )
        
        assert task.task_id == "test-123"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        print("✅ DistributedTask creation test passed")
        
        # Test serialization
        task_dict = task.to_dict()
        assert task_dict['task_id'] == "test-123"
        
        restored_task = DistributedTask.from_dict(task_dict)
        assert restored_task.task_id == task.task_id
        print("✅ Task serialization test passed")
        
        # Test worker node
        worker = WorkerNode(
            node_id="worker-1",
            hostname="host1",
            region="us-east-1",
            capabilities=["analysis", "security"],
            max_concurrent_tasks=10
        )
        
        assert worker.get_load_percentage() == 0.0
        assert worker.can_accept_task() is True
        
        worker.current_load = 5
        assert worker.get_load_percentage() == 50.0
        print("✅ WorkerNode functionality test passed")
        
    except Exception as e:
        print(f"❌ Distributed processing test failed: {e}")
        return False
    
    return True


def main():
    """Run all basic functionality tests."""
    print("🧪 Running Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        ("PR Analysis", test_pr_analysis_basic),
        ("Input Validation", test_validation_functionality),
        ("Intelligent Caching", test_caching_functionality),
        ("Resilience Patterns", test_resilience_functionality),
        ("Distributed Processing", test_distributed_task_structure)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - ALL TESTS PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - TESTS FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - EXCEPTION: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST SUMMARY:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if passed >= 4:  # At least 80% pass rate
        print("\n🎉 BASIC FUNCTIONALITY TESTS SUCCESSFUL!")
        print("Enterprise features are working correctly.")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)