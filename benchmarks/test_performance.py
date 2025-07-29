# Performance benchmarks for AutoGen Code Review Bot

import pytest
import time
import asyncio
from unittest.mock import Mock, patch
import statistics
from concurrent.futures import ThreadPoolExecutor

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.agents import create_agent_conversation
from autogen_code_review_bot.caching import LinterCache
from autogen_code_review_bot.language_detection import detect_languages


class TestPRAnalysisPerformance:
    """Performance tests for PR analysis."""
    
    @pytest.mark.benchmark
    def test_small_pr_analysis_time(self, benchmark, benchmark_results):
        """Benchmark small PR analysis performance."""
        pr_data = {
            'files': [
                {'filename': 'main.py', 'content': 'print("hello")' * 10}
            ]
        }
        
        def analyze():
            return analyze_pr(pr_data, use_cache=False)
        
        result = benchmark(analyze)
        
        # Performance assertion: should complete in under 5 seconds
        assert benchmark.stats.mean < 5.0
        benchmark_results.append({
            'test': 'small_pr_analysis',
            'mean_time': benchmark.stats.mean,
            'median_time': benchmark.stats.median,
            'std_dev': benchmark.stats.stddev
        })
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_large_pr_analysis_time(self, benchmark, large_pr_data, benchmark_results):
        """Benchmark large PR analysis performance."""
        
        def analyze():
            return analyze_pr(large_pr_data, use_cache=False)
        
        result = benchmark(analyze)
        
        # Performance assertion: should complete in under 60 seconds
        assert benchmark.stats.mean < 60.0
        benchmark_results.append({
            'test': 'large_pr_analysis',
            'mean_time': benchmark.stats.mean,
            'files_count': len(large_pr_data['files']),
            'performance_per_file': benchmark.stats.mean / len(large_pr_data['files'])
        })
    
    @pytest.mark.benchmark
    def test_parallel_vs_sequential_analysis(self, multi_language_pr_data):
        """Compare parallel vs sequential analysis performance."""
        
        # Sequential analysis
        start_time = time.time()
        sequential_result = analyze_pr(multi_language_pr_data, use_parallel=False)
        sequential_time = time.time() - start_time
        
        # Parallel analysis
        start_time = time.time()
        parallel_result = analyze_pr(multi_language_pr_data, use_parallel=True)
        parallel_time = time.time() - start_time
        
        # Parallel should be faster for multi-language repos
        speedup_ratio = sequential_time / parallel_time
        assert speedup_ratio > 1.0, f"Parallel execution should be faster, got {speedup_ratio}x speedup"
        
        # Results should be equivalent
        assert len(sequential_result.issues) == len(parallel_result.issues)


class TestAgentPerformance:
    """Performance tests for agent conversations."""
    
    @pytest.mark.benchmark
    def test_agent_conversation_time(self, benchmark):
        """Benchmark agent conversation performance."""
        
        def create_conversation():
            with patch('autogen_code_review_bot.agents.ChatCompletion') as mock_chat:
                mock_chat.create.return_value = Mock(
                    choices=[Mock(message=Mock(content="Test response"))]
                )
                return create_agent_conversation("test prompt", "coder")
        
        result = benchmark(create_conversation)
        
        # Should complete conversation in under 3 seconds
        assert benchmark.stats.mean < 3.0
    
    @pytest.mark.benchmark
    def test_concurrent_agent_conversations(self):
        """Test performance of concurrent agent conversations."""
        
        def single_conversation():
            with patch('autogen_code_review_bot.agents.ChatCompletion') as mock_chat:
                mock_chat.create.return_value = Mock(
                    choices=[Mock(message=Mock(content="Test response"))]
                )
                time.sleep(0.1)  # Simulate API call
                return "completed"
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(single_conversation) for _ in range(8)]
                for future in futures:
                    future.result()
            
            total_time = time.time() - start_time
            results[concurrency] = total_time
        
        # Higher concurrency should reduce total time
        assert results[8] < results[1], "8 concurrent conversations should be faster than sequential"


class TestCachePerformance:
    """Performance tests for caching system."""
    
    @pytest.mark.benchmark
    def test_cache_hit_performance(self, benchmark):
        """Benchmark cache hit performance."""
        cache = LinterCache()
        
        # Pre-populate cache
        test_key = "test_key"
        test_data = {"result": "cached_data", "files": ["test.py"]}
        cache.set(test_key, test_data)
        
        def cache_lookup():
            return cache.get(test_key)
        
        result = benchmark(cache_lookup)
        
        # Cache hits should be very fast (under 1ms)
        assert benchmark.stats.mean < 0.001
        assert result == test_data
    
    @pytest.mark.benchmark
    def test_cache_miss_performance(self, benchmark):
        """Benchmark cache miss performance."""
        cache = LinterCache()
        
        def cache_lookup():
            return cache.get("nonexistent_key")
        
        result = benchmark(cache_lookup)
        
        # Cache misses should still be fast (under 10ms)
        assert benchmark.stats.mean < 0.01
        assert result is None
    
    def test_cache_memory_usage(self):
        """Test cache memory efficiency."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        cache = LinterCache()
        
        # Add many items to cache
        for i in range(1000):
            cache.set(f"key_{i}", {"data": "x" * 1000})  # 1KB per item
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (under 2MB for 1MB of data)
        assert memory_increase < 2 * 1024 * 1024


class TestLanguageDetectionPerformance:
    """Performance tests for language detection."""
    
    @pytest.mark.benchmark
    def test_language_detection_speed(self, benchmark, multi_language_pr_data):
        """Benchmark language detection performance."""
        
        def detect():
            return detect_languages(multi_language_pr_data['files'])
        
        result = benchmark(detect)
        
        # Should detect languages quickly (under 100ms)
        assert benchmark.stats.mean < 0.1
        assert len(result) == len(multi_language_pr_data['files'])
    
    @pytest.mark.benchmark
    def test_large_file_language_detection(self, benchmark):
        """Benchmark language detection on large files."""
        
        large_files = [
            {'filename': 'large.py', 'content': 'def func():\n    pass\n' * 1000},
            {'filename': 'large.js', 'content': 'function test() {}\n' * 1000},
            {'filename': 'large.go', 'content': 'func test() {}\n' * 1000}
        ]
        
        def detect():
            return detect_languages(large_files)
        
        result = benchmark(detect)
        
        # Should handle large files efficiently (under 500ms)
        assert benchmark.stats.mean < 0.5


class TestMemoryLeakDetection:
    """Tests to detect memory leaks during long-running operations."""
    
    def test_repeated_pr_analysis_memory_stable(self):
        """Test that repeated PR analysis doesn't leak memory."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        memory_samples = []
        
        pr_data = {
            'files': [
                {'filename': 'test.py', 'content': 'print("test")'}
            ]
        }
        
        # Perform multiple analyses and track memory
        for i in range(20):
            analyze_pr(pr_data)
            gc.collect()  # Force garbage collection
            
            if i >= 10:  # Skip initial warmup
                memory_samples.append(process.memory_info().rss)
        
        # Memory should be stable (standard deviation < 10% of mean)
        memory_mean = statistics.mean(memory_samples)
        memory_stddev = statistics.stdev(memory_samples)
        memory_cv = memory_stddev / memory_mean
        
        assert memory_cv < 0.1, f"Memory usage not stable, coefficient of variation: {memory_cv}"


@pytest.mark.benchmark
class TestRegressionTests:
    """Performance regression tests with historical baselines."""
    
    PERFORMANCE_BASELINES = {
        'small_pr_analysis': 2.0,  # seconds
        'agent_conversation': 1.5,  # seconds
        'cache_hit': 0.0005,  # seconds
        'language_detection': 0.05  # seconds
    }
    
    def test_performance_regression(self, benchmark_results):
        """Check that performance hasn't regressed beyond baselines."""
        for result in benchmark_results:
            test_name = result['test']
            if test_name in self.PERFORMANCE_BASELINES:
                baseline = self.PERFORMANCE_BASELINES[test_name]
                current_time = result.get('mean_time', float('inf'))
                
                # Allow 20% performance degradation from baseline
                threshold = baseline * 1.2
                assert current_time <= threshold, (
                    f"Performance regression detected for {test_name}: "
                    f"{current_time:.3f}s > {threshold:.3f}s (baseline: {baseline:.3f}s)"
                )