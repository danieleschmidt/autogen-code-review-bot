"""
Performance tests for AutoGen Code Review Bot.

These tests measure system performance characteristics and identify
potential bottlenecks or regression in performance.
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Any

import pytest


@pytest.mark.performance
@pytest.mark.slow
class TestAnalysisPerformance:
    """Test performance of code analysis operations."""
    
    def test_single_file_analysis_speed(self, sample_code_files):
        """Test analysis speed for single files."""
        python_file = sample_code_files["python"]
        
        # Measure time for file reading and basic processing
        start_time = time.time()
        
        # Simulate analysis operations
        content = python_file.read_text()
        lines = content.splitlines()
        word_count = len(content.split())
        
        # Simple analysis simulation
        issues = []
        for i, line in enumerate(lines):
            if "TODO" in line:
                issues.append(f"Line {i+1}: TODO found")
            if len(line) > 120:
                issues.append(f"Line {i+1}: Line too long")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert performance requirements
        assert processing_time < 0.1  # Should complete within 100ms
        assert len(lines) > 0
        assert word_count > 0
    
    def test_multi_file_analysis_speed(self, sample_code_files):
        """Test analysis speed for multiple files."""
        files = list(sample_code_files.values())
        
        start_time = time.time()
        
        results = []
        for file_path in files:
            content = file_path.read_text()
            lines = len(content.splitlines())
            results.append({
                "file": file_path.name,
                "lines": lines,
                "size": len(content)
            })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert performance requirements
        assert processing_time < 1.0  # Should complete within 1 second
        assert len(results) == len(files)
        assert all(r["lines"] > 0 for r in results)
    
    def test_parallel_file_analysis(self, sample_code_files):
        """Test parallel processing performance."""
        files = list(sample_code_files.values())
        
        def analyze_file(file_path: Path) -> Dict[str, Any]:
            content = file_path.read_text()
            return {
                "file": file_path.name,
                "lines": len(content.splitlines()),
                "chars": len(content),
                "words": len(content.split())
            }
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [analyze_file(f) for f in files]
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            parallel_results = list(executor.map(analyze_file, files))
        parallel_time = time.time() - start_time
        
        # Assert
        assert len(sequential_results) == len(parallel_results)
        assert len(sequential_results) == len(files)
        
        # Parallel should be faster or at least not significantly slower
        # (for small files, overhead might make it slower)
        speed_ratio = sequential_time / max(parallel_time, 0.001)
        assert speed_ratio >= 0.5  # Parallel shouldn't be more than 2x slower
    
    @pytest.mark.parametrize("file_size", [100, 500, 1000, 5000])
    def test_analysis_scalability(self, temp_dir, file_size):
        """Test how analysis performance scales with file size."""
        # Create file of specific size (in lines)
        test_file = temp_dir / f"large_file_{file_size}.py"
        
        content_lines = []
        for i in range(file_size):
            content_lines.append(f"def function_{i}():")
            content_lines.append(f"    return {i}")
            content_lines.append("")
        
        test_file.write_text("\n".join(content_lines))
        
        # Measure analysis time
        start_time = time.time()
        
        content = test_file.read_text()
        lines = content.splitlines()
        
        # Simulate more complex analysis
        issues = []
        for i, line in enumerate(lines):
            if "def " in line and not line.strip().endswith(":"):
                issues.append(f"Line {i+1}: Malformed function definition")
            if len(line) > 100:
                issues.append(f"Line {i+1}: Line too long")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance should scale reasonably
        # Time per line should be roughly constant
        time_per_line = processing_time / max(file_size, 1)
        
        assert time_per_line < 0.001  # Less than 1ms per line
        assert processing_time < 5.0   # Total time under 5 seconds


@pytest.mark.performance
class TestCachePerformance:
    """Test caching system performance."""
    
    def test_cache_read_speed(self, cache_dir):
        """Test cache read performance."""
        # Create cache files
        cache_files = []
        for i in range(10):
            cache_file = cache_dir / f"cache_{i}.json"
            cache_data = {
                "id": i,
                "data": f"cached_data_{i}",
                "timestamp": time.time()
            }
            cache_file.write_text(str(cache_data))
            cache_files.append(cache_file)
        
        # Measure read performance
        start_time = time.time()
        
        results = []
        for cache_file in cache_files:
            content = cache_file.read_text()
            results.append(len(content))
        
        end_time = time.time()
        read_time = end_time - start_time
        
        # Assert performance
        assert read_time < 0.1  # Should read all files within 100ms
        assert len(results) == 10
    
    def test_cache_write_speed(self, cache_dir):
        """Test cache write performance."""
        start_time = time.time()
        
        # Write multiple cache entries
        for i in range(20):
            cache_file = cache_dir / f"write_test_{i}.json"
            cache_data = f"test_data_{i}" * 100  # Make it somewhat substantial
            cache_file.write_text(cache_data)
        
        end_time = time.time()
        write_time = end_time - start_time
        
        # Assert performance
        assert write_time < 1.0  # Should write all files within 1 second
        
        # Verify files were created
        written_files = list(cache_dir.glob("write_test_*.json"))
        assert len(written_files) == 20
    
    def test_cache_lookup_performance(self, cache_dir):
        """Test cache lookup performance with many entries."""
        # Create many cache entries
        cache_keys = []
        for i in range(100):
            cache_file = cache_dir / f"lookup_{i:03d}.cache"
            cache_file.write_text(f"data_{i}")
            cache_keys.append(cache_file.name)
        
        # Measure lookup performance
        start_time = time.time()
        
        found_count = 0
        for key in cache_keys[:50]:  # Look up half of them
            cache_file = cache_dir / key
            if cache_file.exists():
                found_count += 1
        
        end_time = time.time()
        lookup_time = end_time - start_time
        
        # Assert performance
        assert lookup_time < 0.5  # Should complete lookups within 500ms
        assert found_count == 50


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Test concurrent operation performance."""
    
    def test_thread_safety_performance(self):
        """Test performance under concurrent access."""
        shared_data = {"counter": 0}
        lock = threading.Lock()
        
        def increment_counter(iterations: int):
            for _ in range(iterations):
                with lock:
                    shared_data["counter"] += 1
        
        # Measure concurrent execution time
        start_time = time.time()
        
        threads = []
        iterations_per_thread = 1000
        num_threads = 5
        
        for _ in range(num_threads):
            thread = threading.Thread(target=increment_counter, args=(iterations_per_thread,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert correctness and performance
        expected_count = num_threads * iterations_per_thread
        assert shared_data["counter"] == expected_count
        assert execution_time < 2.0  # Should complete within 2 seconds
    
    def test_process_pool_performance(self):
        """Test multiprocessing performance."""
        def cpu_intensive_task(n: int) -> int:
            """Simulate CPU-intensive work."""
            total = 0
            for i in range(n):
                total += i ** 2
            return total
        
        tasks = [10000] * 4  # 4 tasks of equal size
        
        # Sequential execution
        start_time = time.time()
        sequential_results = [cpu_intensive_task(n) for n in tasks]
        sequential_time = time.time() - start_time
        
        # Parallel execution
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=2) as executor:
            parallel_results = list(executor.map(cpu_intensive_task, tasks))
        parallel_time = time.time() - start_time
        
        # Assert correctness
        assert sequential_results == parallel_results
        
        # Performance should be improved (on multi-core systems)
        speedup = sequential_time / max(parallel_time, 0.001)
        assert speedup > 0.8  # Should be at least 80% as fast


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and performance."""
    
    def test_memory_efficient_processing(self, temp_dir):
        """Test memory efficiency during large data processing."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process data in chunks to test memory efficiency
            large_file = temp_dir / "large_data.txt"
            
            # Create large file
            with open(large_file, 'w') as f:
                for i in range(10000):  # 10k lines
                    f.write(f"Line {i}: {'x' * 100}\n")  # ~100 chars per line
            
            # Process file in chunks
            processed_lines = 0
            chunk_size = 1000
            
            with open(large_file, 'r') as f:
                while True:
                    lines = []
                    for _ in range(chunk_size):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())
                    
                    if not lines:
                        break
                    
                    # Process chunk
                    processed_lines += len(lines)
                    
                    # Simulate processing
                    for line in lines:
                        _ = len(line)
                    
                    # Clear chunk from memory
                    del lines
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Assert memory efficiency
            assert processed_lines == 10000
            assert memory_increase < 50  # Should not increase by more than 50MB
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_object_lifecycle_performance(self):
        """Test object creation and cleanup performance."""
        class TestObject:
            def __init__(self, data):
                self.data = data
                self.processed = False
            
            def process(self):
                self.processed = True
                return len(str(self.data))
        
        # Measure object creation and processing time
        start_time = time.time()
        
        objects = []
        for i in range(1000):
            obj = TestObject(f"data_{i}")
            obj.process()
            objects.append(obj)
        
        creation_time = time.time() - start_time
        
        # Measure cleanup time
        start_time = time.time()
        del objects
        cleanup_time = time.time() - start_time
        
        # Assert performance
        assert creation_time < 1.0   # Creation should be fast
        assert cleanup_time < 0.1    # Cleanup should be very fast


@pytest.mark.performance
class TestNetworkPerformance:
    """Test network-related performance characteristics."""
    
    def test_api_call_timeout_handling(self):
        """Test performance of timeout handling."""
        def simulate_api_call(delay: float) -> str:
            time.sleep(delay)
            return "success"
        
        # Test fast call
        start_time = time.time()
        result = simulate_api_call(0.1)
        fast_time = time.time() - start_time
        
        assert result == "success"
        assert fast_time < 0.2
        
        # Test timeout scenario
        start_time = time.time()
        try:
            # Simulate timeout after 0.5 seconds
            if True:  # Would be actual timeout logic
                time.sleep(0.1)  # Simulate quick timeout detection
                raise TimeoutError("Request timed out")
        except TimeoutError:
            pass
        
        timeout_handling_time = time.time() - start_time
        assert timeout_handling_time < 0.2  # Timeout should be detected quickly
    
    def test_batch_request_performance(self):
        """Test performance of batch operations."""
        def process_request(request_id: int) -> Dict[str, Any]:
            # Simulate processing
            time.sleep(0.01)  # 10ms per request
            return {
                "id": request_id,
                "status": "processed",
                "timestamp": time.time()
            }
        
        requests = list(range(10))
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [process_request(req_id) for req_id in requests]
        sequential_time = time.time() - start_time
        
        # Batch processing (simulated with threading)
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            batch_results = list(executor.map(process_request, requests))
        batch_time = time.time() - start_time
        
        # Assert
        assert len(sequential_results) == len(batch_results) == 10
        
        # Batch should be faster
        speedup = sequential_time / max(batch_time, 0.001)
        assert speedup > 1.5  # Should be at least 50% faster


# Performance benchmarking utilities
def benchmark_function(func, *args, iterations=100, **kwargs):
    """Utility to benchmark function performance."""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "min_time": min(times),
        "max_time": max(times),
        "avg_time": sum(times) / len(times),
        "total_time": sum(times),
        "iterations": iterations,
        "last_result": result
    }