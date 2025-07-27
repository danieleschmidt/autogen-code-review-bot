"""Performance tests for the AutoGen Code Review Bot."""

import time
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.caching import LinterCache


@pytest.mark.performance
class TestPerformance:
    """Performance tests to ensure the system meets speed requirements."""

    def test_small_pr_analysis_performance(self, temp_dir):
        """Test performance for small PR analysis (< 10 files)."""
        repo_path = temp_dir
        
        # Create 5 small Python files
        src_dir = repo_path / "src"
        src_dir.mkdir(parents=True)
        
        for i in range(5):
            file_path = src_dir / f"module_{i}.py"
            content = f"""
def function_{i}():
    '''A simple function for testing.'''
    return {i} * 2

class Class{i}:
    '''A simple class for testing.'''
    
    def method(self):
        return function_{i}()
"""
            file_path.write_text(content)
        
        # Mock linters to simulate fast execution
        with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
            mock_linters.return_value = {
                "python": {
                    "style": {"passed": True, "issues": []},
                    "security": {"passed": True, "issues": []},
                    "type_check": {"passed": True, "issues": []}
                }
            }
            
            # Measure analysis time
            start_time = time.time()
            result = analyze_pr(str(repo_path), use_cache=False, use_parallel=True)
            end_time = time.time()
            
            analysis_time = end_time - start_time
            
            # Assert performance requirement: < 30 seconds for small PRs
            assert analysis_time < 30.0, f"Analysis took {analysis_time:.2f}s, expected < 30s"
            assert result is not None

    def test_medium_pr_analysis_performance(self, temp_dir):
        """Test performance for medium PR analysis (10-50 files)."""
        repo_path = temp_dir
        
        # Create 25 Python files
        src_dir = repo_path / "src"
        src_dir.mkdir(parents=True)
        
        for i in range(25):
            file_path = src_dir / f"module_{i}.py"
            content = f"""
import json
import requests
from typing import Dict, List, Optional

def process_data_{i}(data: List[Dict]) -> Dict:
    '''Process data for module {i}.'''
    result = {{}}
    for item in data:
        key = f"item_{{item.get('id', i)}}"
        result[key] = item.get('value', 0) * {i}
    return result

class DataProcessor{i}:
    '''Data processor for module {i}.'''
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {{}}
        self.cache = {{}}
    
    def process(self, items: List[Dict]) -> List[Dict]:
        '''Process a list of items.'''
        processed = []
        for item in items:
            if item['id'] not in self.cache:
                self.cache[item['id']] = self._expensive_operation(item)
            processed.append(self.cache[item['id']])
        return processed
    
    def _expensive_operation(self, item: Dict) -> Dict:
        '''Simulate expensive operation.'''
        # Simulate some processing time
        result = item.copy()
        result['processed'] = True
        result['timestamp'] = time.time()
        return result
"""
            file_path.write_text(content)
        
        # Mock linters
        with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
            mock_linters.return_value = {
                "python": {
                    "style": {"passed": True, "issues": []},
                    "security": {"passed": True, "issues": []},
                    "type_check": {"passed": True, "issues": []}
                }
            }
            
            # Measure analysis time
            start_time = time.time()
            result = analyze_pr(str(repo_path), use_cache=False, use_parallel=True)
            end_time = time.time()
            
            analysis_time = end_time - start_time
            
            # Assert performance requirement: < 60 seconds for medium PRs
            assert analysis_time < 60.0, f"Analysis took {analysis_time:.2f}s, expected < 60s"
            assert result is not None

    def test_cache_performance_improvement(self, temp_dir):
        """Test that caching provides significant performance improvement."""
        repo_path = temp_dir
        
        # Create test files
        src_dir = repo_path / "src"
        src_dir.mkdir(parents=True)
        
        for i in range(10):
            file_path = src_dir / f"test_{i}.py"
            file_path.write_text(f"def test_function_{i}(): return {i}")
        
        # Mock Git to return consistent commit hash
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.stdout = "abc123def456"
            mock_subprocess.return_value.returncode = 0
            
            # Mock linters with artificial delay
            def slow_linter(*args, **kwargs):
                time.sleep(0.1)  # Simulate linter execution time
                return {
                    "python": {
                        "style": {"passed": True, "issues": []},
                        "security": {"passed": True, "issues": []}
                    }
                }
            
            with patch('autogen_code_review_bot.pr_analysis.run_linters', side_effect=slow_linter):
                # First run (no cache)
                start_time = time.time()
                result1 = analyze_pr(str(repo_path), use_cache=True)
                first_run_time = time.time() - start_time
                
                # Second run (with cache)
                start_time = time.time()
                result2 = analyze_pr(str(repo_path), use_cache=True)
                second_run_time = time.time() - start_time
                
                # Cache should provide significant speedup
                speedup_ratio = first_run_time / second_run_time if second_run_time > 0 else float('inf')
                
                # Assert at least 3x speedup from caching
                assert speedup_ratio >= 3.0, f"Cache speedup was only {speedup_ratio:.2f}x, expected >= 3x"
                assert result1 == result2

    def test_parallel_processing_performance(self, temp_dir):
        """Test that parallel processing improves performance for multi-language repos."""
        repo_path = temp_dir
        
        # Create files in multiple languages
        languages = {
            "python": ("py", "print('Hello Python')"),
            "javascript": ("js", "console.log('Hello JavaScript');"),
            "typescript": ("ts", "const msg: string = 'Hello TypeScript';"),
        }
        
        # Create multiple files for each language
        for lang, (ext, content) in languages.items():
            lang_dir = repo_path / lang
            lang_dir.mkdir(parents=True)
            
            for i in range(5):
                file_path = lang_dir / f"file_{i}.{ext}"
                file_path.write_text(f"{content}\n// File {i}")
        
        # Mock linters with artificial processing time
        def mock_linter_with_delay(*args, **kwargs):
            time.sleep(0.05)  # Simulate processing time per language
            return {
                lang: {
                    "style": {"passed": True, "issues": []},
                    "security": {"passed": True, "issues": []}
                }
                for lang in languages.keys()
            }
        
        with patch('autogen_code_review_bot.pr_analysis.run_linters', side_effect=mock_linter_with_delay):
            # Sequential processing
            start_time = time.time()
            result_sequential = analyze_pr(str(repo_path), use_parallel=False)
            sequential_time = time.time() - start_time
            
            # Parallel processing
            start_time = time.time()
            result_parallel = analyze_pr(str(repo_path), use_parallel=True)
            parallel_time = time.time() - start_time
            
            # Parallel should be faster for multi-language repos
            speedup_ratio = sequential_time / parallel_time if parallel_time > 0 else float('inf')
            
            # Assert at least 1.5x speedup from parallel processing
            assert speedup_ratio >= 1.5, f"Parallel speedup was only {speedup_ratio:.2f}x, expected >= 1.5x"
            assert result_sequential == result_parallel

    def test_memory_usage_performance(self, temp_dir):
        """Test memory usage stays reasonable for large repositories."""
        import psutil
        import os
        
        repo_path = temp_dir
        
        # Create a large number of files
        src_dir = repo_path / "src"
        src_dir.mkdir(parents=True)
        
        # Create 100 files with substantial content
        for i in range(100):
            file_path = src_dir / f"large_file_{i}.py"
            content = f"""
# Large file {i} with substantial content
import sys
import os
import json
import threading
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataClass{i}:
    '''Data class for file {i}.'''
    id: int
    name: str
    values: List[float]
    metadata: Dict[str, Any]
    
    def process(self) -> Dict[str, Any]:
        '''Process the data.'''
        result = {{
            'id': self.id,
            'name': self.name,
            'sum': sum(self.values),
            'count': len(self.values),
            'avg': sum(self.values) / len(self.values) if self.values else 0
        }}
        result.update(self.metadata)
        return result

def complex_function_{i}(data: List[DataClass{i}]) -> Dict[str, Any]:
    '''Complex function for file {i}.'''
    results = []
    for item in data:
        processed = item.process()
        if processed['avg'] > 10:
            results.append(processed)
    
    return {{
        'processed_count': len(results),
        'total_sum': sum(r['sum'] for r in results),
        'items': results
    }}

# Add more content to make file larger
class LargeClass{i}:
    '''A large class with many methods.'''
    
    def __init__(self):
        self.data = []
        self.cache = {{}}
        self.stats = {{}}
    
    def method_a(self): pass
    def method_b(self): pass
    def method_c(self): pass
    def method_d(self): pass
    def method_e(self): pass
""" * 3  # Triple the content
            file_path.write_text(content)
        
        # Measure memory usage during analysis
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
            mock_linters.return_value = {
                "python": {
                    "style": {"passed": True, "issues": []},
                    "security": {"passed": True, "issues": []}
                }
            }
            
            result = analyze_pr(str(repo_path), use_cache=False)
            
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable (< 500MB increase)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB, expected < 500MB"
        assert result is not None

    def test_cache_size_limits(self):
        """Test that cache size limits are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            
            # Create cache with small size limit
            cache = LinterCache(cache_dir=str(cache_dir), max_size_mb=1)
            
            # Add many cache entries to exceed limit
            for i in range(100):
                key = f"test_key_{i}"
                large_data = {"data": "x" * 1000}  # 1KB per entry
                cache.set(key, large_data)
            
            # Cache should have cleaned up old entries
            cache.cleanup()
            
            # Check that cache size is reasonable
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            total_size_mb = total_size / 1024 / 1024
            
            # Should be close to limit but not exceed significantly
            assert total_size_mb <= 2.0, f"Cache size {total_size_mb:.1f}MB exceeds reasonable limit"