# Pytest configuration for performance benchmarks

import pytest
import json
import os
from pathlib import Path

def pytest_configure(config):
    """Configure pytest for benchmark tests."""
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running benchmarks"
    )

@pytest.fixture
def benchmark_results():
    """Fixture to collect benchmark results."""
    results = []
    yield results
    
    # Save results to file if requested
    if os.getenv('SAVE_BENCHMARK_RESULTS'):
        output_file = Path('benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

@pytest.fixture
def large_pr_data():
    """Fixture providing large PR data for performance testing."""
    return {
        'number': 1234,
        'repository': 'test/large-repo',
        'files_changed': 50,
        'lines_added': 2000,
        'lines_deleted': 500,
        'files': [
            {
                'filename': f'src/module_{i}.py',
                'changes': 100,
                'content': 'def function():\n    pass\n' * 50
            }
            for i in range(50)
        ]
    }

@pytest.fixture
def multi_language_pr_data():
    """Fixture providing multi-language PR data."""
    return {
        'number': 5678,
        'repository': 'test/multi-lang-repo',
        'files': [
            {'filename': 'main.py', 'language': 'python', 'content': 'print("hello")'},
            {'filename': 'app.js', 'language': 'javascript', 'content': 'console.log("hello")'},
            {'filename': 'main.go', 'language': 'go', 'content': 'package main\nfunc main() {}'},
            {'filename': 'app.rb', 'language': 'ruby', 'content': 'puts "hello"'},
            {'filename': 'main.rs', 'language': 'rust', 'content': 'fn main() {}'},
        ]
    }