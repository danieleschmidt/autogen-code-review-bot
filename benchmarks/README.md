# Performance Benchmarks

This directory contains performance testing and benchmarking tools for the AutoGen Code Review Bot.

## Overview

The benchmark suite includes:
- **Unit performance tests**: Individual component benchmarks
- **Integration performance tests**: End-to-end performance testing
- **Load testing**: Concurrent user simulation
- **Memory leak detection**: Long-running stability tests
- **Regression testing**: Performance baseline comparison

## Quick Start

### Install Dependencies

```bash
# Install benchmark-specific dependencies
pip install pytest-benchmark aiohttp psutil

# Install the bot in development mode
pip install -e .[dev]
```

### Run Performance Tests

```bash
# Run all performance benchmarks
pytest benchmarks/ --benchmark-only

# Run with detailed statistics
pytest benchmarks/ --benchmark-only --benchmark-verbose

# Save benchmark results
pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Run only fast benchmarks (exclude slow tests)
pytest benchmarks/ --benchmark-only -m "not slow"
```

### Run Load Tests

```bash
# Start the bot service first
python bot.py &

# Run load tests
python benchmarks/load_test.py

# Run specific load test
python benchmarks/load_test.py --test health
python benchmarks/load_test.py --test webhook
python benchmarks/load_test.py --test metrics

# Test against different URL
python benchmarks/load_test.py --url http://staging.example.com:8080
```

## Benchmark Categories

### 1. PR Analysis Performance (`test_performance.py`)

Tests the core PR analysis functionality:

- **Small PR Analysis**: < 5 seconds target
- **Large PR Analysis**: < 60 seconds target  
- **Parallel vs Sequential**: Measures parallelization benefits
- **Memory Stability**: Ensures no memory leaks

```bash
# Run PR analysis benchmarks
pytest benchmarks/test_performance.py::TestPRAnalysisPerformance --benchmark-only
```

### 2. Agent Performance (`test_performance.py`)

Tests agent conversation performance:

- **Agent Conversation Time**: < 3 seconds target
- **Concurrent Conversations**: Scalability testing
- **Token Usage Efficiency**: Resource optimization

```bash
# Run agent benchmarks
pytest benchmarks/test_performance.py::TestAgentPerformance --benchmark-only
```

### 3. Cache Performance (`test_performance.py`)

Tests caching system efficiency:

- **Cache Hit Performance**: < 1ms target
- **Cache Miss Performance**: < 10ms target
- **Memory Usage**: Efficiency validation

```bash
# Run cache benchmarks
pytest benchmarks/test_performance.py::TestCachePerformance --benchmark-only
```

### 4. Load Testing (`load_test.py`)

Simulates concurrent user load:

- **Health Check Load**: 10 concurrent users, 20 requests each
- **Webhook Load**: 5 concurrent users, 10 requests each
- **Metrics Load**: 20 concurrent users, 50 requests each

```bash
# Full load test suite
python benchmarks/load_test.py

# Results saved to load_test_results.json
```

## Performance Targets

### Response Time Targets

| Component | Target | Critical |
|-----------|--------|----------|
| Small PR Analysis | < 5s | < 10s |
| Large PR Analysis | < 60s | < 120s |
| Agent Conversation | < 3s | < 5s |
| Cache Hit | < 1ms | < 5ms |
| Health Check | < 100ms | < 500ms |

### Throughput Targets

| Endpoint | Target RPS | Critical RPS |
|----------|------------|--------------|
| Health Check | > 50 | > 10 |
| Metrics | > 20 | > 5 |
| Webhook | > 5 | > 1 |

### Resource Targets

| Resource | Target | Critical |
|----------|--------|----------|
| Memory Usage | < 512MB | < 1GB |
| CPU Usage | < 50% | < 80% |
| Error Rate | < 1% | < 5% |

## Regression Testing

The benchmark suite includes regression detection:

```python
PERFORMANCE_BASELINES = {
    'small_pr_analysis': 2.0,  # seconds
    'agent_conversation': 1.5,  # seconds
    'cache_hit': 0.0005,       # seconds
    'language_detection': 0.05  # seconds
}
```

Regression tests fail if performance degrades beyond 20% of baseline.

## Continuous Integration

Add to CI pipeline (`.github/workflows/performance.yml`):

```yaml
name: Performance Tests
on:
  pull_request:
  push:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install pytest-benchmark aiohttp psutil
      
      - name: Run performance tests
        run: |
          pytest benchmarks/ --benchmark-only --benchmark-json=benchmark.json
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark.json
      
      - name: Performance regression check
        run: |
          python scripts/check_performance_regression.py benchmark.json
```

## Profiling

For detailed performance profiling:

```bash
# Profile with cProfile
python -m cProfile -o profile.stats bot.py

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler bot.py

# Line-by-line profiling
pip install line_profiler
kernprof -l -v bot.py
```

## Monitoring Integration

Performance metrics can be integrated with monitoring:

```python
# Custom performance metrics
from prometheus_client import Histogram

performance_histogram = Histogram(
    'benchmark_duration_seconds',
    'Benchmark execution time',
    ['benchmark_name']
)

# Usage in benchmark
with performance_histogram.labels(benchmark_name='pr_analysis').time():
    analyze_pr(pr_data)
```

## Best Practices

### Writing Performance Tests

1. **Isolate tests**: Each test should be independent
2. **Use realistic data**: Test with production-like data sizes
3. **Warm up**: Allow for JIT compilation and cache warming
4. **Multiple iterations**: Run tests multiple times for accuracy
5. **Resource cleanup**: Ensure tests don't leak resources

### Interpreting Results

1. **Focus on percentiles**: 95th/99th percentiles matter more than averages
2. **Consider variance**: High standard deviation indicates inconsistent performance
3. **Resource correlation**: High CPU/memory usage affects response times
4. **Scalability**: Test performance under different load levels

### Performance Optimization

1. **Profile first**: Identify bottlenecks before optimizing
2. **Measure impact**: Validate optimizations with benchmarks
3. **Parallel processing**: Utilize multiple cores for CPU-bound tasks
4. **Caching**: Cache expensive operations
5. **Connection pooling**: Reuse HTTP connections
6. **Async I/O**: Use asyncio for I/O-bound operations

## Troubleshooting

### Common Issues

1. **Inconsistent results**: Check for background processes, use isolated environment
2. **Memory leaks**: Run with `--benchmark-disable-gc` to detect GC pressure
3. **Network timeouts**: Increase timeout values for load tests
4. **Resource exhaustion**: Monitor system resources during tests

### Debug Commands

```bash
# Check system resources
top -p $(pgrep -f "python.*bot.py")

# Monitor network connections
netstat -an | grep 8080

# Check memory usage
ps aux | grep python

# Monitor file descriptors
lsof -p $(pgrep -f "python.*bot.py")
```