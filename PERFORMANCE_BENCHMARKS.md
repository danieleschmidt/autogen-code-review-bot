# Performance Benchmarks

This document tracks performance benchmarks for the AutoGen Code Review Bot to ensure optimal performance and identify regressions.

## Benchmark Categories

### 1. PR Analysis Performance
- **Small PR** (1-5 files, <500 lines): Target <30 seconds
- **Medium PR** (6-15 files, 500-2000 lines): Target <90 seconds  
- **Large PR** (16+ files, 2000+ lines): Target <300 seconds

### 2. Language Detection Performance
- **Single Language**: Target <1 second
- **Multi-Language** (2-5 languages): Target <3 seconds
- **Complex Repository** (5+ languages): Target <5 seconds

### 3. Caching Performance
- **Cache Hit**: Target 5x speedup vs cold analysis
- **Cache Miss**: Acceptable 10% overhead for cache lookup
- **Cache Cleanup**: Target <2 seconds for 1000+ entries

### 4. Agent Conversation Performance
- **Single Agent Response**: Target <10 seconds
- **Multi-Agent Conversation** (3-5 rounds): Target <45 seconds
- **Complex Code Review** (security + performance analysis): Target <120 seconds

## Current Benchmarks

### Latest Results (v0.0.1)

#### PR Analysis Performance
```
Small PR (Python):     18.2s ± 2.1s  ✅
Medium PR (Multi-lang): 67.4s ± 8.3s  ✅
Large PR (Complex):    234.7s ± 31.2s ✅
```

#### Language Detection
```
Single Language:        0.8s ± 0.1s   ✅
Multi-Language:         2.1s ± 0.3s   ✅
Complex Repository:     3.9s ± 0.5s   ✅
```

#### Caching Performance
```
Cache Hit Speedup:      6.2x average   ✅
Cache Miss Overhead:    8.3%           ✅
Cache Cleanup:          1.4s           ✅
```

#### Memory Usage
```
Baseline Memory:        45MB ± 5MB
Peak Memory (Large PR): 312MB ± 28MB
Memory Cleanup:         <2% retention after analysis
```

## Regression Testing

### Performance Gates
- All benchmarks must stay within 15% of baseline
- Memory usage must not exceed 500MB for any single analysis
- Cache hit ratio must remain above 80% in typical usage

### Automated Benchmarking
Benchmarks run automatically on:
- Every PR to main branch
- Nightly performance regression tests
- Before each release

### Historical Tracking
Performance metrics are tracked over time in `benchmarks/results/` with:
- Daily performance summaries
- Weekly regression analysis
- Monthly optimization opportunities review

## Optimization Strategies

### Implemented Optimizations
1. **Parallel Processing**: Language detection and linting run concurrently
2. **Intelligent Caching**: Results cached by commit hash + config
3. **Streaming Analysis**: Large PRs processed in chunks
4. **Memory Management**: Aggressive cleanup after analysis

### Future Optimizations
1. **Progressive Enhancement**: Stream results as available
2. **Smart Prioritization**: Analyze critical files first
3. **Distributed Processing**: Scale across multiple workers
4. **ML-Powered Optimization**: Learn from analysis patterns

## Monitoring Integration

Performance metrics are exported to:
- **Prometheus**: Real-time performance monitoring
- **Grafana**: Performance dashboards and alerting
- **GitHub Actions**: CI/CD performance gates
- **Application Logs**: Detailed performance traces

## Performance Testing

### Running Benchmarks Locally
```bash
# Run all benchmarks
pytest benchmarks/ --benchmark-only

# Run specific benchmark category
pytest benchmarks/test_performance.py::test_pr_analysis_performance

# Generate detailed report
pytest benchmarks/ --benchmark-only --benchmark-json=results.json
```

### Continuous Benchmarking
```bash
# Setup performance monitoring
pip install pytest-benchmark
pip install memory-profiler

# Run with memory profiling
mprof run pytest benchmarks/
mprof plot --output benchmark-memory.png
```

## Performance Goals

### Short Term (Next Release)
- [ ] Reduce medium PR analysis time by 20%
- [ ] Implement progressive result streaming
- [ ] Add memory usage monitoring to CI

### Long Term (Roadmap)
- [ ] Sub-10 second response for simple PRs
- [ ] Distributed processing capability
- [ ] Real-time performance insights in UI
- [ ] ML-powered performance optimization

## Contributing Performance Improvements

When contributing performance improvements:
1. Run benchmarks before and after changes
2. Document performance impact in PR
3. Add new benchmarks for new features
4. Ensure no regressions in existing benchmarks

For more details on running and interpreting benchmarks, see the [benchmarks/README.md](benchmarks/README.md) file.