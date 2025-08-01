# Testing Guide

Comprehensive testing strategy and guidelines for AutoGen Code Review Bot.

## Testing Philosophy

Our testing approach follows the testing pyramid:
- **Unit Tests (70%)**: Fast, isolated component tests
- **Integration Tests (20%)**: Component interaction tests  
- **E2E Tests (10%)**: Full workflow tests

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Fast execution (< 1 second per test)
- High code coverage target (95%+)

### Integration Tests (`tests/integration/`)
- Test component interactions
- Mock external services but use real internal components
- Medium execution time (1-30 seconds per test)
- Focus on data flow and API contracts

### End-to-End Tests (`tests/e2e/`)
- Test complete user workflows
- Use real services when possible (with test data)
- Slower execution (30+ seconds per test)
- Focus on business scenarios

### Performance Tests (`benchmarks/`)
- Measure response times and resource usage
- Regression testing against baselines
- Load testing and stress testing
- Memory leak detection

### Security Tests (`tests/security/`)
- Input validation and sanitization
- Authentication and authorization
- Secret handling and exposure
- Vulnerability scanning

## Running Tests

### Basic Commands

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests only
pytest tests/e2e/               # E2E tests only

# Run with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest benchmarks/ --benchmark-only
```

### Test Selection

```bash
# Run by marker
pytest -m unit                  # Unit tests only
pytest -m integration          # Integration tests only
pytest -m "not slow"           # Exclude slow tests

# Run by pattern
pytest tests/ -k "test_github"  # Tests with "github" in name
pytest tests/ -k "agent"       # Tests with "agent" in name

# Run specific file or function
pytest tests/test_agents.py
pytest tests/test_agents.py::test_coder_agent
```

### Debugging Tests

```bash
# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Enter debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Run without capture (show print statements)
pytest -s
```

## Writing Good Tests

### Test Structure (AAA Pattern)

```python
def test_example():
    # Arrange - Set up test data and mocks
    input_data = {"key": "value"}
    expected_result = "expected"
    
    # Act - Execute the function under test
    result = function_under_test(input_data)
    
    # Assert - Verify the results
    assert result == expected_result
```

### Naming Conventions

- Test files: `test_<module_name>.py`
- Test functions: `test_<behavior_being_tested>`
- Test classes: `Test<ComponentName>`

```python
# Good test names
def test_coder_agent_analyzes_python_code():
def test_github_client_handles_rate_limiting():
def test_cache_returns_none_for_missing_key():

# Bad test names  
def test_agent():
def test_github():
def test_cache():
```

## Best Practices Summary

1. **Write tests first** (TDD when possible)
2. **Keep tests independent** (no shared state)
3. **Use descriptive names** (test behavior, not implementation)
4. **Mock external dependencies** (databases, APIs, file system)
5. **Test edge cases** (empty inputs, errors, boundary conditions)
6. **Maintain test quality** (refactor tests like production code)
7. **Run tests frequently** (on every change)
8. **Monitor test metrics** (coverage, execution time, flakiness)