# Test Fixtures

This directory contains test data and fixtures used by the test suite.

## Directory Structure

```
fixtures/
├── github/           # GitHub API response fixtures
├── code_samples/     # Sample code files for testing
├── configs/          # Sample configuration files
├── pr_data/          # Pull request test data
└── agent_responses/  # Mock agent conversation data
```

## Usage

Test fixtures can be loaded using the pytest fixtures defined in `conftest.py`:

```python
def test_pr_analysis(sample_pr_data, sample_github_files):
    # Use the fixture data in your test
    result = analyze_pr(sample_pr_data, sample_github_files)
    assert result is not None
```

## Adding New Fixtures

1. Create appropriately named JSON or text files
2. Update `conftest.py` if new fixture functions are needed
3. Document the fixture purpose and structure
4. Ensure sensitive data is sanitized or mocked

## Fixture Guidelines

- **No Secrets**: Never include real API keys, tokens, or credentials
- **Realistic Data**: Fixtures should represent realistic production data
- **Minimal Size**: Keep fixtures as small as possible while remaining useful
- **Clear Names**: Use descriptive names that indicate the fixture purpose
- **Documentation**: Document complex fixtures with inline comments
EOF < /dev/null
