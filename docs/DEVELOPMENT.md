# Development Guide

This guide helps developers set up and work with the AutoGen Code Review Bot.

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd autogen-code-review-bot
   make init  # Complete development setup
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start Development**
   ```bash
   make dev-start  # Start bot in development mode
   ```

## Development Environment

### Devcontainer Support

This project includes full devcontainer support for consistent development environments:

- **VS Code**: Use "Reopen in Container" for full setup
- **GitHub Codespaces**: One-click development environment
- **Local Docker**: `docker-compose -f .devcontainer/docker-compose.yml up`

### Manual Setup

```bash
# Install Python 3.8+
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
make install

# Setup pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Code Quality

Run before committing:
```bash
make format     # Format code with ruff
make lint       # Run linting checks
make test       # Run test suite
make security   # Security scans
```

Or run everything:
```bash
make ci-local   # Full CI pipeline locally
```

### 2. Testing

```bash
make test              # Full test suite with coverage
make test-fast         # Quick tests without coverage
make test-integration  # Integration tests only
make test-benchmark    # Performance benchmarks
```

### 3. Pre-commit Hooks

Automatically runs on each commit:
- Code formatting (ruff)
- Import sorting (isort)
- Security scanning (bandit)
- Secret detection (detect-secrets)
- YAML validation
- Markdown linting

Force run on all files:
```bash
make pre-commit
```

## Architecture Overview

```
src/autogen_code_review_bot/
├── agents.py              # Core agent implementations
├── github_integration.py  # GitHub API integration
├── pr_analysis.py         # Pull request analysis
├── caching.py             # Intelligent caching system
├── monitoring.py          # Metrics and observability
└── ...
```

### Key Components

1. **Dual Agent System**
   - Coder Agent: Focuses on functionality and bugs
   - Reviewer Agent: Emphasizes quality and security

2. **GitHub Integration**
   - Webhook handling for PR events
   - Comment posting and status updates
   - Repository access and file analysis

3. **Multi-language Support**
   - Python, JavaScript, TypeScript, Go, Rust, Ruby
   - Configurable linter mapping
   - Language detection utilities

4. **Performance Optimization**
   - Intelligent caching by commit hash
   - Parallel processing for multi-language repos
   - Configurable worker pools

## Configuration

### Agent Configuration

```yaml
# agent_config.yaml
agents:
  coder:
    model: "gpt-4"
    temperature: 0.3
    focus_areas: ["functionality", "bugs", "edge_cases"]
  
  reviewer:
    model: "gpt-4"
    temperature: 0.1
    focus_areas: ["security", "performance", "standards"]
```

### Linter Configuration

```yaml
# linters.yaml
linters:
  python: ruff
  javascript: eslint
  typescript: eslint
  go: golangci-lint
  ruby: rubocop
```

## IDE Integration

### VS Code

The repository includes comprehensive VS Code configuration:

- **Settings**: Auto-formatting, linting, testing
- **Tasks**: Common development tasks
- **Launch**: Debug configurations for bot and tests
- **Extensions**: Recommended extensions for Python development

### Other IDEs

- **PyCharm**: Import as Python project, configure interpreters
- **Vim/Neovim**: LSP support via pyright/pylsp
- **Emacs**: Python mode with elpy or eglot

## Debugging

### Local Debugging

1. **Bot Debugging**
   ```bash
   python bot.py --config agent_config.yaml --debug
   ```

2. **PR Analysis**
   ```bash
   python -m autogen_code_review_bot.cli review-pr --pr-number 123 --repo owner/repo --debug
   ```

3. **VS Code Debugging**
   - Use provided launch configurations
   - Set breakpoints in Python code
   - Debug tests with integrated test runner

### Performance Profiling

```bash
make profile  # Generate performance profile
# View with: python -m pstats profile.stats
```

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/test_*.py`)
   - Individual component testing
   - Mock external dependencies
   - Fast execution

2. **Integration Tests** (`tests/test_*_integration.py`)
   - Component interaction testing
   - Real API calls (with rate limiting)
   - Slower execution

3. **Benchmarks** (`benchmarks/`)
   - Performance regression testing
   - Load testing scenarios
   - Memory usage profiling

### Coverage Requirements

- **Minimum**: 95% code coverage
- **Exclusions**: Test files, `__init__.py`, debug code
- **Reports**: HTML, XML, terminal output

### Test Data

- **Fixtures**: `tests/fixtures/` directory
- **Mock Data**: Realistic GitHub API responses
- **Test Repositories**: Minimal repos for integration testing

## Security Considerations

### Development Security

1. **Secret Management**
   - Use `.env` files (never commit)
   - Secrets baseline for detection
   - GitHub Actions secrets for CI/CD

2. **Code Security**
   - Bandit security linting
   - Dependency vulnerability scanning
   - Input validation and sanitization

3. **Docker Security**
   - Non-root user containers
   - Minimal base images
   - Security scanning in CI

### Security Testing

```bash
make security           # Run all security checks
make security-update    # Update security baseline
```

## Contributing

### Code Style

- **Formatter**: ruff (configured in `pyproject.toml`)
- **Line Length**: 88 characters
- **Import Style**: isort with black compatibility
- **Docstrings**: Google style

### Commit Messages

Follow conventional commits:
```
feat: add new language support
fix: resolve caching race condition  
docs: update API documentation
test: add integration tests for webhooks
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Quality Checks**
   ```bash
   make ci-local  # Run full CI pipeline
   ```

4. **Submit PR**
   - Use provided PR template
   - Link related issues
   - Request appropriate reviewers

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
   ```

2. **Test Failures**
   ```bash
   make clean     # Clean cache files
   make install   # Reinstall dependencies
   make test      # Run tests again
   ```

3. **Pre-commit Issues**
   ```bash
   pre-commit clean       # Clean pre-commit cache
   pre-commit install     # Reinstall hooks
   pre-commit run --all-files  # Force run on all files
   ```

4. **Docker Issues**
   ```bash
   docker system prune -a  # Clean Docker cache
   make build-docker       # Rebuild images
   ```

### Debug Logging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### Getting Help

1. **Documentation**: Check `docs/` directory
2. **Issues**: Search existing GitHub issues
3. **Discussions**: Use GitHub Discussions
4. **Community**: Join project communication channels

## Performance Tips

### Development Performance

1. **Fast Testing**
   ```bash
   make test-fast     # Skip coverage for speed
   pytest tests/ -x   # Stop on first failure
   ```

2. **Parallel Testing**
   ```bash
   pytest tests/ -n auto  # Use all CPU cores
   ```

3. **Cache Optimization**
   - Keep cache enabled in development
   - Use `--cache-clear` only when needed
   - Monitor cache hit rates

### Production Performance

1. **Caching Strategy**
   - Configure appropriate TTL
   - Monitor cache size limits
   - Use Redis for distributed caching

2. **Parallel Processing**
   - Tune worker pool sizes
   - Monitor resource usage
   - Use async operations where possible

## Release Process

### Version Bumping

```bash
make version-patch  # 0.1.0 -> 0.1.1
make version-minor  # 0.1.1 -> 0.2.0
make version-major  # 0.2.0 -> 1.0.0
```

### Building Releases

```bash
make build         # Build Python package
make build-docker  # Build Docker images
```

### Deployment

See `docs/DEPLOYMENT_GUIDE.md` for production deployment instructions.
EOF < /dev/null
