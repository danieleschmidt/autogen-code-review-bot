# AutoGen-Code-Review-Bot

Two-agent "coder" + "reviewer" loop using Microsoft AutoGen for automated PR critiques and code quality enhancement.

## Features

- **Dual-Agent Architecture**: Specialized "Coder" and "Reviewer" agents with distinct personalities and expertise
- **Automated PR Analysis**: Comprehensive pull request review with security, performance, and style feedback
- **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Rust, Ruby, and more
- **GitHub Integration**: Seamless webhook integration for automatic PR reviews
- **Configurable Rules**: Customizable review criteria and coding standards
- **Learning System**: Agents improve through feedback loops and historical review data

## Quick Start

```bash
# Install dependencies and dev tools
pip install -e .[dev]

# Configure GitHub webhook
python setup_webhook.py --repo your-org/your-repo

# Start the review bot
python bot.py --config config/default.yaml
```

### Pre-commit Hooks
Run code style and secret scanning checks before committing:

```bash
pre-commit run --all-files
```

## Agent Roles

### Coder Agent
- Focuses on functionality and implementation details
- Suggests code improvements and refactoring opportunities
- Identifies potential bugs and edge cases

### Reviewer Agent
- Emphasizes code quality, maintainability, and best practices
- Checks for security vulnerabilities and performance issues
- Ensures adherence to team coding standards

## Configuration

```yaml
# config/review_config.yaml
agents:
  coder:
    model: "gpt-4"
    temperature: 0.3
    focus_areas: ["functionality", "bugs", "edge_cases"]
  
  reviewer:
    model: "gpt-4"
    temperature: 0.1
    focus_areas: ["security", "performance", "standards"]

github:
  webhook_secret: "your_webhook_secret"  # pragma: allowlist secret
  bot_token: "your_github_token"  # pragma: allowlist secret

review_criteria:
  security_scan: true
  performance_check: true
  test_coverage: true
  documentation: true
```

### Linter Configuration

The bot uses language-specific linters to check code style. A YAML file can
override the default mapping of languages to linting tools:

```yaml
# linters.yaml
linters:
  python: ruff
  javascript: eslint
  typescript: eslint
  go: golangci-lint
  ruby: rubocop
```

Pass the path to this file when invoking `analyze_pr` to customize which
linters run for each language.
Unspecified languages fall back to the built-in defaults for Python,
JavaScript, and TypeScript.

Example usage:

```python
from autogen_code_review_bot import analyze_pr

result = analyze_pr("/path/to/repo", config_path="linters.yaml")
print(result.style.output)
```

## Usage Examples

### Manual Review
```bash
# Review a specific PR
python review_pr.py --pr-number 123 --repo owner/repo

# Review local changes
python review_local.py --path ./src --diff HEAD~1
```

### GitHub Integration
The bot automatically triggers on:
- New pull requests
- Push events to PR branches
- Review requests

You can also run the analysis manually and post the results to a pull request:

```python
from autogen_code_review_bot.github_integration import analyze_and_comment

# GITHUB_TOKEN environment variable must be set
analyze_and_comment('/path/to/repo', 'owner/repo', 123)
```

## Sample Review Output

```markdown
## 🤖 AutoGen Code Review

### Coder Agent Findings:
- ✅ Logic implementation looks solid
- ⚠️ Consider edge case handling in line 45
- 💡 Suggestion: Extract method for better readability

### Reviewer Agent Findings:
- 🔒 Security: Potential SQL injection risk in query builder
- 🚀 Performance: Database query could be optimized
- 📝 Documentation: Add docstrings for public methods

### Overall Score: 7.5/10
```

## Advanced Features

- **Custom Rule Engine**: Define project-specific review rules
- **Multi-Agent Conversations**: Agents discuss and refine feedback
- **Integration Tests**: Automatic test generation suggestions
- **Code Metrics**: Track complexity, maintainability scores
- **Learning Mode**: Agents adapt to team preferences over time

## Deployment

### Docker
```bash
docker build -t autogen-review-bot .
docker run -d --env-file .env autogen-review-bot
```

### GitHub Actions
```yaml
name: AutoGen Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run AutoGen Review
        uses: ./action.yml
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Agent behavior customization
- Adding new programming language support
- Extending review capabilities

## License

MIT License - see [LICENSE](LICENSE) file for details.

See [CHANGELOG](CHANGELOG.md) for release history and [CODEOWNERS](.github/CODEOWNERS) for maintainers.
