# Development Guide

## Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd autogen-code-review-bot
pip install -e .

# Development dependencies
pip install -r requirements-dev.txt
```

## Development Workflow
1. **Setup**: Follow [Python Development Guide](https://docs.python.org/3/tutorial/)
2. **Testing**: Run `pytest` before commits
3. **Linting**: Use `ruff .` for code quality
4. **Type Checking**: Run `mypy src/` if configured

## Project Structure
- `src/autogen_code_review_bot/`: Main package
- `tests/`: Test suite
- `docs/`: Documentation

## Environment Setup
See [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html) for isolation.

## IDE Configuration
Recommended: VS Code with Python extension and `.editorconfig` support.

## Debugging
Use standard Python debugging tools. See [Python Debugging](https://docs.python.org/3/library/pdb.html).