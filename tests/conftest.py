"""Pytest configuration and shared fixtures for the test suite."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

import pytest
import yaml


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_github_token() -> str:
    """Provide a mock GitHub token for testing."""
    return "ghp_test_token_1234567890abcdef"


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration for testing."""
    return {
        "agents": {
            "coder": {
                "model": "gpt-4",
                "temperature": 0.3,
                "focus_areas": ["functionality", "bugs", "edge_cases"]
            },
            "reviewer": {
                "model": "gpt-4", 
                "temperature": 0.1,
                "focus_areas": ["security", "performance", "standards"]
            }
        },
        "github": {
            "webhook_secret": "test_secret",
            "bot_token": "test_token"
        },
        "review_criteria": {
            "security_scan": True,
            "performance_check": True,
            "test_coverage": True,
            "documentation": True
        }
    }


@pytest.fixture
def config_file(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary config file for testing."""
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def mock_pr_data() -> Dict[str, Any]:
    """Provide mock PR data for testing."""
    return {
        "number": 123,
        "title": "Add new feature",
        "body": "This PR adds a new feature to the codebase",
        "user": {"login": "test_user"},
        "head": {
            "sha": "abc123def456",
            "ref": "feature/new-feature"
        },
        "base": {
            "sha": "def456abc123",
            "ref": "main"
        },
        "changed_files": 5,
        "additions": 100,
        "deletions": 20
    }


@pytest.fixture
def mock_github_client():
    """Provide a mock GitHub client for testing."""
    mock_client = Mock()
    mock_client.get_repo.return_value = Mock()
    mock_client.get_user.return_value = Mock()
    return mock_client


@pytest.fixture
def sample_code_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    code_content = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    return a + b


def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self) -> list:
        """Return calculation history."""
        return self.history.copy()
'''
    
    file_path = temp_dir / "sample_code.py"
    with open(file_path, "w") as f:
        f.write(code_content)
    return file_path


@pytest.fixture
def mock_linter_results():
    """Provide mock linter results for testing."""
    return {
        "style": {
            "tool": "ruff",
            "passed": True,
            "issues": [],
            "output": "All checks passed"
        },
        "security": {
            "tool": "bandit",
            "passed": True,
            "issues": [],
            "output": "No security issues found"
        },
        "type_check": {
            "tool": "mypy",
            "passed": True,
            "issues": [],
            "output": "Success: no issues found"
        }
    }


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "GITHUB_TOKEN": "test_token",
        "OPENAI_API_KEY": "test_openai_key",
        "CACHE_ENABLED": "true",
        "DEBUG": "true"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(autouse=True)
def clean_cache():
    """Clean up cache after each test."""
    yield
    # Cleanup logic would go here if needed


# Markers for different test types
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.performance = pytest.mark.performance