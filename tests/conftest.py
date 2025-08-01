"""
Pytest configuration and fixtures for AutoGen Code Review Bot tests.

This module provides shared fixtures and configuration for all test modules.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_repo_path(temp_dir: Path) -> Path:
    """Create a sample repository structure for testing."""
    repo_path = temp_dir / "sample_repo"
    repo_path.mkdir()
    
    # Create sample Python files
    (repo_path / "main.py").write_text("""
def hello_world():
    print("Hello, World\!")
    
if __name__ == "__main__":
    hello_world()
""")
    
    (repo_path / "utils.py").write_text("""
def add_numbers(a, b):
    return a + b

def divide_numbers(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""")
    
    # Create sample JavaScript file
    (repo_path / "script.js").write_text("""
function greet(name) {
    console.log(`Hello, ${name}\!`);
}

function calculate(a, b) {
    return a + b;
}
""")
    
    return repo_path


@pytest.fixture
def sample_pr_data() -> Dict[str, Any]:
    """Sample pull request data for testing."""
    return {
        "number": 123,
        "title": "Add new feature",
        "body": "This PR adds a new feature to the application.",
        "state": "open",
        "head": {
            "sha": "abc123def456",
            "ref": "feature/new-feature"
        },
        "base": {
            "sha": "def456abc123",
            "ref": "main"
        },
        "user": {
            "login": "developer"
        },
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T01:00:00Z"
    }


@pytest.fixture
def sample_github_files() -> list:
    """Sample GitHub files data for testing."""
    return [
        {
            "filename": "main.py",
            "status": "modified",
            "additions": 5,
            "deletions": 2,
            "changes": 7,
            "patch": """@@ -1,3 +1,6 @@
+import sys
+
 def hello_world():
-    print("Hello, World\!")
+    print("Hello, World\!")
+    sys.exit(0)
     
 if __name__ == "__main__":"""
        },
        {
            "filename": "utils.py",
            "status": "added",
            "additions": 8,
            "deletions": 0,
            "changes": 8,
            "patch": """@@ -0,0 +1,8 @@
+def add_numbers(a, b):
+    return a + b
+
+def divide_numbers(a, b):
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
+    return a / b"""
        }
    ]


@pytest.fixture
def mock_github_client():
    """Mock GitHub client for testing."""
    client = Mock()
    
    # Mock repository methods
    client.get_repo.return_value = Mock()
    client.get_repo.return_value.get_pull.return_value = Mock()
    client.get_repo.return_value.get_contents.return_value = Mock()
    
    # Mock pull request methods
    pr_mock = client.get_repo.return_value.get_pull.return_value
    pr_mock.get_files.return_value = []
    pr_mock.create_issue_comment.return_value = Mock()
    pr_mock.create_review.return_value = Mock()
    
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()
    
    # Mock chat completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "Mock AI response"
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 100
    
    client.chat.completions.create.return_value = mock_response
    
    return client


@pytest.fixture
def sample_agent_config() -> Dict[str, Any]:
    """Sample agent configuration for testing."""
    return {
        "agents": {
            "coder": {
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": 1000,
                "focus_areas": ["functionality", "bugs", "edge_cases"]
            },
            "reviewer": {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 1000,
                "focus_areas": ["security", "performance", "standards"]
            }
        },
        "review_criteria": {
            "security_scan": True,
            "performance_check": True,
            "test_coverage": True,
            "documentation": True
        }
    }


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    cache = Mock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.clear.return_value = True
    return cache


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Set up test environment variables."""
    test_env_vars = {
        "GITHUB_TOKEN": "test_token",
        "OPENAI_API_KEY": "test_openai_key",
        "APP_ENV": "test",
        "DEBUG": "true",
        "CACHE_ENABLED": "false",
        "LOG_LEVEL": "DEBUG"
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sample_linter_config() -> Dict[str, str]:
    """Sample linter configuration for testing."""
    return {
        "python": "ruff",
        "javascript": "eslint",
        "typescript": "eslint",
        "go": "golangci-lint",
        "rust": "clippy",
        "ruby": "rubocop"
    }


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for testing."""
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Mock linter output"
    mock_result.stderr = ""
    return mock_result


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security-related"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance-related"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark e2e tests
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        
        # Mark unit tests (default)
        if not any(mark.name in ["integration", "e2e"] for mark in item.own_markers):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests based on name patterns
        if any(keyword in item.name for keyword in ["benchmark", "load", "stress"]):
            item.add_marker(pytest.mark.slow)


# Custom test result reporting
@pytest.fixture(autouse=True)
def test_execution_tracker(request):
    """Track test execution for reporting."""
    test_name = request.node.name
    test_file = request.node.fspath.basename
    
    # Store test metadata for potential use in reporting
    request.node.test_metadata = {
        "name": test_name,
        "file": test_file,
        "markers": [mark.name for mark in request.node.iter_markers()]
    }
EOF < /dev/null
