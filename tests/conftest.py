"""
Pytest configuration and fixtures for AutoGen Code Review Bot tests.

This module provides shared fixtures and configuration for all test modules.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

import pytest
import yaml


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "agents": {
            "coder": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "focus_areas": ["functionality", "bugs", "edge_cases"]
            },
            "reviewer": {
                "model": "gpt-3.5-turbo", 
                "temperature": 0.1,
                "focus_areas": ["security", "performance", "standards"]
            }
        },
        "github": {
            "webhook_secret": "test_webhook_secret",
            "bot_token": "test_bot_token",
            "api_url": "https://api.github.com"
        },
        "review_criteria": {
            "security_scan": True,
            "performance_check": True,
            "test_coverage": True,
            "documentation": True
        },
        "cache": {
            "enabled": True,
            "directory": "test_cache",
            "ttl_hours": 1
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_repo(temp_dir: Path) -> Path:
    """Create a temporary Git repository for testing."""
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir()
    
    # Initialize git repo
    os.system(f"cd {repo_dir} && git init")
    os.system(f"cd {repo_dir} && git config user.name 'Test User'")
    os.system(f"cd {repo_dir} && git config user.email 'test@example.com'")
    
    # Create sample files
    (repo_dir / "main.py").write_text("""
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
""")
    
    (repo_dir / "README.md").write_text("# Test Repository")
    
    (repo_dir / "requirements.txt").write_text("""
requests>=2.31.0
pyyaml>=6.0.0
""")
    
    # Initial commit
    os.system(f"cd {repo_dir} && git add .")
    os.system(f"cd {repo_dir} && git commit -m 'Initial commit'")
    
    return repo_dir


@pytest.fixture
def sample_pr_data() -> Dict[str, Any]:
    """Provide sample PR data for testing."""
    return {
        "number": 123,
        "title": "Add new feature",
        "body": "This PR adds a new feature to the codebase.",
        "head": {
            "sha": "abc123",
            "ref": "feature-branch"
        },
        "base": {
            "sha": "def456", 
            "ref": "main"
        },
        "changed_files": [
            {
                "filename": "src/new_feature.py",
                "status": "added",
                "additions": 50,
                "deletions": 0,
                "patch": "+def new_feature():\n+    return 'Hello World'"
            },
            {
                "filename": "tests/test_new_feature.py",
                "status": "added", 
                "additions": 20,
                "deletions": 0,
                "patch": "+def test_new_feature():\n+    assert new_feature() == 'Hello World'"
            }
        ]
    }


@pytest.fixture
def mock_github_client():
    """Provide a mock GitHub client."""
    client = Mock()
    client.get_pull_request.return_value = {
        "number": 123,
        "title": "Test PR",
        "body": "Test description",
        "state": "open"
    }
    client.post_comment.return_value = {"id": 456}
    client.get_files.return_value = []
    return client


@pytest.fixture
def mock_openai_client():
    """Provide a mock OpenAI client."""
    client = Mock()
    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    return client


@pytest.fixture
def mock_autogen_agent():
    """Provide a mock AutoGen agent."""
    agent = Mock()
    agent.generate_reply.return_value = "Mock agent response"
    agent.send.return_value = True
    return agent


@pytest.fixture
def sample_code_files(temp_dir: Path) -> Dict[str, Path]:
    """Create sample code files for testing."""
    files = {}
    
    # Python file
    python_file = temp_dir / "sample.py"
    python_file.write_text("""
import os
import sys

def process_data(data):
    # TODO: Add input validation
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)
    
    def process(self):
        return process_data(self.data)
""")
    files["python"] = python_file
    
    # JavaScript file
    js_file = temp_dir / "sample.js"
    js_file.write_text("""
function processData(data) {
    // TODO: Add error handling
    const result = [];
    for (const item of data) {
        if (item > 0) {
            result.push(item * 2);
        }
    }
    return result;
}

class DataProcessor {
    constructor() {
        this.data = [];
    }
    
    addData(item) {
        this.data.push(item);
    }
    
    process() {
        return processData(this.data);
    }
}

module.exports = { DataProcessor, processData };
""")
    files["javascript"] = js_file
    
    # Go file
    go_file = temp_dir / "sample.go"
    go_file.write_text("""
package main

import "fmt"

func processData(data []int) []int {
    // TODO: Add error handling
    var result []int
    for _, item := range data {
        if item > 0 {
            result = append(result, item*2)
        }
    }
    return result
}

type DataProcessor struct {
    data []int
}

func (dp *DataProcessor) AddData(item int) {
    dp.data = append(dp.data, item)
}

func (dp *DataProcessor) Process() []int {
    return processData(dp.data)
}

func main() {
    processor := &DataProcessor{}
    processor.AddData(1)
    processor.AddData(2)
    result := processor.Process()
    fmt.Println(result)
}
""")
    files["go"] = go_file
    
    return files


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    test_env = {
        "GITHUB_TOKEN": "test_token",
        "GITHUB_WEBHOOK_SECRET": "test_secret",
        "OPENAI_API_KEY": "test_openai_key",
        "BOT_CONFIG_PATH": "test_config.yaml",
        "LOG_LEVEL": "DEBUG",
        "CACHE_ENABLED": "true"
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def cache_dir(temp_dir: Path) -> Path:
    """Provide a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def config_file(temp_dir: Path, test_config: Dict[str, Any]) -> Path:
    """Create a temporary config file."""
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    return config_file


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Add any singleton reset logic here
    yield
    # Cleanup after test


# Pytest markers for test categorization
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)