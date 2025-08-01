"""
Testing utilities for AutoGen Code Review Bot.

This module provides helper functions and classes for testing.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock


class TestRepoBuilder:
    """Helper class to build test repositories with specific structures."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def add_python_file(self, name: str, content: str) -> Path:
        """Add a Python file to the test repo."""
        file_path = self.base_path / f"{name}.py"
        file_path.write_text(content)
        return file_path
    
    def add_javascript_file(self, name: str, content: str) -> Path:
        """Add a JavaScript file to the test repo."""
        file_path = self.base_path / f"{name}.js"
        file_path.write_text(content)
        return file_path
    
    def add_config_file(self, name: str, config: Dict[str, Any]) -> Path:
        """Add a configuration file to the test repo."""
        file_path = self.base_path / f"{name}.json"
        file_path.write_text(json.dumps(config, indent=2))
        return file_path
    
    def add_directory(self, name: str) -> Path:
        """Add a directory to the test repo."""
        dir_path = self.base_path / name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def get_path(self) -> Path:
        """Get the path to the test repo."""
        return self.base_path


class MockGitHubResponse:
    """Helper class to create mock GitHub API responses."""
    
    @staticmethod
    def create_pr_response(
        number: int = 123,
        title: str = "Test PR",
        state: str = "open"
    ) -> Dict[str, Any]:
        """Create a mock PR response."""
        return {
            "number": number,
            "title": title,
            "body": "Test PR description",
            "state": state,
            "head": {
                "sha": "abc123def456",
                "ref": "feature/test"
            },
            "base": {
                "sha": "def456abc123",
                "ref": "main"
            },
            "user": {
                "login": "testuser"
            },
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T01:00:00Z"
        }
    
    @staticmethod
    def create_file_response(
        filename: str,
        status: str = "modified",
        additions: int = 5,
        deletions: int = 2
    ) -> Dict[str, Any]:
        """Create a mock file response."""
        return {
            "filename": filename,
            "status": status,
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
            "patch": f"Mock patch for {filename}"
        }
    
    @staticmethod
    def create_comment_response(
        comment_id: int = 1,
        body: str = "Test comment"
    ) -> Dict[str, Any]:
        """Create a mock comment response."""
        return {
            "id": comment_id,
            "body": body,
            "user": {
                "login": "bot"
            },
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }


class MockOpenAIResponse:
    """Helper class to create mock OpenAI API responses."""
    
    @staticmethod
    def create_chat_response(
        content: str = "Mock AI response",
        tokens: int = 100
    ) -> Mock:
        """Create a mock chat completion response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = content
        response.usage = Mock()
        response.usage.total_tokens = tokens
        response.usage.prompt_tokens = tokens // 2
        response.usage.completion_tokens = tokens // 2
        return response


class TestDataGenerator:
    """Helper class to generate test data."""
    
    @staticmethod
    def generate_code_samples() -> Dict[str, str]:
        """Generate sample code in different languages."""
        return {
            "python": '''
def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers)

def validate_email(email):
    """Validate email format."""
    import re
    pattern = r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$'
    return re.match(pattern, email) is not None
''',
            "javascript": '''
function calculateSum(numbers) {
    if (\!numbers || numbers.length === 0) {
        return 0;
    }
    return numbers.reduce((sum, num) => sum + num, 0);
}

function validateEmail(email) {
    const pattern = /^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$/;
    return pattern.test(email);
}
''',
            "typescript": '''
interface User {
    id: number;
    name: string;
    email: string;
}

function processUser(user: User): boolean {
    if (\!user.name || \!user.email) {
        return false;
    }
    console.log(`Processing user: ${user.name}`);
    return true;
}
''',
            "go": '''
package main

import (
    "fmt"
    "regexp"
)

func calculateSum(numbers []int) int {
    sum := 0
    for _, num := range numbers {
        sum += num
    }
    return sum
}

func validateEmail(email string) bool {
    pattern := `^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$`
    matched, _ := regexp.MatchString(pattern, email)
    return matched
}
'''
        }
    
    @staticmethod
    def generate_pr_scenarios() -> List[Dict[str, Any]]:
        """Generate different PR scenarios for testing."""
        return [
            {
                "name": "small_feature_pr",
                "files_changed": 3,
                "additions": 50,
                "deletions": 10,
                "languages": ["python"]
            },
            {
                "name": "large_refactoring_pr",
                "files_changed": 25,
                "additions": 500,
                "deletions": 300,
                "languages": ["python", "javascript"]
            },
            {
                "name": "security_fix_pr",
                "files_changed": 5,
                "additions": 20,
                "deletions": 30,
                "languages": ["python"],
                "has_security_changes": True
            },
            {
                "name": "documentation_pr",
                "files_changed": 10,
                "additions": 200,
                "deletions": 50,
                "languages": ["markdown"],
                "documentation_only": True
            }
        ]


class AssertionHelpers:
    """Helper functions for common test assertions."""
    
    @staticmethod
    def assert_valid_analysis_result(result: Dict[str, Any]):
        """Assert that an analysis result has the expected structure."""
        assert isinstance(result, dict)
        assert "status" in result
        assert "analysis" in result
        assert "timestamp" in result
        
        if result["status"] == "success":
            assert "coder_feedback" in result["analysis"]
            assert "reviewer_feedback" in result["analysis"]
        else:
            assert "error" in result
    
    @staticmethod
    def assert_valid_github_comment(comment: str):
        """Assert that a GitHub comment has the expected format."""
        assert isinstance(comment, str)
        assert len(comment) > 0
        assert "## 🤖 AutoGen Code Review" in comment
        assert "### Coder Agent Findings:" in comment or "### Reviewer Agent Findings:" in comment
    
    @staticmethod
    def assert_performance_within_limits(
        execution_time: float,
        max_time: float,
        memory_usage: int = None,
        max_memory: int = None
    ):
        """Assert that performance metrics are within acceptable limits."""
        assert execution_time <= max_time, f"Execution time {execution_time}s exceeded limit {max_time}s"
        
        if memory_usage is not None and max_memory is not None:
            assert memory_usage <= max_memory, f"Memory usage {memory_usage} exceeded limit {max_memory}"
    
    @staticmethod
    def assert_no_secrets_leaked(text: str):
        """Assert that no secrets are present in the text."""
        import re
        
        # Common secret patterns
        secret_patterns = [
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub Personal Access Token
            r'sk-[a-zA-Z0-9]{48}',   # OpenAI API Key
            r'xoxb-[a-zA-Z0-9-]{11,}',  # Slack Bot Token
            r'AIza[0-9A-Za-z-_]{35}',   # Google API Key
        ]
        
        for pattern in secret_patterns:
            matches = re.findall(pattern, text)
            assert not matches, f"Potential secret found: {matches}"


class TestEnvironmentManager:
    """Helper class to manage test environment setup and teardown."""
    
    def __init__(self):
        self.temp_dirs = []
        self.original_env = {}
    
    def create_temp_repo(self) -> TestRepoBuilder:
        """Create a temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return TestRepoBuilder(Path(temp_dir))
    
    def set_env_var(self, key: str, value: str):
        """Set an environment variable and track it for cleanup."""
        import os
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    def cleanup(self):
        """Clean up all temporary resources."""
        import os
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()
        
        # Restore original environment variables
        for key, original_value in self.original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        self.original_env.clear()


# Decorator for test timing
def time_test(func):
    """Decorator to time test execution."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Test {func.__name__} executed in {execution_time:.3f} seconds")
        
        return result
    
    return wrapper


# Context manager for capturing logs
class LogCapture:
    """Context manager to capture log messages during tests."""
    
    def __init__(self, logger_name: str = None):
        self.logger_name = logger_name
        self.records = []
        self.handler = None
    
    def __enter__(self):
        import logging
        
        logger = logging.getLogger(self.logger_name)
        self.handler = TestLogHandler(self.records)
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)
        
        return self.records
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import logging
        
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)


class TestLogHandler(logging.Handler):
    """Custom log handler for capturing log records in tests."""
    
    def __init__(self, records_list: List):
        super().__init__()
        self.records = records_list
    
    def emit(self, record):
        self.records.append(record)
EOF < /dev/null
