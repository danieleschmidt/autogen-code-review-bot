"""
Example integration tests for AutoGen Code Review Bot.

Integration tests verify that multiple components work together correctly.
They test the interaction between different modules and external services.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml


@pytest.mark.integration
class TestGitHubIntegration:
    """Test GitHub API integration."""
    
    def test_pr_analysis_workflow(self, temp_repo, sample_pr_data, mock_github_client):
        """Test complete PR analysis workflow."""
        # Arrange
        pr_number = sample_pr_data["number"]
        repo_path = temp_repo
        
        # Mock the GitHub client responses
        mock_github_client.get_pull_request.return_value = sample_pr_data
        mock_github_client.get_files.return_value = sample_pr_data["changed_files"]
        
        # Act - This would normally call the actual analysis function
        # For now, we'll simulate the workflow
        pr_data = mock_github_client.get_pull_request(pr_number)
        files = mock_github_client.get_files(pr_number)
        
        # Assert
        assert pr_data["number"] == pr_number
        assert len(files) == 2
        assert any(f["filename"].endswith(".py") for f in files)
        
        # Verify GitHub client was called correctly
        mock_github_client.get_pull_request.assert_called_once_with(pr_number)
        mock_github_client.get_files.assert_called_once_with(pr_number)
    
    def test_webhook_processing(self, sample_pr_data):
        """Test webhook event processing."""
        # Arrange
        webhook_payload = {
            "action": "opened",
            "pull_request": sample_pr_data,
            "repository": {
                "name": "test-repo",
                "owner": {"login": "test-owner"}
            }
        }
        
        # Act - Simulate webhook processing
        action = webhook_payload["action"]
        pr_data = webhook_payload["pull_request"]
        
        # Assert
        assert action == "opened"
        assert pr_data["number"] == 123
        assert pr_data["title"] == "Add new feature"
    
    @patch('requests.post')
    def test_comment_posting(self, mock_post, sample_pr_data):
        """Test posting comments to GitHub PR."""
        # Arrange
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": 789}
        
        comment_text = "## 🤖 AutoGen Code Review\n\nGreat work!"
        
        # Act - Simulate posting comment
        import requests
        response = requests.post(
            "https://api.github.com/repos/owner/repo/issues/123/comments",
            json={"body": comment_text},
            headers={"Authorization": "token test_token"}
        )
        
        # Assert
        assert response.status_code == 201
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert comment_text in str(call_args)


@pytest.mark.integration
class TestAgentInteraction:
    """Test agent interaction and conversation flow."""
    
    def test_dual_agent_conversation(self, mock_autogen_agent):
        """Test conversation between Coder and Reviewer agents."""
        # Arrange
        coder_agent = mock_autogen_agent
        reviewer_agent = mock_autogen_agent
        
        # Configure agent responses
        coder_agent.generate_reply.return_value = "The code looks functionally correct."
        reviewer_agent.generate_reply.return_value = "I agree, but suggest adding error handling."
        
        # Act - Simulate agent conversation
        coder_response = coder_agent.generate_reply("Analyze this code")
        reviewer_response = reviewer_agent.generate_reply(coder_response)
        
        # Assert
        assert "functionally correct" in coder_response
        assert "error handling" in reviewer_response
        
        # Verify agents were called
        coder_agent.generate_reply.assert_called_once()
        reviewer_agent.generate_reply.assert_called_once()
    
    def test_agent_memory_persistence(self, temp_dir):
        """Test agent conversation memory."""
        # Arrange
        conversation_file = temp_dir / "conversation.json"
        conversation_data = [
            {"agent": "coder", "message": "Initial analysis"},
            {"agent": "reviewer", "message": "Feedback on analysis"}
        ]
        
        # Act - Save and load conversation
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data, f)
        
        with open(conversation_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Assert
        assert len(loaded_data) == 2
        assert loaded_data[0]["agent"] == "coder"
        assert loaded_data[1]["agent"] == "reviewer"


@pytest.mark.integration
class TestLanguageDetectionAndLinting:
    """Test language detection and linting integration."""
    
    def test_multi_language_analysis(self, sample_code_files):
        """Test analysis of multiple programming languages."""
        # Arrange
        files = sample_code_files
        
        # Act - Simulate language detection
        detected_languages = {}
        for lang, file_path in files.items():
            if file_path.suffix == ".py":
                detected_languages[str(file_path)] = "python"
            elif file_path.suffix == ".js":
                detected_languages[str(file_path)] = "javascript"
            elif file_path.suffix == ".go":
                detected_languages[str(file_path)] = "go"
        
        # Assert
        assert len(detected_languages) == 3
        assert "python" in detected_languages.values()
        assert "javascript" in detected_languages.values()
        assert "go" in detected_languages.values()
    
    @patch('subprocess.run')
    def test_linter_execution(self, mock_subprocess, sample_code_files):
        """Test linter execution for different languages."""
        # Arrange
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "No issues found"
        
        python_file = sample_code_files["python"]
        
        # Act - Simulate running ruff on Python file
        import subprocess
        result = subprocess.run(
            ["ruff", "check", str(python_file)],
            capture_output=True,
            text=True
        )
        
        # Assert
        assert result.returncode == 0
        mock_subprocess.assert_called_once()


@pytest.mark.integration
class TestCacheIntegration:
    """Test caching system integration."""
    
    def test_cache_lifecycle(self, cache_dir, temp_repo):
        """Test complete cache lifecycle."""
        # Arrange
        cache_key = "test_repo_abc123"
        cache_file = cache_dir / f"{cache_key}.json"
        
        analysis_result = {
            "commit_hash": "abc123",
            "analysis": "Code looks good",
            "timestamp": "2025-07-27T12:00:00Z"
        }
        
        # Act - Write to cache
        with open(cache_file, 'w') as f:
            json.dump(analysis_result, f)
        
        # Read from cache
        with open(cache_file, 'r') as f:
            cached_result = json.load(f)
        
        # Assert
        assert cached_result["commit_hash"] == "abc123"
        assert cached_result["analysis"] == "Code looks good"
        assert cache_file.exists()
    
    def test_cache_invalidation(self, cache_dir):
        """Test cache invalidation based on TTL."""
        # Arrange
        import time
        cache_file = cache_dir / "expired_cache.json"
        
        # Create cache entry
        cache_data = {
            "data": "test",
            "timestamp": time.time() - 3600  # 1 hour ago
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Act - Check if cache is expired (TTL = 30 minutes)
        ttl_seconds = 30 * 60  # 30 minutes
        current_time = time.time()
        cache_age = current_time - cache_data["timestamp"]
        is_expired = cache_age > ttl_seconds
        
        # Assert
        assert is_expired is True


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration loading and validation."""
    
    def test_config_loading(self, config_file, test_config):
        """Test configuration file loading."""
        # Act
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        # Assert
        assert loaded_config == test_config
        assert loaded_config["agents"]["coder"]["model"] == "gpt-3.5-turbo"
    
    def test_environment_override(self, config_file, environment_variables):
        """Test environment variable configuration override."""
        # Arrange
        import os
        
        # Act
        github_token = os.environ.get("GITHUB_TOKEN")
        log_level = os.environ.get("LOG_LEVEL")
        
        # Assert
        assert github_token == "test_token"
        assert log_level == "DEBUG"
    
    def test_config_validation(self, test_config):
        """Test configuration validation."""
        # Act - Validate required fields
        required_fields = ["agents", "github", "review_criteria"]
        
        # Assert
        for field in required_fields:
            assert field in test_config
        
        # Validate agent configuration
        assert "coder" in test_config["agents"]
        assert "reviewer" in test_config["agents"]
        
        # Validate GitHub configuration
        assert "webhook_secret" in test_config["github"]
        assert "bot_token" in test_config["github"]


@pytest.mark.integration 
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_full_pr_review_workflow(self, temp_repo, sample_pr_data, mock_github_client, mock_autogen_agent):
        """Test complete PR review from webhook to comment."""
        # Arrange
        repo_path = temp_repo
        pr_number = sample_pr_data["number"]
        
        # Mock all external dependencies
        mock_github_client.get_pull_request.return_value = sample_pr_data
        mock_github_client.get_files.return_value = sample_pr_data["changed_files"]
        mock_github_client.post_comment.return_value = {"id": 999}
        
        mock_autogen_agent.generate_reply.return_value = "Code review completed"
        
        # Act - Simulate full workflow
        # 1. Receive webhook
        webhook_data = {"action": "opened", "pull_request": sample_pr_data}
        
        # 2. Get PR details
        pr_data = mock_github_client.get_pull_request(pr_number)
        
        # 3. Get changed files
        files = mock_github_client.get_files(pr_number)
        
        # 4. Run agent analysis
        analysis = mock_autogen_agent.generate_reply("Analyze PR")
        
        # 5. Post comment
        comment_result = mock_github_client.post_comment(pr_number, analysis)
        
        # Assert
        assert pr_data["number"] == pr_number
        assert len(files) == 2
        assert analysis == "Code review completed"
        assert comment_result["id"] == 999
        
        # Verify all components were called
        mock_github_client.get_pull_request.assert_called_once_with(pr_number)
        mock_github_client.get_files.assert_called_once_with(pr_number)
        mock_autogen_agent.generate_reply.assert_called_once()
        mock_github_client.post_comment.assert_called_once_with(pr_number, analysis)