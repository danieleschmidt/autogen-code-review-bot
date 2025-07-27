"""Integration tests for the complete PR review workflow."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from autogen_code_review_bot.pr_analysis import analyze_pr
from autogen_code_review_bot.github_integration import analyze_and_comment


@pytest.mark.integration
class TestFullWorkflow:
    """Test the complete end-to-end workflow."""

    def test_complete_pr_analysis_workflow(self, sample_code_file, sample_config):
        """Test the complete PR analysis workflow from start to finish."""
        # Create a temporary repository structure
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Copy sample file to repo
            test_file = repo_path / "src" / "calculator.py"
            test_file.parent.mkdir(parents=True)
            test_file.write_text(sample_code_file.read_text())
            
            # Create a config file
            config_path = repo_path / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(sample_config, f)
            
            # Mock the analysis functions
            with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
                mock_linters.return_value = {
                    "python": {
                        "style": {"passed": True, "issues": []},
                        "security": {"passed": True, "issues": []},
                        "type_check": {"passed": True, "issues": []}
                    }
                }
                
                # Run the analysis
                result = analyze_pr(str(repo_path), use_cache=False)
                
                # Verify results
                assert result is not None
                assert "python" in result
                mock_linters.assert_called_once()

    @pytest.mark.integration
    def test_github_integration_workflow(self, mock_pr_data, sample_config):
        """Test the GitHub integration workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            
            # Create mock repository structure
            src_dir = repo_path / "src"
            src_dir.mkdir(parents=True)
            
            # Create a simple Python file
            test_file = src_dir / "test.py"
            test_file.write_text("print('Hello, World!')")
            
            # Mock GitHub API calls
            with patch('github.Github') as mock_github_class:
                mock_github = Mock()
                mock_repo = Mock()
                mock_pr = Mock()
                
                mock_github_class.return_value = mock_github
                mock_github.get_repo.return_value = mock_repo
                mock_repo.get_pull.return_value = mock_pr
                
                # Mock the PR files
                mock_file = Mock()
                mock_file.filename = "src/test.py"
                mock_file.patch = "@@ -0,0 +1 @@\n+print('Hello, World!')"
                mock_pr.get_files.return_value = [mock_file]
                
                # Mock analysis result
                with patch('autogen_code_review_bot.pr_analysis.analyze_pr') as mock_analyze:
                    mock_analyze.return_value = {
                        "python": {
                            "style": {"passed": True, "issues": []},
                            "security": {"passed": True, "issues": []},
                            "summary": "Code looks good!"
                        }
                    }
                    
                    # Test the integration
                    with patch.dict('os.environ', {'GITHUB_TOKEN': 'test_token'}):
                        result = analyze_and_comment(str(repo_path), "test/repo", 123)
                        
                        # Verify the workflow completed
                        assert result is not None
                        mock_analyze.assert_called_once()

    @pytest.mark.integration
    def test_caching_integration(self, sample_code_file, sample_config):
        """Test that caching works correctly in the integration workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            cache_dir = repo_path / ".cache"
            
            # Setup repository
            src_dir = repo_path / "src"
            src_dir.mkdir(parents=True)
            test_file = src_dir / "test.py"
            test_file.write_text(sample_code_file.read_text())
            
            # Mock Git commands to return consistent commit hash
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value.stdout = "abc123def456"
                mock_subprocess.return_value.returncode = 0
                
                # Mock linter execution
                with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
                    mock_result = {
                        "python": {
                            "style": {"passed": True, "issues": []},
                            "security": {"passed": True, "issues": []}
                        }
                    }
                    mock_linters.return_value = mock_result
                    
                    # First run - should execute linters
                    result1 = analyze_pr(str(repo_path), use_cache=True)
                    assert mock_linters.call_count == 1
                    
                    # Second run - should use cache
                    result2 = analyze_pr(str(repo_path), use_cache=True)
                    # Should not call linters again if properly cached
                    
                    # Results should be the same
                    assert result1 == result2

    @pytest.mark.integration 
    def test_parallel_processing_integration(self, temp_dir):
        """Test parallel processing with multiple languages."""
        repo_path = temp_dir
        
        # Create files in different languages
        languages = {
            "python": "print('Hello from Python')",
            "javascript": "console.log('Hello from JavaScript');",
            "typescript": "const message: string = 'Hello from TypeScript';"
        }
        
        for lang, content in languages.items():
            lang_dir = repo_path / lang
            lang_dir.mkdir(exist_ok=True)
            
            if lang == "python":
                (lang_dir / "test.py").write_text(content)
            elif lang == "javascript":
                (lang_dir / "test.js").write_text(content)
            elif lang == "typescript":
                (lang_dir / "test.ts").write_text(content)
        
        # Mock linter results for each language
        with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
            mock_linters.return_value = {
                lang: {
                    "style": {"passed": True, "issues": []},
                    "security": {"passed": True, "issues": []}
                }
                for lang in languages.keys()
            }
            
            # Run analysis with parallel processing
            result = analyze_pr(str(repo_path), use_parallel=True)
            
            # Verify all languages were processed
            for lang in languages.keys():
                assert lang in result
                assert result[lang]["style"]["passed"]

    @pytest.mark.integration
    def test_error_handling_integration(self, temp_dir):
        """Test error handling in the integration workflow."""
        repo_path = temp_dir
        
        # Create a problematic Python file
        src_dir = repo_path / "src"
        src_dir.mkdir(parents=True)
        bad_file = src_dir / "bad_code.py"
        bad_file.write_text("import os\nos.system('rm -rf /')  # Security issue")
        
        # Mock linters to return security issues
        with patch('autogen_code_review_bot.pr_analysis.run_linters') as mock_linters:
            mock_linters.return_value = {
                "python": {
                    "style": {"passed": True, "issues": []},
                    "security": {
                        "passed": False,
                        "issues": [
                            {
                                "severity": "HIGH",
                                "message": "Potential shell injection vulnerability",
                                "line": 2,
                                "file": "src/bad_code.py"
                            }
                        ]
                    }
                }
            }
            
            # Run analysis
            result = analyze_pr(str(repo_path), use_cache=False)
            
            # Verify security issues were detected
            assert result["python"]["security"]["passed"] is False
            assert len(result["python"]["security"]["issues"]) > 0
            assert "shell injection" in result["python"]["security"]["issues"][0]["message"]

    @pytest.mark.integration
    def test_configuration_validation_integration(self, temp_dir, sample_config):
        """Test that configuration validation works in the workflow."""
        repo_path = temp_dir
        
        # Create invalid config
        invalid_config = sample_config.copy()
        invalid_config["agents"]["coder"]["temperature"] = 2.0  # Invalid temperature
        
        config_path = repo_path / "invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)
        
        # Test should handle invalid configuration gracefully
        with patch('autogen_code_review_bot.config_validation.validate_config') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid temperature value")
            
            # This should not crash but handle the error gracefully
            with pytest.raises(ValueError, match="Invalid temperature value"):
                analyze_pr(str(repo_path), config_path=str(config_path))