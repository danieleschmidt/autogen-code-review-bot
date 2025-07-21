"""Tests for test coverage metrics and reporting."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, call
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot.coverage_metrics import (
    CoverageReporter,
    CoverageResult,
    CoverageConfig,
    run_coverage_analysis,
    generate_coverage_report,
    validate_coverage_threshold,
    discover_test_files,
    CoverageError
)


class TestCoverageConfig:
    """Test coverage configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CoverageConfig()
        
        assert config.minimum_coverage == 85.0
        assert config.source_dirs == ["src"]
        assert config.test_dirs == ["tests"]
        assert config.exclude_patterns == ["*/test_*", "*/tests/*", "*/__pycache__/*"]
        assert config.fail_under == 85.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CoverageConfig(
            minimum_coverage=90.0,
            source_dirs=["app", "lib"],
            test_dirs=["test", "spec"],
            exclude_patterns=["*/migrations/*"],
            fail_under=90.0
        )
        
        assert config.minimum_coverage == 90.0
        assert config.source_dirs == ["app", "lib"]
        assert config.test_dirs == ["test", "spec"]
        assert config.exclude_patterns == ["*/migrations/*"]
        assert config.fail_under == 90.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = CoverageConfig(minimum_coverage=75.0, fail_under=75.0)
        assert config.minimum_coverage == 75.0
        
        # Test negative coverage threshold
        with pytest.raises(ValueError, match="Coverage threshold must be between 0 and 100"):
            CoverageConfig(minimum_coverage=-5.0)
        
        # Test coverage over 100%
        with pytest.raises(ValueError, match="Coverage threshold must be between 0 and 100"):
            CoverageConfig(minimum_coverage=105.0)
        
        # Test empty source directories
        with pytest.raises(ValueError, match="Source directories cannot be empty"):
            CoverageConfig(source_dirs=[])


class TestCoverageResult:
    """Test coverage result data structure."""
    
    def test_coverage_result_creation(self):
        """Test creating coverage result."""
        result = CoverageResult(
            total_coverage=87.5,
            line_coverage=90.0,
            branch_coverage=85.0,
            files_analyzed=25,
            lines_covered=450,
            lines_total=500,
            branches_covered=170,
            branches_total=200
        )
        
        assert result.total_coverage == 87.5
        assert result.line_coverage == 90.0
        assert result.branch_coverage == 85.0
        assert result.files_analyzed == 25
        assert result.lines_covered == 450
        assert result.lines_total == 500
        assert result.branches_covered == 170
        assert result.branches_total == 200
    
    def test_coverage_result_meets_threshold(self):
        """Test threshold checking."""
        good_result = CoverageResult(total_coverage=90.0, line_coverage=90.0, branch_coverage=90.0)
        bad_result = CoverageResult(total_coverage=75.0, line_coverage=75.0, branch_coverage=75.0)
        
        assert good_result.meets_threshold(85.0)
        assert not bad_result.meets_threshold(85.0)
    
    def test_coverage_result_to_dict(self):
        """Test converting result to dictionary."""
        result = CoverageResult(
            total_coverage=87.5,
            line_coverage=90.0,
            branch_coverage=85.0,
            files_analyzed=25
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["total_coverage"] == 87.5
        assert result_dict["line_coverage"] == 90.0
        assert result_dict["branch_coverage"] == 85.0
        assert result_dict["files_analyzed"] == 25


class TestTestDiscovery:
    """Test test file discovery functionality."""
    
    def test_discover_test_files_basic(self):
        """Test basic test file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "tests"
            test_dir.mkdir()
            
            # Create test files
            (test_dir / "test_example.py").write_text("def test_something(): pass")
            (test_dir / "test_another.py").write_text("def test_other(): pass")
            (test_dir / "not_a_test.py").write_text("def helper(): pass")
            
            test_files = discover_test_files([str(test_dir)])
            test_files = [Path(f).name for f in test_files]
            
            assert "test_example.py" in test_files
            assert "test_another.py" in test_files
            assert "not_a_test.py" not in test_files
    
    def test_discover_test_files_multiple_dirs(self):
        """Test test discovery across multiple directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test directories
            tests1 = Path(temp_dir) / "tests"
            tests2 = Path(temp_dir) / "unit_tests"
            tests1.mkdir()
            tests2.mkdir()
            
            (tests1 / "test_a.py").write_text("def test_a(): pass")
            (tests2 / "test_b.py").write_text("def test_b(): pass")
            
            test_files = discover_test_files([str(tests1), str(tests2)])
            test_names = [Path(f).name for f in test_files]
            
            assert "test_a.py" in test_names
            assert "test_b.py" in test_names
            assert len(test_files) == 2
    
    def test_discover_test_files_nested_directories(self):
        """Test test discovery in nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            test_dir = Path(temp_dir) / "tests"
            unit_dir = test_dir / "unit"
            integration_dir = test_dir / "integration"
            
            for dir_path in [test_dir, unit_dir, integration_dir]:
                dir_path.mkdir(parents=True)
            
            (test_dir / "test_main.py").write_text("def test_main(): pass")
            (unit_dir / "test_unit.py").write_text("def test_unit(): pass")
            (integration_dir / "test_integration.py").write_text("def test_integration(): pass")
            
            test_files = discover_test_files([str(test_dir)])
            test_names = [Path(f).name for f in test_files]
            
            assert "test_main.py" in test_names
            assert "test_unit.py" in test_names
            assert "test_integration.py" in test_names
            assert len(test_files) == 3
    
    def test_discover_test_files_no_directory(self):
        """Test handling of non-existent directories."""
        test_files = discover_test_files(["/nonexistent/directory"])
        assert test_files == []


class TestCoverageReporter:
    """Test coverage reporter functionality."""
    
    def test_coverage_reporter_init(self):
        """Test coverage reporter initialization."""
        config = CoverageConfig()
        reporter = CoverageReporter(config)
        
        assert reporter.config == config
        assert reporter.source_dirs == ["src"]
        assert reporter.test_dirs == ["tests"]
    
    @patch('subprocess.run')
    def test_run_coverage_successful(self, mock_run):
        """Test successful coverage run."""
        # Mock successful pytest-cov execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"totals": {"percent_covered": 87.5, "num_statements": 500, "missing_lines": 62}}',
            stderr=""
        )
        
        config = CoverageConfig()
        reporter = CoverageReporter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = reporter.run_coverage(temp_dir)
            
            assert isinstance(result, CoverageResult)
            mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_coverage_failure(self, mock_run):
        """Test coverage run failure handling."""
        # Mock failed pytest-cov execution
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Error running coverage"
        )
        
        config = CoverageConfig()
        reporter = CoverageReporter(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(CoverageError, match="Coverage analysis failed"):
                reporter.run_coverage(temp_dir)
    
    def test_parse_coverage_output_json(self):
        """Test parsing JSON coverage output."""
        config = CoverageConfig()
        reporter = CoverageReporter(config)
        
        # Mock coverage.py JSON output format
        json_output = {
            "totals": {
                "percent_covered": 87.5,
                "percent_covered_display": "88%",
                "covered_lines": 450,
                "num_statements": 500,
                "missing_lines": 50,
                "excluded_lines": 10
            },
            "files": {
                "src/module1.py": {
                    "summary": {
                        "percent_covered": 90.0,
                        "covered_lines": 45,
                        "num_statements": 50
                    }
                }
            }
        }
        
        result = reporter._parse_coverage_json(json.dumps(json_output))
        
        assert result.total_coverage == 87.5
        assert result.lines_covered == 450
        assert result.lines_total == 500
        assert result.files_analyzed == 1
    
    def test_generate_html_report(self):
        """Test HTML report generation."""
        config = CoverageConfig()
        reporter = CoverageReporter(config)
        
        result = CoverageResult(
            total_coverage=87.5,
            line_coverage=90.0,
            branch_coverage=85.0,
            files_analyzed=25
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            html_path = reporter.generate_html_report(result, temp_dir)
            
            assert Path(html_path).exists()
            assert Path(html_path).suffix == ".html"
            
            # Check HTML content contains key information
            content = Path(html_path).read_text()
            assert "87.5%" in content
            assert "Coverage Report" in content


class TestCoverageValidation:
    """Test coverage threshold validation."""
    
    def test_validate_coverage_threshold_pass(self):
        """Test validation passes when coverage meets threshold."""
        result = CoverageResult(total_coverage=90.0, line_coverage=90.0, branch_coverage=90.0)
        
        is_valid, message = validate_coverage_threshold(result, 85.0)
        
        assert is_valid
        assert "Coverage of 90.0% meets threshold" in message
    
    def test_validate_coverage_threshold_fail(self):
        """Test validation fails when coverage below threshold."""
        result = CoverageResult(total_coverage=80.0, line_coverage=80.0, branch_coverage=80.0)
        
        is_valid, message = validate_coverage_threshold(result, 85.0)
        
        assert not is_valid
        assert "Coverage of 80.0% below threshold" in message
    
    def test_validate_coverage_threshold_exact(self):
        """Test validation at exact threshold."""
        result = CoverageResult(total_coverage=85.0, line_coverage=85.0, branch_coverage=85.0)
        
        is_valid, message = validate_coverage_threshold(result, 85.0)
        
        assert is_valid
        assert "Coverage of 85.0% meets threshold" in message


class TestCoverageIntegration:
    """Test integration functions."""
    
    @patch('autogen_code_review_bot.coverage_metrics.CoverageReporter')
    def test_run_coverage_analysis_success(self, mock_reporter_class):
        """Test successful coverage analysis integration."""
        # Mock reporter and result
        mock_reporter = Mock()
        mock_result = CoverageResult(total_coverage=90.0, line_coverage=90.0, branch_coverage=90.0)
        mock_reporter.run_coverage.return_value = mock_result
        mock_reporter_class.return_value = mock_reporter
        
        config = CoverageConfig()
        result = run_coverage_analysis("test_repo", config)
        
        assert result == mock_result
        mock_reporter.run_coverage.assert_called_once_with("test_repo")
    
    @patch('autogen_code_review_bot.coverage_metrics.CoverageReporter')
    def test_generate_coverage_report_with_html(self, mock_reporter_class):
        """Test coverage report generation with HTML output."""
        # Mock reporter and result
        mock_reporter = Mock()
        mock_result = CoverageResult(total_coverage=90.0, line_coverage=90.0, branch_coverage=90.0)
        mock_reporter.run_coverage.return_value = mock_result
        mock_reporter.generate_html_report.return_value = "/path/to/report.html"
        mock_reporter_class.return_value = mock_reporter
        
        config = CoverageConfig()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result, html_path = generate_coverage_report("test_repo", config, temp_dir)
            
            assert result == mock_result
            assert html_path == "/path/to/report.html"
            mock_reporter.generate_html_report.assert_called_once()


class TestCoverageError:
    """Test coverage error handling."""
    
    def test_coverage_error_creation(self):
        """Test custom coverage error creation."""
        error = CoverageError("Coverage analysis failed", details={"exit_code": 1})
        
        assert str(error) == "Coverage analysis failed"
        assert error.details == {"exit_code": 1}
    
    def test_coverage_error_without_details(self):
        """Test coverage error without details."""
        error = CoverageError("Simple error")
        
        assert str(error) == "Simple error"
        assert error.details == {}


class TestCoverageConfigFromFile:
    """Test loading coverage configuration from files."""
    
    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "coverage.yaml"
            config_content = """
            minimum_coverage: 90.0
            source_dirs:
              - src
              - lib
            test_dirs:
              - tests
              - spec
            exclude_patterns:
              - "*/migrations/*"
              - "*/test_*"
            fail_under: 90.0
            """
            config_file.write_text(config_content)
            
            config = CoverageConfig.from_file(str(config_file))
            
            assert config.minimum_coverage == 90.0
            assert config.source_dirs == ["src", "lib"]
            assert config.test_dirs == ["tests", "spec"]
            assert "*/migrations/*" in config.exclude_patterns
            assert config.fail_under == 90.0
    
    def test_load_config_missing_file(self):
        """Test handling of missing configuration file."""
        config = CoverageConfig.from_file("/nonexistent/config.yaml")
        
        # Should return default configuration
        assert config.minimum_coverage == 85.0
        assert config.source_dirs == ["src"]
        assert config.test_dirs == ["tests"]
    
    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid.yaml"
            config_file.write_text("invalid: yaml: content: [")
            
            config = CoverageConfig.from_file(str(config_file))
            
            # Should return default configuration
            assert config.minimum_coverage == 85.0


class TestCoverageIntegrationWithCI:
    """Test integration with CI systems."""
    
    def test_coverage_output_for_ci(self):
        """Test coverage output format suitable for CI systems."""
        result = CoverageResult(
            total_coverage=87.5,
            line_coverage=90.0,
            branch_coverage=85.0,
            files_analyzed=25,
            lines_covered=450,
            lines_total=500
        )
        
        ci_output = result.to_ci_format()
        
        # Should contain key metrics for CI parsing
        assert "total_coverage=87.5" in ci_output
        assert "line_coverage=90.0" in ci_output
        assert "files_analyzed=25" in ci_output
        assert "coverage_status=PASS" in ci_output or "coverage_status=FAIL" in ci_output
    
    def test_coverage_junit_xml_output(self):
        """Test JUnit XML format output for CI integration."""
        result = CoverageResult(
            total_coverage=87.5,
            line_coverage=90.0,
            branch_coverage=85.0,
            files_analyzed=25
        )
        
        xml_output = result.to_junit_xml()
        
        # Should be valid XML with test suite information
        assert "<testsuite" in xml_output
        assert 'name="coverage"' in xml_output
        assert "<testcase" in xml_output
        assert "87.5" in xml_output