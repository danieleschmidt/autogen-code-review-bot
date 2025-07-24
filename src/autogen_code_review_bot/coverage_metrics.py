"""Test coverage metrics and reporting functionality."""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml

from .logging_config import get_logger
from .system_config import get_system_config

logger = get_logger(__name__)


class CoverageError(Exception):
    """Exception raised when coverage analysis fails."""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class CoverageConfig:
    """Configuration for coverage analysis."""
    
    minimum_coverage: float = 85.0
    source_dirs: List[str] = field(default_factory=lambda: ["src"])
    test_dirs: List[str] = field(default_factory=lambda: ["tests"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/test_*", "*/tests/*", "*/__pycache__/*"
    ])
    fail_under: float = 85.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.minimum_coverage <= 100:
            raise ValueError("Coverage threshold must be between 0 and 100")
        if not 0 <= self.fail_under <= 100:
            raise ValueError("Fail under threshold must be between 0 and 100")
        if not self.source_dirs:
            raise ValueError("Source directories cannot be empty")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'CoverageConfig':
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            CoverageConfig instance
        """
        try:
            if not Path(config_path).exists():
                logger.warning("Coverage config file not found, using defaults", 
                              extra={"config_path": config_path})
                return cls()
            
            from .config_validation import validate_config_file, ConfigError, ValidationError
            
            try:
                # Validate configuration using our validation framework
                config_data = validate_config_file(config_path, "coverage")
                
                if not config_data:
                    logger.warning("Empty coverage config file, using defaults")
                    return cls()
                
                # Extract coverage-specific configuration
                coverage_config = config_data.get("coverage", {})
                
                logger.info("Loaded and validated coverage configuration from file", 
                           extra={"config_path": config_path, "config": coverage_config})
                
                return cls(**coverage_config)
                
            except (ConfigError, ValidationError) as e:
                logger.error(f"Coverage configuration validation failed: {e}", 
                           extra={"config_path": config_path})
                logger.warning("Using default coverage configuration due to validation errors")
                return cls()
            
        except (yaml.YAMLError, TypeError, ValueError) as e:
            logger.error("Failed to load coverage config, using defaults", 
                        extra={"config_path": config_path, "error": str(e)})
            return cls()


@dataclass
class CoverageResult:
    """Result of coverage analysis."""
    
    total_coverage: float = 0.0
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    files_analyzed: int = 0
    lines_covered: int = 0
    lines_total: int = 0
    branches_covered: int = 0
    branches_total: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def meets_threshold(self, threshold: float) -> bool:
        """Check if coverage meets the specified threshold."""
        return self.total_coverage >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "total_coverage": self.total_coverage,
            "line_coverage": self.line_coverage,
            "branch_coverage": self.branch_coverage,
            "files_analyzed": self.files_analyzed,
            "lines_covered": self.lines_covered,
            "lines_total": self.lines_total,
            "branches_covered": self.branches_covered,
            "branches_total": self.branches_total,
            "timestamp": self.timestamp
        }
    
    def to_ci_format(self) -> str:
        """Format result for CI system consumption."""
        status = "PASS" if self.meets_threshold(85.0) else "FAIL"
        return (
            f"total_coverage={self.total_coverage}\n"
            f"line_coverage={self.line_coverage}\n"
            f"branch_coverage={self.branch_coverage}\n"
            f"files_analyzed={self.files_analyzed}\n"
            f"coverage_status={status}"
        )
    
    def to_junit_xml(self) -> str:
        """Generate JUnit XML format for CI integration."""
        status = "PASS" if self.meets_threshold(85.0) else "FAIL"
        
        xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="coverage" tests="1" failures="{0 if status == 'PASS' else 1}" time="0">
    <testcase name="coverage_threshold" classname="coverage">
        {'<failure message="Coverage below threshold">Coverage: ' + str(self.total_coverage) + '%</failure>' if status == 'FAIL' else ''}
    </testcase>
</testsuite>'''
        return xml


class CoverageReporter:
    """Test coverage analysis and reporting."""
    
    def __init__(self, config: CoverageConfig):
        """Initialize coverage reporter.
        
        Args:
            config: Coverage configuration
        """
        self.config = config
        self.source_dirs = config.source_dirs
        self.test_dirs = config.test_dirs
    
    def run_coverage(self, repo_path: str) -> CoverageResult:
        """Run coverage analysis on the repository.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            CoverageResult with analysis results
            
        Raises:
            CoverageError: If coverage analysis fails
        """
        logger.info("Starting coverage analysis", 
                   extra={"repo_path": repo_path, "config": self.config.to_dict()})
        
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise CoverageError(f"Repository path does not exist: {repo_path}")
        
        # Discover test files
        test_files = discover_test_files([str(repo_path / td) for td in self.test_dirs])
        
        if not test_files:
            logger.warning("No test files found", 
                          extra={"test_dirs": self.test_dirs, "repo_path": str(repo_path)})
            return CoverageResult(files_analyzed=0)
        
        logger.info("Discovered test files", extra={"test_count": len(test_files)})
        
        # Run pytest with coverage
        try:
            result = self._run_pytest_coverage(repo_path, test_files)
            
            logger.info("Coverage analysis completed", 
                       extra={
                           "total_coverage": result.total_coverage,
                           "files_analyzed": result.files_analyzed,
                           "meets_threshold": result.meets_threshold(self.config.minimum_coverage)
                       })
            
            return result
            
        except Exception as e:
            logger.error("Coverage analysis failed", extra={"error": str(e)})
            raise CoverageError(f"Coverage analysis failed: {e}", {"exception": str(e)})
    
    def _run_pytest_coverage(self, repo_path: Path, test_files: List[str]) -> CoverageResult:
        """Run pytest with coverage collection.
        
        Args:
            repo_path: Path to repository
            test_files: List of test files to run
            
        Returns:
            CoverageResult with parsed results
        """
        # Build coverage command
        cmd = [
            "python", "-m", "pytest",
            "--cov=" + ",".join(self.source_dirs),
            "--cov-report=json",
            "--cov-report=term-missing",
            "--cov-fail-under=" + str(self.config.fail_under),
            "-v"
        ] + test_files
        
        logger.debug("Running coverage command", extra={"command": " ".join(cmd)})
        
        try:
            # Run coverage analysis
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=get_system_config().coverage_timeout
            )
            
            if result.returncode != 0:
                # Coverage failure might be due to threshold not met
                logger.warning("Coverage command returned non-zero", 
                              extra={"exit_code": result.returncode, "stderr": result.stderr})
            
            # Try to parse JSON output even if exit code is non-zero
            return self._parse_coverage_output(result.stdout, result.stderr)
            
        except subprocess.TimeoutExpired:
            raise CoverageError("Coverage analysis timed out")
        except subprocess.SubprocessError as e:
            raise CoverageError(f"Failed to run coverage analysis: {e}")
    
    def _parse_coverage_output(self, stdout: str, stderr: str) -> CoverageResult:
        """Parse coverage output and extract metrics.
        
        Args:
            stdout: Standard output from coverage command
            stderr: Standard error from coverage command
            
        Returns:
            CoverageResult with parsed metrics
        """
        # Look for JSON output in various places
        json_data = None
        
        # Try to find .coverage.json file (pytest-cov default)
        coverage_files = [
            "coverage.json",
            ".coverage.json",
            "htmlcov/coverage.json"
        ]
        
        for coverage_file in coverage_files:
            if Path(coverage_file).exists():
                try:
                    with open(coverage_file, 'r') as f:
                        json_data = json.load(f)
                    break
                except (json.JSONDecodeError, OSError):
                    continue
        
        if json_data:
            return self._parse_coverage_json(json.dumps(json_data))
        else:
            # Fall back to parsing text output
            return self._parse_coverage_text(stdout + "\n" + stderr)
    
    def _parse_coverage_json(self, json_output: str) -> CoverageResult:
        """Parse JSON coverage output.
        
        Args:
            json_output: JSON string with coverage data
            
        Returns:
            CoverageResult with parsed data
        """
        try:
            data = json.loads(json_output)
            totals = data.get("totals", {})
            files = data.get("files", {})
            
            return CoverageResult(
                total_coverage=totals.get("percent_covered", 0.0),
                line_coverage=totals.get("percent_covered", 0.0),
                branch_coverage=totals.get("percent_covered_branches", 0.0),
                files_analyzed=len(files),
                lines_covered=totals.get("covered_lines", 0),
                lines_total=totals.get("num_statements", 0),
                branches_covered=totals.get("covered_branches", 0),
                branches_total=totals.get("num_branches", 0)
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse coverage JSON", extra={"error": str(e)})
            return CoverageResult()
    
    def _parse_coverage_text(self, text_output: str) -> CoverageResult:
        """Parse text coverage output as fallback.
        
        Args:
            text_output: Text output from coverage command
            
        Returns:
            CoverageResult with parsed data
        """
        # Look for coverage percentage in text output
        import re
        
        # Common patterns for coverage output
        patterns = [
            r"TOTAL\s+\d+\s+\d+\s+(\d+)%",  # pytest-cov format
            r"Total coverage:\s+(\d+\.?\d*)%",  # Alternative format
            r"Coverage:\s+(\d+\.?\d*)%"  # Generic format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_output)
            if match:
                coverage = float(match.group(1))
                logger.info("Parsed coverage from text output", extra={"coverage": coverage})
                return CoverageResult(
                    total_coverage=coverage,
                    line_coverage=coverage,
                    files_analyzed=1  # Rough estimate
                )
        
        # If no patterns match, try to find any percentage
        percentage_match = re.search(r"(\d+\.?\d*)%", text_output)
        if percentage_match:
            coverage = float(percentage_match.group(1))
            return CoverageResult(
                total_coverage=coverage,
                line_coverage=coverage,
                files_analyzed=1
            )
        
        logger.warning("Could not parse coverage from output", 
                      extra={"output_sample": text_output[:200]})
        return CoverageResult()
    
    def generate_html_report(self, result: CoverageResult, output_dir: str) -> str:
        """Generate HTML coverage report.
        
        Args:
            result: Coverage analysis result
            output_dir: Directory to write HTML report
            
        Returns:
            Path to generated HTML report
        """
        output_path = Path(output_dir) / f"coverage_report_{int(time.time())}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Coverage Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Coverage Report</h1>
        <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Total Coverage: 
            <span class="{'pass' if result.total_coverage >= 85 else 'fail'}">{result.total_coverage:.1f}%</span>
            </strong>
        </div>
        <div class="metric">Line Coverage: {result.line_coverage:.1f}%</div>
        <div class="metric">Branch Coverage: {result.branch_coverage:.1f}%</div>
        <div class="metric">Files Analyzed: {result.files_analyzed}</div>
        <div class="metric">Lines Covered: {result.lines_covered} / {result.lines_total}</div>
    </div>
    
    <div class="thresholds">
        <h2>Thresholds</h2>
        <p>Minimum Coverage Required: {self.config.minimum_coverage}%</p>
        <p>Status: <span class="{'pass' if result.meets_threshold(self.config.minimum_coverage) else 'fail'}">
        {'PASS' if result.meets_threshold(self.config.minimum_coverage) else 'FAIL'}
        </span></p>
    </div>
</body>
</html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info("Generated HTML coverage report", extra={"report_path": str(output_path)})
        return str(output_path)


def discover_test_files(test_dirs: List[str]) -> List[str]:
    """Discover test files in specified directories.
    
    Args:
        test_dirs: List of directories to search for tests
        
    Returns:
        List of test file paths
    """
    test_files = []
    
    for test_dir in test_dirs:
        test_dir_path = Path(test_dir)
        if not test_dir_path.exists():
            logger.debug("Test directory does not exist", extra={"test_dir": test_dir})
            continue
        
        # Find Python files starting with 'test_'
        for test_file in test_dir_path.rglob("test_*.py"):
            if test_file.is_file():
                test_files.append(str(test_file))
    
    logger.debug("Test file discovery completed", 
                extra={"directories": test_dirs, "files_found": len(test_files)})
    return test_files


def validate_coverage_threshold(result: CoverageResult, threshold: float) -> Tuple[bool, str]:
    """Validate coverage against threshold.
    
    Args:
        result: Coverage analysis result
        threshold: Minimum coverage threshold
        
    Returns:
        Tuple of (is_valid, message)
    """
    is_valid = result.meets_threshold(threshold)
    
    if is_valid:
        message = f"Coverage of {result.total_coverage:.1f}% meets threshold of {threshold:.1f}%"
        logger.info("Coverage validation passed", 
                   extra={"coverage": result.total_coverage, "threshold": threshold})
    else:
        message = f"Coverage of {result.total_coverage:.1f}% below threshold of {threshold:.1f}%"
        logger.warning("Coverage validation failed", 
                      extra={"coverage": result.total_coverage, "threshold": threshold})
    
    return is_valid, message


def run_coverage_analysis(repo_path: str, config: CoverageConfig = None) -> CoverageResult:
    """Run complete coverage analysis.
    
    Args:
        repo_path: Path to repository
        config: Coverage configuration (uses default if not provided)
        
    Returns:
        CoverageResult with analysis results
    """
    if config is None:
        config = CoverageConfig()
    
    reporter = CoverageReporter(config)
    return reporter.run_coverage(repo_path)


def generate_coverage_report(repo_path: str, config: CoverageConfig = None, 
                           html_output_dir: str = None) -> Tuple[CoverageResult, Optional[str]]:
    """Generate complete coverage report with optional HTML output.
    
    Args:
        repo_path: Path to repository
        config: Coverage configuration
        html_output_dir: Directory for HTML report (optional)
        
    Returns:
        Tuple of (CoverageResult, html_report_path)
    """
    if config is None:
        config = CoverageConfig()
    
    reporter = CoverageReporter(config)
    result = reporter.run_coverage(repo_path)
    
    html_path = None
    if html_output_dir:
        html_path = reporter.generate_html_report(result, html_output_dir)
    
    return result, html_path