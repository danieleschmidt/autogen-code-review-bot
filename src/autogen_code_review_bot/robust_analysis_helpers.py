#!/usr/bin/env python3
"""
Robust analysis helper functions with comprehensive error handling.
"""

import subprocess
from pathlib import Path
from typing import List

from .intelligent_cache_system import with_cache
from .linter_config import LinterConfig
from .logging_config import get_logger
from .models import AnalysisSection
from .robust_error_handling import (
    ErrorSeverity,
    robust_operation,
    safe_execute,
    validate_file_path,
)

logger = get_logger(__name__)


def is_ignored_path(path: Path) -> bool:
    """Check if path should be ignored during analysis."""
    ignored_patterns = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
    }

    return any(pattern in str(path) for pattern in ignored_patterns)


@with_cache(analysis_type="security", ttl_seconds=1800)
@robust_operation(
    component="security_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.HIGH,
    retry_count=1,
    fallback_value=AnalysisSection(
        tool="security-analyzer",
        output="Security analysis failed - using fallback safe result",
        return_code=1,
        metadata={"severity": "unknown", "tools_run": 0, "fallback": True},
    ),
)
def run_security_analysis(repo_path: str, languages: List[str]) -> AnalysisSection:
    """Run security analysis using available tools with robust error handling."""
    # Validate inputs
    validate_file_path(repo_path)
    if not languages:
        return AnalysisSection(
            tool="security-analyzer",
            output="No languages detected for security analysis",
            return_code=0,
            metadata={"severity": "low", "tools_run": 0},
        )

    results = []
    tools_attempted = 0

    # Python security analysis
    if "python" in languages:
        tools_attempted += 1
        bandit_result = safe_execute(
            _run_bandit_scan,
            repo_path,
            component="security_analysis",
            operation="bandit_scan",
            fallback_value="Bandit: Tool execution failed",
        )
        if bandit_result:
            results.append(bandit_result)

    # Generic secret scanning
    tools_attempted += 1
    secret_result = safe_execute(
        _run_secret_scan,
        repo_path,
        component="security_analysis",
        operation="secret_scan",
        fallback_value="Secret Detection: Tool execution failed",
    )
    if secret_result:
        results.append(secret_result)

    # Safety check for Python dependencies
    if "python" in languages:
        tools_attempted += 1
        safety_result = safe_execute(
            _run_safety_check,
            repo_path,
            component="security_analysis",
            operation="safety_check",
            fallback_value="Safety: Tool execution failed",
        )
        if safety_result:
            results.append(safety_result)

    output = "\n\n".join(results) if results else "No security issues detected"
    severity = "high" if any("CRITICAL" in r or "HIGH" in r for r in results) else "low"

    return AnalysisSection(
        tool="security-analyzer",
        output=output,
        return_code=0,
        metadata={
            "severity": severity,
            "tools_run": tools_attempted,
            "tools_succeeded": len(results),
        },
    )


def _run_bandit_scan(repo_path: str) -> str:
    """Run bandit security scan."""
    result = subprocess.run(
        ["bandit", "-r", repo_path, "-f", "txt"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=repo_path,
    )
    if result.stdout:
        return f"Bandit (Python Security): {result.stdout[:1000]}"
    return ""


def _run_secret_scan(repo_path: str) -> str:
    """Run secret detection scan."""
    result = subprocess.run(
        ["detect-secrets", "scan", "--all-files", repo_path],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_path,
    )
    if result.stdout and "No secrets were detected" not in result.stdout:
        return f"Secret Detection: {result.stdout[:500]}"
    return ""


def _run_safety_check(repo_path: str) -> str:
    """Run safety dependency check."""
    result = subprocess.run(
        ["safety", "check", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_path,
    )
    if result.stdout and result.stdout != "[]":
        return f"Safety (Dependencies): {result.stdout[:500]}"
    return ""


@with_cache(analysis_type="style", ttl_seconds=900)
@robust_operation(
    component="style_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.MEDIUM,
    retry_count=2,
    fallback_value=AnalysisSection(
        tool="style-analyzer",
        output="Style analysis failed - using fallback result",
        return_code=1,
        metadata={"issues_count": 0, "languages_analyzed": 0, "fallback": True},
    ),
)
def run_style_analysis(
    repo_path: str, languages: List[str], linter_config: LinterConfig
) -> AnalysisSection:
    """Run style analysis using language-specific linters with robust error handling."""
    # Validate inputs
    validate_file_path(repo_path)
    if not languages:
        return AnalysisSection(
            tool="style-analyzer",
            output="No languages detected for style analysis",
            return_code=0,
            metadata={"issues_count": 0, "languages_analyzed": 0},
        )

    results = []

    # Python style analysis
    if "python" in languages:
        python_result = safe_execute(
            _run_python_style_check,
            repo_path,
            linter_config,
            component="style_analysis",
            operation="python_linting",
            fallback_value="Python linter: Tool execution failed",
        )
        if python_result:
            results.append(python_result)

    # JavaScript/TypeScript style analysis
    if any(lang in languages for lang in ["javascript", "typescript"]):
        js_result = safe_execute(
            _run_js_style_check,
            repo_path,
            component="style_analysis",
            operation="js_linting",
            fallback_value="ESLint: Tool execution failed",
        )
        if js_result:
            results.append(js_result)

    # Go style analysis
    if "go" in languages:
        go_result = safe_execute(
            _run_go_style_check,
            repo_path,
            component="style_analysis",
            operation="go_linting",
            fallback_value="golangci-lint: Tool execution failed",
        )
        if go_result:
            results.append(go_result)

    output = "\n\n".join(results) if results else "No style issues detected"
    issues_count = len(
        [r for r in results if "error" in r.lower() or "warning" in r.lower()]
    )

    return AnalysisSection(
        tool="style-analyzer",
        output=output,
        return_code=0,
        metadata={"issues_count": issues_count, "languages_analyzed": len(languages)},
    )


def _run_python_style_check(repo_path: str, linter_config: LinterConfig) -> str:
    """Run Python style checking."""
    linter = getattr(linter_config, "python", "ruff")

    if linter == "ruff":
        result = subprocess.run(
            ["ruff", "check", repo_path, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=repo_path,
        )
        if result.stdout:
            return f"Ruff (Python): {result.stdout[:1000]}"
    elif linter == "flake8":
        result = subprocess.run(
            ["flake8", repo_path],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=repo_path,
        )
        if result.stdout:
            return f"Flake8 (Python): {result.stdout[:1000]}"

    return ""


def _run_js_style_check(repo_path: str) -> str:
    """Run JavaScript/TypeScript style checking."""
    result = subprocess.run(
        ["eslint", "--ext", ".js,.ts,.jsx,.tsx", repo_path],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=repo_path,
    )
    if result.stdout:
        return f"ESLint (JS/TS): {result.stdout[:1000]}"
    return ""


def _run_go_style_check(repo_path: str) -> str:
    """Run Go style checking."""
    result = subprocess.run(
        ["golangci-lint", "run", repo_path],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=repo_path,
    )
    if result.stdout:
        return f"golangci-lint (Go): {result.stdout[:1000]}"
    return ""


@with_cache(analysis_type="performance", ttl_seconds=1200)
@robust_operation(
    component="performance_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.LOW,
    retry_count=1,
    fallback_value=AnalysisSection(
        tool="performance-analyzer",
        output="Performance analysis completed with limited results",
        return_code=0,
        metadata={"hotspots": 0, "metrics_collected": 0, "fallback": True},
    ),
)
def run_performance_analysis(repo_path: str, languages: List[str]) -> AnalysisSection:
    """Run performance analysis and complexity metrics with robust error handling."""
    # Validate inputs
    validate_file_path(repo_path)
    if not languages:
        return AnalysisSection(
            tool="performance-analyzer",
            output="No languages detected for performance analysis",
            return_code=0,
            metadata={"hotspots": 0, "metrics_collected": 0},
        )

    results = []

    # Python complexity analysis
    if "python" in languages:
        complexity_result = safe_execute(
            _run_complexity_analysis,
            repo_path,
            component="performance_analysis",
            operation="complexity_check",
            fallback_value="",
        )
        if complexity_result:
            results.append(complexity_result)

    # Generic file size analysis
    file_size_result = safe_execute(
        _run_file_size_analysis,
        repo_path,
        component="performance_analysis",
        operation="file_size_check",
        fallback_value="",
    )
    if file_size_result:
        results.append(file_size_result)

    # Simple code metrics
    metrics_result = safe_execute(
        _run_code_metrics,
        repo_path,
        component="performance_analysis",
        operation="code_metrics",
        fallback_value="",
    )
    if metrics_result:
        results.append(metrics_result)

    output = (
        "\n\n".join(results)
        if results
        else "Performance analysis completed - no major concerns"
    )
    hotspots = len(
        [r for r in results if "complex" in r.lower() or "large" in r.lower()]
    )

    return AnalysisSection(
        tool="performance-analyzer",
        output=output,
        return_code=0,
        metadata={"hotspots": hotspots, "metrics_collected": len(results)},
    )


def _run_complexity_analysis(repo_path: str) -> str:
    """Run complexity analysis."""
    result = subprocess.run(
        ["radon", "cc", repo_path, "-s"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_path,
    )
    if result.stdout:
        return f"Radon Complexity (Python): {result.stdout[:1000]}"
    return ""


def _run_file_size_analysis(repo_path: str) -> str:
    """Run file size analysis."""
    large_files = []
    for file_path in Path(repo_path).rglob("*"):
        if file_path.is_file() and not is_ignored_path(file_path):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 1.0:  # Files larger than 1MB
                large_files.append(f"{file_path.name}: {size_mb:.1f}MB")

    if large_files:
        return "Large files detected:\n" + "\n".join(large_files[:10])
    return ""


def _run_code_metrics(repo_path: str) -> str:
    """Run basic code metrics."""
    total_lines = 0
    total_files = 0

    for file_path in Path(repo_path).rglob("*"):
        if (
            file_path.is_file()
            and not is_ignored_path(file_path)
            and file_path.suffix in {".py", ".js", ".ts", ".go", ".java", ".cpp", ".c"}
        ):

            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    total_files += 1
            except Exception:
                continue

    if total_files > 0:
        avg_lines = total_lines / total_files
        return f"Code metrics: {total_files} files, {total_lines} lines, avg {avg_lines:.1f} lines/file"
    return ""
