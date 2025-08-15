#!/usr/bin/env python3
"""
Analysis helper functions for the AutoGen Code Review Bot.

This module contains the actual implementation of security, style, and performance analysis.
"""

import subprocess
from pathlib import Path
from typing import List

from .linter_config import LinterConfig
from .models import AnalysisSection
from .robust_error_handling import ErrorSeverity, robust_operation


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


@robust_operation(
    component="security_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.HIGH,
    retry_count=1,
)
def run_security_analysis(repo_path: str, languages: List[str]) -> AnalysisSection:
    """Run security analysis using available tools."""
    try:
        results = []

        # Python security analysis
        if "python" in languages:
            try:
                result = subprocess.run(
                    ["bandit", "-r", repo_path, "-f", "txt"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"Bandit (Python Security): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("Bandit: Tool not available or timed out")

        # Generic secret scanning
        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--all-files", repo_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_path,
            )
            if result.stdout and "No secrets were detected" not in result.stdout:
                results.append(f"Secret Detection: {result.stdout[:500]}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results.append("Secret Detection: Tool not available or timed out")

        # Safety check for Python dependencies
        if "python" in languages:
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=repo_path,
                )
                if result.stdout and result.stdout != "[]":
                    results.append(f"Safety (Dependencies): {result.stdout[:500]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("Safety: Tool not available or timed out")

        output = "\n\n".join(results) if results else "No security issues detected"
        severity = (
            "high" if any("CRITICAL" in r or "HIGH" in r for r in results) else "low"
        )

        return AnalysisSection(
            tool="security-analyzer",
            output=output,
            return_code=0,
            metadata={"severity": severity, "tools_run": len(results)},
        )

    except Exception as e:
        return AnalysisSection(
            tool="security-analyzer",
            output=f"Security analysis failed: {e}",
            return_code=1,
            metadata={"error": str(e)},
        )


@robust_operation(
    component="style_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.MEDIUM,
    retry_count=2,
)
def run_style_analysis(
    repo_path: str, languages: List[str], linter_config: LinterConfig
) -> AnalysisSection:
    """Run style analysis using language-specific linters."""
    try:
        results = []

        # Python style analysis
        if "python" in languages:
            linter = getattr(linter_config, "python", "ruff")
            try:
                if linter == "ruff":
                    result = subprocess.run(
                        ["ruff", "check", repo_path, "--output-format", "text"],
                        capture_output=True,
                        text=True,
                        timeout=180,
                        cwd=repo_path,
                    )
                    if result.stdout:
                        results.append(f"Ruff (Python): {result.stdout[:1000]}")
                elif linter == "flake8":
                    result = subprocess.run(
                        ["flake8", repo_path],
                        capture_output=True,
                        text=True,
                        timeout=180,
                        cwd=repo_path,
                    )
                    if result.stdout:
                        results.append(f"Flake8 (Python): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append(
                    f"Python linter ({linter}): Tool not available or timed out"
                )

        # JavaScript/TypeScript style analysis
        if "javascript" in languages or "typescript" in languages:
            try:
                result = subprocess.run(
                    ["eslint", "--ext", ".js,.ts,.jsx,.tsx", repo_path],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"ESLint (JS/TS): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("ESLint: Tool not available or timed out")

        # Go style analysis
        if "go" in languages:
            try:
                result = subprocess.run(
                    ["golangci-lint", "run", repo_path],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"golangci-lint (Go): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("golangci-lint: Tool not available or timed out")

        output = "\n\n".join(results) if results else "No style issues detected"
        issues_count = len(
            [r for r in results if "error" in r.lower() or "warning" in r.lower()]
        )

        return AnalysisSection(
            tool="style-analyzer",
            output=output,
            return_code=0,
            metadata={
                "issues_count": issues_count,
                "languages_analyzed": len(languages),
            },
        )

    except Exception as e:
        return AnalysisSection(
            tool="style-analyzer",
            output=f"Style analysis failed: {e}",
            return_code=1,
            metadata={"error": str(e)},
        )


@robust_operation(
    component="performance_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.LOW,
    retry_count=1,
)
def run_performance_analysis(repo_path: str, languages: List[str]) -> AnalysisSection:
    """Run performance analysis and complexity metrics."""
    try:
        results = []

        # Python complexity analysis
        if "python" in languages:
            try:
                result = subprocess.run(
                    ["radon", "cc", repo_path, "-s"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"Radon Complexity (Python): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("Radon: Tool not available or timed out")

        # Generic file size analysis
        try:
            large_files = []
            for file_path in Path(repo_path).rglob("*"):
                if file_path.is_file() and not is_ignored_path(file_path):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > 1.0:  # Files larger than 1MB
                        large_files.append(f"{file_path.name}: {size_mb:.1f}MB")

            if large_files:
                results.append("Large files detected:\n" + "\n".join(large_files[:10]))
        except Exception:
            pass

        # Simple code metrics
        try:
            total_lines = 0
            total_files = 0

            for file_path in Path(repo_path).rglob("*"):
                if (
                    file_path.is_file()
                    and not is_ignored_path(file_path)
                    and file_path.suffix
                    in {".py", ".js", ".ts", ".go", ".java", ".cpp", ".c"}
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
                results.append(
                    f"Code metrics: {total_files} files, {total_lines} lines, avg {avg_lines:.1f} lines/file"
                )
        except Exception:
            pass

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

    except Exception as e:
        return AnalysisSection(
            tool="performance-analyzer",
            output=f"Performance analysis failed: {e}",
            return_code=1,
            metadata={"error": str(e)},
        )
