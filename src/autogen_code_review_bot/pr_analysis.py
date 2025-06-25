"""Automated pull request analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from subprocess import CalledProcessError, run
from typing import List


@dataclass
class AnalysisSection:
    """Individual section of the PR analysis report."""

    tool: str
    output: str


@dataclass
class PRAnalysisResult:
    """Container for all analysis sections."""

    security: AnalysisSection
    style: AnalysisSection
    performance: AnalysisSection


def _run_command(cmd: List[str], cwd: str) -> str:
    """Execute ``cmd`` in ``cwd`` and return combined output."""

    try:
        completed = run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
        return completed.stdout.strip()
    except CalledProcessError as exc:  # pragma: no cover - runtime feedback
        output = exc.stdout or ""
        err = exc.stderr or ""
        return (output + "\n" + err).strip()


def analyze_pr(repo_path: str) -> PRAnalysisResult:
    """Run static analysis tools against ``repo_path`` and return the results."""

    def ensure(tool: str) -> str:
        return "not installed" if which(tool) is None else ""

    # Style analysis via ruff
    style_output = ensure("ruff") or _run_command(
        ["ruff", "check", repo_path], cwd=repo_path
    )

    # Security analysis via bandit
    security_output = ensure("bandit") or _run_command(
        ["bandit", "-r", repo_path, "-q"], cwd=repo_path
    )

    # Performance estimation via radon cyclomatic complexity
    performance_output = ensure("radon") or _run_command(
        ["radon", "cc", "-s", "-a", repo_path], cwd=repo_path
    )

    return PRAnalysisResult(
        security=AnalysisSection(tool="bandit", output=security_output),
        style=AnalysisSection(tool="ruff", output=style_output),
        performance=AnalysisSection(tool="radon", output=performance_output),
    )
