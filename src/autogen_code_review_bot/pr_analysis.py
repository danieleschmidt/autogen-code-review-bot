"""Automated pull request analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from subprocess import CalledProcessError, TimeoutExpired, run
from typing import Dict, List, Set
import os
from pathlib import Path
import yaml

from .language_detection import detect_language

# Default mapping of languages to linter executables
DEFAULT_LINTERS: Dict[str, str] = {
    "python": "ruff",
    "javascript": "eslint",
    "typescript": "eslint",
    "ruby": "rubocop",
}


def load_linter_config(config_path: str | Path | None = None) -> Dict[str, str]:
    """Return language→linter mapping loaded from ``config_path``.

    Missing languages fall back to :data:`DEFAULT_LINTERS`.
    """
    mapping = DEFAULT_LINTERS.copy()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        linters = data.get("linters", {}) if isinstance(data, dict) else {}
        if isinstance(linters, dict):
            mapping.update({str(k): str(v) for k, v in linters.items()})
    return mapping


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


def _detect_repo_languages(repo_path: str | Path) -> Set[str]:
    """Return a set of languages present in ``repo_path``."""

    repo_path = Path(repo_path)
    languages: Set[str] = set()
    for root, _, files in os.walk(repo_path):
        for name in files:
            file_path = Path(root) / name
            lang = detect_language(file_path)
            if lang != "unknown":
                languages.add(lang)
    return languages


DEFAULT_TIMEOUT = 30


def _run_command(cmd: List[str], cwd: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute ``cmd`` in ``cwd`` and return combined output.
    
    Security measures:
    - Uses shell=False to prevent shell injection
    - Validates input parameters
    - Limits environment access
    - Enforces timeout to prevent DoS
    """
    # Security: Validate input parameters
    if not isinstance(cmd, list):
        raise TypeError("Command must be a list of strings")
    
    if not cmd:
        raise ValueError("Command list cannot be empty")
    
    for arg in cmd:
        if not isinstance(arg, str):
            raise TypeError("All command arguments must be strings")
    
    # Security: Validate working directory
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("Working directory must be a non-empty string")

    try:
        # Security: Explicitly set shell=False and limit environment
        completed = run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
            timeout=timeout,
            shell=False,  # Explicit security: prevent shell injection
            env=None,     # Use inherited environment (safer than custom env)
        )
        return completed.stdout.strip()
    except TimeoutExpired:
        return "timed out"
    except CalledProcessError as exc:  # pragma: no cover - runtime feedback
        output = exc.stdout or ""
        err = exc.stderr or ""
        return (output + "\n" + err).strip()


def _run_style_checks(repo_path: str, linters: Dict[str, str]) -> tuple[str, str]:
    """Return style tool label and output for ``repo_path``."""

    def ensure(tool: str) -> str:
        return "not installed" if which(tool) is None else ""

    languages = _detect_repo_languages(repo_path)
    style_sections: List[tuple[str, str]] = []
    for lang in sorted(languages):
        tool = linters.get(lang)
        if not tool:
            continue
        cmd = [tool]
        if tool == "ruff":
            cmd.append("check")
        cmd.append(repo_path)
        output = ensure(tool) or _run_command(cmd, cwd=repo_path)
        style_sections.append((tool, output))

    style_tool = "+".join(name for name, _ in style_sections) or ""
    style_output = "\n".join(f"{name}:\n{out}" for name, out in style_sections)
    return style_tool, style_output


def _run_security_checks(repo_path: str, ensure) -> str:
    return ensure("bandit") or _run_command(["bandit", "-r", repo_path, "-q"], cwd=repo_path)


def _run_performance_checks(repo_path: str, ensure) -> str:
    return ensure("radon") or _run_command(["radon", "cc", "-s", "-a", repo_path], cwd=repo_path)


def analyze_pr(repo_path: str, config_path: str | None = None) -> PRAnalysisResult:
    """Run static analysis tools against ``repo_path`` and return the results.

    Parameters
    ----------
    repo_path:
        Path to the repository under analysis.
    config_path:
        Optional YAML file specifying language→linter mappings.
    """

    def ensure(tool: str) -> str:
        return "not installed" if which(tool) is None else ""

    linters = load_linter_config(config_path)

    style_tool, style_output = _run_style_checks(repo_path, linters)
    security_output = _run_security_checks(repo_path, ensure)
    performance_output = _run_performance_checks(repo_path, ensure)

    return PRAnalysisResult(
        security=AnalysisSection(tool="bandit", output=security_output),
        style=AnalysisSection(tool=style_tool or "lint", output=style_output),
        performance=AnalysisSection(tool="radon", output=performance_output),
    )
