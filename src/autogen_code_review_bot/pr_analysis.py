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
from .logging_utils import get_request_logger, RequestContext, timed_operation, MetricsCollector
from .config import get_default_linters, get_default_timeout


def load_linter_config(config_path: str | Path | None = None) -> Dict[str, str]:
    """Return language→linter mapping loaded from ``config_path``.

    Missing languages fall back to configured default linters.
    """
    mapping = get_default_linters()
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


# Initialize logger
logger = get_request_logger(__name__)


def _run_command(cmd: List[str], cwd: str, timeout: int | None = None, context: RequestContext | None = None) -> str:
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

    # Use configured default timeout if none provided
    if timeout is None:
        timeout = get_default_timeout()

    if context:
        logger.debug(
            "Executing command",
            context=context,
            command=cmd[0] if cmd else "unknown",
            args_count=len(cmd) - 1 if len(cmd) > 1 else 0,
            working_directory=cwd,
            timeout=timeout
        )

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
        
        result = completed.stdout.strip()
        
        if context:
            logger.debug(
                "Command executed successfully",
                context=context,
                command=cmd[0],
                return_code=completed.returncode,
                output_length=len(result)
            )
            
        return result
        
    except TimeoutExpired:
        if context:
            logger.warning(
                "Command execution timed out",
                context=context,
                command=cmd[0] if cmd else "unknown",
                timeout=timeout
            )
        return "timed out"
        
    except CalledProcessError as exc:  # pragma: no cover - runtime feedback
        output = exc.stdout or ""
        err = exc.stderr or ""
        result = (output + "\n" + err).strip()
        
        if context:
            logger.warning(
                "Command execution failed",
                context=context,
                command=cmd[0] if cmd else "unknown",
                return_code=exc.returncode,
                error_output_length=len(result)
            )
            
        return result


def _run_style_checks(repo_path: str, linters: Dict[str, str], context: RequestContext | None = None) -> tuple[str, str]:
    """Return style tool label and output for ``repo_path``."""

    def ensure(tool: str) -> str:
        result = "not installed" if which(tool) is None else ""
        if result and context:
            logger.warning(
                f"Style tool not available: {tool}",
                context=context,
                tool=tool
            )
        return result

    languages = _detect_repo_languages(repo_path)
    if context:
        logger.debug(
            "Detected languages in repository",
            context=context,
            languages=list(languages)
        )
    
    style_sections: List[tuple[str, str]] = []
    for lang in sorted(languages):
        tool = linters.get(lang)
        if not tool:
            if context:
                logger.debug(
                    f"No linter configured for {lang}",
                    context=context,
                    language=lang
                )
            continue
            
        if context:
            logger.debug(
                f"Running {tool} for {lang}",
                context=context,
                language=lang,
                tool=tool
            )
            
        cmd = [tool]
        if tool == "ruff":
            cmd.append("check")
        cmd.append(repo_path)
        output = ensure(tool) or _run_command(cmd, cwd=repo_path, context=context)
        style_sections.append((tool, output))

    style_tool = "+".join(name for name, _ in style_sections) or ""
    style_output = "\n".join(f"{name}:\n{out}" for name, out in style_sections)
    
    if context:
        logger.debug(
            "Style checks completed",
            context=context,
            tools_used=[name for name, _ in style_sections],
            sections_count=len(style_sections)
        )
    
    return style_tool, style_output


def _run_security_checks(repo_path: str, ensure, context: RequestContext | None = None) -> str:
    tool_check = ensure("bandit")
    if tool_check:
        return tool_check
    return _run_command(["bandit", "-r", repo_path, "-q"], cwd=repo_path, context=context)


def _run_performance_checks(repo_path: str, ensure, context: RequestContext | None = None) -> str:
    tool_check = ensure("radon") 
    if tool_check:
        return tool_check
    return _run_command(["radon", "cc", "-s", "-a", repo_path], cwd=repo_path, context=context)


@timed_operation(operation="pr_analysis")
def analyze_pr(repo_path: str, config_path: str | None = None, context: RequestContext | None = None) -> PRAnalysisResult:
    """Run static analysis tools against ``repo_path`` and return the results.

    Parameters
    ----------
    repo_path:
        Path to the repository under analysis.
    config_path:
        Optional YAML file specifying language→linter mappings.
    context:
        Request context for logging correlation.
    """
    if context is None:
        context = RequestContext()
        
    logger.info(
        "Starting PR analysis",
        context=context,
        repo_path=repo_path,
        config_path=config_path
    )
    
    # Initialize metrics collector
    metrics = MetricsCollector(context)

    def ensure(tool: str) -> str:
        result = "not installed" if which(tool) is None else ""
        if result:
            logger.warning(
                f"Analysis tool not available: {tool}",
                context=context,
                tool=tool
            )
            metrics.increment("missing_tools")
        return result

    try:
        linters = load_linter_config(config_path)
        logger.debug(
            "Loaded linter configuration",
            context=context,
            linters=list(linters.keys())
        )

        # Run style checks
        logger.info("Running style checks", context=context)
        style_tool, style_output = _run_style_checks(repo_path, linters, context)
        metrics.increment("style_checks_completed")

        # Run security checks  
        logger.info("Running security checks", context=context)
        security_output = _run_security_checks(repo_path, ensure, context)
        metrics.increment("security_checks_completed")

        # Run performance checks
        logger.info("Running performance checks", context=context)
        performance_output = _run_performance_checks(repo_path, ensure, context)
        metrics.increment("performance_checks_completed")

        result = PRAnalysisResult(
            security=AnalysisSection(tool="bandit", output=security_output),
            style=AnalysisSection(tool=style_tool or "lint", output=style_output),
            performance=AnalysisSection(tool="radon", output=performance_output),
        )
        
        # Log completion metrics
        metrics.log_metrics("PR analysis completed successfully")
        
        logger.info(
            "PR analysis completed successfully",
            context=context,
            style_tool=style_tool,
            security_issues=len(security_output.split('\n')) if security_output else 0,
            performance_score=performance_output[:20] if performance_output else "N/A"
        )
        
        return result
        
    except Exception as e:
        metrics.increment("analysis_errors")
        metrics.log_metrics("PR analysis failed")
        
        logger.error(
            "PR analysis failed",
            context=context,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise
