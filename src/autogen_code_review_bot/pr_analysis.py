"""Automated pull request analysis utilities."""

from __future__ import annotations

from shutil import which
from subprocess import CalledProcessError, TimeoutExpired, run  # nosec B404 - subprocess usage is secure with validation
from typing import Dict, List, Set
import os
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from .language_detection import detect_language
from .models import AnalysisSection, PRAnalysisResult
from .caching import LinterCache, get_commit_hash

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
    Security: Validates config file path and contents.
    """
    mapping = DEFAULT_LINTERS.copy()
    if config_path:
        # Validate config file path for security
        if not _validate_path_safety(str(config_path)):
            return mapping  # Return defaults for unsafe paths
            
        try:
            config_path = Path(config_path)
            if not config_path.exists() or not config_path.is_file():
                return mapping
                
            # Limit file size to prevent DoS
            if config_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return mapping
                
            with open(config_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, yaml.YAMLError, PermissionError):
            return mapping  # Return defaults on any error
            
        linters = data.get("linters", {}) if isinstance(data, dict) else {}
        if isinstance(linters, dict):
            # Validate linter values against allowlist
            safe_linters = {}
            for k, v in linters.items():
                if isinstance(k, str) and isinstance(v, str):
                    linter_name = Path(v).name
                    if linter_name in ALLOWED_EXECUTABLES:
                        safe_linters[k] = v
            mapping.update(safe_linters)
    return mapping




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

# Allowlist of safe executables to prevent command injection
ALLOWED_EXECUTABLES = {
    "ruff", "eslint", "rubocop", "bandit", "radon", "golangci-lint",
    "flake8", "pylint", "mypy", "black", "isort", "prettier"
}


def _validate_command_safety(cmd: List[str]) -> bool:
    """Validate that command is safe to execute.
    
    Args:
        cmd: Command list to validate
        
    Returns:
        True if command is safe, False otherwise
    """
    if not cmd or not isinstance(cmd, list):
        return False
        
    # Check if executable is in allowlist
    executable = cmd[0]
    executable_name = Path(executable).name
    
    if executable_name not in ALLOWED_EXECUTABLES:
        return False
    
    # Check for shell metacharacters in arguments
    dangerous_chars = ['&', '|', ';', '$', '`', '>', '<', '"', "'", '\\']
    for arg in cmd:
        if any(char in str(arg) for char in dangerous_chars):
            return False
            
    return True


def _validate_path_safety(path: str) -> bool:
    """Validate that path is safe and doesn't contain traversal attempts.
    
    Args:
        path: File system path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    if not path or not isinstance(path, str):
        return False
        
    # Resolve path to detect traversal attempts
    try:
        resolved_path = Path(path).resolve()
        # Check for obvious traversal patterns
        if ".." in str(resolved_path) or str(resolved_path).startswith("/etc"):
            return False
        return True
    except (OSError, ValueError):
        return False


def _run_command(cmd: List[str], cwd: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute ``cmd`` in ``cwd`` and return combined output.
    
    Security: Validates command safety and path traversal prevention.
    """
    # Validate command safety
    if not _validate_command_safety(cmd):
        return f"unsafe command rejected: {cmd[0] if cmd else 'empty'}"
        
    # Validate working directory path
    if not _validate_path_safety(cwd):
        return "unsafe working directory path rejected"
        
    # Ensure working directory exists
    if not Path(cwd).is_dir():
        return "working directory does not exist"

    try:
        completed = run(  # nosec B603 - command is validated against allowlist
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
            timeout=timeout,
            shell=False,  # Explicitly disable shell to prevent injection
        )
        return completed.stdout.strip()
    except TimeoutExpired:
        return "timed out"
    except CalledProcessError as exc:  # pragma: no cover - runtime feedback
        output = exc.stdout or ""
        err = exc.stderr or ""
        return (output + "\n" + err).strip()
    except (OSError, FileNotFoundError) as exc:
        return f"execution error: {exc}"


def _run_single_linter(lang: str, tool: str, repo_path: str) -> tuple[str, str]:
    """Run a single linter for a language and return (tool_name, output)."""
    def ensure(tool: str) -> str:
        return "not installed" if which(tool) is None else ""
    
    if not tool:
        return "", ""
    
    cmd = [tool]
    if tool == "ruff":
        cmd.append("check")
    cmd.append(repo_path)
    output = ensure(tool) or _run_command(cmd, cwd=repo_path)
    return tool, output


def _run_style_checks(repo_path: str, linters: Dict[str, str]) -> tuple[str, str]:
    """Return style tool label and output for ``repo_path`` (sequential version)."""
    return _run_style_checks_parallel(repo_path, linters, max_workers=1)


def _run_style_checks_parallel(repo_path: str, linters: Dict[str, str], max_workers: int = 3) -> tuple[str, str]:
    """Return style tool label and output for ``repo_path`` with parallel execution."""
    languages = _detect_repo_languages(repo_path)
    
    # Prepare tasks for parallel execution
    tasks = []
    for lang in sorted(languages):
        tool = linters.get(lang)
        if tool:
            tasks.append((lang, tool))
    
    if not tasks:
        return "", ""
    
    style_sections: List[tuple[str, str]] = []
    
    if max_workers == 1:
        # Sequential execution for compatibility
        for lang, tool in tasks:
            result = _run_single_linter(lang, tool, repo_path)
            if result[0]:  # Only add if tool name is not empty
                style_sections.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(_run_single_linter, lang, tool, repo_path): (lang, tool)
                for lang, tool in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                try:
                    tool_name, output = future.result()
                    if tool_name:  # Only add if tool name is not empty
                        style_sections.append((tool_name, output))
                except Exception as exc:
                    lang, tool = future_to_task[future]
                    # Log error but continue with other tools
                    style_sections.append((tool, f"execution error: {exc}"))
    
    # Sort results for consistent output
    style_sections.sort(key=lambda x: x[0])
    
    style_tool = "+".join(name for name, _ in style_sections) or ""
    style_output = "\n".join(f"{name}:\n{out}" for name, out in style_sections)
    return style_tool, style_output


def _run_security_checks(repo_path: str, ensure) -> str:
    return ensure("bandit") or _run_command(["bandit", "-r", repo_path, "-q"], cwd=repo_path)


def _run_performance_checks(repo_path: str, ensure) -> str:
    return ensure("radon") or _run_command(["radon", "cc", "-s", "-a", repo_path], cwd=repo_path)


def _run_all_checks_parallel(repo_path: str, linters: Dict[str, str]) -> PRAnalysisResult:
    """Run all analysis checks (security, style, performance) in parallel."""
    def ensure(tool: str) -> str:
        # Validate tool is in allowlist before checking installation
        if Path(tool).name not in ALLOWED_EXECUTABLES:
            return f"tool '{tool}' not allowed"
        return "not installed" if which(tool) is None else ""
    
    # Define check functions
    def run_security():
        return _run_security_checks(repo_path, ensure)
    
    def run_style():
        return _run_style_checks_parallel(repo_path, linters, max_workers=3)
    
    def run_performance():
        return _run_performance_checks(repo_path, ensure)
    
    # Execute all checks in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        security_future = executor.submit(run_security)
        style_future = executor.submit(run_style)
        performance_future = executor.submit(run_performance)
        
        # Collect results
        try:
            security_output = security_future.result()
            style_tool, style_output = style_future.result()
            performance_output = performance_future.result()
        except Exception as exc:
            # If any check fails, return error result
            return _create_error_result(f"parallel execution error: {exc}")
    
    return PRAnalysisResult(
        security=AnalysisSection(tool="bandit", output=security_output),
        style=AnalysisSection(tool=style_tool or "lint", output=style_output),
        performance=AnalysisSection(tool="radon", output=performance_output),
    )


def analyze_pr(repo_path: str, config_path: str | None = None, use_cache: bool = True, use_parallel: bool = True) -> PRAnalysisResult:
    """Run static analysis tools against ``repo_path`` and return the results.

    Parameters
    ----------
    repo_path:
        Path to the repository under analysis. Must be a valid, safe directory path.
    config_path:
        Optional YAML file specifying language→linter mappings. Must be a safe file path.
    use_cache:
        Whether to use caching for performance optimization. Default: True.
    use_parallel:
        Whether to use parallel execution for better performance. Default: True.
    
    Security
    --------
    Validates all input paths and rejects unsafe commands/paths.
    """
    # Validate repository path for security
    if not repo_path or not isinstance(repo_path, str):
        return _create_error_result("invalid repository path")
    
    if not _validate_path_safety(repo_path):
        return _create_error_result("unsafe repository path rejected")
    
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists() or not repo_path_obj.is_dir():
        return _create_error_result("repository path does not exist or is not a directory")

    linters = load_linter_config(config_path)
    
    # Try to use cache if enabled
    cache = None
    commit_hash = None
    config_hash = None
    if use_cache:
        cache = LinterCache()
        commit_hash = get_commit_hash(repo_path)
        if commit_hash:
            config_hash = cache.get_config_hash(linters)
            cached_result = cache.get(commit_hash, config_hash)
            if cached_result:
                return cached_result

    # Run analysis - use parallel execution if enabled
    if use_parallel:
        result = _run_all_checks_parallel(repo_path, linters)
    else:
        # Sequential execution (original implementation)
        def ensure(tool: str) -> str:
            # Validate tool is in allowlist before checking installation
            if Path(tool).name not in ALLOWED_EXECUTABLES:
                return f"tool '{tool}' not allowed"
            return "not installed" if which(tool) is None else ""

        style_tool, style_output = _run_style_checks(repo_path, linters)
        security_output = _run_security_checks(repo_path, ensure)
        performance_output = _run_performance_checks(repo_path, ensure)

        result = PRAnalysisResult(
            security=AnalysisSection(tool="bandit", output=security_output),
            style=AnalysisSection(tool=style_tool or "lint", output=style_output),
            performance=AnalysisSection(tool="radon", output=performance_output),
        )
    
    # Cache the result if caching is enabled and we have a commit hash
    if use_cache and cache and commit_hash and config_hash:
        cache.set(commit_hash, config_hash, result)
    
    return result


def _create_error_result(error_msg: str) -> PRAnalysisResult:
    """Create a PRAnalysisResult with error messages."""
    error_section = AnalysisSection(tool="error", output=error_msg)
    return PRAnalysisResult(
        security=error_section,
        style=error_section,
        performance=error_section,
    )
