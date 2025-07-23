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
from .caching import LinterCache, get_commit_hash, InvalidationStrategy
from .logging_config import get_logger
from .agents import run_agent_conversation
from .monitoring import MetricsEmitter
from .exceptions import AnalysisError, ValidationError, ToolError

# Default mapping of languages to linter executables
DEFAULT_LINTERS: Dict[str, str] = {
    "python": "ruff",
    "javascript": "eslint",
    "typescript": "eslint",
    "ruby": "rubocop",
}

# Initialize logger and metrics for PR analysis operations
logger = get_logger(__name__)
metrics = MetricsEmitter()


def load_linter_config(config_path: str | Path | None = None) -> Dict[str, str]:
    """Return language→linter mapping loaded from ``config_path``.

    Missing languages fall back to :data:`DEFAULT_LINTERS`.
    Security: Validates config file path and contents using comprehensive validation.
    """
    from .config_validation import validate_config_file, ConfigError, ValidationError
    
    mapping = DEFAULT_LINTERS.copy()
    if config_path:
        try:
            config_path = Path(config_path)
            if not config_path.exists() or not config_path.is_file():
                logger.warning(f"Linter config file not found: {config_path}, using defaults")
                return mapping
                
            # Limit file size to prevent DoS
            if config_path.stat().st_size > 1024 * 1024:  # 1MB limit
                logger.warning(f"Linter config file too large: {config_path}, using defaults")
                return mapping
            
            # Use our new validation system
            validated_config = validate_config_file(str(config_path), "linter")
            
            linters = validated_config.get("linters", {})
            if linters:
                mapping.update(linters)
                logger.info(f"Loaded and validated linter config with {len(linters)} mappings", 
                           extra={"config_path": str(config_path), "linters": linters})
            else:
                logger.debug("No linter mappings in config file, using defaults")
                
        except (ConfigError, ValidationError) as e:
            logger.error(f"Linter config validation failed: {e}", 
                        extra={"config_path": str(config_path)})
            # Return defaults on validation error - ensures system keeps working
            return mapping
        except (OSError, PermissionError) as e:
            logger.warning(f"Unable to read linter config file: {e}", 
                          extra={"config_path": str(config_path)})
            return mapping
    
    return mapping




def _detect_repo_languages(repo_path: str | Path, max_files: int = 10000) -> Set[str]:
    """Return a set of languages present in ``repo_path``.
    
    Args:
        repo_path: Path to the repository to analyze
        max_files: Maximum number of files to scan (default: 10,000)
        
    Returns:
        Set of detected programming languages
    """

    repo_path = Path(repo_path)
    languages: Set[str] = set()
    file_count = 0
    
    for root, _, files in os.walk(repo_path):
        for name in files:
            # Early exit condition to prevent excessive memory usage
            if file_count >= max_files:
                logger.warning(
                    f"Language detection stopped - reached file limit of {max_files}",
                    extra={
                        "files_scanned": file_count,
                        "max_files": max_files,
                        "repo_path": str(repo_path),
                        "languages_found": list(languages)
                    }
                )
                break
                
            file_path = Path(root) / name
            lang = detect_language(file_path)
            if lang != "unknown":
                languages.add(lang)
            file_count += 1
        else:
            # Continue outer loop if inner loop wasn't broken
            continue
        # Break outer loop if inner loop was broken
        break
    
    logger.debug("Language detection completed", 
                extra={
                    "languages": list(languages), 
                    "files_scanned": file_count,
                    "repo_path": str(repo_path),
                    "limit_reached": file_count >= max_files
                })
    return languages


def _get_repo_size_info(repo_path: str | Path) -> tuple[int, int]:
    """Get repository size information for streaming decision.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Tuple of (file_count, total_size_bytes)
    """
    repo_path = Path(repo_path)
    file_count = 0
    total_size = 0
    
    try:
        for root, _, files in os.walk(repo_path):
            for name in files:
                file_path = Path(root) / name
                try:
                    stat_info = file_path.stat()
                    file_count += 1
                    total_size += stat_info.st_size
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
    except (OSError, PermissionError):
        # If we can't walk the directory, return zeros
        logger.warning("Failed to access repository for size calculation", 
                      extra={"repo_path": str(repo_path)})
        return 0, 0
    
    logger.debug("Repository size analysis completed", 
                extra={
                    "repo_path": str(repo_path),
                    "file_count": file_count,
                    "total_size_mb": total_size / (1024 * 1024)
                })
    return file_count, total_size


def _should_use_streaming(file_count: int, total_size: int) -> bool:
    """Determine if we should use streaming analysis for a repository.
    
    Args:
        file_count: Number of files in the repository
        total_size: Total size in bytes
        
    Returns:
        True if streaming should be used, False otherwise
    """
    # Thresholds for streaming decision
    MAX_FILES_NORMAL = 1000        # More than 1000 files triggers streaming
    MAX_SIZE_NORMAL = 10 * 1024 * 1024  # More than 10MB triggers streaming
    
    should_stream = file_count > MAX_FILES_NORMAL or total_size > MAX_SIZE_NORMAL
    
    logger.debug("Streaming decision made", 
                extra={
                    "file_count": file_count,
                    "total_size_mb": total_size / (1024 * 1024),
                    "should_stream": should_stream,
                    "reason": "file_count" if file_count > MAX_FILES_NORMAL else "size" if total_size > MAX_SIZE_NORMAL else "normal"
                })
    
    return should_stream


def _detect_repo_languages_chunked(repo_path: str | Path, chunk_size: int = 100, progress_callback=None) -> Set[str]:
    """Detect repository languages using chunked processing for large repos.
    
    Args:
        repo_path: Path to the repository
        chunk_size: Number of files to process in each chunk
        progress_callback: Optional callback function called with (processed, total)
        
    Returns:
        Set of detected languages
    """
    repo_path = Path(repo_path)
    languages: Set[str] = set()
    
    # Collect all files first
    all_files = []
    try:
        for root, _, files in os.walk(repo_path):
            for name in files:
                file_path = Path(root) / name
                all_files.append(file_path)
    except (OSError, PermissionError) as e:
        logger.warning("Failed to walk repository for language detection", 
                      extra={"repo_path": str(repo_path), "error": str(e)})
        return languages
    
    total_files = len(all_files)
    processed = 0
    
    logger.info("Starting chunked language detection", 
               extra={
                   "repo_path": str(repo_path),
                   "total_files": total_files,
                   "chunk_size": chunk_size
               })
    
    # Process files in chunks
    for i in range(0, total_files, chunk_size):
        chunk_files = all_files[i:i + chunk_size]
        
        for file_path in chunk_files:
            try:
                lang = detect_language(file_path)
                if lang != "unknown":
                    languages.add(lang)
                processed += 1
            except (OSError, PermissionError):
                # Skip files we can't access
                processed += 1
                continue
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(processed, total_files)
        
        logger.debug("Language detection chunk completed", 
                    extra={
                        "processed": processed,
                        "total": total_files,
                        "chunk_end": i + chunk_size,
                        "languages_so_far": list(languages)
                    })
    
    logger.info("Chunked language detection completed", 
               extra={
                   "repo_path": str(repo_path),
                   "languages": list(languages),
                   "files_processed": processed,
                   "total_files": total_files
               })
    
    return languages


def _analyze_pr_streaming(repo_path: str, linters: Dict[str, str], use_cache: bool = True) -> PRAnalysisResult:
    """Analyze a PR using streaming approach for large repositories.
    
    Args:
        repo_path: Path to the repository
        linters: Language to linter mapping
        use_cache: Whether to use caching
        
    Returns:
        PRAnalysisResult with analysis results
    """
    logger.info("Starting streaming PR analysis", 
               extra={"repo_path": repo_path, "use_cache": use_cache})
    
    try:
        # Use chunked language detection for large repos
        languages = _detect_repo_languages_chunked(repo_path, chunk_size=200)
        
        # Apply cache logic if enabled
        cache = None
        commit_hash = None
        config_hash = None
        
        if use_cache:
            logger.debug("Initializing cache for streaming analysis")
            cache = LinterCache()
            
            # Set up cache invalidation strategy
            cache.invalidation_strategy = InvalidationStrategy(cache.cache_dir)
            
            commit_hash = get_commit_hash(repo_path)
            if commit_hash:
                config_hash = cache.get_config_hash(linters)
                logger.debug("Checking cache for streaming analysis results", 
                            extra={
                                "commit_hash": commit_hash[:8], 
                                "config_hash": config_hash[:8]
                            })
                
                # Get list of tools for invalidation check
                tools = list(linters.values())
                cached_result = cache.get_with_invalidation_check(commit_hash, config_hash, tools)
                if cached_result:
                    logger.info("Cache hit for streaming analysis", 
                               extra={"commit_hash": commit_hash[:8]})
                    return cached_result
            else:
                logger.debug("No commit hash available for streaming analysis")
        
        # Run the actual analysis (reuse existing parallel implementation)
        logger.info("Running analysis with streaming optimizations")
        result = _run_all_checks_parallel(repo_path, linters)
        
        logger.info("Streaming analysis completed successfully",
                    extra={
                        "languages_detected": list(languages),
                        "has_security_issues": bool(result.security.output.strip()),
                        "has_style_issues": bool(result.style.output.strip()),
                        "has_performance_issues": bool(result.performance.output.strip())
                    })
        
        # Cache the result if caching is enabled
        if use_cache and cache and commit_hash and config_hash:
            logger.debug("Caching streaming analysis results", 
                        extra={"commit_hash": commit_hash[:8]})
            cache.set(commit_hash, config_hash, result)
        
        return result
        
    except Exception as exc:
        logger.error("Streaming analysis failed", extra={"error": str(exc)})
        raise AnalysisError(f"Streaming analysis error: {exc}") from exc


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
    
    # Resolve executable path to detect traversal attempts
    try:
        resolved_executable = Path(executable).resolve()
        executable_name = resolved_executable.name
        
        # Only allow if executable name matches allowlist exactly
        if executable_name not in ALLOWED_EXECUTABLES:
            return False
            
        # Reject any path that contains directory traversal
        # Only allow simple executable names or absolute paths to standard locations
        if '/' in executable or '\\' in executable:
            # For security, reject any path-based executable references
            # Only allow direct executable names that will be resolved via PATH
            return False
            
    except (OSError, ValueError):
        return False
    
    # Check for shell metacharacters in arguments
    dangerous_chars = ['&', '|', ';', '$', '`', '>', '<', '"', "'", '\\', '\x00']
    for arg in cmd:
        if any(char in str(arg) for char in dangerous_chars):
            return False
            
    return True


def _validate_path_safety(path: str, project_root: str = None) -> bool:
    """Validate that path is safe and doesn't contain traversal attempts.
    
    Args:
        path: File system path to validate
        project_root: Optional project root to validate against
        
    Returns:
        True if path is safe, False otherwise
    """
    if not path or not isinstance(path, str):
        return False
    
    # Check for null bytes and other dangerous characters
    if '\x00' in path or '\r' in path or '\n' in path or '\t' in path:
        return False
        
    # Check for URL encoding attempts
    if '%' in path and any(x in path.lower() for x in ['%2e', '%2f', '%5c']):
        return False
    
    # Check path length to prevent buffer overflow attempts
    if len(path) > 4096:
        return False
        
    try:
        # Resolve path to detect traversal attempts and symlink following
        resolved_path = Path(path).resolve()
        path_str = str(resolved_path)
        
        # Define sensitive directories that should never be accessible
        sensitive_prefixes = [
            "/etc/", "/proc/", "/sys/", "/dev/",
            "/home/", "/var/log/", "/usr/bin/", "/usr/sbin/",
            "/bin/", "/sbin/", "/.ssh/", "/tmp/",
            "C:\\Windows\\", "C:\\Users\\", "C:\\Program Files\\"
        ]
        
        # Check if resolved path points to sensitive locations
        # Only check if path starts with these sensitive directories, but allow
        # exceptions when there's a valid project_root containing the path
        for prefix in sensitive_prefixes:
            if path_str.startswith(prefix):
                # If we have a project_root, check if this is within allowed boundaries
                if project_root:
                    try:
                        project_resolved = Path(project_root).resolve()
                        resolved_path.relative_to(project_resolved)
                        # Path is within project boundaries, allow it even if in sensitive location
                        continue
                    except ValueError:
                        # Path is outside project root and in sensitive location - reject
                        return False
                else:
                    # No project root context and in sensitive location - reject
                    return False
        
        # Special case for /root/ - only block if not in a development context
        # Allow /root/repo/ and subdirectories for development, but block direct /root/ access
        if path_str.startswith("/root/") and not any(allowed in path_str for allowed in ["/root/repo", "/root/workspace", "/root/project"]):
            # If we have a project_root, check if this is within allowed boundaries
            if project_root:
                try:
                    project_resolved = Path(project_root).resolve()
                    resolved_path.relative_to(project_resolved)
                    # Path is within project boundaries, allow it
                except ValueError:
                    # Path is outside project root - reject
                    return False
            else:
                # No project root context - reject direct /root/ access
                return False
        
        # If project_root is provided, ensure path is within project boundaries
        if project_root:
            try:
                project_resolved = Path(project_root).resolve()
                # Check if resolved path is within project root
                try:
                    resolved_path.relative_to(project_resolved)
                except ValueError:
                    # Path is outside project root
                    return False
            except (OSError, ValueError):
                return False
        
        # Additional checks for traversal patterns that might survive resolution
        dangerous_patterns = ['../', '..\\', '..\\//', '..\\..', '....']
        for pattern in dangerous_patterns:
            if pattern in path or pattern in path_str:
                return False
                
        # Check for symlink that might point outside allowed areas
        original_path = Path(path)
        if original_path.is_symlink():
            # For symlinks, we need to be extra careful
            # Only allow symlinks that resolve to safe locations within project
            if project_root:
                try:
                    resolved_path.relative_to(Path(project_root).resolve())
                except ValueError:
                    return False
            else:
                # Without project root context, reject all symlinks for safety
                return False
        
        return True
        
    except (OSError, ValueError, RuntimeError):
        # Any error in path resolution is treated as unsafe
        return False


def _run_command(cmd: List[str], cwd: str, timeout: int = DEFAULT_TIMEOUT, project_root: str | None = None) -> str:
    """Execute ``cmd`` in ``cwd`` and return combined output.
    
    Security: Validates command safety and path traversal prevention.
    
    Args:
        cmd: Command and arguments to execute
        cwd: Working directory for command execution
        timeout: Command timeout in seconds
        project_root: Project root path for security validation
        
    Returns:
        Combined stdout/stderr output from command
        
    Raises:
        ValidationError: If command or path validation fails
        ToolError: If command execution fails
    """
    # Validate command safety
    if not _validate_command_safety(cmd):
        logger.error("Unsafe command rejected", 
                    extra={"command": cmd[0] if cmd else 'empty', "full_cmd": cmd})
        raise ValidationError(f"Unsafe command rejected: {cmd[0] if cmd else 'empty'}")
        
    # Validate working directory path
    if not _validate_path_safety(cwd, project_root):
        logger.error("Unsafe working directory path rejected", 
                    extra={"cwd": cwd, "project_root": project_root})
        raise ValidationError(f"Unsafe working directory path rejected: {cwd}")
        
    # Ensure working directory exists
    if not Path(cwd).is_dir():
        logger.error("Working directory does not exist", extra={"cwd": cwd})
        raise ValidationError(f"Working directory does not exist: {cwd}")

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
        logger.debug("Command executed successfully", 
                    extra={"command": cmd[0], "cwd": cwd, "output_length": len(completed.stdout)})
        return completed.stdout.strip()
    except TimeoutExpired as exc:
        logger.error("Command timed out", 
                    extra={"command": cmd[0], "timeout": timeout, "cwd": cwd})
        raise ToolError(f"Command '{cmd[0]}' timed out after {timeout}s") from exc
    except CalledProcessError as exc:  # pragma: no cover - runtime feedback
        output = exc.stdout or ""
        err = exc.stderr or ""
        combined_output = (output + "\n" + err).strip()
        logger.warning("Command returned non-zero exit code", 
                      extra={
                          "command": cmd[0], 
                          "exit_code": exc.returncode,
                          "cwd": cwd,
                          "output_length": len(combined_output)
                      })
        # For linter tools, non-zero exit often means issues found, not failure
        return combined_output
    except (OSError, FileNotFoundError) as exc:
        logger.error("Command execution failed", 
                    extra={"command": cmd[0], "cwd": cwd, "error": str(exc)})
        raise ToolError(f"Failed to execute '{cmd[0]}': {exc}") from exc


def _run_single_linter(lang: str, tool: str, repo_path: str) -> tuple[str, str]:
    """Run a single linter for a language and return (tool_name, output).
    
    Args:
        lang: Programming language name
        tool: Linter tool name
        repo_path: Repository path to analyze
        
    Returns:
        Tuple of (tool_name, output_text)
        
    Raises:
        ToolError: If linter execution fails
        ValidationError: If paths or commands are invalid
    """
    def ensure(tool: str) -> str:
        return "not installed" if which(tool) is None else ""
    
    if not tool:
        return "", ""
    
    # Check if tool is installed
    installation_check = ensure(tool)
    if installation_check:
        logger.warning("Linter tool not available", 
                      extra={"tool": tool, "language": lang, "status": installation_check})
        return tool, installation_check
    
    cmd = [tool]
    if tool == "ruff":
        cmd.append("check")
    cmd.append(repo_path)
    
    try:
        output = _run_command(cmd, cwd=repo_path, project_root=repo_path)
        return tool, output
    except (ToolError, ValidationError) as exc:
        logger.error("Linter execution failed", 
                    extra={"tool": tool, "language": lang, "error": str(exc)})
        # Re-raise with more context
        raise ToolError(f"Failed to run {tool} for {lang}: {exc}") from exc


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
    """Run security analysis checks on the repository.
    
    Args:
        repo_path: Path to repository to analyze
        ensure: Function to check tool availability
        
    Returns:
        Security analysis output
        
    Raises:
        ToolError: If security tool execution fails
        ValidationError: If paths are invalid
    """
    installation_check = ensure("bandit")
    if installation_check:
        logger.warning("Security tool not available", 
                      extra={"tool": "bandit", "status": installation_check})
        return installation_check
    
    try:
        return _run_command(["bandit", "-r", repo_path, "-q"], cwd=repo_path, project_root=repo_path)
    except (ToolError, ValidationError) as exc:
        logger.error("Security analysis failed", extra={"tool": "bandit", "error": str(exc)})
        raise ToolError(f"Security analysis failed: {exc}") from exc


def _run_performance_checks(repo_path: str, ensure) -> str:
    """Run performance analysis checks on the repository.
    
    Args:
        repo_path: Path to repository to analyze
        ensure: Function to check tool availability
        
    Returns:
        Performance analysis output
        
    Raises:
        ToolError: If performance tool execution fails
        ValidationError: If paths are invalid
    """
    installation_check = ensure("radon")
    if installation_check:
        logger.warning("Performance tool not available", 
                      extra={"tool": "radon", "status": installation_check})
        return installation_check
    
    try:
        return _run_command(["radon", "cc", "-s", "-a", repo_path], cwd=repo_path, project_root=repo_path)
    except (ToolError, ValidationError) as exc:
        logger.error("Performance analysis failed", extra={"tool": "radon", "error": str(exc)})
        raise ToolError(f"Performance analysis failed: {exc}") from exc


def _run_timed_check(check_type: str, check_func, *args):
    """Run a check function with timing and metrics recording.
    
    Args:
        check_type: Type of check (security, style, performance) for logging/metrics
        check_func: Function to execute
        *args: Arguments to pass to check_func
        
    Returns:
        Result from check_func
    """
    import time
    start_time = time.time()
    logger.debug(f"Starting {check_type} analysis")
    result = check_func(*args)
    duration = time.time() - start_time
    metrics.record_histogram("pr_analysis_check_duration_seconds", duration, tags={"check_type": check_type})
    logger.debug(f"{check_type.capitalize()} analysis completed", extra={"duration_seconds": duration})
    return result


def _run_all_checks_parallel(repo_path: str, linters: Dict[str, str]) -> PRAnalysisResult:
    """Run all analysis checks (security, style, performance) in parallel."""
    logger.debug("Starting parallel analysis checks", extra={"repo_path": repo_path, "linters": linters})
    
    def ensure(tool: str) -> str:
        # Validate tool is in allowlist before checking installation
        if Path(tool).name not in ALLOWED_EXECUTABLES:
            return f"tool '{tool}' not allowed"
        return "not installed" if which(tool) is None else ""
    
    # Define check functions using the timing utility
    def run_security():
        return _run_timed_check("security", _run_security_checks, repo_path, ensure)
    
    def run_style():
        return _run_timed_check("style", _run_style_checks_parallel, repo_path, linters, 3)
    
    def run_performance():
        return _run_timed_check("performance", _run_performance_checks, repo_path, ensure)
    
    # Execute all checks in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        logger.debug("Submitting parallel analysis tasks")
        # Submit all tasks
        security_future = executor.submit(run_security)
        style_future = executor.submit(run_style)
        performance_future = executor.submit(run_performance)
        
        # Collect results
        try:
            security_output = security_future.result()
            style_tool, style_output = style_future.result()
            performance_output = performance_future.result()
            logger.debug("All parallel analysis tasks completed successfully")
        except (ToolError, ValidationError, AnalysisError) as exc:
            # Re-raise specific analysis errors with context
            logger.error("Analysis task failed", 
                        extra={"error_type": type(exc).__name__, "error": str(exc)})
            metrics.record_counter("pr_analysis_errors_total", 1, tags={"error_type": type(exc).__name__})
            raise AnalysisError(f"Analysis failed during parallel execution: {exc}") from exc
        except Exception as exc:
            # Handle unexpected errors
            logger.error("Unexpected error in parallel execution", 
                        extra={"error_type": type(exc).__name__, "error": str(exc)})
            metrics.record_counter("pr_analysis_errors_total", 1, tags={"error_type": "unexpected"})
            raise AnalysisError(f"Unexpected error during analysis: {exc}") from exc
    
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
    # Start timing and metrics collection
    import time
    analysis_start_time = time.time()
    
    # Record throughput metrics
    metrics.record_counter("pr_analysis_requests_total", 1, 
                          tags={"cache_enabled": str(use_cache), "parallel_enabled": str(use_parallel)})
    
    logger.info("Starting PR analysis", 
                extra={
                    "repo_path": repo_path, 
                    "use_cache": use_cache, 
                    "use_parallel": use_parallel,
                    "config_path": config_path
                })
    
    # Validate repository path for security
    if not repo_path or not isinstance(repo_path, str):
        logger.error("Invalid repository path provided", extra={"repo_path": repo_path})
        # Record error metrics
        metrics.record_counter("pr_analysis_errors_total", 1, tags={"error_type": "invalid_path"})
        raise ValidationError("Invalid repository path: must be a non-empty string")
    
    if not _validate_path_safety(repo_path):
        logger.error("Unsafe repository path rejected", extra={"repo_path": repo_path})
        # Record security error metrics
        metrics.record_counter("pr_analysis_errors_total", 1, tags={"error_type": "unsafe_path"})
        raise ValidationError(f"Unsafe repository path rejected: {repo_path}")
    
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists() or not repo_path_obj.is_dir():
        logger.error("Repository path validation failed", 
                    extra={
                        "repo_path": repo_path, 
                        "exists": repo_path_obj.exists(), 
                        "is_dir": repo_path_obj.is_dir()
                    })
        # Record validation error metrics
        metrics.record_counter("pr_analysis_errors_total", 1, tags={"error_type": "path_validation"})
        raise ValidationError(f"Repository path does not exist or is not a directory: {repo_path}")

    logger.debug("Loading linter configuration", extra={"config_path": config_path})
    linters = load_linter_config(config_path)
    logger.debug("Linter configuration loaded", extra={"linters": linters})
    
    # Try to use cache if enabled
    cache = None
    commit_hash = None
    config_hash = None
    if use_cache:
        logger.debug("Initializing cache for analysis")
        cache = LinterCache()
        
        # Set up cache invalidation strategy
        cache.invalidation_strategy = InvalidationStrategy(cache.cache_dir)
        
        commit_hash = get_commit_hash(repo_path)
        if commit_hash:
            config_hash = cache.get_config_hash(linters)
            logger.debug("Checking cache for existing results", 
                        extra={
                            "commit_hash": commit_hash[:8], 
                            "config_hash": config_hash[:8]
                        })
            
            # Get list of tools for invalidation check
            tools = list(linters.values())
            cached_result = cache.get_with_invalidation_check(commit_hash, config_hash, tools)
            if cached_result:
                logger.info("Cache hit - returning cached results", 
                           extra={"commit_hash": commit_hash[:8]})
                # Record cache hit metrics
                metrics.record_counter("pr_analysis_cache_hits_total", 1)
                analysis_duration = time.time() - analysis_start_time
                metrics.record_histogram("pr_analysis_duration_seconds", analysis_duration, tags={"cache_hit": "true"})
                return cached_result
        else:
            logger.debug("No commit hash available, cache disabled")

    # Check if we should use streaming for large repositories
    file_count, total_size = _get_repo_size_info(repo_path)
    if _should_use_streaming(file_count, total_size):
        logger.info("Using streaming analysis for large repository", 
                   extra={
                       "file_count": file_count,
                       "total_size_mb": total_size / (1024 * 1024),
                       "reason": "repository_size_optimization"
                   })
        # Record streaming analysis metrics
        metrics.record_counter("pr_analysis_streaming_total", 1, tags={"file_count": str(file_count), "size_mb": str(int(total_size / (1024 * 1024)))})
        streaming_result = _analyze_pr_streaming(repo_path, linters, use_cache)
        analysis_duration = time.time() - analysis_start_time
        metrics.record_histogram("pr_analysis_duration_seconds", analysis_duration, tags={"method": "streaming"})
        return streaming_result

    # Run analysis - use parallel execution if enabled
    analysis_method_start = time.time()
    
    if use_parallel:
        logger.info("Running analysis with parallel execution")
        result = _run_all_checks_parallel(repo_path, linters)
        # Record parallel execution metrics
        parallel_duration = time.time() - analysis_method_start
        metrics.record_histogram("pr_analysis_method_duration_seconds", parallel_duration, tags={"method": "parallel"})
        metrics.record_counter("pr_analysis_method_total", 1, tags={"method": "parallel"})
    else:
        logger.info("Running analysis with sequential execution")
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
        # Record sequential execution metrics
        sequential_duration = time.time() - analysis_method_start
        metrics.record_histogram("pr_analysis_method_duration_seconds", sequential_duration, tags={"method": "sequential"})
        metrics.record_counter("pr_analysis_method_total", 1, tags={"method": "sequential"})
    
    # Calculate and record completion metrics
    analysis_duration = time.time() - analysis_start_time
    metrics.record_histogram("pr_analysis_duration_seconds", analysis_duration, tags={"cache_hit": "false"})
    metrics.record_counter("pr_analysis_completed_total", 1)
    
    # Record issue detection metrics
    has_security_issues = bool(result.security.output.strip())
    has_style_issues = bool(result.style.output.strip())
    has_performance_issues = bool(result.performance.output.strip())
    
    if has_security_issues:
        metrics.record_counter("pr_analysis_issues_detected_total", 1, tags={"issue_type": "security"})
    if has_style_issues:
        metrics.record_counter("pr_analysis_issues_detected_total", 1, tags={"issue_type": "style"})
    if has_performance_issues:
        metrics.record_counter("pr_analysis_issues_detected_total", 1, tags={"issue_type": "performance"})
    
    # Record quality score (inverse of issues found)
    total_issues = sum([has_security_issues, has_style_issues, has_performance_issues])
    quality_score = max(0, 100 - (total_issues * 33))  # 0-100 scale
    metrics.record_gauge("pr_analysis_quality_score", quality_score)
    
    logger.info("Analysis completed successfully",
                extra={
                    "has_security_issues": has_security_issues,
                    "has_style_issues": has_style_issues,
                    "has_performance_issues": has_performance_issues,
                    "analysis_duration_seconds": analysis_duration,
                    "quality_score": quality_score
                })
    
    # Cache the result if caching is enabled and we have a commit hash
    if use_cache and cache and commit_hash and config_hash:
        logger.debug("Caching analysis results", extra={"commit_hash": commit_hash[:8]})
        cache.set(commit_hash, config_hash, result)
        # Record cache storage metrics
        metrics.record_counter("pr_analysis_cache_stores_total", 1)
    
    return result


def format_analysis_with_agents(result: PRAnalysisResult, config_path: str | None = None) -> str:
    """Format analysis result with agent conversation for enhanced feedback.
    
    Args:
        result: The PR analysis result
        config_path: Optional path to agent configuration file
        
    Returns:
        Formatted analysis with agent conversation
    """
    logger.info("Formatting analysis with agent conversation", 
               extra={"has_config": config_path is not None})
    
    if not config_path:
        # Fallback to basic formatting if no agent config
        return format_analysis_result(result)
    
    try:
        # Prepare code context from analysis results
        code_context = _extract_code_context(result)
        
        # Run agent conversation
        conversation_result = run_agent_conversation(code_context, config_path)
        
        # Combine original analysis with agent conversation
        formatted_result = f"## Code Review Analysis\n\n"
        formatted_result += format_analysis_result(result)
        formatted_result += f"\n\n## Agent Discussion\n\n{conversation_result}\n"
        
        logger.info("Successfully formatted analysis with agent conversation")
        return formatted_result
        
    except Exception as e:
        logger.error("Failed to run agent conversation, falling back to basic format", 
                    extra={"error": str(e), "error_type": type(e).__name__})
        return format_analysis_result(result)


def _extract_code_context(result: PRAnalysisResult) -> str:
    """Extract relevant code context from analysis result for agent review."""
    context_parts = []
    
    if result.security.output.strip():
        context_parts.append(f"Security findings:\n{result.security.output}")
    
    if result.style.output.strip():
        context_parts.append(f"Style findings:\n{result.style.output}")
        
    if result.performance.output.strip():
        context_parts.append(f"Performance findings:\n{result.performance.output}")
    
    if not context_parts:
        context_parts.append("No issues detected in analysis.")
    
    return "\n\n".join(context_parts)


def format_analysis_result(result: PRAnalysisResult) -> str:
    """Format analysis result into readable text."""
    sections = []
    
    if result.security.output.strip():
        sections.append(f"**Security ({result.security.tool})**:\n{result.security.output}")
    
    if result.style.output.strip():
        sections.append(f"**Style ({result.style.tool})**:\n{result.style.output}")
        
    if result.performance.output.strip():
        sections.append(f"**Performance ({result.performance.tool})**:\n{result.performance.output}")
    
    if not sections:
        return "✅ No issues found in the analysis."
    
    return "\n\n".join(sections)

def _create_error_result(error_msg: str) -> PRAnalysisResult:
    """Create a PRAnalysisResult with error messages."""
    error_section = AnalysisSection(tool="error", output=error_msg)
    return PRAnalysisResult(
        security=error_section,
        style=error_section,
        performance=error_section,
    )
