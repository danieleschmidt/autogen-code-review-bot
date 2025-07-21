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
from .agent_conversations import conduct_agent_discussion, RefinementCriteria

# Default mapping of languages to linter executables
DEFAULT_LINTERS: Dict[str, str] = {
    "python": "ruff",
    "javascript": "eslint",
    "typescript": "eslint",
    "ruby": "rubocop",
}

# Initialize logger for PR analysis operations
logger = get_logger(__name__)


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
    file_count = 0
    for root, _, files in os.walk(repo_path):
        for name in files:
            file_path = Path(root) / name
            lang = detect_language(file_path)
            if lang != "unknown":
                languages.add(lang)
            file_count += 1
    
    logger.debug("Language detection completed", 
                extra={
                    "languages": list(languages), 
                    "files_scanned": file_count,
                    "repo_path": str(repo_path)
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
        return _create_error_result(f"streaming analysis error: {exc}")


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
    logger.debug("Starting parallel analysis checks", extra={"repo_path": repo_path, "linters": linters})
    
    def ensure(tool: str) -> str:
        # Validate tool is in allowlist before checking installation
        if Path(tool).name not in ALLOWED_EXECUTABLES:
            return f"tool '{tool}' not allowed"
        return "not installed" if which(tool) is None else ""
    
    # Define check functions
    def run_security():
        logger.debug("Starting security analysis")
        result = _run_security_checks(repo_path, ensure)
        logger.debug("Security analysis completed")
        return result
    
    def run_style():
        logger.debug("Starting style analysis")
        result = _run_style_checks_parallel(repo_path, linters, max_workers=3)
        logger.debug("Style analysis completed")
        return result
    
    def run_performance():
        logger.debug("Starting performance analysis")
        result = _run_performance_checks(repo_path, ensure)
        logger.debug("Performance analysis completed")
        return result
    
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
        except Exception as exc:
            # If any check fails, return error result
            logger.error("Parallel execution failed", extra={"error": str(exc)})
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
        return _create_error_result("invalid repository path")
    
    if not _validate_path_safety(repo_path):
        logger.error("Unsafe repository path rejected", extra={"repo_path": repo_path})
        return _create_error_result("unsafe repository path rejected")
    
    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists() or not repo_path_obj.is_dir():
        logger.error("Repository path validation failed", 
                    extra={
                        "repo_path": repo_path, 
                        "exists": repo_path_obj.exists(), 
                        "is_dir": repo_path_obj.is_dir()
                    })
        return _create_error_result("repository path does not exist or is not a directory")

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
        return _analyze_pr_streaming(repo_path, linters, use_cache)

    # Run analysis - use parallel execution if enabled
    if use_parallel:
        logger.info("Running analysis with parallel execution")
        result = _run_all_checks_parallel(repo_path, linters)
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
    
    logger.info("Analysis completed successfully",
                extra={
                    "has_security_issues": bool(result.security.output.strip()),
                    "has_style_issues": bool(result.style.output.strip()),
                    "has_performance_issues": bool(result.performance.output.strip())
                })
    
    # Cache the result if caching is enabled and we have a commit hash
    if use_cache and cache and commit_hash and config_hash:
        logger.debug("Caching analysis results", extra={"commit_hash": commit_hash[:8]})
        cache.set(commit_hash, config_hash, result)
    
    return result


def analyze_pr_with_conversations(
    repo_path: str, 
    config_path: str | None = None, 
    use_cache: bool = True, 
    use_parallel: bool = True,
    enable_conversations: bool = False,
    conversation_rounds: int = 2
) -> PRAnalysisResult:
    """Analyze PR with optional agent conversations for enhanced review quality.
    
    This function extends the basic PR analysis with agent conversations,
    allowing multiple AI agents to discuss and refine their feedback.
    
    Parameters
    ----------
    repo_path:
        Path to the repository under analysis.
    config_path:
        Optional YAML file specifying linter and agent configurations.
    use_cache:
        Whether to use caching for performance optimization.
    use_parallel:
        Whether to use parallel execution.
    enable_conversations:
        Whether to enable agent conversations for refinement.
    conversation_rounds:
        Maximum rounds of agent discussion (default: 2).
    
    Returns
    -------
    PRAnalysisResult:
        Analysis results, potentially enhanced with agent discussions.
    """
    logger.info("Starting PR analysis with conversation capability", 
                extra={
                    "repo_path": repo_path,
                    "enable_conversations": enable_conversations,
                    "conversation_rounds": conversation_rounds
                })
    
    # Run standard PR analysis first
    result = analyze_pr(repo_path, config_path, use_cache, use_parallel)
    
    # If conversations are disabled or we have errors, return standard result
    if not enable_conversations or not config_path:
        logger.debug("Agent conversations disabled or no config provided")
        return result
    
    # Check if there are significant issues that warrant agent discussion
    has_issues = any([
        result.security.output.strip(),
        result.style.output.strip(), 
        result.performance.output.strip()
    ])
    
    if not has_issues:
        logger.info("No significant issues found, skipping agent conversations")
        return result
    
    # Prepare code snippet for discussion (use problematic files)
    code_snippet = _extract_problematic_code(repo_path, result)
    
    if not code_snippet:
        logger.debug("No specific code issues to discuss")
        return result
    
    try:
        # Conduct agent discussion
        logger.info("Starting agent conversation for code refinement",
                   extra={"code_length": len(code_snippet)})
        
        criteria = RefinementCriteria(
            max_rounds=conversation_rounds,
            consensus_threshold=0.7,
            timeout_seconds=120.0  # 2 minutes for practical usage
        )
        
        discussion_result = conduct_agent_discussion(
            code_snippet=code_snippet,
            config_path=config_path,
            max_rounds=conversation_rounds,
            criteria=criteria
        )
        
        if discussion_result.consensus_reached and discussion_result.final_messages:
            logger.info("Agent conversation completed with consensus",
                       extra={
                           "round_count": discussion_result.round_count,
                           "confidence": discussion_result.confidence_score
                       })
            
            # Enhance results with conversation insights
            enhanced_result = _integrate_conversation_results(result, discussion_result)
            return enhanced_result
        else:
            logger.info("Agent conversation completed without strong consensus",
                       extra={"round_count": discussion_result.round_count})
    
    except Exception as e:
        logger.warning("Agent conversation failed, falling back to standard analysis",
                      extra={"error": str(e)})
    
    return result


def _extract_problematic_code(repo_path: str, analysis_result: PRAnalysisResult) -> str:
    """Extract code snippets that have analysis issues for agent discussion."""
    problematic_files = []
    
    # Parse linter outputs to find specific files with issues
    outputs = [
        analysis_result.security.output,
        analysis_result.style.output,
        analysis_result.performance.output
    ]
    
    for output in outputs:
        if not output.strip():
            continue
            
        # Look for file paths in the output (basic heuristic)
        lines = output.split('\n')
        for line in lines:
            if ':' in line and ('/' in line or '\\' in line):
                # Extract potential file path
                parts = line.split(':')
                if parts:
                    potential_path = parts[0].strip()
                    full_path = Path(repo_path) / potential_path
                    if full_path.exists() and full_path.is_file():
                        problematic_files.append(str(full_path))
    
    if not problematic_files:
        # Fallback: use first Python file found
        repo_path_obj = Path(repo_path)
        python_files = list(repo_path_obj.glob('**/*.py'))
        if python_files:
            problematic_files = [str(python_files[0])]
    
    # Read and combine code from problematic files (limited size)
    code_snippets = []
    total_length = 0
    max_code_length = 2000  # Limit for practical agent discussion
    
    for file_path in problematic_files[:3]:  # Max 3 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if total_length + len(content) > max_code_length:
                    remaining = max_code_length - total_length
                    content = content[:remaining] + "\n... (truncated)"
                
                code_snippets.append(f"# File: {Path(file_path).name}\n{content}")
                total_length += len(content)
                
                if total_length >= max_code_length:
                    break
                    
        except (IOError, UnicodeDecodeError) as e:
            logger.debug(f"Could not read file {file_path}: {e}")
            continue
    
    return '\n\n'.join(code_snippets) if code_snippets else ""


def _integrate_conversation_results(
    analysis_result: PRAnalysisResult, 
    discussion_result
) -> PRAnalysisResult:
    """Integrate agent conversation results into the analysis result."""
    
    # Create enhanced sections with agent insights
    conversation_summary = _format_conversation_summary(discussion_result)
    
    # Add conversation insights to each section
    enhanced_security = AnalysisSection(
        tool=f"{analysis_result.security.tool} + agents",
        output=f"{analysis_result.security.output}\n\n--- Agent Discussion ---\n{conversation_summary}"
    )
    
    enhanced_style = AnalysisSection(
        tool=f"{analysis_result.style.tool} + agents", 
        output=f"{analysis_result.style.output}\n\n--- Agent Discussion ---\n{conversation_summary}"
    )
    
    enhanced_performance = AnalysisSection(
        tool=f"{analysis_result.performance.tool} + agents",
        output=f"{analysis_result.performance.output}\n\n--- Agent Discussion ---\n{conversation_summary}"
    )
    
    return PRAnalysisResult(
        security=enhanced_security,
        style=enhanced_style,
        performance=enhanced_performance
    )


def _format_conversation_summary(discussion_result) -> str:
    """Format the agent conversation results for display."""
    if not discussion_result.final_messages:
        return "No agent discussion messages available."
    
    summary_parts = [
        f"Agent Discussion Summary (Rounds: {discussion_result.round_count})",
        f"Consensus: {'Yes' if discussion_result.consensus_reached else 'No'}",
        f"Confidence: {discussion_result.confidence_score:.2f}",
        ""
    ]
    
    # Add key messages from the discussion
    for i, message in enumerate(discussion_result.final_messages[-4:], 1):  # Last 4 messages
        confidence_str = f" (confidence: {message.confidence:.2f})" if message.confidence else ""
        summary_parts.append(
            f"{i}. {message.agent_name}: {message.content[:150]}{'...' if len(message.content) > 150 else ''}{confidence_str}"
        )
    
    return '\n'.join(summary_parts)


def _create_error_result(error_msg: str) -> PRAnalysisResult:
    """Create a PRAnalysisResult with error messages."""
    error_section = AnalysisSection(tool="error", output=error_msg)
    return PRAnalysisResult(
        security=error_section,
        style=error_section,
        performance=error_section,
    )
