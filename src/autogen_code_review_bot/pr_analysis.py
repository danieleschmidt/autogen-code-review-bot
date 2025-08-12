#!/usr/bin/env python3
"""
Core PR Analysis Engine for AutoGen Code Review Bot.

This module provides the main analysis functionality for pull requests,
integrating multiple linting tools, security scanners, and AI agents.
"""

import os
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from .agents import load_agents_from_yaml, run_dual_review
from .enhanced_agents import run_enhanced_dual_review
from .robust_analysis_helpers import run_security_analysis, run_style_analysis, run_performance_analysis, is_ignored_path
from .robust_error_handling import robust_operation, ErrorSeverity, validate_file_path, health_checker
from .exceptions import AnalysisError
from .language_detection import detect_language
from .linter_config import LinterConfig
from .logging_config import get_logger
from .metrics import get_metrics_registry
from .models import AnalysisSection, PRAnalysisResult

logger = get_logger(__name__)
metrics = get_metrics_registry()



@robust_operation(
    component="pr_analysis",
    operation="full_analysis",
    severity=ErrorSeverity.HIGH,
    retry_count=1,
    raise_on_failure=True
)
def analyze_pr(repo_path: str,
               config_path: Optional[str] = None,
               use_cache: bool = True,
               use_parallel: bool = True) -> PRAnalysisResult:
    """
    Comprehensive PR analysis with security, style, and performance checks.
    
    Args:
        repo_path: Path to repository to analyze
        config_path: Optional path to linter configuration
        use_cache: Whether to use caching for linter results
        use_parallel: Whether to run analysis in parallel
    
    Returns:
        PRAnalysisResult with analysis sections
    """
    start_time = datetime.now(timezone.utc)
    
    # Validate inputs
    validate_file_path(repo_path)

    try:
        logger.info("Starting PR analysis", extra={
            "repo_path": repo_path,
            "config_path": config_path,
            "use_cache": use_cache,
            "use_parallel": use_parallel
        })

        # Real implementation for Generation 1
        repo_path_obj = Path(repo_path)
        if not repo_path_obj.exists():
            raise AnalysisError(f"Repository path does not exist: {repo_path}")

        # Detect languages in the repository
        all_files = list(repo_path_obj.rglob('*'))
        code_files = [str(f) for f in all_files if f.is_file() and not is_ignored_path(f)]
        detected_languages = detect_language(code_files)
        
        logger.info(f"Detected languages: {detected_languages}")
        
        # Load linter configuration
        linter_config = load_linter_config(config_path)
        
        if use_parallel and len(detected_languages) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=3) as executor:
                security_future = executor.submit(run_security_analysis, repo_path, detected_languages)
                style_future = executor.submit(run_style_analysis, repo_path, detected_languages, linter_config)
                performance_future = executor.submit(run_performance_analysis, repo_path, detected_languages)
                
                security_result = security_future.result()
                style_result = style_future.result()
                performance_result = performance_future.result()
        else:
            # Sequential execution
            security_result = run_security_analysis(repo_path, detected_languages)
            style_result = run_style_analysis(repo_path, detected_languages, linter_config)
            performance_result = run_performance_analysis(repo_path, detected_languages)

        # Create analysis result
        result = PRAnalysisResult(
            security=security_result,
            style=style_result,
            performance=performance_result,
            metadata={
                "analysis_timestamp": start_time.isoformat(),
                "repo_path": repo_path,
                "analysis_duration": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "cache_used": use_cache,
                "parallel_execution": use_parallel
            }
        )

        logger.info("PR analysis completed successfully", extra={
            "duration_seconds": result.metadata["analysis_duration"]
        })

        return result

    except Exception as e:
        logger.error(f"PR analysis failed: {e}")
        raise AnalysisError(f"PR analysis failed: {e}")


def load_linter_config(config_path: Optional[str] = None) -> LinterConfig:
    """Load linter configuration from YAML file or use defaults."""
    if config_path and Path(config_path).exists():
        try:
            return LinterConfig.from_yaml(config_path)
        except Exception as e:
            logger.warning(f"Failed to load linter config {config_path}: {e}")

    return LinterConfig()


def format_analysis_with_agents(result: PRAnalysisResult, agent_config_path: str) -> str:
    """Format analysis results using enhanced AI agent conversation."""
    try:
        # Prepare analysis summary for agents
        analysis_summary = f"""
=== SECURITY ANALYSIS ===
Tool: {result.security.tool}
Severity: {result.security.metadata.get('severity', 'unknown')}
Output: {result.security.output}

=== STYLE ANALYSIS ===  
Tool: {result.style.tool}
Issues Count: {result.style.metadata.get('issues_count', 0)}
Languages Analyzed: {result.style.metadata.get('languages_analyzed', 0)}
Output: {result.style.output}

=== PERFORMANCE ANALYSIS ===
Tool: {result.performance.tool}
Hotspots: {result.performance.metadata.get('hotspots', 0)}
Metrics Collected: {result.performance.metadata.get('metrics_collected', 0)}
Output: {result.performance.output}
"""

        # Run enhanced dual agent review
        agent_feedback = run_enhanced_dual_review(analysis_summary, agent_config_path)

        # Format final output
        return f"""
# 🤖 AutoGen Code Review Results

## Analysis Summary
- **Analysis Duration**: {result.metadata.get('analysis_duration', 'N/A')} seconds
- **Security Severity**: {result.security.metadata.get('severity', 'unknown')}

## AI Agent Discussion
{agent_feedback}

## Raw Analysis Results

### 🔒 Security Analysis ({result.security.tool})
{result.security.output}

### 🎨 Style Analysis ({result.style.tool})
{result.style.output}

### ⚡ Performance Analysis ({result.performance.tool})
{result.performance.output}
"""

    except Exception as e:
        logger.error(f"Agent formatting failed: {e}")
        # Fallback to standard formatting
        return f"""
# 🤖 AutoGen Code Review Results

⚠️ Agent conversation failed: {e}

## Raw Analysis Results

### 🔒 Security Analysis ({result.security.tool})
{result.security.output}

### 🎨 Style Analysis ({result.style.tool})
{result.style.output}

### ⚡ Performance Analysis ({result.performance.tool})
{result.performance.output}
"""
