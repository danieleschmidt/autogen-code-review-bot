from __future__ import annotations

import json
import os
import time
from typing import Any, TYPE_CHECKING, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import random

from .logging_config import get_logger, log_operation_start, log_operation_end, ContextLogger
from .monitoring import MetricsEmitter
from .system_config import get_system_config
from .token_security import TokenMasker, mask_token_in_url, safe_exception_str
logger = get_logger(__name__)
metrics = MetricsEmitter()


def _calculate_exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True) -> float:
    """Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter to avoid thundering herd
        
    Returns:
        Delay time in seconds
    """
    # Calculate exponential delay: base_delay * (2^attempt)
    delay = base_delay * (2 ** attempt)
    
    # Cap at maximum delay
    delay = min(delay, max_delay)
    
    # Add jitter to avoid thundering herd problems
    if jitter:
        # Add random jitter of ±25% of the delay
        jitter_amount = delay * 0.25
        delay += random.uniform(-jitter_amount, jitter_amount)
        delay = max(0.1, delay)  # Ensure minimum delay
    
    return delay


# Error classes for better error handling
class GitHubError(Exception):
    """Base class for GitHub API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_headers: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_headers = response_headers or {}
        self.timestamp = datetime.now(timezone.utc)


class RateLimitError(GitHubError):
    """Raised when GitHub API rate limit is exceeded."""
    
    def __init__(self, message: str, reset_time: Optional[int] = None, remaining: Optional[int] = None):
        super().__init__(message, status_code=429)
        self.reset_time = reset_time
        self.remaining = remaining
        
        if reset_time:
            reset_dt = datetime.fromtimestamp(reset_time, timezone.utc)
            wait_minutes = max(0, (reset_dt - self.timestamp).total_seconds() / 60)
            self.message = f"{message} Rate limit resets at {reset_dt.strftime('%H:%M UTC')} (in {wait_minutes:.1f} minutes)"
        else:
            self.message = message


class GitHubConnectionError(GitHubError):
    """Raised when connection to GitHub fails."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"GitHub connection failed: {message}. Please check network connectivity and try again.")
        self.original_error = original_error


class CircuitBreakerError(GitHubError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str = "GitHub API circuit breaker is open due to repeated failures"):
        super().__init__(message)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state management."""
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_seconds: int = 300  # 5 minutes
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit breaker state."""
        current_time = time.time()
        
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and (current_time - self.last_failure_time) > self.timeout_seconds:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        
        return True
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"


# Global circuit breaker instance
_circuit_breaker = CircuitBreakerState()

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .pr_analysis import PRAnalysisResult


def _get_token(token: str | None) -> str:
    """Return ``token`` or the value from ``GITHUB_TOKEN`` environment variable."""

    env_token = os.getenv("GITHUB_TOKEN")
    final = token or env_token
    if not final:
        raise ValueError("GitHub token not provided")
    return final


def _request_with_retries(
    method: str,
    url: str,
    *,
    token: str | None,
    data: Any | None = None,
    params: dict[str, Any] | None = None,
    retries: Optional[int] = None,
) -> requests.Response:
    """Return a ``requests`` response with enhanced error handling and retry logic."""
    global _circuit_breaker
    
    config = get_system_config()
    if retries is None:
        retries = config.default_retry_attempts
    
    # Check circuit breaker
    if not _circuit_breaker.should_allow_request():
        metrics.record_counter("github_api_errors_total", 1, 
                             tags={"error_type": "circuit_breaker_open", "api_operation": method.upper()})
        raise CircuitBreakerError()
    
    token_val = _get_token(token)
    
    # Start operation tracking for API request
    request_context = log_operation_start(
        logger,
        "github_api_request",
        method=method.upper(),
        url=mask_token_in_url(url, token_val),
        retries=retries
    )
    
    last_error = None
    
    for attempt in range(retries):
        try:
            logger.debug("Making GitHub API request", 
                        method=method.upper(),
                        attempt=attempt + 1,
                        max_retries=retries,
                        circuit_breaker_state=_circuit_breaker.state)
            
            resp = requests.request(
                method,
                url,
                headers=_headers(token_val),
                data=data,
                params=params,
                timeout=config.github_api_timeout,
            )
            
            # Check for specific HTTP status codes before raising
            if resp.status_code == 429:
                # Rate limit handling
                reset_time = resp.headers.get('X-RateLimit-Reset')
                remaining = resp.headers.get('X-RateLimit-Remaining', '0')
                
                reset_timestamp = int(reset_time) if reset_time else None
                metrics.record_counter("github_api_errors_total", 1, 
                                     tags={"error_type": "rate_limit", "api_operation": method.upper()})
                
                if attempt < retries - 1:
                    # Calculate sleep time based on reset time or sophisticated exponential backoff
                    if reset_timestamp:
                        # For rate limits, respect the reset time but cap at reasonable values
                        time_until_reset = reset_timestamp - int(time.time())
                        sleep_time = min(max(time_until_reset, 1), 60)  # 1s min, 60s max
                    else:
                        # Use sophisticated exponential backoff for rate limits
                        sleep_time = _calculate_exponential_backoff(attempt, base_delay=2.0, max_delay=60.0)
                    
                    logger.warning("Rate limit exceeded, waiting", 
                                 sleep_seconds=sleep_time,
                                 reset_time=reset_timestamp,
                                 remaining=remaining)
                    time.sleep(sleep_time)
                    continue
                else:
                    raise RateLimitError(
                        "GitHub API rate limit exceeded", 
                        reset_time=reset_timestamp,
                        remaining=int(remaining)
                    )
            
            elif 400 <= resp.status_code < 500:
                # Client errors (don't retry except for rate limits)
                error_msg = f"GitHub API client error: {resp.status_code}"
                if resp.status_code == 401:
                    error_msg = "GitHub API authentication failed. Please check your token."
                elif resp.status_code == 403:
                    error_msg = "GitHub API access forbidden. Please check your token permissions."
                elif resp.status_code == 404:
                    error_msg = "GitHub API resource not found. Please check the repository and PR number."
                
                metrics.record_counter("github_api_errors_total", 1, 
                                     tags={"error_type": f"client_error_{resp.status_code}", "api_operation": method.upper()})
                raise GitHubError(error_msg, status_code=resp.status_code, response_headers=dict(resp.headers))
            
            elif resp.status_code >= 500:
                # Server errors (retry with exponential backoff)
                error_msg = f"GitHub API server error: {resp.status_code}"
                if attempt < retries - 1:
                    # Use sophisticated exponential backoff for server errors
                    sleep_time = _calculate_exponential_backoff(attempt, base_delay=1.0, max_delay=30.0)
                    logger.warning("GitHub API server error, retrying", 
                                 status_code=resp.status_code,
                                 sleep_seconds=sleep_time,
                                 attempt=attempt + 1)
                    metrics.record_counter("github_api_retries_total", 1, 
                                         tags={"error_type": f"server_error_{resp.status_code}", "api_operation": method.upper()})
                    time.sleep(sleep_time)
                    continue
                else:
                    metrics.record_counter("github_api_errors_total", 1, 
                                         tags={"error_type": f"server_error_{resp.status_code}", "api_operation": method.upper()})
                    raise GitHubError(error_msg, status_code=resp.status_code, response_headers=dict(resp.headers))
            
            # If we get here, request was successful
            resp.raise_for_status()  # This should not raise for 2xx codes
            
            logger.debug("GitHub API request successful",
                        status_code=resp.status_code,
                        response_size=len(resp.content),
                        attempt=attempt + 1)
            
            # Record success in circuit breaker and metrics
            _circuit_breaker.record_success()
            metrics.record_counter("github_api_requests_total", 1, 
                                 tags={"status": "success", "api_operation": method.upper()})
            metrics.record_histogram("github_api_duration_seconds", time.time() - request_context.get('start_time', time.time()))
            
            log_operation_end(logger, request_context, success=True, 
                            status_code=resp.status_code,
                            attempt=attempt + 1)
            return resp
            
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Network-level errors
            last_error = GitHubConnectionError(safe_exception_str(exc), original_error=exc)
            metrics.record_counter("github_api_errors_total", 1, 
                                 tags={"error_type": "connection_error", "api_operation": method.upper()})
            
            if attempt < retries - 1:
                # Use sophisticated exponential backoff for connection errors
                sleep_time = _calculate_exponential_backoff(attempt, base_delay=0.5, max_delay=15.0)
                logger.warning("GitHub API connection failed, retrying",
                             error=safe_exception_str(exc),
                             attempt=attempt + 1,
                             sleep_seconds=sleep_time)
                metrics.record_counter("github_api_retries_total", 1, 
                                     tags={"error_type": "connection_error", "api_operation": method.upper()})
                time.sleep(sleep_time)
                continue
            
        except requests.RequestException as exc:
            # Other request exceptions
            last_error = GitHubError(f"GitHub API request failed: {safe_exception_str(exc)}")
            metrics.record_counter("github_api_errors_total", 1, 
                                 tags={"error_type": "request_error", "api_operation": method.upper()})
            
            if attempt < retries - 1:
                # Use sophisticated exponential backoff for general request errors
                sleep_time = _calculate_exponential_backoff(attempt, base_delay=0.5, max_delay=15.0)
                logger.warning("GitHub API request failed, retrying",
                             error=safe_exception_str(exc),
                             attempt=attempt + 1,
                             sleep_seconds=sleep_time)
                metrics.record_counter("github_api_retries_total", 1, 
                                     tags={"error_type": "request_error", "api_operation": method.upper()})
                time.sleep(sleep_time)
                continue
    
    # If we've exhausted all retries, record circuit breaker failure and raise
    _circuit_breaker.record_failure()
    log_operation_end(logger, request_context, success=False, 
                    error=safe_exception_str(last_error) if last_error else "Unknown error", total_attempts=retries)
    
    if last_error:
        raise last_error
    else:
        raise GitHubError("GitHub API request failed after all retry attempts")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def get_pull_request_diff(repo: str, pr_number: int, token: str | None = None) -> str:
    """Return the diff for ``pr_number`` in ``repo``."""
    logger.info("Fetching pull request diff", 
               repository=repo, 
               pr_number=pr_number)
    
    config = get_system_config()
    url = f"{config.github_api_url}/repos/{repo}/pulls/{pr_number}"
    resp = _request_with_retries(
        "get",
        url,
        token=token,
        params={"media_type": "diff"},
    )
    
    logger.debug("Pull request diff retrieved", 
                repository=repo,
                pr_number=pr_number,
                diff_size=len(resp.text))
    
    return resp.text


def post_comment(
    repo: str, pr_number: int, body: str, token: str | None = None
) -> Any:
    """Post ``body`` as a comment on the pull request with fallback for large comments."""
    logger.info("Posting comment to pull request",
               repository=repo,
               pr_number=pr_number,
               comment_length=len(body))

    config = get_system_config()
    url = f"{config.github_api_url}/repos/{repo}/issues/{pr_number}/comments"
    
    try:
        resp = _request_with_retries(
            "post",
            url,
            token=token,
            data=json.dumps({"body": body}),
        )
        
        result = resp.json()
        logger.info("Comment posted successfully",
                   repository=repo,
                   pr_number=pr_number,
                   comment_id=result.get('id'))
        
        metrics.record_counter("github_comments_posted_total", 1, tags={"status": "success", "fallback": "false"})
        return result
        
    except GitHubError as e:
        # Check if error is due to comment size and try fallback
        if "too large" in str(e).lower() or len(body) > 65000:  # GitHub comment limit
            logger.warning("Comment too large, trying fallback summary",
                         repository=repo,
                         pr_number=pr_number,
                         original_length=len(body))
            
            # Create a summarized version
            summary_body = _create_comment_summary(body)
            
            try:
                resp = _request_with_retries(
                    "post",
                    url,
                    token=token,
                    data=json.dumps({"body": summary_body}),
                )
                
                result = resp.json()
                logger.info("Fallback summary comment posted successfully",
                           repository=repo,
                           pr_number=pr_number,
                           comment_id=result.get('id'),
                           summary_length=len(summary_body))
                
                metrics.record_counter("github_comments_posted_total", 1, tags={"status": "success", "fallback": "true"})
                return result
                
            except GitHubError as fallback_error:
                logger.error("Failed to post even summary comment",
                           repository=repo,
                           pr_number=pr_number,
                           error=str(fallback_error))
                metrics.record_counter("github_comments_posted_total", 1, tags={"status": "error", "fallback": "failed"})
                raise
        else:
            metrics.record_counter("github_comments_posted_total", 1, tags={"status": "error", "fallback": "false"})
            raise


def _create_comment_summary(full_body: str) -> str:
    """Create a summarized version of a comment that's too large."""
    lines = full_body.split('\n')
    
    # Keep header and first few lines of each section
    summary_lines = []
    current_section = ""
    section_line_count = 0
    max_lines_per_section = 10
    
    for line in lines:
        if line.startswith('##') or line.startswith('###'):
            # New section header
            current_section = line
            summary_lines.append(line)
            section_line_count = 0
        elif section_line_count < max_lines_per_section:
            summary_lines.append(line)
            section_line_count += 1
        elif section_line_count == max_lines_per_section:
            summary_lines.append("... (content truncated due to size limits)")
            section_line_count += 1
    
    summary = '\n'.join(summary_lines)
    
    # Add footer explaining truncation
    summary += "\n\n---\n⚠️ **Note**: This comment was automatically truncated due to size limits. " \
               "Run the analysis locally for complete details."
    
    return summary


def format_analysis_result(result: PRAnalysisResult) -> str:
    """Return ``result`` formatted for a GitHub comment."""

    return (
        "## 🤖 AutoGen Code Review\n"
        f"### Security ({result.security.tool})\n{result.security.output}\n\n"
        f"### Style ({result.style.tool})\n{result.style.output}\n\n"
        f"### Performance ({result.performance.tool})\n{result.performance.output}"
    )


def analyze_and_comment(
    repo_path: str,
    repo: str,
    pr_number: int,
    token: str | None = None,
    config_path: str | None = None,
) -> Any:
    """Analyze the repo and post the results as a comment on the PR with enhanced error handling."""
    
    # Start full analysis and comment operation
    operation_context = log_operation_start(
        logger,
        "analyze_and_comment",
        repository=repo,
        pr_number=pr_number,
        repo_path=repo_path,
        config_path=config_path
    )
    
    analysis_result = None
    comment_result = None
    
    try:
        from .pr_analysis import analyze_pr

        logger.info("Starting PR analysis and comment workflow",
                   repository=repo,
                   pr_number=pr_number)
        
        # Attempt to fetch PR diff for context (optional, non-blocking)
        try:
            diff_content = get_pull_request_diff(repo, pr_number, token)
            logger.debug("PR diff fetched successfully", 
                        repository=repo,
                        pr_number=pr_number,
                        diff_size=len(diff_content))
        except (GitHubError, GitHubConnectionError) as diff_error:
            logger.warning("Failed to fetch PR diff, continuing with local analysis only",
                         repository=repo,
                         pr_number=pr_number,
                         error=str(diff_error))
            metrics.record_counter("github_diff_fetch_failures_total", 1, 
                                 tags={"error_type": type(diff_error).__name__})
        
        # Perform analysis (this should always be attempted)
        try:
            analysis_result = analyze_pr(repo_path, config_path)
            
            logger.info("Analysis completed successfully",
                       security_tool=analysis_result.security.tool,
                       style_tool=analysis_result.style.tool,
                       performance_tool=analysis_result.performance.tool)
            
        except Exception as analysis_error:
            logger.error("PR analysis failed",
                        repository=repo,
                        pr_number=pr_number,
                        error=str(analysis_error),
                        error_type=type(analysis_error).__name__)
            
            metrics.record_counter("pr_analysis_failures_total", 1, 
                                 tags={"error_type": type(analysis_error).__name__})
            
            # Create a fallback result to still post a comment
            from .models import AnalysisSection, PRAnalysisResult
            analysis_result = PRAnalysisResult(
                security=AnalysisSection(tool="error", output=f"Analysis failed: {str(analysis_error)}"),
                style=AnalysisSection(tool="error", output="Analysis not completed due to error"),
                performance=AnalysisSection(tool="error", output="Analysis not completed due to error")
            )
        
        # Attempt to post comment with multiple fallback strategies
        if analysis_result:
            try:
                body = format_analysis_result(analysis_result)
                comment_result = post_comment(repo, pr_number, body, token)
                
                logger.info("Analysis and comment workflow completed successfully",
                           repository=repo,
                           pr_number=pr_number,
                           comment_id=comment_result.get('id'))
                
            except RateLimitError as rate_error:
                logger.error("Rate limit exceeded when posting comment",
                           repository=repo,
                           pr_number=pr_number,
                           reset_time=getattr(rate_error, 'reset_time', None))
                
                # For rate limits, we could implement a delayed retry queue
                # For now, we'll just log and raise
                metrics.record_counter("github_comment_failures_total", 1, 
                                     tags={"error_type": "rate_limit"})
                raise
                
            except GitHubConnectionError as conn_error:
                logger.error("Connection error when posting comment",
                           repository=repo,
                           pr_number=pr_number,
                           error=str(conn_error))
                
                metrics.record_counter("github_comment_failures_total", 1, 
                                     tags={"error_type": "connection"})
                
                # For connection errors, we could implement local result storage
                # For now, we'll log and raise
                raise
                
            except GitHubError as github_error:
                logger.error("GitHub API error when posting comment",
                           repository=repo,
                           pr_number=pr_number,
                           error=str(github_error),
                           status_code=getattr(github_error, 'status_code', None))
                
                metrics.record_counter("github_comment_failures_total", 1, 
                                     tags={"error_type": "github_api"})
                
                # Try a minimal fallback comment
                try:
                    minimal_body = f"## 🤖 AutoGen Code Review\\n\\n" \
                                  f"Analysis completed but failed to post full results.\\n" \
                                  f"Error: {str(github_error)[:200]}..."
                    
                    comment_result = post_comment(repo, pr_number, minimal_body, token)
                    logger.info("Posted minimal fallback comment",
                               repository=repo,
                               pr_number=pr_number,
                               comment_id=comment_result.get('id'))
                    
                except Exception as minimal_error:
                    logger.error("Even minimal comment posting failed",
                               repository=repo,
                               pr_number=pr_number,
                               error=str(minimal_error))
                    raise github_error  # Raise original error
        
        # Record success metrics
        if comment_result:
            metrics.record_counter("analyze_and_comment_completed_total", 1, tags={"status": "success"})
            log_operation_end(logger, operation_context, success=True,
                             comment_id=comment_result.get('id'))
        else:
            metrics.record_counter("analyze_and_comment_completed_total", 1, tags={"status": "partial"})
            log_operation_end(logger, operation_context, success=False, error="No comment posted")
        
        return comment_result or {"status": "analysis_completed_no_comment"}
        
    except CircuitBreakerError as cb_error:
        logger.error("Circuit breaker open, GitHub API unavailable",
                    repository=repo,
                    pr_number=pr_number)
        
        metrics.record_counter("analyze_and_comment_completed_total", 1, tags={"status": "circuit_breaker"})
        log_operation_end(logger, operation_context, success=False, error="Circuit breaker open")
        
        # Could implement local storage of results for later retry
        raise
        
    except Exception as e:
        logger.error("Analysis and comment workflow failed with unexpected error",
                    repository=repo,
                    pr_number=pr_number,
                    error=str(e),
                    error_type=type(e).__name__)
        
        metrics.record_counter("analyze_and_comment_completed_total", 1, tags={"status": "error"})
        log_operation_end(logger, operation_context, success=False, error=str(e))
        raise
