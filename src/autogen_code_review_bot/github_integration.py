from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import requests

from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    RetryStrategy,
    get_circuit_breaker,
)
from .config import get_github_api_url, get_http_timeout
from .logging_utils import RequestContext, get_request_logger
from .metrics import record_operation_metrics, with_metrics

logger = get_request_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .pr_analysis import PRAnalysisResult


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


def _get_token(token: str | None) -> str:
    """Return ``token`` or the value from ``GITHUB_TOKEN`` environment variable."""

    env_token = os.getenv("GITHUB_TOKEN")
    final = token or env_token
    if not final:
        raise ValueError("GitHub token not provided")
    return final


# Configure circuit breaker for GitHub API
_github_circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=60.0,    # Try recovery after 60 seconds
    success_threshold=3,      # Close after 3 successes
    request_timeout=30.0,     # 30 second request timeout
    max_retries=3,           # Maximum 3 retry attempts
    base_delay=0.5,          # Start with 0.5 second delay
    max_delay=30.0,          # Maximum 30 second delay
    jitter_factor=0.2,       # 20% jitter to prevent thundering herd
    monitoring_window=50     # Track last 50 requests
)

_github_circuit_breaker = get_circuit_breaker("github_api", _github_circuit_breaker_config)
_retry_strategy = RetryStrategy(_github_circuit_breaker_config)


@with_metrics(operation="github_api_request")
def _request_with_retries(
    method: str,
    url: str,
    *,
    token: str | None,
    data: Any | None = None,
    params: dict[str, Any] | None = None,
    retries: int = 3,
    context: RequestContext | None = None,
) -> requests.Response:
    """Return a ``requests`` response using enhanced retry logic with circuit breaker.
    
    Features:
    - Circuit breaker protection against cascading failures
    - Differentiated error handling (rate limits, client errors, server errors)
    - Exponential backoff with jitter
    - Respect for Retry-After headers
    - Enhanced metrics and logging
    """
    if context is None:
        context = RequestContext()

    start_time = time.time()
    token_val = _get_token(token)

    logger.info(
        "Starting enhanced GitHub API request",
        context=context,
        method=method,
        url=url,
        max_retries=retries,
        circuit_breaker_state=_github_circuit_breaker.state.value
    )

    def make_request() -> requests.Response:
        """Internal function to make the actual request."""
        resp = requests.request(
            method,
            url,
            headers=_headers(token_val),
            data=data,
            params=params,
            timeout=get_http_timeout(),
        )

        # Check for specific HTTP status codes before raising
        if resp.status_code == 429:
            # Rate limit handling
            reset_time = resp.headers.get('X-RateLimit-Reset')
            remaining = resp.headers.get('X-RateLimit-Remaining', '0')

            reset_timestamp = int(reset_time) if reset_time else None

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

            raise GitHubError(error_msg, status_code=resp.status_code, response_headers=dict(resp.headers))

        elif resp.status_code >= 500:
            # Server errors (should be retried)
            error_msg = f"GitHub API server error: {resp.status_code}"
            raise GitHubError(error_msg, status_code=resp.status_code, response_headers=dict(resp.headers))

        # If we get here, request was successful
        resp.raise_for_status()  # This should not raise for 2xx codes
        return resp

    last_exception = None

    for attempt in range(retries + 1):  # +1 for initial attempt
        try:
            logger.debug(
                f"GitHub API request attempt {attempt + 1}",
                context=context,
                attempt=attempt + 1,
                max_attempts=retries + 1,
                circuit_breaker_state=_github_circuit_breaker.state.value
            )

            # Use circuit breaker protection
            resp = _github_circuit_breaker.call(make_request, context=context)

            logger.info(
                "GitHub API request successful",
                context=context,
                status_code=resp.status_code,
                attempt=attempt + 1,
                duration_ms=(time.time() - start_time) * 1000
            )

            # Record success metrics
            record_operation_metrics(
                operation="github_api_success",
                duration_ms=(time.time() - start_time) * 1000,
                status="success",
                context=context
            )

            return resp

        except CircuitBreakerError as exc:
            logger.error(
                "GitHub API request blocked by circuit breaker",
                context=context,
                circuit_breaker_state=_github_circuit_breaker.state.value,
                error=str(exc)
            )
            record_operation_metrics(
                operation="github_api_circuit_breaker_blocked",
                duration_ms=(time.time() - start_time) * 1000,
                status="circuit_breaker_open",
                context=context
            )
            raise

        except RateLimitError as exc:
            last_exception = exc

            logger.warning(
                "Rate limit exceeded",
                context=context,
                reset_time=exc.reset_time,
                remaining=exc.remaining,
                attempt=attempt + 1,
                max_attempts=retries + 1
            )

            # If this is the last attempt, don't sleep
            if attempt == retries:
                break

            # Calculate retry delay respecting rate limit reset time
            retry_after = None
            if exc.reset_time:
                # For rate limits, respect the reset time but cap at reasonable values
                time_until_reset = exc.reset_time - int(time.time())
                retry_after = min(max(time_until_reset, 1), 60)  # 1s min, 60s max

            delay = _retry_strategy.calculate_delay(attempt, retry_after)

            logger.info(
                f"Retrying GitHub API request after {delay:.2f}s delay",
                context=context,
                delay_seconds=delay,
                rate_limit_reset=exc.reset_time,
                attempt=attempt + 1,
                next_attempt=attempt + 2
            )

            time.sleep(delay)

        except (requests.ConnectionError, requests.Timeout) as exc:
            # Network-level errors
            last_exception = GitHubConnectionError(str(exc), original_error=exc)

            logger.warning(
                "GitHub API connection failed",
                context=context,
                error=str(exc),
                error_type=type(exc).__name__,
                attempt=attempt + 1,
                max_attempts=retries + 1
            )

            # Check if we should retry this error type
            if not _retry_strategy.should_retry(attempt, exc):
                logger.info(
                    "Not retrying GitHub API request",
                    context=context,
                    error_type=type(exc).__name__,
                    reason="error_type_not_retryable"
                )
                break

            # If this is the last attempt, don't sleep
            if attempt == retries:
                break

            delay = _retry_strategy.calculate_delay(attempt, None)

            logger.info(
                f"Retrying GitHub API request after {delay:.2f}s delay",
                context=context,
                delay_seconds=delay,
                attempt=attempt + 1,
                next_attempt=attempt + 2
            )

            time.sleep(delay)

        except requests.RequestException as exc:
            last_exception = exc
            error_type = type(exc).__name__

            # Enhanced error classification
            status_code = getattr(exc.response, 'status_code', None) if hasattr(exc, 'response') else None

            logger.warning(
                "GitHub API request failed",
                context=context,
                error=str(exc),
                error_type=error_type,
                status_code=status_code,
                attempt=attempt + 1,
                max_attempts=retries + 1
            )

            # Check if we should retry this error type
            if not _retry_strategy.should_retry(attempt, exc):
                logger.info(
                    "Not retrying GitHub API request",
                    context=context,
                    error_type=error_type,
                    status_code=status_code,
                    reason="error_type_not_retryable"
                )
                break

            # If this is the last attempt, don't sleep
            if attempt == retries:
                break

            # Calculate retry delay
            retry_after = None
            if hasattr(exc, 'response') and exc.response:
                retry_after = _retry_strategy.extract_retry_after(exc.response)

            delay = _retry_strategy.calculate_delay(attempt, retry_after)

            logger.info(
                f"Retrying GitHub API request after {delay:.2f}s delay",
                context=context,
                delay_seconds=delay,
                retry_after_header=retry_after,
                attempt=attempt + 1,
                next_attempt=attempt + 2
            )

            time.sleep(delay)

    # If we get here, all retries failed
    logger.error(
        "GitHub API request failed after all retries",
        context=context,
        error=str(last_exception) if last_exception else "Unknown error",
        total_attempts=retries + 1,
        circuit_breaker_state=_github_circuit_breaker.state.value
    )

    record_operation_metrics(
        operation="github_api_failure",
        duration_ms=(time.time() - start_time) * 1000,
        status="max_retries_exceeded",
        context=context
    )

    if last_exception:
        raise last_exception
    else:
        raise requests.RequestException("All retry attempts failed")


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def get_pull_request_diff(
    repo: str,
    pr_number: int,
    token: str | None = None,
    context: RequestContext | None = None
) -> str:
    """Return the diff for ``pr_number`` in ``repo``."""
    if context is None:
        context = RequestContext()

    logger.info(
        "Fetching pull request diff",
        context=context,
        repo=repo,
        pr_number=pr_number
    )

    url = f"{get_github_api_url()}/repos/{repo}/pulls/{pr_number}"
    resp = _request_with_retries(
        "get",
        url,
        token=token,
        params={"media_type": "diff"},
        context=context,
    )

    logger.info(
        "Successfully fetched pull request diff",
        context=context,
        repo=repo,
        pr_number=pr_number,
        diff_size=len(resp.text)
    )

    return resp.text


def post_comment(
    repo: str,
    pr_number: int,
    body: str,
    token: str | None = None,
    context: RequestContext | None = None
) -> Any:
    """Post ``body`` as a comment on the pull request with fallback for large comments."""
    if context is None:
        context = RequestContext()

    logger.info(
        "Posting comment to pull request",
        context=context,
        repo=repo,
        pr_number=pr_number,
        comment_length=len(body)
    )

    url = f"{get_github_api_url()}/repos/{repo}/issues/{pr_number}/comments"

    try:
        resp = _request_with_retries(
            "post",
            url,
            token=token,
            data=json.dumps({"body": body}),
            context=context,
        )

        result = resp.json()

        logger.info(
            "Successfully posted comment to pull request",
            context=context,
            repo=repo,
            pr_number=pr_number,
            comment_id=result.get("id")
        )

        record_operation_metrics(
            operation="github_comment_posted",
            duration_ms=0,
            status="success",
            context=context
        )

        return result

    except GitHubError as e:
        # Check if error is due to comment size and try fallback
        if "too large" in str(e).lower() or len(body) > 65000:  # GitHub comment limit
            logger.warning(
                "Comment too large, trying fallback summary",
                context=context,
                repo=repo,
                pr_number=pr_number,
                original_length=len(body)
            )

            # Create a summarized version
            summary_body = _create_comment_summary(body)

            try:
                resp = _request_with_retries(
                    "post",
                    url,
                    token=token,
                    data=json.dumps({"body": summary_body}),
                    context=context,
                )

                result = resp.json()

                logger.info(
                    "Fallback summary comment posted successfully",
                    context=context,
                    repo=repo,
                    pr_number=pr_number,
                    comment_id=result.get("id"),
                    summary_length=len(summary_body)
                )

                record_operation_metrics(
                    operation="github_comment_posted",
                    duration_ms=0,
                    status="success_with_fallback",
                    context=context
                )

                return result

            except GitHubError as fallback_error:
                logger.error(
                    "Failed to post even summary comment",
                    context=context,
                    repo=repo,
                    pr_number=pr_number,
                    error=str(fallback_error)
                )

                record_operation_metrics(
                    operation="github_comment_posted",
                    duration_ms=0,
                    status="failure",
                    context=context
                )
                raise
        else:
            record_operation_metrics(
                operation="github_comment_posted",
                duration_ms=0,
                status="failure",
                context=context
            )
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
    context: RequestContext | None = None,
) -> Any:
    """Analyze the repo and post the results as a comment on the PR with enhanced error handling."""
    if context is None:
        context = RequestContext()

    logger.info(
        "Starting analyze and comment workflow",
        context=context,
        repo_path=repo_path,
        repo=repo,
        pr_number=pr_number
    )

    from .pr_analysis import analyze_pr

    analysis_result = None
    comment_result = None

    try:
        # Attempt to fetch PR diff for context (optional, non-blocking)
        try:
            diff_content = get_pull_request_diff(repo, pr_number, token, context)
            logger.debug(
                "PR diff fetched successfully",
                context=context,
                repo=repo,
                pr_number=pr_number,
                diff_size=len(diff_content)
            )
        except (GitHubError, GitHubConnectionError) as diff_error:
            logger.warning(
                "Failed to fetch PR diff, continuing with local analysis only",
                context=context,
                repo=repo,
                pr_number=pr_number,
                error=str(diff_error)
            )

            record_operation_metrics(
                operation="github_diff_fetch_failure",
                duration_ms=0,
                status="failure",
                context=context
            )

        # Perform analysis (this should always be attempted)
        try:
            analysis_result = analyze_pr(repo_path, config_path, context=context)

            logger.info(
                "Analysis completed successfully",
                context=context,
                security_tool=analysis_result.security.tool,
                style_tool=analysis_result.style.tool,
                performance_tool=analysis_result.performance.tool
            )

        except Exception as analysis_error:
            logger.error(
                "PR analysis failed",
                context=context,
                repo=repo,
                pr_number=pr_number,
                error=str(analysis_error),
                error_type=type(analysis_error).__name__
            )

            record_operation_metrics(
                operation="pr_analysis_failure",
                duration_ms=0,
                status="failure",
                context=context
            )

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
                comment_result = post_comment(repo, pr_number, body, token, context)

                logger.info(
                    "Analysis and comment workflow completed successfully",
                    context=context,
                    repo=repo,
                    pr_number=pr_number,
                    comment_id=comment_result.get("id")
                )

            except RateLimitError as rate_error:
                logger.error(
                    "Rate limit exceeded when posting comment",
                    context=context,
                    repo=repo,
                    pr_number=pr_number,
                    reset_time=getattr(rate_error, 'reset_time', None)
                )

                record_operation_metrics(
                    operation="github_comment_rate_limited",
                    duration_ms=0,
                    status="rate_limited",
                    context=context
                )
                raise

            except GitHubConnectionError as conn_error:
                logger.error(
                    "Connection error when posting comment",
                    context=context,
                    repo=repo,
                    pr_number=pr_number,
                    error=str(conn_error)
                )

                record_operation_metrics(
                    operation="github_comment_connection_error",
                    duration_ms=0,
                    status="connection_error",
                    context=context
                )
                raise

            except GitHubError as github_error:
                logger.error(
                    "GitHub API error when posting comment",
                    context=context,
                    repo=repo,
                    pr_number=pr_number,
                    error=str(github_error),
                    status_code=getattr(github_error, 'status_code', None)
                )

                record_operation_metrics(
                    operation="github_comment_api_error",
                    duration_ms=0,
                    status="api_error",
                    context=context
                )

                # Try a minimal fallback comment
                try:
                    minimal_body = f"## 🤖 AutoGen Code Review\n\n" \
                                  f"Analysis completed but failed to post full results.\n" \
                                  f"Error: {str(github_error)[:200]}..."

                    comment_result = post_comment(repo, pr_number, minimal_body, token, context)

                    logger.info(
                        "Posted minimal fallback comment",
                        context=context,
                        repo=repo,
                        pr_number=pr_number,
                        comment_id=comment_result.get("id")
                    )

                except Exception as minimal_error:
                    logger.error(
                        "Even minimal comment posting failed",
                        context=context,
                        repo=repo,
                        pr_number=pr_number,
                        error=str(minimal_error)
                    )
                    raise github_error  # Raise original error

        # Record success metrics
        if comment_result:
            record_operation_metrics(
                operation="analyze_and_comment_completed",
                duration_ms=0,
                status="success",
                context=context
            )
        else:
            record_operation_metrics(
                operation="analyze_and_comment_completed",
                duration_ms=0,
                status="partial",
                context=context
            )

        return comment_result or {"status": "analysis_completed_no_comment"}

    except CircuitBreakerError:
        logger.error(
            "Circuit breaker open, GitHub API unavailable",
            context=context,
            repo=repo,
            pr_number=pr_number
        )

        record_operation_metrics(
            operation="analyze_and_comment_circuit_breaker",
            duration_ms=0,
            status="circuit_breaker_open",
            context=context
        )
        raise

    except Exception as e:
        logger.error(
            "Analysis and comment workflow failed with unexpected error",
            context=context,
            repo=repo,
            pr_number=pr_number,
            error=str(e),
            error_type=type(e).__name__
        )

        record_operation_metrics(
            operation="analyze_and_comment_error",
            duration_ms=0,
            status="error",
            context=context
        )
        raise
