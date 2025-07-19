from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TYPE_CHECKING

import requests

from .logging_utils import get_request_logger, RequestContext
from .config import get_github_api_url, get_http_timeout
from .metrics import record_operation_metrics, with_metrics
from .circuit_breaker import (
    get_circuit_breaker, 
    CircuitBreakerConfig, 
    CircuitBreakerError,
    RetryStrategy
)

logger = get_request_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .pr_analysis import PRAnalysisResult


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
        return requests.request(
            method,
            url,
            headers=_headers(token_val),
            data=data,
            params=params,
            timeout=get_http_timeout(),
        )
    
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
            resp.raise_for_status()
            
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
    """Post ``body`` as a comment on the pull request."""
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
    
    return result


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
    """Analyze the repo and post the results as a comment on the PR."""
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

    result = analyze_pr(repo_path, config_path, context=context)
    body = format_analysis_result(result)
    
    comment_result = post_comment(repo, pr_number, body, token, context=context)
    
    logger.info(
        "Completed analyze and comment workflow",
        context=context,
        repo=repo,
        pr_number=pr_number,
        comment_id=comment_result.get("id")
    )
    
    return comment_result
