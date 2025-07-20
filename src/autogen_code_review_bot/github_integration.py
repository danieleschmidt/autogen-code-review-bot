from __future__ import annotations

import json
import os
import time
from typing import Any, TYPE_CHECKING

import requests

from .logging_config import get_logger, log_operation_start, log_operation_end, ContextLogger

API_URL = "https://api.github.com"
logger = get_logger(__name__)

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
    retries: int = 3,
) -> requests.Response:
    """Return a ``requests`` response using ``method`` with simple retry logic."""
    token_val = _get_token(token)
    
    # Start operation tracking for API request
    request_context = log_operation_start(
        logger,
        "github_api_request",
        method=method.upper(),
        url=url.replace(token_val, "***") if token_val in url else url,
        retries=retries
    )
    
    for attempt in range(retries):
        try:
            logger.debug("Making GitHub API request", 
                        method=method.upper(),
                        attempt=attempt + 1,
                        max_retries=retries)
            
            resp = requests.request(
                method,
                url,
                headers=_headers(token_val),
                data=data,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            
            logger.debug("GitHub API request successful",
                        status_code=resp.status_code,
                        response_size=len(resp.content))
            
            log_operation_end(logger, request_context, success=True, 
                            status_code=resp.status_code,
                            attempt=attempt + 1)
            return resp
            
        except requests.RequestException as exc:
            logger.warning("GitHub API request failed",
                         error=str(exc),
                         attempt=attempt + 1,
                         max_retries=retries,
                         will_retry=attempt < retries - 1)
            
            if attempt == retries - 1:
                log_operation_end(logger, request_context, success=False, 
                                error=str(exc), total_attempts=retries)
                raise
            
            sleep_time = 2 ** attempt * 0.5
            logger.debug("Retrying GitHub API request", 
                        sleep_seconds=sleep_time,
                        next_attempt=attempt + 2)
            time.sleep(sleep_time)


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def get_pull_request_diff(repo: str, pr_number: int, token: str | None = None) -> str:
    """Return the diff for ``pr_number`` in ``repo``."""
    logger.info("Fetching pull request diff", 
               repository=repo, 
               pr_number=pr_number)
    
    url = f"{API_URL}/repos/{repo}/pulls/{pr_number}"
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
    """Post ``body`` as a comment on the pull request."""
    logger.info("Posting comment to pull request",
               repository=repo,
               pr_number=pr_number,
               comment_length=len(body))

    url = f"{API_URL}/repos/{repo}/issues/{pr_number}/comments"
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
) -> Any:
    """Analyze the repo and post the results as a comment on the PR."""
    
    # Start full analysis and comment operation
    operation_context = log_operation_start(
        logger,
        "analyze_and_comment",
        repository=repo,
        pr_number=pr_number,
        repo_path=repo_path,
        config_path=config_path
    )
    
    try:
        from .pr_analysis import analyze_pr

        logger.info("Starting PR analysis and comment workflow",
                   repository=repo,
                   pr_number=pr_number)
        
        result = analyze_pr(repo_path, config_path)
        
        logger.info("Analysis completed, formatting results",
                   security_tool=result.security.tool,
                   style_tool=result.style.tool,
                   performance_tool=result.performance.tool)
        
        body = format_analysis_result(result)
        comment_result = post_comment(repo, pr_number, body, token)
        
        logger.info("Analysis and comment workflow completed successfully",
                   repository=repo,
                   pr_number=pr_number,
                   comment_id=comment_result.get('id'))
        
        log_operation_end(logger, operation_context, success=True,
                         comment_id=comment_result.get('id'))
        
        return comment_result
        
    except Exception as e:
        logger.error("Analysis and comment workflow failed",
                    repository=repo,
                    pr_number=pr_number,
                    error=str(e),
                    error_type=type(e).__name__)
        log_operation_end(logger, operation_context, success=False, error=str(e))
        raise
