from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TYPE_CHECKING

import requests

from .logging_utils import get_request_logger, RequestContext
from .config import get_github_api_url, get_http_timeout

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
    """Return a ``requests`` response using ``method`` with simple retry logic."""
    if context is None:
        context = RequestContext()
        
    token_val = _get_token(token)
    
    logger.info(
        "Starting GitHub API request",
        context=context,
        method=method,
        url=url,
        retries=retries
    )
    
    for attempt in range(retries):
        try:
            logger.debug(
                f"GitHub API request attempt {attempt + 1}",
                context=context,
                attempt=attempt + 1,
                max_retries=retries
            )
            
            resp = requests.request(
                method,
                url,
                headers=_headers(token_val),
                data=data,
                params=params,
                timeout=get_http_timeout(),
            )
            resp.raise_for_status()
            
            logger.info(
                "GitHub API request successful",
                context=context,
                status_code=resp.status_code,
                attempt=attempt + 1
            )
            
            return resp
        except requests.RequestException as exc:
            logger.warning(
                "GitHub API request failed",
                context=context,
                error=str(exc),
                attempt=attempt + 1,
                max_retries=retries,
                will_retry=attempt < retries - 1
            )
            
            if attempt == retries - 1:
                logger.error(
                    "GitHub API request failed after all retries",
                    context=context,
                    error=str(exc),
                    total_attempts=retries
                )
                raise
                
            sleep_time = 2 ** attempt * 0.5
            logger.debug(
                f"Retrying after {sleep_time}s delay",
                context=context,
                sleep_time=sleep_time
            )
            time.sleep(sleep_time)


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
