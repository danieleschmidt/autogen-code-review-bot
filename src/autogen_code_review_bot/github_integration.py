from __future__ import annotations

import json
import logging
import time
from typing import Any, TYPE_CHECKING

import requests

API_URL = "https://api.github.com"
logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .pr_analysis import PRAnalysisResult


def _request_with_retries(method: str, url: str, *, token: str, data: Any | None = None, params: dict[str, Any] | None = None, retries: int = 3) -> requests.Response:
    """Return a ``requests`` response using ``method`` with simple retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=_headers(token),
                data=data,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            logger.warning("GitHub API request failed: %s", exc)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 0.5)


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def get_pull_request_diff(repo: str, pr_number: int, token: str) -> str:
    """Return the diff for ``pr_number`` in ``repo``."""

    url = f"{API_URL}/repos/{repo}/pulls/{pr_number}"
    resp = _request_with_retries(
        "get",
        url,
        token=token,
        params={"media_type": "diff"},
    )
    return resp.text


def post_comment(repo: str, pr_number: int, body: str, token: str) -> Any:
    """Post ``body`` as a comment on the pull request."""

    url = f"{API_URL}/repos/{repo}/issues/{pr_number}/comments"
    resp = _request_with_retries(
        "post",
        url,
        token=token,
        data=json.dumps({"body": body}),
    )
    return resp.json()


def format_analysis_result(result: PRAnalysisResult) -> str:
    """Return ``result`` formatted for a GitHub comment."""

    return (
        "## 🤖 AutoGen Code Review\n"
        f"### Security ({result.security.tool})\n{result.security.output}\n\n"
        f"### Style ({result.style.tool})\n{result.style.output}\n\n"
        f"### Performance ({result.performance.tool})\n{result.performance.output}"
    )


def analyze_and_comment(
    repo_path: str, repo: str, pr_number: int, token: str, config_path: str | None = None
) -> Any:
    """Analyze the repo and post the results as a comment on the PR."""

    from .pr_analysis import analyze_pr

    result = analyze_pr(repo_path, config_path)
    body = format_analysis_result(result)
    return post_comment(repo, pr_number, body, token)
