from __future__ import annotations

import json
from typing import Any
import requests

API_URL = "https://api.github.com"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def get_pull_request_diff(repo: str, pr_number: int, token: str) -> str:
    """Return the diff for ``pr_number`` in ``repo``.

    ``repo`` should be in the form ``"owner/name"``.
    """
    url = f"{API_URL}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(
        url,
        headers=_headers(token),
        params={"media_type": "diff"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.text


def post_comment(repo: str, pr_number: int, body: str, token: str) -> Any:
    """Post ``body`` as a comment on the pull request."""
    url = f"{API_URL}/repos/{repo}/issues/{pr_number}/comments"
    resp = requests.post(
        url,
        headers=_headers(token),
        data=json.dumps({"body": body}),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
