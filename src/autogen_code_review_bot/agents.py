"""Author and Reviewer agents — each wraps a Backend and knows its own system prompt."""

from __future__ import annotations

import re
from typing import List, Optional

from .backends import Backend
from .models import ReviewComment, ReviewFeedback, ReviewStatus


# ---------------------------------------------------------------------------
# Author agent
# ---------------------------------------------------------------------------

AUTHOR_SYSTEM = """\
You are an expert Python developer.
When given a coding task, respond with ONLY the Python code (no explanations, no markdown fences).
When given prior review feedback, revise your code to address every comment.
Always write clean, well-named, properly typed Python.
"""


class AuthorAgent:
    """Writes and revises Python code based on task descriptions and reviewer feedback."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._history: List[dict] = []

    def write(self, task: str) -> str:
        """Write initial code for a task."""
        self._history = [{"role": "user", "content": task}]
        code = self._backend.complete(AUTHOR_SYSTEM, self._history)
        code = _strip_markdown_fences(code)
        self._history.append({"role": "assistant", "content": code})
        return code

    def revise(self, feedback: ReviewFeedback) -> str:
        """Revise code given reviewer feedback."""
        revision_request = (
            "Please revise your code to address the following review feedback:\n\n"
            + feedback.to_prompt_text()
        )
        self._history.append({"role": "user", "content": revision_request})
        code = self._backend.complete(AUTHOR_SYSTEM, self._history)
        code = _strip_markdown_fences(code)
        self._history.append({"role": "assistant", "content": code})
        return code


# ---------------------------------------------------------------------------
# Reviewer agent
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM = """\
You are a strict but fair Python code reviewer.
You review code against four criteria:
  1. correctness   — does it actually work? are there bugs?
  2. style         — PEP 8, clear naming, no dead code
  3. edge_cases    — does it handle None, empty input, boundary values?
  4. security      — no injection risks, no unsafe eval, safe defaults

Respond in this EXACT format (do not add extra sections):

STATUS: pass   (or STATUS: fail)
SUMMARY: one sentence overall verdict
COMMENTS:
- [correctness] description of issue (or "none")
- [style] description of issue (or "none")
- [edge_cases] description of issue (or "none")
- [security] description of issue (or "none")

If there are no issues in a category write "none" for that line.
STATUS must be "pass" only when ALL categories are "none".
"""


class ReviewerAgent:
    """Reviews Python code and returns structured feedback."""

    def __init__(self, backend: Backend):
        self._backend = backend

    def review(self, code: str, task_description: str) -> ReviewFeedback:
        """Review code against the original task and return structured feedback."""
        prompt = (
            f"Task the author was asked to solve:\n{task_description}\n\n"
            f"Code to review:\n```python\n{code}\n```"
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._backend.complete(REVIEWER_SYSTEM, messages)
        return _parse_review_response(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` fences if present."""
    text = text.strip()
    # Remove opening fence (```python or ```)
    text = re.sub(r"^```(?:python)?\s*\n?", "", text, flags=re.MULTILINE)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


_COMMENT_LINE_RE = re.compile(
    r"-\s*\[(?P<cat>[^\]]+)\]\s*(?P<msg>.+)", re.IGNORECASE
)


def _parse_review_response(text: str) -> ReviewFeedback:
    """Parse the structured reviewer response into a ReviewFeedback object."""
    lines = text.strip().splitlines()

    status = ReviewStatus.FAIL  # default safe
    summary = ""
    comments: List[ReviewComment] = []

    in_comments = False
    for line in lines:
        line_stripped = line.strip()

        if line_stripped.upper().startswith("STATUS:"):
            val = line_stripped.split(":", 1)[1].strip().lower()
            status = ReviewStatus.PASS if val == "pass" else ReviewStatus.FAIL
            continue

        if line_stripped.upper().startswith("SUMMARY:"):
            summary = line_stripped.split(":", 1)[1].strip()
            continue

        if line_stripped.upper().startswith("COMMENTS:"):
            in_comments = True
            continue

        if in_comments:
            m = _COMMENT_LINE_RE.match(line_stripped)
            if m:
                cat = m.group("cat").lower().strip()
                msg = m.group("msg").strip()
                if msg.lower() != "none":
                    comments.append(ReviewComment(category=cat, message=msg))

    return ReviewFeedback(status=status, comments=comments, summary=summary)
