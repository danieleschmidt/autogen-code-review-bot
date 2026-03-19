"""Tests for AuthorAgent and ReviewerAgent."""

import pytest
from autogen_code_review_bot.agents import (
    AuthorAgent,
    ReviewerAgent,
    _strip_markdown_fences,
    _parse_review_response,
)
from autogen_code_review_bot.backends import MockBackend
from autogen_code_review_bot.models import ReviewStatus


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

def test_strip_markdown_fences_python():
    raw = "```python\ndef foo():\n    pass\n```"
    assert _strip_markdown_fences(raw) == "def foo():\n    pass"


def test_strip_markdown_fences_plain():
    raw = "```\ndef foo():\n    pass\n```"
    assert _strip_markdown_fences(raw) == "def foo():\n    pass"


def test_strip_markdown_fences_no_fences():
    raw = "def foo():\n    pass"
    assert _strip_markdown_fences(raw) == raw


def test_parse_review_response_pass():
    response = """\
STATUS: pass
SUMMARY: Code is clean and correct
COMMENTS:
- [correctness] none
- [style] none
- [edge_cases] none
- [security] none
"""
    fb = _parse_review_response(response)
    assert fb.status == ReviewStatus.PASS
    assert fb.summary == "Code is clean and correct"
    assert fb.comments == []


def test_parse_review_response_fail():
    response = """\
STATUS: fail
SUMMARY: Several issues found
COMMENTS:
- [correctness] Missing return type hint
- [style] Variable names not descriptive
- [edge_cases] Does not handle empty string
- [security] none
"""
    fb = _parse_review_response(response)
    assert fb.status == ReviewStatus.FAIL
    assert len(fb.comments) == 3
    categories = [c.category for c in fb.comments]
    assert "correctness" in categories
    assert "style" in categories
    assert "edge_cases" in categories
    assert "security" not in categories


# ---------------------------------------------------------------------------
# AuthorAgent tests
# ---------------------------------------------------------------------------

def test_author_write():
    backend = MockBackend(responses=["def validate_email(email): pass"])
    author = AuthorAgent(backend)
    code = author.write("Write a function that validates email addresses")
    assert "validate_email" in code
    assert backend.call_count == 1


def test_author_write_strips_fences():
    backend = MockBackend(responses=["```python\ndef foo(): pass\n```"])
    author = AuthorAgent(backend)
    code = author.write("Write something")
    assert "```" not in code


def test_author_revise():
    backend = MockBackend(responses=[
        "def foo(): return None",
        "def foo(x: int) -> int:\n    return x + 1",
    ])
    author = AuthorAgent(backend)
    author.write("Write foo")
    from autogen_code_review_bot.models import ReviewFeedback, ReviewComment, ReviewStatus
    feedback = ReviewFeedback(
        status=ReviewStatus.FAIL,
        summary="Missing types",
        comments=[ReviewComment(category="style", message="Add type hints")],
    )
    revised = author.revise(feedback)
    assert "int" in revised
    assert backend.call_count == 2


def test_author_history_grows():
    backend = MockBackend(responses=["v1", "v2"])
    author = AuthorAgent(backend)
    author.write("task")
    from autogen_code_review_bot.models import ReviewFeedback, ReviewStatus
    fb = ReviewFeedback(status=ReviewStatus.FAIL, summary="bad")
    author.revise(fb)
    # history: user(task), assistant(v1), user(revise), assistant(v2)
    assert len(author._history) == 4


# ---------------------------------------------------------------------------
# ReviewerAgent tests
# ---------------------------------------------------------------------------

def test_reviewer_review_pass():
    response = """\
STATUS: pass
SUMMARY: Excellent
COMMENTS:
- [correctness] none
- [style] none
- [edge_cases] none
- [security] none
"""
    backend = MockBackend(responses=[response])
    reviewer = ReviewerAgent(backend)
    fb = reviewer.review("def foo(): pass", "Write foo")
    assert fb.passed is True


def test_reviewer_review_fail():
    response = """\
STATUS: fail
SUMMARY: Has bugs
COMMENTS:
- [correctness] Does not validate input
- [style] none
- [edge_cases] Crashes on None
- [security] none
"""
    backend = MockBackend(responses=[response])
    reviewer = ReviewerAgent(backend)
    fb = reviewer.review("def foo(x): return x + 1", "Write foo")
    assert not fb.passed
    assert len(fb.comments) == 2
