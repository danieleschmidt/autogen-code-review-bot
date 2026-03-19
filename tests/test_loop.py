"""Tests for the two-agent review loop."""

import pytest
from autogen_code_review_bot.loop import run_review_loop, ReviewResult
from autogen_code_review_bot.backends import MockBackend
from autogen_code_review_bot.models import ReviewStatus


PASS_REVIEW = """\
STATUS: pass
SUMMARY: All good
COMMENTS:
- [correctness] none
- [style] none
- [edge_cases] none
- [security] none
"""

FAIL_REVIEW = """\
STATUS: fail
SUMMARY: Needs work
COMMENTS:
- [correctness] none
- [style] Missing docstring
- [edge_cases] Does not handle None
- [security] none
"""


def make_loop(author_responses, reviewer_responses):
    return run_review_loop(
        "Write a function",
        author_backend=MockBackend(responses=author_responses),
        reviewer_backend=MockBackend(responses=reviewer_responses),
    )


def test_loop_passes_on_first_try():
    result = make_loop(
        author_responses=["def foo(): pass"],
        reviewer_responses=[PASS_REVIEW],
    )
    assert result.passed is True
    assert result.iteration_count == 1


def test_loop_passes_after_revision():
    result = make_loop(
        author_responses=["def foo(): pass", "def foo():\n    '''docstring'''\n    pass"],
        reviewer_responses=[FAIL_REVIEW, PASS_REVIEW],
    )
    assert result.passed is True
    assert result.iteration_count == 2
    assert "docstring" in result.final_code


def test_loop_fails_after_max_iterations():
    result = run_review_loop(
        "Write a function",
        author_backend=MockBackend(responses=["def foo(): pass"] * 10),
        reviewer_backend=MockBackend(responses=[FAIL_REVIEW] * 10),
        max_iterations=3,
    )
    assert result.passed is False
    assert result.iteration_count == 3


def test_loop_result_summary():
    result = make_loop(["def foo(): pass"], [PASS_REVIEW])
    summary = result.summary()
    assert "PASSED" in summary
    assert "def foo(): pass" in summary


def test_loop_shared_backend():
    """Both agents can use the same backend instance."""
    # author gets first call (code), reviewer gets second call (review)
    shared = MockBackend(responses=["def foo(): pass", PASS_REVIEW])
    result = run_review_loop("Write foo", backend=shared)
    assert result.passed is True
    assert shared.call_count == 2


def test_loop_raises_without_backend():
    with pytest.raises(ValueError, match="backend"):
        run_review_loop("Write a function")


def test_loop_iteration_records():
    result = make_loop(
        author_responses=["v1", "v2"],
        reviewer_responses=[FAIL_REVIEW, PASS_REVIEW],
    )
    assert len(result.iterations) == 2
    assert result.iterations[0].code == "v1"
    assert result.iterations[1].code == "v2"
    assert result.iterations[0].feedback.status == ReviewStatus.FAIL
    assert result.iterations[1].feedback.status == ReviewStatus.PASS


def test_loop_verbose(capsys):
    make_loop(
        author_responses=["def foo(): pass"],
        reviewer_responses=[PASS_REVIEW],
    )
    # No output without verbose
    captured = capsys.readouterr()
    assert captured.out == ""


def test_loop_verbose_on(capsys):
    run_review_loop(
        "Write a function",
        author_backend=MockBackend(responses=["def foo(): pass"]),
        reviewer_backend=MockBackend(responses=[PASS_REVIEW]),
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "iteration 1" in captured.out.lower()
    assert "pass" in captured.out.lower()
