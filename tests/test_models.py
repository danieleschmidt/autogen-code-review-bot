"""Tests for data models."""

import pytest
from autogen_code_review_bot.models import ReviewComment, ReviewFeedback, ReviewStatus


def test_review_comment_str_no_line():
    c = ReviewComment(category="style", message="Line too long")
    assert "[style/warning]" in str(c)
    assert "Line too long" in str(c)


def test_review_comment_str_with_line():
    c = ReviewComment(category="correctness", message="Off-by-one", line=10, severity="error")
    assert "line 10" in str(c)
    assert "[correctness/error]" in str(c)


def test_review_feedback_passed():
    fb = ReviewFeedback(status=ReviewStatus.PASS, summary="Looks good")
    assert fb.passed is True


def test_review_feedback_failed():
    fb = ReviewFeedback(status=ReviewStatus.FAIL, summary="Has issues")
    assert fb.passed is False


def test_review_feedback_to_prompt_text():
    fb = ReviewFeedback(
        status=ReviewStatus.FAIL,
        summary="Multiple issues",
        comments=[ReviewComment(category="style", message="Bad naming")],
    )
    text = fb.to_prompt_text()
    assert "FAIL" in text
    assert "Multiple issues" in text
    assert "Bad naming" in text
