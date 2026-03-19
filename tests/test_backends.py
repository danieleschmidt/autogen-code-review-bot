"""Tests for backend implementations."""

import pytest
from autogen_code_review_bot.backends import MockBackend


def test_mock_backend_single_response():
    b = MockBackend(responses=["hello"])
    assert b.complete("sys", []) == "hello"
    assert b.call_count == 1


def test_mock_backend_multiple_responses():
    b = MockBackend(responses=["first", "second", "third"])
    assert b.complete("sys", []) == "first"
    assert b.complete("sys", []) == "second"
    assert b.complete("sys", []) == "third"


def test_mock_backend_repeats_last():
    b = MockBackend(responses=["only"])
    b.complete("sys", [])
    assert b.complete("sys", []) == "only"
    assert b.call_count == 2


def test_mock_backend_response_fn():
    calls = []

    def fn(system, messages):
        calls.append((system, messages))
        return f"reply-{len(calls)}"

    b = MockBackend(response_fn=fn)
    assert b.complete("s", []) == "reply-1"
    assert b.complete("s", []) == "reply-2"
    assert len(calls) == 2


def test_mock_backend_raises_if_both_provided():
    with pytest.raises(ValueError):
        MockBackend(responses=["x"], response_fn=lambda s, m: "y")


def test_mock_backend_empty_responses():
    b = MockBackend(responses=[])
    result = b.complete("sys", [])
    assert "No response" in result
