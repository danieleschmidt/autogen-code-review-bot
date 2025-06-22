import pytest

from autogen_review_bot.core import process_text


def test_success():
    assert process_text("hello") == "HELLO"


def test_edge_case_null_input():
    with pytest.raises(ValueError):
        process_text(None)
