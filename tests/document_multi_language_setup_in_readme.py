import pytest
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


@pytest.mark.parametrize(
    "phrase",
    [
        "Linter Configuration",
        "linters:",
        "python: ruff",
    ],
)
def test_instructions_present(phrase):
    text = README.read_text(encoding="utf-8")
    assert phrase in text


def test_example_snippet():
    text = README.read_text(encoding="utf-8")
    assert "```yaml" in text and "linters:" in text
