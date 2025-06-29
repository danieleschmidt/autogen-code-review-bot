import pytest
from pathlib import Path
from autogen_code_review_bot.language_detection import detect_language

@pytest.mark.parametrize(
    "filename,expected",
    [
        ("script.py", "python"),
        ("app.js", "javascript"),
        ("lib.ts", "typescript"),
        ("main.go", "go"),
        ("mod.rs", "rust"),
        ("tool.rb", "ruby"),
    ],
)
def test_detects_known_extensions(filename, expected):
    assert detect_language(filename) == expected


def test_accepts_path_object():
    assert detect_language(Path("foo.py")) == "python"


def test_unknown_extension():
    assert detect_language(Path("readme.txt")) == "unknown"

