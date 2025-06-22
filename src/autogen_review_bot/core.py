from __future__ import annotations


def process_text(text: str | None) -> str:
    """Return the text uppercased. Raises ValueError if text is None."""
    if text is None:
        raise ValueError("text cannot be None")
    return text.upper()
