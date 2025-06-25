import pathlib
from typing import Dict

EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
}


def detect_language(filename: str) -> str:
    """Return the programming language based on ``filename`` extension."""
    ext = pathlib.Path(filename).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, "unknown")
