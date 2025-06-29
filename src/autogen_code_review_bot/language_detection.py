from __future__ import annotations

from pathlib import Path

# Mapping of file extensions to language identifiers
EXTENSION_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.go': 'go',
    '.rs': 'rust',
    '.rb': 'ruby',
}


def detect_language(filename: str | Path) -> str:
    """Return the language associated with ``filename`` based on its extension.

    ``filename`` may be a string or :class:`~pathlib.Path` instance.
    """
    ext = Path(filename).suffix.lower()
    return EXTENSION_MAP.get(ext, 'unknown')
