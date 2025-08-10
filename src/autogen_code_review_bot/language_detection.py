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


def detect_language(filenames):
    """Detect languages from filenames (single file or list of files).
    
    Args:
        filenames: Either a single filename (str/Path) or list of filenames
        
    Returns:
        str or list: For single file, returns language string.
                     For list, returns list of unique languages found.
    """
    # Handle single filename
    if isinstance(filenames, (str, Path)):
        ext = Path(filenames).suffix.lower()
        return EXTENSION_MAP.get(ext, 'unknown')

    # Handle list of filenames
    if isinstance(filenames, list):
        if not filenames:
            return []

        languages = set()
        for filename in filenames:
            if isinstance(filename, (str, Path)):
                ext = Path(filename).suffix.lower()
                lang = EXTENSION_MAP.get(ext, 'unknown')
                if lang != 'unknown':
                    languages.add(lang)

        return list(languages) if languages else ['unknown']

    return 'unknown'
