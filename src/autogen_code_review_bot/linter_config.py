#!/usr/bin/env python3
"""
Linter configuration for AutoGen Code Review Bot.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LinterConfig:
    """Configuration for language-specific linters."""
    python: str = "ruff"
    javascript: str = "eslint"
    typescript: str = "eslint"
    go: str = "golangci-lint"
    rust: str = "clippy"
    java: str = "checkstyle"
    cpp: str = "clang-tidy"
    ruby: str = "rubocop"
    php: str = "phpcs"
    swift: str = "swiftlint"
    kotlin: str = "ktlint"
    scala: str = "scalastyle"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'LinterConfig':
        """Load linter configuration from YAML file."""
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Linter config file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        linters = data.get('linters', {})
        return cls(**{k: v for k, v in linters.items() if hasattr(cls, k)})