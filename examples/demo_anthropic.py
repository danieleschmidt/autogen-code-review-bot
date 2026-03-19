#!/usr/bin/env python3
"""
Demo: two-agent code review loop with real Anthropic backend.

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/demo_anthropic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot import run_review_loop
from autogen_code_review_bot.backends import AnthropicBackend

TASK = (
    "Write a Python function `validate_email(email: str) -> bool` "
    "that returns True if the email address is valid, False otherwise. "
    "Handle edge cases: None input, empty string, missing @, missing domain."
)


def main():
    try:
        backend = AnthropicBackend(model="claude-3-5-haiku-20241022")
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install anthropic")
        return 1

    print("=" * 60)
    print("AutoGen Code Review Loop — Anthropic Demo")
    print("=" * 60)
    print(f"\nTask: {TASK}\n")
    print("Running (this may take a moment)...\n")

    result = run_review_loop(
        TASK,
        backend=backend,
        max_iterations=5,
        verbose=True,
    )

    print()
    print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
