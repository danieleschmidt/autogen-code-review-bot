#!/usr/bin/env python3
"""
Demo: two-agent code review loop with scripted mock backend.

Simulates a realistic review scenario for an email validation function:
- Iteration 1: Author writes naive code → Reviewer rejects (missing edge cases)
- Iteration 2: Author adds checks → Reviewer rejects (security concern with regex)  
- Iteration 3: Author uses stdlib → Reviewer passes

Run:
    python examples/demo_mock.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot import run_review_loop
from autogen_code_review_bot.backends import MockBackend

TASK = "Write a Python function `validate_email(email: str) -> bool` that returns True if the email address is valid, False otherwise. Handle edge cases properly."

# Scripted author responses (improving over iterations)
AUTHOR_RESPONSES = [
    # Iteration 1: naive, no edge cases
    """\
def validate_email(email: str) -> bool:
    return "@" in email
""",

    # Iteration 2: regex but still some issues
    """\
import re

def validate_email(email: str) -> bool:
    pattern = r'.+@.+\\..+'
    return bool(re.match(pattern, email))
""",

    # Iteration 3: proper implementation
    """\
import re
from typing import Optional


def validate_email(email: str) -> bool:
    \"\"\"
    Validate an email address.

    Returns True if the email looks valid, False otherwise.
    Handles None, empty strings, and common malformed inputs.
    \"\"\"
    if not email or not isinstance(email, str):
        return False

    # RFC 5322-inspired pattern (practical subset)
    pattern = r'^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$'
    return bool(re.fullmatch(pattern, email.strip()))
""",
]

# Scripted reviewer responses (escalating then passing)
REVIEWER_RESPONSES = [
    # Iteration 1: fail — only checks for @
    """\
STATUS: fail
SUMMARY: Only checks for "@" — far too permissive
COMMENTS:
- [correctness] "notanemail@" passes, "@nodomain" passes — both are invalid
- [style] none
- [edge_cases] Does not handle None or empty string — will crash on None
- [security] none
""",

    # Iteration 2: fail — regex too loose, no None guard
    """\
STATUS: fail
SUMMARY: Better but still has edge case and style issues
COMMENTS:
- [correctness] none
- [style] Missing docstring, import should be at top level
- [edge_cases] Does not handle None input — will raise TypeError
- [security] none
""",

    # Iteration 3: pass
    """\
STATUS: pass
SUMMARY: Clean, correct, handles edge cases, well documented
COMMENTS:
- [correctness] none
- [style] none
- [edge_cases] none
- [security] none
""",
]


def main():
    print("=" * 60)
    print("AutoGen Code Review Loop — Mock Demo")
    print("=" * 60)
    print(f"\nTask: {TASK}\n")

    result = run_review_loop(
        TASK,
        author_backend=MockBackend(responses=AUTHOR_RESPONSES),
        reviewer_backend=MockBackend(responses=REVIEWER_RESPONSES),
        max_iterations=5,
        verbose=True,
    )

    print()
    print(result.summary())

    # Quick functional test of the final code
    print("\n--- Running final code sanity checks ---")
    namespace = {}
    exec(result.final_code, namespace)
    validate_email = namespace["validate_email"]

    test_cases = [
        ("user@example.com", True),
        ("bad-email", False),
        ("@nodomain.com", False),
        ("user@", False),
        ("", False),
        (None, False),
        ("user.name+tag@sub.domain.org", True),
    ]

    all_ok = True
    for email, expected in test_cases:
        got = validate_email(email)
        status = "✅" if got == expected else "❌"
        if got != expected:
            all_ok = False
        print(f"  {status} validate_email({email!r}) → {got} (expected {expected})")

    print()
    print("All checks passed!" if all_ok else "⚠️  Some checks failed")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
