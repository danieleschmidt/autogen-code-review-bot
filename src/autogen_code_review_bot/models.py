"""Data models for the code review loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ReviewStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class ReviewComment:
    """A single review comment with optional location info."""

    category: str  # "correctness" | "style" | "edge_case" | "security"
    message: str
    line: Optional[int] = None
    severity: str = "warning"  # "error" | "warning" | "info"

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.category}/{self.severity}]{loc} {self.message}"


@dataclass
class ReviewFeedback:
    """Structured feedback from the reviewer agent."""

    status: ReviewStatus
    comments: List[ReviewComment] = field(default_factory=list)
    summary: str = ""

    @property
    def passed(self) -> bool:
        return self.status == ReviewStatus.PASS

    def to_prompt_text(self) -> str:
        """Format feedback as text suitable for injecting into author's next prompt."""
        lines = [f"Review result: {self.status.value.upper()}"]
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.comments:
            lines.append("Comments:")
            for c in self.comments:
                lines.append(f"  - {c}")
        return "\n".join(lines)
