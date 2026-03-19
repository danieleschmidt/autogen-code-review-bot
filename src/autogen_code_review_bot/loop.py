"""
Two-agent iterative review loop.

Flow:
  Author writes code → Reviewer reviews → if FAIL, Author revises → repeat
  Loop ends when Reviewer passes OR max_iterations reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .agents import AuthorAgent, ReviewerAgent
from .backends import Backend, MockBackend
from .models import ReviewFeedback, ReviewStatus


@dataclass
class IterationRecord:
    """Snapshot of one author-reviewer exchange."""

    iteration: int
    code: str
    feedback: ReviewFeedback


@dataclass
class ReviewResult:
    """Final result of a complete review loop."""

    task: str
    final_code: str
    final_feedback: ReviewFeedback
    iterations: List[IterationRecord] = field(default_factory=list)
    passed: bool = False

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    def summary(self) -> str:
        verdict = "✅ PASSED" if self.passed else "❌ FAILED (max iterations reached)"
        lines = [
            f"Task: {self.task}",
            f"Result: {verdict}",
            f"Iterations: {self.iteration_count}",
            "",
        ]
        for rec in self.iterations:
            lines.append(f"--- Iteration {rec.iteration} ---")
            lines.append(f"Reviewer: {rec.feedback.status.value.upper()} — {rec.feedback.summary}")
            for c in rec.feedback.comments:
                lines.append(f"  {c}")
            lines.append("")
        lines.append("=== Final Code ===")
        lines.append(self.final_code)
        return "\n".join(lines)


def run_review_loop(
    task: str,
    *,
    author_backend: Optional[Backend] = None,
    reviewer_backend: Optional[Backend] = None,
    backend: Optional[Backend] = None,
    max_iterations: int = 5,
    verbose: bool = False,
) -> ReviewResult:
    """
    Run the two-agent review loop for a given coding task.

    Parameters
    ----------
    task:
        Natural-language description of the coding task.
    author_backend:
        Backend for the AuthorAgent.  Falls back to ``backend``.
    reviewer_backend:
        Backend for the ReviewerAgent.  Falls back to ``backend``.
    backend:
        Convenience shortcut — used when author and reviewer share one backend.
        Ignored if ``author_backend`` / ``reviewer_backend`` are set explicitly.
    max_iterations:
        Maximum number of author→reviewer cycles before giving up.
    verbose:
        If True, print iteration progress to stdout.

    Returns
    -------
    ReviewResult
        Full result including iteration history, final code, and verdict.
    """
    if backend is None and author_backend is None and reviewer_backend is None:
        raise ValueError(
            "Provide at least one of: backend, author_backend, reviewer_backend"
        )

    _author_be = author_backend or backend
    _reviewer_be = reviewer_backend or backend

    author = AuthorAgent(_author_be)
    reviewer = ReviewerAgent(_reviewer_be)

    iterations: List[IterationRecord] = []
    code = author.write(task)

    for i in range(1, max_iterations + 1):
        feedback = reviewer.review(code, task)

        iterations.append(IterationRecord(iteration=i, code=code, feedback=feedback))

        if verbose:
            print(f"[iteration {i}] reviewer: {feedback.status.value.upper()} — {feedback.summary}")
            for c in feedback.comments:
                print(f"              {c}")

        if feedback.passed:
            return ReviewResult(
                task=task,
                final_code=code,
                final_feedback=feedback,
                iterations=iterations,
                passed=True,
            )

        if i < max_iterations:
            code = author.revise(feedback)

    # Exhausted iterations — return last state
    return ReviewResult(
        task=task,
        final_code=code,
        final_feedback=iterations[-1].feedback,
        iterations=iterations,
        passed=False,
    )
