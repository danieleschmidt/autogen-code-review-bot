"""Data models for PR analysis results."""

from dataclasses import dataclass


@dataclass
class AnalysisSection:
    """Individual section of the PR analysis report."""

    tool: str
    output: str


@dataclass
class PRAnalysisResult:
    """Container for all analysis sections."""

    security: AnalysisSection
    style: AnalysisSection
    performance: AnalysisSection