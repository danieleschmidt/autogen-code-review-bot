"""Data models for PR analysis results."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class AnalysisSection:
    """Individual section of the PR analysis report."""

    tool: str
    output: str
    return_code: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PRAnalysisResult:
    """Container for all analysis sections."""

    security: AnalysisSection
    style: AnalysisSection
    performance: AnalysisSection
    metadata: Optional[Dict[str, Any]] = None