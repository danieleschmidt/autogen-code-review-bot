"""Agent implementations for code review."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import yaml


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    model: str
    temperature: float = 0.1
    focus_areas: List[str] = field(default_factory=list)


class BaseAgent:
    """Base functionality shared by all agents."""

    def __init__(self, name: str, config: AgentConfig) -> None:
        self.name = name
        self.config = config

    @property
    def personality(self) -> str:
        return ", ".join(self.config.focus_areas)

    def review(self, code: str) -> str:  # pragma: no cover - placeholder
        """Return agent-specific review comments for the given code snippet."""
        raise NotImplementedError


class CoderAgent(BaseAgent):
    """Agent focused on implementation details and bug detection."""

    def review(self, code: str) -> str:
        snippet = code.strip().splitlines()[0][:80] if code else ""
        return f"[Coder] ({self.personality}) Suggestions based on snippet: {snippet}"


class ReviewerAgent(BaseAgent):
    """Agent focused on code quality, security, and performance."""

    def review(self, code: str) -> str:
        snippet = code.strip().splitlines()[0][:80] if code else ""
        return f"[Reviewer] ({self.personality}) Feedback based on snippet: {snippet}"


def load_agents_from_yaml(path: str) -> Dict[str, BaseAgent]:
    """Load agent configuration from a YAML file and return agent instances."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    agents = {}
    for role in ("coder", "reviewer"):
        cfg = data.get("agents", {}).get(role, {})
        if not cfg:
            continue
        agent_config = AgentConfig(**cfg)
        agent_cls = CoderAgent if role == "coder" else ReviewerAgent
        agents[role] = agent_cls(role, agent_config)

    return agents


def run_dual_review(code: str, config_path: str) -> Dict[str, str]:
    """Run the dual-agent review process using the configuration at ``config_path``."""
    agents = load_agents_from_yaml(config_path)
    coder: CoderAgent = agents.get("coder")  # type: ignore
    reviewer: ReviewerAgent = agents.get("reviewer")  # type: ignore

    feedback = {}
    if coder:
        feedback["coder"] = coder.review(code)
    if reviewer:
        feedback["reviewer"] = reviewer.review(code)
    return feedback
