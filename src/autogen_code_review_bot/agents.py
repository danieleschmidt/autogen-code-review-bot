"""Agent implementations for code review."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .agent_templates import default_templates


# Configuration error class (will be imported from config_validation when needed)
class ConfigError(Exception):
    """Configuration error for agent loading."""
    pass


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
    from .config_validation import ConfigError, ValidationError, validate_config_file
    from .logging_config import get_logger

    logger = get_logger(__name__)

    try:
        # Validate configuration using our validation framework
        data = validate_config_file(path, "agent")

        logger.info("Successfully loaded and validated agent configuration",
                   extra={"config_path": path})

    except (ConfigError, ValidationError) as e:
        logger.error(f"Agent configuration validation failed: {e}",
                    extra={"config_path": path})
        raise ConfigError(f"Invalid agent configuration in {path}: {e}")
    except (OSError, PermissionError) as e:
        logger.error(f"Unable to read agent configuration file: {e}",
                    extra={"config_path": path})
        raise ConfigError(f"Cannot read agent configuration file {path}: {e}")

    agents = {}
    agents_config = data.get("agents", {})

    for role in ("coder", "reviewer"):
        cfg = agents_config.get(role, {})
        if not cfg:
            logger.debug(f"No configuration found for {role} agent, skipping")
            continue

        try:
            agent_config = AgentConfig(**cfg)
            agent_cls = CoderAgent if role == "coder" else ReviewerAgent
            agents[role] = agent_cls(role, agent_config)
            logger.debug(f"Successfully created {role} agent",
                        extra={"model": agent_config.model, "focus_areas": agent_config.focus_areas})
        except TypeError as e:
            logger.error(f"Invalid configuration for {role} agent: {e}",
                        extra={"config": cfg})
            raise ConfigError(f"Invalid {role} agent configuration: {e}")

    if not agents:
        logger.warning("No agents were configured")

    # Load response templates if configured
    if "response_templates" in data:
        try:
            default_templates.load_from_config(data["response_templates"])
            logger.info("Successfully loaded custom response templates",
                       extra={"available_templates": default_templates.get_available_templates()})
        except Exception as e:
            logger.warning(f"Failed to load response templates, using defaults: {e}")

    return agents


@dataclass
class ConversationTurn:
    """Represents a single turn in an agent conversation."""

    agent_name: str
    message: str
    context: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"ConversationTurn({self.agent_name}: {self.message[:50]}...)"


class AgentConversation:
    """Manages a conversation between multiple agents."""

    def __init__(self, agents: List[BaseAgent], code: str = "", max_turns: int = 10):
        self.agents = agents
        self.code = code
        self.turns: List[ConversationTurn] = []
        self.is_resolved = False
        self.resolution: Optional[str] = None
        self.max_turns = max_turns

    def add_turn(self, agent_name: str, message: str, context: Dict = None) -> None:
        """Add a turn to the conversation."""
        if context is None:
            context = {}
        turn = ConversationTurn(agent_name, message, context)
        self.turns.append(turn)

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_context(self, for_agent: str) -> List[str]:
        """Get conversation context for a specific agent."""
        return [f"{turn.agent_name}: {turn.message}" for turn in self.turns]

    def should_continue(self) -> bool:
        """Determine if the conversation should continue."""
        return not self.is_resolved and len(self.turns) < self.max_turns

    def resolve(self, resolution: str) -> None:
        """Mark the conversation as resolved."""
        self.is_resolved = True
        self.resolution = resolution


class ConversationManager:
    """Manages agent conversations and determines when agents should discuss."""

    def __init__(self, temperature: float = 0.1, max_conversation_turns: int = 5):
        self.temperature = temperature
        self.max_conversation_turns = max_conversation_turns

    def start_conversation(self, agents: List[BaseAgent], code: str) -> AgentConversation:
        """Start a new conversation between agents."""
        return AgentConversation(agents, code, self.max_conversation_turns)

    def run_conversation(self, agents: List[BaseAgent], code: str) -> str:
        """Run a complete conversation between agents."""
        # Get initial reviews from all agents
        initial_reviews = {}
        for agent in agents:
            initial_reviews[agent.name] = agent.review(code)

        # Check if agents should discuss
        if not self._should_agents_discuss(initial_reviews):
            result = "No discussion needed. Individual reviews:\n\n"
            for agent_name, review in initial_reviews.items():
                result += f"**{agent_name.title()}**: {review}\n\n"
            return result

        # Start conversation
        conversation = self.start_conversation(agents, code)

        # Add initial reviews as conversation turns
        for agent_name, review in initial_reviews.items():
            conversation.add_turn(agent_name, review)

        # Continue conversation until resolved or max turns
        while conversation.should_continue():
            for agent in agents:
                if not conversation.should_continue():
                    break

                context = conversation.get_context(agent.name)
                response = self._generate_agent_response(agent, code, context)
                conversation.add_turn(agent.name, response)

                # Check if conversation should be resolved
                if self._detect_resolution(conversation.turns[-3:]):  # Check last 3 turns
                    conversation.resolve("Agents reached consensus")
                    break

        # Generate summary
        return self._generate_conversation_summary(conversation)

    def _should_agents_discuss(self, reviews: Dict[str, str]) -> bool:
        """Determine if agents should engage in discussion."""
        if len(reviews) < 2:
            return False

        # Simple heuristic: look for conflicting keywords
        positive_words = ['good', 'solid', 'clean', 'excellent', 'well', 'nice']
        negative_words = ['bug', 'issue', 'problem', 'error', 'security', 'vulnerability', 'concern']

        review_sentiments = []
        for review in reviews.values():
            review_lower = review.lower()
            positive_count = sum(1 for word in positive_words if word in review_lower)
            negative_count = sum(1 for word in negative_words if word in review_lower)

            if negative_count > positive_count:
                review_sentiments.append('negative')
            elif positive_count > negative_count:
                review_sentiments.append('positive')
            else:
                review_sentiments.append('neutral')

        # Discuss if there are conflicting sentiments
        return len(set(review_sentiments)) > 1

    def _detect_resolution(self, recent_turns: List[ConversationTurn]) -> bool:
        """Detect if agents have reached resolution in recent turns."""
        if len(recent_turns) < 2:
            return False

        agreement_keywords = [
            'agree', 'agreed', 'consensus', 'resolved', 'settled',
            'right', 'correct', 'exactly', 'perfect', 'sounds good',
            'makes sense', 'good point', 'you\'re right'
        ]

        recent_messages = [turn.message.lower() for turn in recent_turns[-2:]]

        return any(
            any(keyword in message for keyword in agreement_keywords)
            for message in recent_messages
        )

    def _generate_agent_response(self, agent: BaseAgent, code: str, context: List[str]) -> str:
        """Generate a response from an agent based on context using configurable templates."""
        # This is a simplified implementation - in a real system,
        # this would call an LLM API with the agent's personality

        context_str = "\n".join(context[-3:])  # Last 3 turns for context

        # Determine agent type and select appropriate template category
        if isinstance(agent, CoderAgent):
            agent_type = "coder"
            # Select template category based on context or randomly
            template_categories = ["improvement_focused", "assessment", "agreement"]
        else:  # ReviewerAgent
            agent_type = "reviewer"
            template_categories = ["concern_focused", "findings", "opinion"]

        # Randomly select a template category
        template_category = random.choice(template_categories)

        try:
            return default_templates.get_response(agent_type, template_category)
        except ValueError:
            # Fallback to a generic response if template system fails
            return f"Agent {agent.name} provided feedback on the code implementation."

    def _generate_conversation_summary(self, conversation: AgentConversation) -> str:
        """Generate a summary of the conversation."""
        result = "## Conversation Summary\n\n"

        if conversation.is_resolved:
            result += f"**Resolution**: {conversation.resolution}\n\n"
        else:
            result += "**Status**: Conversation reached maximum turns without explicit resolution\n\n"

        result += "**Discussion Highlights**:\n"
        for i, turn in enumerate(conversation.turns, 1):
            result += f"{i}. **{turn.agent_name.title()}**: {turn.message}\n"

        return result


def run_dual_review(code: str, config_path: str) -> Dict[str, str]:
    """Run the dual-agent review process using the configuration at ``config_path``."""
    agents = load_agents_from_yaml(config_path)
    coder: Optional[CoderAgent] = agents.get("coder")
    reviewer: Optional[ReviewerAgent] = agents.get("reviewer")

    feedback = {}
    if coder is not None:
        feedback["coder"] = coder.review(code)
    if reviewer is not None:
        feedback["reviewer"] = reviewer.review(code)
    return feedback


def run_agent_conversation(code: str, config_path: str) -> str:
    """Run an agent conversation for enhanced code review."""
    agents_dict = load_agents_from_yaml(config_path)
    agents_list = list(agents_dict.values())

    if len(agents_list) < 2:
        # Fall back to simple review if not enough agents
        dual_result = run_dual_review(code, config_path)
        if not dual_result:
            return "No agents available for code review. Please check your configuration."
        return str(dual_result)

    manager = ConversationManager()
    try:
        result = manager.run_conversation(agents_list, code)
        if not result or not result.strip():
            return "Agent conversation completed but produced no output."
        return result
    except Exception as e:
        # Fallback to dual review if conversation fails
        dual_result = run_dual_review(code, config_path)
        if not dual_result:
            return f"Agent conversation failed and no fallback agents available: {str(e)}"
        return f"Agent conversation failed, fallback result: {str(dual_result)}"
