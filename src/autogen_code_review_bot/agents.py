"""Agent implementations for code review."""

from __future__ import annotations

import time
import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
        """Generate a response from an agent based on context."""
        # This is a simplified implementation - in a real system, 
        # this would call an LLM API with the agent's personality
        
        context_str = "\n".join(context[-3:])  # Last 3 turns for context
        
        # Simple response generation based on agent type and context
        if isinstance(agent, CoderAgent):
            responses = [
                f"Looking at the code implementation, I notice potential improvements in {random.choice(['performance', 'error handling', 'edge cases'])}",
                f"From a coding perspective, this {random.choice(['looks solid', 'needs refactoring', 'has bugs'])}",
                f"I {random.choice(['agree', 'disagree'])} with the previous assessment regarding the implementation"
            ]
        else:  # ReviewerAgent
            responses = [
                f"From a review standpoint, I'm {random.choice(['concerned about', 'satisfied with'])} the {random.choice(['security', 'maintainability', 'readability'])} aspects",
                f"The code review indicates {random.choice(['good practices', 'areas for improvement', 'security concerns'])}",
                f"I {random.choice(['concur', 'have reservations'])} about the current approach"
            ]
        
        return random.choice(responses)
    
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
    coder: CoderAgent = agents.get("coder")  # type: ignore
    reviewer: ReviewerAgent = agents.get("reviewer")  # type: ignore

    feedback = {}
    if coder:
        feedback["coder"] = coder.review(code)
    if reviewer:
        feedback["reviewer"] = reviewer.review(code)
    return feedback


def run_agent_conversation(code: str, config_path: str) -> str:
    """Run an agent conversation for enhanced code review."""
    agents_dict = load_agents_from_yaml(config_path)
    agents_list = list(agents_dict.values())
    
    if len(agents_list) < 2:
        # Fall back to simple review if not enough agents
        return str(run_dual_review(code, config_path))
    
    manager = ConversationManager()
    return manager.run_conversation(agents_list, code)
