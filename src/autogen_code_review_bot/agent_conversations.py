"""Agent conversation system for collaborative code review refinement."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .agents import BaseAgent, load_agents_from_yaml


class MessageType(str, Enum):
    """Types of messages in agent conversations."""

    REVIEW = "review"
    QUESTION = "question"
    RESPONSE = "response"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    REFINEMENT = "refinement"


@dataclass
class Message:
    """A message in an agent conversation."""

    agent_name: str
    content: str
    message_type: MessageType
    timestamp: float = field(default_factory=time.time)
    confidence: Optional[float] = None

    def __post_init__(self):
        """Validate message fields after initialization."""
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")

        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ConversationState:
    """State of an ongoing conversation between agents."""

    code_snippet: str
    participants: List[str]
    messages: List[Message] = field(default_factory=list)
    round_number: int = 1
    status: str = "active"  # active, completed, timeout, error

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def get_messages_by_round(self, round_num: int) -> List[Message]:
        """Get messages from a specific round."""
        # For simplicity, we'll consider messages in chronological order
        # In a more sophisticated implementation, we'd track round per message
        messages_per_round = len(self.messages) // self.round_number if self.round_number > 0 else 0
        start_idx = (round_num - 1) * messages_per_round
        end_idx = round_num * messages_per_round
        return self.messages[start_idx:end_idx] if start_idx < len(self.messages) else []

    def get_latest_messages_by_agent(self) -> Dict[str, Message]:
        """Get the latest message from each agent."""
        latest_messages = {}
        for message in reversed(self.messages):
            if message.agent_name not in latest_messages:
                latest_messages[message.agent_name] = message
        return latest_messages


class ConversationHistory:
    """Manages conversation history and persistence."""

    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self._lock = threading.RLock()

    def add_conversation(self, conversation: ConversationState) -> str:
        """Add a conversation and return its ID."""
        conversation_id = str(uuid.uuid4())

        with self._lock:
            self.conversations[conversation_id] = conversation

        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get a conversation by ID."""
        with self._lock:
            return self.conversations.get(conversation_id)

    def update_conversation(self, conversation_id: str, conversation: ConversationState) -> bool:
        """Update an existing conversation."""
        with self._lock:
            if conversation_id in self.conversations:
                self.conversations[conversation_id] = conversation
                return True
            return False

    def get_active_conversations(self) -> Dict[str, ConversationState]:
        """Get all active conversations."""
        with self._lock:
            return {
                conv_id: conv for conv_id, conv in self.conversations.items()
                if conv.status == "active"
            }


@dataclass
class RefinementCriteria:
    """Criteria for determining when to stop conversation refinement."""

    max_rounds: int = 5
    consensus_threshold: float = 0.8
    timeout_seconds: float = 300.0  # 5 minutes
    min_confidence: float = 0.6

    def should_continue_refinement(self, round_num: int) -> bool:
        """Check if refinement should continue based on round number."""
        return round_num <= self.max_rounds

    def has_sufficient_consensus(self, messages: List[Message]) -> bool:
        """Check if agents have reached sufficient consensus."""
        if not messages:
            return False

        # Count agreement messages
        agreement_messages = [
            msg for msg in messages
            if msg.message_type == MessageType.AGREEMENT
        ]

        # Calculate consensus ratio
        if len(messages) < 2:  # Need at least 2 messages for consensus
            return False

        consensus_ratio = len(agreement_messages) / len(messages)
        return consensus_ratio >= self.consensus_threshold

    def is_timeout_exceeded(self, start_time: float) -> bool:
        """Check if conversation has exceeded timeout."""
        return time.time() - start_time > self.timeout_seconds

    def meets_confidence_threshold(self, messages: List[Message]) -> bool:
        """Check if messages meet minimum confidence threshold."""
        confident_messages = [
            msg for msg in messages
            if msg.confidence is not None and msg.confidence >= self.min_confidence
        ]

        if not messages:
            return False

        return len(confident_messages) / len(messages) >= 0.5  # At least 50% confident


@dataclass
class ConsensusResult:
    """Result of a conversation consensus process."""

    conversation_id: str
    final_messages: List[Message]
    consensus_reached: bool
    confidence_score: float
    round_count: int
    reason: str = ""  # Reason for ending conversation


class ConversationManager:
    """Manages agent conversations and refinement processes."""

    def __init__(self, criteria: Optional[RefinementCriteria] = None):
        self.history = ConversationHistory()
        self.criteria = criteria or RefinementCriteria()
        self._active_agents: Dict[str, Dict[str, BaseAgent]] = {}
        self._lock = threading.RLock()

    def start_conversation(
        self,
        code_snippet: str,
        config_path: str,
        participants: Optional[List[str]] = None
    ) -> str:
        """Start a new conversation between agents."""
        # Load agents from configuration
        agents = load_agents_from_yaml(config_path)

        if participants is None:
            participants = list(agents.keys())

        # Store agents for this conversation
        conversation_id = str(uuid.uuid4())
        with self._lock:
            self._active_agents[conversation_id] = agents

        # Create conversation state
        conversation = ConversationState(
            code_snippet=code_snippet,
            participants=participants
        )

        # Add to history
        final_conversation_id = self.history.add_conversation(conversation)

        # Update agents mapping with actual conversation ID
        if final_conversation_id != conversation_id:
            with self._lock:
                self._active_agents[final_conversation_id] = self._active_agents.pop(conversation_id)

        return final_conversation_id

    def conduct_discussion(
        self,
        conversation_id: str,
        max_rounds: Optional[int] = None
    ) -> Optional[ConsensusResult]:
        """Conduct a discussion between agents."""
        conversation = self.history.get_conversation(conversation_id)
        if not conversation:
            return None

        agents = self._active_agents.get(conversation_id, {})
        if not agents:
            return None

        start_time = time.time()
        max_rounds = max_rounds or self.criteria.max_rounds

        # Initial round - get initial reviews
        for agent_name in conversation.participants:
            if agent_name in agents:
                agent = agents[agent_name]
                review = agent.review(conversation.code_snippet)

                message = Message(
                    agent_name=agent_name,
                    content=review,
                    message_type=MessageType.REVIEW,
                    confidence=0.8  # Default confidence for initial reviews
                )
                conversation.add_message(message)

        # Refinement rounds
        while conversation.round_number < max_rounds:
            # Check stopping criteria
            if self.criteria.is_timeout_exceeded(start_time):
                conversation.status = "timeout"
                break

            if self.criteria.has_sufficient_consensus(conversation.messages):
                conversation.status = "completed"
                break

            # Next round of discussion
            conversation.round_number += 1
            round_messages = []

            # Get responses from each agent based on previous messages
            for agent_name in conversation.participants:
                if agent_name in agents:
                    agent = agents[agent_name]

                    # Create context from previous messages
                    context = self._create_context_for_agent(conversation, agent_name)
                    response = self._generate_agent_response(agent, context)

                    message_type = self._determine_message_type(response, conversation.messages)

                    message = Message(
                        agent_name=agent_name,
                        content=response,
                        message_type=message_type,
                        confidence=self._estimate_confidence(response)
                    )

                    conversation.add_message(message)
                    round_messages.append(message)

            # Update conversation in history
            self.history.update_conversation(conversation_id, conversation)

        # Calculate final consensus result
        consensus_reached = self.criteria.has_sufficient_consensus(conversation.messages)
        confidence_score = self._calculate_overall_confidence(conversation.messages)

        result = ConsensusResult(
            conversation_id=conversation_id,
            final_messages=conversation.messages,
            consensus_reached=consensus_reached,
            confidence_score=confidence_score,
            round_count=conversation.round_number,
            reason=conversation.status
        )

        return result

    def _create_context_for_agent(self, conversation: ConversationState, agent_name: str) -> str:
        """Create context string for agent based on conversation history."""
        context_parts = [f"Code under review:\n{conversation.code_snippet}\n"]

        # Add other agents' messages
        for message in conversation.messages:
            if message.agent_name != agent_name:
                context_parts.append(
                    f"{message.agent_name}: {message.content}"
                )

        return "\n".join(context_parts)

    def _generate_agent_response(self, agent: BaseAgent, context: str) -> str:
        """Generate a response from an agent based on context."""
        # For now, use the existing review method
        # In a full implementation, this would be more sophisticated
        response = agent.review(context)

        # Add some refinement based on context length
        if "CRITICAL" in context.upper() or "DANGEROUS" in context.upper():
            return f"{response}. I agree this needs immediate attention."
        elif "AGREE" in context.upper() or "APPROVED" in context.upper():
            return f"{response}. I concur with the assessment."
        else:
            return response

    def _determine_message_type(self, response: str, previous_messages: List[Message]) -> MessageType:
        """Determine the type of message based on content."""
        response_lower = response.lower()

        if any(word in response_lower for word in ["agree", "concur", "approved", "correct"]):
            return MessageType.AGREEMENT
        elif any(word in response_lower for word in ["disagree", "incorrect", "wrong"]):
            return MessageType.DISAGREEMENT
        elif "?" in response:
            return MessageType.QUESTION
        elif len(previous_messages) == 0:
            return MessageType.REVIEW
        else:
            return MessageType.RESPONSE

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on response content."""
        response_lower = response.lower()

        # High confidence indicators
        high_confidence_words = ["critical", "certain", "definitely", "always", "never"]
        if any(word in response_lower for word in high_confidence_words):
            return 0.9

        # Medium confidence indicators
        medium_confidence_words = ["likely", "probably", "should", "recommend"]
        if any(word in response_lower for word in medium_confidence_words):
            return 0.7

        # Low confidence indicators
        low_confidence_words = ["maybe", "might", "could", "perhaps", "possibly"]
        if any(word in response_lower for word in low_confidence_words):
            return 0.5

        # Default confidence
        return 0.6

    def _calculate_overall_confidence(self, messages: List[Message]) -> float:
        """Calculate overall confidence score from all messages."""
        if not messages:
            return 0.0

        confidence_scores = [
            msg.confidence for msg in messages
            if msg.confidence is not None
        ]

        if not confidence_scores:
            return 0.5  # Default confidence

        return sum(confidence_scores) / len(confidence_scores)

    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        """Get a summary of a conversation."""
        conversation = self.history.get_conversation(conversation_id)
        if not conversation:
            return None

        return {
            "conversation_id": conversation_id,
            "code_snippet": conversation.code_snippet,
            "participants": conversation.participants,
            "message_count": len(conversation.messages),
            "rounds": conversation.round_number,
            "status": conversation.status,
            "final_consensus": self.criteria.has_sufficient_consensus(conversation.messages)
        }

    def cleanup_completed_conversations(self, max_age_hours: float = 24.0) -> int:
        """Clean up old completed conversations."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_count = 0

        with self._lock:
            conversations_to_remove = []

            for conv_id, conversation in self.history.conversations.items():
                if conversation.status != "active":
                    # Check if any message is older than cutoff
                    if conversation.messages:
                        oldest_message_time = min(msg.timestamp for msg in conversation.messages)
                        if oldest_message_time < cutoff_time:
                            conversations_to_remove.append(conv_id)

            # Remove old conversations
            for conv_id in conversations_to_remove:
                del self.history.conversations[conv_id]
                if conv_id in self._active_agents:
                    del self._active_agents[conv_id]
                removed_count += 1

        return removed_count


def conduct_agent_discussion(
    code_snippet: str,
    config_path: str,
    max_rounds: int = 3,
    criteria: Optional[RefinementCriteria] = None
) -> ConsensusResult:
    """
    High-level function to conduct an agent discussion.
    
    Args:
        code_snippet: Code to review and discuss
        config_path: Path to agent configuration file
        max_rounds: Maximum discussion rounds
        criteria: Custom refinement criteria
    
    Returns:
        ConsensusResult with discussion outcome
    """
    manager = ConversationManager(criteria)
    conversation_id = manager.start_conversation(code_snippet, config_path)
    result = manager.conduct_discussion(conversation_id, max_rounds)

    if result is None:
        # Fallback result if discussion failed
        return ConsensusResult(
            conversation_id=conversation_id,
            final_messages=[],
            consensus_reached=False,
            confidence_score=0.0,
            round_count=0,
            reason="Failed to conduct discussion"
        )

    return result
