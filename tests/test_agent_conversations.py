"""Tests for agent conversation system."""

import time
import threading
from unittest.mock import Mock, patch
from pathlib import Path

from autogen_code_review_bot.agents import (
    load_agents_from_yaml, 
    CoderAgent, 
    ReviewerAgent,
    AgentConfig
)
from autogen_code_review_bot.agent_conversations import (
    ConversationManager,
    ConversationState,
    Message,
    MessageType,
    ConversationHistory,
    RefinementCriteria,
    ConsensusResult
)


def test_message_creation():
    """Test Message dataclass creation and properties."""
    msg = Message(
        agent_name="coder",
        content="This code looks good",
        message_type=MessageType.REVIEW
    )
    assert msg.agent_name == "coder"
    assert msg.content == "This code looks good"
    assert msg.message_type == MessageType.REVIEW
    assert isinstance(msg.timestamp, float)
    assert msg.confidence is None
    

def test_conversation_state():
    """Test ConversationState management."""
    state = ConversationState(
        code_snippet="print('hello')",
        participants=["coder", "reviewer"]
    )
    
    assert state.code_snippet == "print('hello')"
    assert state.participants == ["coder", "reviewer"]
    assert state.round_number == 1
    assert state.status == "active"
    assert len(state.messages) == 0
    
    # Test adding messages
    msg = Message("coder", "Initial review", MessageType.REVIEW)
    state.add_message(msg)
    
    assert len(state.messages) == 1
    assert state.messages[0] == msg


def test_conversation_history():
    """Test conversation history tracking."""
    history = ConversationHistory()
    
    # Add conversation
    state = ConversationState("code", ["coder"])
    msg = Message("coder", "comment", MessageType.REVIEW)
    state.add_message(msg)
    
    conv_id = history.add_conversation(state)
    assert conv_id in history.conversations
    
    # Retrieve conversation
    retrieved = history.get_conversation(conv_id)
    assert retrieved is not None
    assert retrieved.code_snippet == "code"
    
    # Test non-existent conversation
    assert history.get_conversation("nonexistent") is None


def test_refinement_criteria():
    """Test refinement criteria evaluation."""
    criteria = RefinementCriteria(
        max_rounds=3,
        consensus_threshold=0.8,
        timeout_seconds=30.0,
        min_confidence=0.7
    )
    
    # Test round limit
    assert not criteria.should_continue_refinement(round_num=4)
    assert criteria.should_continue_refinement(round_num=2)
    
    # Test consensus check
    messages = [
        Message("coder", "good code", MessageType.REVIEW, confidence=0.9),
        Message("reviewer", "agree", MessageType.AGREEMENT, confidence=0.85)
    ]
    
    # High consensus should stop refinement
    assert not criteria.has_sufficient_consensus(messages)  # Only 2 messages, need more for consensus
    
    # Add more agreeing messages
    messages.extend([
        Message("coder", "confirmed", MessageType.AGREEMENT, confidence=0.8),
        Message("reviewer", "final approval", MessageType.AGREEMENT, confidence=0.9)
    ])
    
    assert criteria.has_sufficient_consensus(messages)


def test_conversation_manager_initialization():
    """Test ConversationManager initialization."""
    manager = ConversationManager()
    
    assert isinstance(manager.history, ConversationHistory)
    assert isinstance(manager.criteria, RefinementCriteria)
    assert manager.criteria.max_rounds == 5  # Default value


def test_conversation_manager_start_conversation(tmp_path):
    """Test starting a conversation between agents."""
    # Create test config
    cfg = {
        'agents': {
            'coder': {'model': 'gpt-4', 'temperature': 0.1, 'focus_areas': ['bugs']},
            'reviewer': {'model': 'gpt-4', 'temperature': 0.1, 'focus_areas': ['security']},
        }
    }
    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text("agents:\n  coder:\n    model: gpt-4\n    focus_areas: [bugs]\n  reviewer:\n    model: gpt-4\n    focus_areas: [security]\n")
    
    manager = ConversationManager()
    code = "print('hello world')"
    
    conv_id = manager.start_conversation(
        code_snippet=code,
        config_path=str(config_path)
    )
    
    assert conv_id is not None
    conversation = manager.history.get_conversation(conv_id)
    assert conversation is not None
    assert conversation.code_snippet == code
    assert len(conversation.participants) == 2
    assert "coder" in conversation.participants
    assert "reviewer" in conversation.participants


def test_conversation_manager_conduct_discussion(tmp_path):
    """Test conducting a discussion between agents."""
    # Create test config
    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text("agents:\n  coder:\n    model: gpt-4\n    focus_areas: [bugs]\n  reviewer:\n    model: gpt-4\n    focus_areas: [security]\n")
    
    manager = ConversationManager()
    code = "x = input(); eval(x)"  # Problematic code for discussion
    
    # Mock the agent review methods to return predictable responses
    with patch.object(CoderAgent, 'review') as mock_coder, \
         patch.object(ReviewerAgent, 'review') as mock_reviewer:
        
        mock_coder.side_effect = [
            "This code takes user input directly",
            "I agree there are security concerns"
        ]
        mock_reviewer.side_effect = [
            "CRITICAL: eval() is dangerous - arbitrary code execution",
            "We need input validation and safer alternatives"
        ]
        
        conv_id = manager.start_conversation(code, str(config_path))
        result = manager.conduct_discussion(conv_id, max_rounds=2)
        
        assert result is not None
        assert result.conversation_id == conv_id
        assert len(result.final_messages) > 0
        assert result.consensus_reached is not None
        
        conversation = manager.history.get_conversation(conv_id)
        assert len(conversation.messages) >= 2  # At least initial reviews


def test_conversation_manager_thread_safety():
    """Test that ConversationManager is thread-safe."""
    manager = ConversationManager()
    results = []
    errors = []
    
    def create_conversation(thread_id):
        try:
            conv_id = manager.history.add_conversation(
                ConversationState(f"code_{thread_id}", ["coder"])
            )
            results.append(conv_id)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = []
    for i in range(10):
        thread = threading.Thread(target=create_conversation, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Check results
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    assert len(results) == 10
    assert len(set(results)) == 10  # All IDs should be unique


def test_consensus_result():
    """Test ConsensusResult dataclass."""
    messages = [
        Message("coder", "looks good", MessageType.REVIEW, confidence=0.8),
        Message("reviewer", "approved", MessageType.AGREEMENT, confidence=0.9)
    ]
    
    result = ConsensusResult(
        conversation_id="test-123",
        final_messages=messages,
        consensus_reached=True,
        confidence_score=0.85,
        round_count=2
    )
    
    assert result.conversation_id == "test-123"
    assert len(result.final_messages) == 2
    assert result.consensus_reached is True
    assert result.confidence_score == 0.85
    assert result.round_count == 2


def test_message_type_enum():
    """Test MessageType enum values."""
    assert MessageType.REVIEW == "review"
    assert MessageType.QUESTION == "question"
    assert MessageType.RESPONSE == "response"
    assert MessageType.AGREEMENT == "agreement"
    assert MessageType.DISAGREEMENT == "disagreement"
    assert MessageType.REFINEMENT == "refinement"


def test_conversation_timeout_handling():
    """Test handling of conversation timeouts."""
    criteria = RefinementCriteria(timeout_seconds=0.1)  # Very short timeout
    
    # Simulate a long-running conversation
    start_time = time.time()
    time.sleep(0.15)  # Sleep longer than timeout
    
    assert criteria.is_timeout_exceeded(start_time)
    
    # Test with normal timeout
    recent_start = time.time()
    assert not criteria.is_timeout_exceeded(recent_start)


def test_conversation_state_serialization():
    """Test that conversation state can be properly serialized."""
    state = ConversationState(
        code_snippet="def test(): pass",
        participants=["coder", "reviewer"]
    )
    
    msg = Message(
        agent_name="coder",
        content="Function looks minimal",
        message_type=MessageType.REVIEW,
        confidence=0.7
    )
    state.add_message(msg)
    
    # Test that all fields are accessible for serialization
    assert hasattr(state, 'code_snippet')
    assert hasattr(state, 'participants')
    assert hasattr(state, 'messages')
    assert hasattr(state, 'round_number')
    assert hasattr(state, 'status')
    
    # Test message serialization
    assert hasattr(msg, 'agent_name')
    assert hasattr(msg, 'content')
    assert hasattr(msg, 'message_type')
    assert hasattr(msg, 'timestamp')
    assert hasattr(msg, 'confidence')


def test_empty_conversation_handling():
    """Test handling of empty conversations."""
    manager = ConversationManager()
    
    # Test empty code snippet
    state = ConversationState(code_snippet="", participants=[])
    conv_id = manager.history.add_conversation(state)
    
    conversation = manager.history.get_conversation(conv_id)
    assert conversation.code_snippet == ""
    assert len(conversation.participants) == 0
    
    # Should handle gracefully without errors
    assert conversation.status == "active"


def test_criteria_edge_cases():
    """Test edge cases in refinement criteria."""
    criteria = RefinementCriteria()
    
    # Test with empty message list
    assert not criteria.has_sufficient_consensus([])
    
    # Test with messages without confidence scores
    messages = [
        Message("coder", "review", MessageType.REVIEW),  # No confidence
        Message("reviewer", "response", MessageType.RESPONSE)
    ]
    
    # Should handle None confidence values gracefully
    result = criteria.has_sufficient_consensus(messages)
    assert isinstance(result, bool)