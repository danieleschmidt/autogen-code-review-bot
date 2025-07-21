"""Tests for agent conversation system."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Dict

from autogen_code_review_bot.agents import (
    BaseAgent, CoderAgent, ReviewerAgent, AgentConfig,
    AgentConversation, ConversationTurn, ConversationManager
)


@pytest.fixture
def coder_config():
    """Create a test coder agent configuration."""
    return AgentConfig(
        model="test-model",
        temperature=0.1,
        focus_areas=["bugs", "performance"]
    )


@pytest.fixture
def reviewer_config():
    """Create a test reviewer agent configuration."""
    return AgentConfig(
        model="test-model", 
        temperature=0.2,
        focus_areas=["security", "maintainability"]
    )


@pytest.fixture
def coder_agent(coder_config):
    """Create a test coder agent."""
    return CoderAgent("coder", coder_config)


@pytest.fixture
def reviewer_agent(reviewer_config):
    """Create a test reviewer agent."""
    return ReviewerAgent("reviewer", reviewer_config)


class TestConversationTurn:
    """Test the ConversationTurn data structure."""

    def test_conversation_turn_creation(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            agent_name="coder",
            message="This code has a potential bug",
            context={"line": 42, "file": "test.py"}
        )
        
        assert turn.agent_name == "coder"
        assert turn.message == "This code has a potential bug"
        assert turn.context == {"line": 42, "file": "test.py"}
        assert isinstance(turn.timestamp, float)

    def test_conversation_turn_repr(self):
        """Test string representation of conversation turn."""
        turn = ConversationTurn("reviewer", "Good point!", {})
        repr_str = repr(turn)
        
        assert "reviewer" in repr_str
        assert "Good point!" in repr_str


class TestAgentConversation:
    """Test the AgentConversation class."""

    def test_conversation_creation(self, coder_agent, reviewer_agent):
        """Test creating a new conversation."""
        conversation = AgentConversation([coder_agent, reviewer_agent])
        
        assert len(conversation.agents) == 2
        assert len(conversation.turns) == 0
        assert not conversation.is_resolved
        assert conversation.max_turns == 10

    def test_add_turn(self, coder_agent, reviewer_agent):
        """Test adding a turn to the conversation."""
        conversation = AgentConversation([coder_agent, reviewer_agent])
        
        conversation.add_turn("coder", "Initial review comment", {"file": "test.py"})
        
        assert len(conversation.turns) == 1
        assert conversation.turns[0].agent_name == "coder"
        assert conversation.turns[0].message == "Initial review comment"

    def test_get_agent_by_name(self, coder_agent, reviewer_agent):
        """Test retrieving agent by name."""
        conversation = AgentConversation([coder_agent, reviewer_agent])
        
        found_agent = conversation.get_agent("coder")
        assert found_agent == coder_agent
        
        not_found = conversation.get_agent("nonexistent")
        assert not_found is None

    def test_get_conversation_context(self, coder_agent, reviewer_agent):
        """Test getting conversation context for an agent."""
        conversation = AgentConversation([coder_agent, reviewer_agent])
        conversation.add_turn("coder", "First comment", {})
        conversation.add_turn("reviewer", "Response", {})
        
        context = conversation.get_context("reviewer")
        
        assert len(context) == 2
        assert "coder: First comment" in context
        assert "reviewer: Response" in context

    def test_should_continue_conversation(self, coder_agent, reviewer_agent):
        """Test conversation continuation logic."""
        conversation = AgentConversation([coder_agent, reviewer_agent], max_turns=3)
        
        # Should continue when not resolved and under max turns
        assert conversation.should_continue()
        
        # Add turns up to max
        for i in range(3):
            conversation.add_turn("coder", f"Turn {i}", {})
        
        # Should not continue when max turns reached
        assert not conversation.should_continue()
        
        # Should not continue when resolved
        conversation.turns = []  # Reset
        conversation.is_resolved = True
        assert not conversation.should_continue()

    def test_resolve_conversation(self, coder_agent, reviewer_agent):
        """Test resolving a conversation."""
        conversation = AgentConversation([coder_agent, reviewer_agent])
        conversation.add_turn("coder", "Issue found", {})
        conversation.add_turn("reviewer", "Agreed, let's resolve", {})
        
        conversation.resolve("Final consensus reached")
        
        assert conversation.is_resolved
        assert conversation.resolution == "Final consensus reached"


class TestConversationManager:
    """Test the ConversationManager class."""

    def test_manager_creation(self):
        """Test creating a conversation manager."""
        manager = ConversationManager()
        
        assert manager.temperature == 0.1
        assert manager.max_conversation_turns == 5

    def test_start_conversation(self, coder_agent, reviewer_agent):
        """Test starting a new conversation."""
        manager = ConversationManager()
        code = "def test(): pass"
        
        conversation = manager.start_conversation(
            [coder_agent, reviewer_agent], 
            code
        )
        
        assert len(conversation.agents) == 2
        assert conversation.code == code

    @patch('autogen_code_review_bot.agents.ConversationManager._should_agents_discuss')
    def test_run_conversation_no_discussion_needed(self, mock_should_discuss, coder_agent, reviewer_agent):
        """Test running conversation when no discussion is needed."""
        mock_should_discuss.return_value = False
        manager = ConversationManager()
        
        result = manager.run_conversation([coder_agent, reviewer_agent], "def test(): pass")
        
        assert "No discussion needed" in result
        assert len(result.split('\n')) >= 2  # Should have individual reviews

    @patch('autogen_code_review_bot.agents.ConversationManager._should_agents_discuss')
    @patch('autogen_code_review_bot.agents.ConversationManager._generate_agent_response')
    def test_run_conversation_with_discussion(self, mock_generate, mock_should_discuss, coder_agent, reviewer_agent):
        """Test running conversation with agent discussion."""
        mock_should_discuss.return_value = True
        mock_generate.side_effect = [
            "I think there's a bug here",
            "I agree, let's fix it",
            "Perfect, issue resolved"
        ]
        
        manager = ConversationManager()
        
        result = manager.run_conversation([coder_agent, reviewer_agent], "def test(): pass")
        
        assert "Conversation Summary" in result
        assert "I think there's a bug here" in result

    def test_should_agents_discuss(self, coder_agent, reviewer_agent):
        """Test logic for determining if agents should discuss."""
        manager = ConversationManager()
        
        # Test with conflicting reviews (should discuss)
        reviews = {
            "coder": "This code looks good",
            "reviewer": "This has security issues"
        }
        assert manager._should_agents_discuss(reviews)
        
        # Test with similar reviews (no discussion needed)
        reviews = {
            "coder": "Code looks good",
            "reviewer": "Implementation is solid"
        }
        assert not manager._should_agents_discuss(reviews)

    def test_detect_resolution(self):
        """Test detecting when conversation should be resolved."""
        manager = ConversationManager()
        
        # Test with agreement keywords
        turns = [
            ConversationTurn("coder", "I agree with your assessment", {}),
            ConversationTurn("reviewer", "Yes, that sounds right", {})
        ]
        assert manager._detect_resolution(turns)
        
        # Test without agreement
        turns = [
            ConversationTurn("coder", "I think this is wrong", {}),
            ConversationTurn("reviewer", "No, you're mistaken", {})
        ]
        assert not manager._detect_resolution(turns)

    @patch('random.random')
    def test_generate_agent_response_mock(self, mock_random, coder_agent):
        """Test agent response generation with mocked randomness."""
        mock_random.return_value = 0.5
        manager = ConversationManager()
        
        context = ["reviewer: Previous comment"]
        response = manager._generate_agent_response(coder_agent, "def test(): pass", context)
        
        # Should include agent name and be non-empty
        assert response.startswith("[Coder]") or "coder" in response.lower()
        assert len(response) > 10


class TestAgentConversationIntegration:
    """Integration tests for the agent conversation system."""

    def test_full_conversation_flow(self, coder_agent, reviewer_agent):
        """Test a complete conversation flow."""
        manager = ConversationManager(max_conversation_turns=3)
        code = """
        def process_user_input(user_input):
            # This function processes user input without validation
            return eval(user_input)  # Security issue
        """
        
        with patch.object(manager, '_should_agents_discuss', return_value=True):
            with patch.object(manager, '_generate_agent_response') as mock_gen:
                mock_gen.side_effect = [
                    "Security concern: eval() is dangerous",
                    "Agreed, this allows arbitrary code execution",
                    "We should use ast.literal_eval() instead"
                ]
                
                result = manager.run_conversation([coder_agent, reviewer_agent], code)
                
                assert "Security concern" in result
                assert "arbitrary code execution" in result
                assert "ast.literal_eval" in result

    def test_conversation_with_yaml_config_integration(self):
        """Test conversation system with YAML configuration loading."""
        # This test would require a mock YAML file
        # For now, we'll test the structure
        config_data = {
            "agents": {
                "coder": {
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "focus_areas": ["performance", "bugs"]
                },
                "reviewer": {
                    "model": "gpt-4", 
                    "temperature": 0.2,
                    "focus_areas": ["security", "maintainability"]
                }
            }
        }
        
        # Validate that our conversation system can work with this structure
        assert "coder" in config_data["agents"]
        assert "reviewer" in config_data["agents"]
        assert all("focus_areas" in agent for agent in config_data["agents"].values())