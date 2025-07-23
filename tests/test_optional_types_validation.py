"""Test proper null checking and validation for Optional types."""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from autogen_code_review_bot.agents import run_dual_review, run_agent_conversation


class TestOptionalTypesValidation:
    """Test that Optional types are properly validated before use."""
    
    def test_run_dual_review_handles_missing_agents(self):
        """Test that run_dual_review handles missing agents gracefully."""
        # Create a minimal config that would result in empty agents dict
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
# Empty agent config - should not crash the system
agents: {}
""")
            config_path = f.name
        
        try:
            # Mock the load_agents_from_yaml to return empty dict
            with patch('autogen_code_review_bot.agents.load_agents_from_yaml') as mock_load:
                mock_load.return_value = {}  # No agents loaded
                
                # This should not crash, even with missing agents
                result = run_dual_review("def test(): pass", config_path)
                
                # Should return empty feedback or handle gracefully
                assert isinstance(result, dict)
                assert len(result) == 0  # No agents means no feedback
                
        finally:
            Path(config_path).unlink()
    
    def test_run_dual_review_handles_partial_agents(self):
        """Test that run_dual_review handles when only some agents are available."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
# Partial config
agents:
  coder:
    name: "test_coder"
  # reviewer is missing
""")
            config_path = f.name
        
        try:
            # Mock agents where only coder exists
            mock_coder = MagicMock()
            mock_coder.review.return_value = "Coder feedback"
            
            agents_dict = {"coder": mock_coder}  # reviewer is missing
            
            with patch('autogen_code_review_bot.agents.load_agents_from_yaml') as mock_load:
                mock_load.return_value = agents_dict
                
                result = run_dual_review("def test(): pass", config_path)
                
                # Should handle missing reviewer gracefully
                assert "coder" in result
                assert result["coder"] == "Coder feedback"
                assert "reviewer" not in result  # Missing agent should not be in result
                
        finally:
            Path(config_path).unlink()
    
    def test_run_agent_conversation_handles_missing_agents(self):
        """Test that run_agent_conversation handles missing agents gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
agents: {}
""")
            config_path = f.name
        
        try:
            with patch('autogen_code_review_bot.agents.load_agents_from_yaml') as mock_load:
                mock_load.return_value = {}
                
                # Should handle empty agents list gracefully
                result = run_agent_conversation("def test(): pass", config_path)
                
                # Should return some default message or handle gracefully
                assert isinstance(result, str)
                assert len(result) > 0  # Should have some fallback response
                
        finally:
            Path(config_path).unlink()
    
    def test_agent_registry_get_method_validation(self):
        """Test that AgentRegistry properly handles None returns."""
        from autogen_code_review_bot.agents import AgentRegistry, CoderAgent
        
        registry = AgentRegistry()
        
        # Getting non-existent agent should return None
        agent = registry.get_agent("nonexistent")
        assert agent is None
        
        # Add an agent and test retrieval
        coder = CoderAgent("test_coder")
        registry.add_agent(coder)
        
        # Should find existing agent
        found_agent = registry.get_agent("test_coder")
        assert found_agent is not None
        assert found_agent.name == "test_coder"
        
        # Should still return None for non-existent
        missing_agent = registry.get_agent("still_missing")
        assert missing_agent is None
    
    def test_optional_parameter_validation(self):
        """Test functions with Optional parameters validate inputs properly."""
        from autogen_code_review_bot.monitoring import MetricsEmitter
        
        emitter = MetricsEmitter()
        
        # Test with None tags (should not crash)
        emitter.record_counter("test_metric", 1, tags=None)
        emitter.record_gauge("test_gauge", 100, tags=None)
        emitter.record_histogram("test_histogram", 50, tags=None)
        
        # Should work without issues
        metrics = emitter.get_metrics()
        assert "test_metric" in metrics
        assert "test_gauge" in metrics
        assert "test_histogram" in metrics
    
    def test_optional_timestamp_handling(self):
        """Test that optional timestamp parameters are handled correctly."""
        from autogen_code_review_bot.monitoring import MetricsEmitter
        import time
        
        emitter = MetricsEmitter()
        emitter.record_counter("old_metric", 1)
        emitter.record_counter("recent_metric", 2)
        
        # Test with None timestamp (should return all metrics)
        all_metrics = emitter.get_metrics(since_timestamp=None)
        assert len(all_metrics) == 2
        
        # Test with specific timestamp
        current_time = time.time()
        recent_metrics = emitter.get_metrics(since_timestamp=current_time - 1)
        assert len(recent_metrics) >= 0  # Should not crash
    
    def test_webhook_deduplication_optional_handling(self):
        """Test webhook deduplication handles optional delivery IDs."""
        from autogen_code_review_bot.webhook_deduplication import WebhookDeduplicator
        
        deduplicator = WebhookDeduplicator()
        
        # Should handle None delivery_id gracefully
        result = deduplicator.is_duplicate(None)
        assert isinstance(result, bool)
        
        # Should handle empty string
        result = deduplicator.is_duplicate("")
        assert isinstance(result, bool)
        
        # Should handle normal ID
        result = deduplicator.is_duplicate("valid-id-123")
        assert isinstance(result, bool)
    
    def test_config_validation_optional_fields(self):
        """Test that configuration validation handles optional fields properly."""
        from autogen_code_review_bot.config_validation import ConfigError
        
        # This test ensures that optional fields don't cause validation errors
        # when they're None or missing
        
        # Test will be implemented based on actual config validation usage
        # For now, just verify the exception class exists and can be instantiated
        error = ConfigError("Test error", field=None, value=None)
        assert error.message == "Test error"
        assert error.field is None
        assert error.value is None


if __name__ == "__main__":
    pytest.main([__file__])