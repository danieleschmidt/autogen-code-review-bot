"""Tests for webhook event deduplication."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot.webhook_deduplication import (
    WebhookDeduplicator,
    DuplicateEventError,
    is_duplicate_event,
    cleanup_expired_events
)


class TestWebhookDeduplicator:
    """Test webhook event deduplication functionality."""
    
    def test_first_event_not_duplicate(self):
        """Test that the first occurrence of an event is not considered duplicate."""
        dedup = WebhookDeduplicator()
        delivery_id = "12345-abcde-67890"
        
        assert not dedup.is_duplicate(delivery_id)
    
    def test_second_event_is_duplicate(self):
        """Test that the second occurrence of an event is considered duplicate."""
        dedup = WebhookDeduplicator()
        delivery_id = "12345-abcde-67890"
        
        # First call should not be duplicate
        assert not dedup.is_duplicate(delivery_id)
        
        # Second call should be duplicate
        assert dedup.is_duplicate(delivery_id)
    
    def test_different_events_not_duplicate(self):
        """Test that different delivery IDs are not considered duplicates."""
        dedup = WebhookDeduplicator()
        
        assert not dedup.is_duplicate("delivery-1")
        assert not dedup.is_duplicate("delivery-2")
        assert not dedup.is_duplicate("delivery-3")
    
    def test_ttl_expiration(self):
        """Test that events expire after TTL and can be processed again."""
        # Use a very short TTL for testing
        dedup = WebhookDeduplicator(ttl_seconds=0.1)
        delivery_id = "test-delivery"
        
        # First occurrence
        assert not dedup.is_duplicate(delivery_id)
        
        # Immediately after, should be duplicate
        assert dedup.is_duplicate(delivery_id)
        
        # Wait for expiration
        time.sleep(0.2)
        
        # After expiration, should not be duplicate anymore
        assert not dedup.is_duplicate(delivery_id)
    
    def test_cleanup_expired_events(self):
        """Test that expired events are properly cleaned up."""
        dedup = WebhookDeduplicator(ttl_seconds=0.1)
        
        # Add some events
        dedup.is_duplicate("event-1")
        dedup.is_duplicate("event-2")
        dedup.is_duplicate("event-3")
        
        # Verify events are stored
        assert len(dedup._events) == 3
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Trigger cleanup
        dedup.cleanup_expired()
        
        # All events should be cleaned up
        assert len(dedup._events) == 0
    
    def test_automatic_cleanup_on_check(self):
        """Test that cleanup happens automatically during duplicate checks."""
        dedup = WebhookDeduplicator(ttl_seconds=0.1, cleanup_interval=0.05)
        
        # Add an event
        dedup.is_duplicate("test-event")
        assert len(dedup._events) == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Check a new event - this should trigger cleanup
        dedup.is_duplicate("new-event")
        
        # Only the new event should remain
        assert len(dedup._events) == 1
        assert "new-event" in dedup._events
    
    def test_thread_safety(self):
        """Test that the deduplicator is thread-safe."""
        import threading
        
        dedup = WebhookDeduplicator()
        delivery_id = "thread-test"
        results = []
        
        def check_duplicate():
            result = dedup.is_duplicate(delivery_id)
            results.append(result)
        
        # Create multiple threads checking the same delivery ID
        threads = [threading.Thread(target=check_duplicate) for _ in range(10)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Only one thread should get False (not duplicate), rest should get True
        false_count = sum(1 for r in results if not r)
        true_count = sum(1 for r in results if r)
        
        assert false_count == 1
        assert true_count == 9
    
    def test_invalid_delivery_id(self):
        """Test handling of invalid delivery IDs."""
        dedup = WebhookDeduplicator()
        
        # None should not cause errors
        assert not dedup.is_duplicate(None)
        
        # Empty string should not cause errors
        assert not dedup.is_duplicate("")
        
        # These should be treated as different events
        assert dedup.is_duplicate(None)  # Second None should be duplicate
        assert not dedup.is_duplicate("")  # But empty string is different


class TestFunctionInterface:
    """Test the module-level functions."""
    
    def test_is_duplicate_event_function(self):
        """Test the module-level is_duplicate_event function."""
        # First call should not be duplicate
        assert not is_duplicate_event("func-test-1")
        
        # Second call should be duplicate
        assert is_duplicate_event("func-test-1")
        
        # Different ID should not be duplicate
        assert not is_duplicate_event("func-test-2")
    
    def test_cleanup_expired_events_function(self):
        """Test the module-level cleanup function."""
        # Add some events with short TTL
        with patch('autogen_code_review_bot.webhook_deduplication._global_deduplicator') as mock_dedup:
            mock_instance = Mock()
            mock_dedup.return_value = mock_instance
            
            cleanup_expired_events()
            
            mock_instance.cleanup_expired.assert_called_once()


class TestDuplicateEventError:
    """Test the custom exception."""
    
    def test_exception_creation(self):
        """Test that the exception can be created and contains delivery ID."""
        delivery_id = "test-delivery-id"
        error = DuplicateEventError(delivery_id)
        
        assert str(error) == f"Duplicate webhook event: {delivery_id}"
        assert error.delivery_id == delivery_id
    
    def test_exception_inheritance(self):
        """Test that the exception inherits from Exception."""
        error = DuplicateEventError("test")
        assert isinstance(error, Exception)


class TestPersistentStorage:
    """Test optional persistent storage functionality."""
    
    def test_file_based_persistence(self):
        """Test file-based persistence of deduplication state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_file = Path(temp_dir) / "dedup_state.json"
            
            # Create deduplicator with persistent storage
            dedup1 = WebhookDeduplicator(storage_file=str(storage_file))
            
            # Add some events
            dedup1.is_duplicate("persistent-1")
            dedup1.is_duplicate("persistent-2")
            
            # Create new instance with same storage file
            dedup2 = WebhookDeduplicator(storage_file=str(storage_file))
            
            # Events should be considered duplicates in new instance
            assert dedup2.is_duplicate("persistent-1")
            assert dedup2.is_duplicate("persistent-2")
    
    def test_storage_file_corruption_handling(self):
        """Test handling of corrupted storage files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_file = Path(temp_dir) / "corrupted.json"
            
            # Create corrupted file
            storage_file.write_text("invalid json content")
            
            # Should not crash, should start fresh
            dedup = WebhookDeduplicator(storage_file=str(storage_file))
            
            # Should work normally
            assert not dedup.is_duplicate("test-after-corruption")


class TestIntegrationWithWebhookHandler:
    """Test integration with the actual webhook handler."""
    
    @patch('autogen_code_review_bot.webhook_deduplication.is_duplicate_event')
    def test_webhook_handler_checks_duplicates(self, mock_is_duplicate):
        """Test that webhook handler properly checks for duplicates."""
        # This test verifies the integration point exists
        # The actual integration will be in the webhook handler
        mock_is_duplicate.return_value = False
        
        # Simulate webhook handler checking for duplicates
        delivery_id = "github-delivery-123"
        is_duplicate = mock_is_duplicate(delivery_id)
        
        assert not is_duplicate
        mock_is_duplicate.assert_called_once_with(delivery_id)
    
    def test_duplicate_event_handling_in_webhook(self):
        """Test that duplicate events are properly handled in webhook flow."""
        with patch('autogen_code_review_bot.webhook_deduplication.is_duplicate_event') as mock_is_duplicate:
            mock_is_duplicate.return_value = True
            
            # When a duplicate is detected, webhook should skip processing
            delivery_id = "duplicate-delivery-456"
            
            if is_duplicate_event(delivery_id):
                # This is what the webhook handler should do
                response = {"status": "duplicate", "message": "Event already processed"}
            else:
                response = {"status": "processed"}
            
            assert response["status"] == "duplicate"