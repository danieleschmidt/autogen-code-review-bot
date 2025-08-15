"""Webhook event deduplication to prevent duplicate processing."""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .logging_config import get_logger
from .system_config import get_system_config

logger = get_logger(__name__)


class DuplicateEventError(Exception):
    """Exception raised when a duplicate webhook event is detected."""

    def __init__(self, delivery_id: str):
        self.delivery_id = delivery_id
        super().__init__(f"Duplicate webhook event: {delivery_id}")


@dataclass
class EventRecord:
    """Record of a processed webhook event."""

    delivery_id: str
    timestamp: float
    processed_at: float = None

    def __post_init__(self):
        if self.processed_at is None:
            self.processed_at = time.time()

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this event record has expired."""
        return (time.time() - self.processed_at) > ttl_seconds


class WebhookDeduplicator:
    """Thread-safe webhook event deduplicator."""

    def __init__(
        self,
        ttl_seconds: Optional[float] = None,
        cleanup_interval: Optional[float] = None,
        storage_file: Optional[str] = None,
    ):
        """Initialize the deduplicator.

        Args:
            ttl_seconds: Time to live for event records in seconds (uses system config if None)
            cleanup_interval: How often to run cleanup in seconds (uses system config if None)
            storage_file: Optional file path for persistent storage
        """
        config = get_system_config()
        self.ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else config.webhook_deduplication_ttl
        )
        cleanup_interval = (
            cleanup_interval
            if cleanup_interval is not None
            else config.cache_cleanup_interval
        )
        self.cleanup_interval = cleanup_interval
        self.storage_file = storage_file
        self._events: Dict[str, EventRecord] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Load from persistent storage if available
        if self.storage_file:
            self._load_from_storage()

    def is_duplicate(self, delivery_id: Optional[str]) -> bool:
        """Check if an event is a duplicate and mark it as processed.

        Args:
            delivery_id: GitHub delivery ID from X-GitHub-Delivery header

        Returns:
            True if this is a duplicate event, False if it's the first occurrence
        """
        if not delivery_id:
            # Treat None/empty as a special case - still track them
            delivery_id = str(delivery_id)  # Convert None to "None", "" stays ""

        with self._lock:
            # Periodic cleanup
            if (time.time() - self._last_cleanup) > self.cleanup_interval:
                self._cleanup_expired_unlocked()

            # Check if we've seen this delivery ID before
            if delivery_id in self._events:
                existing = self._events[delivery_id]
                if not existing.is_expired(self.ttl_seconds):
                    logger.info(
                        "Duplicate webhook event detected",
                        extra={
                            "delivery_id": delivery_id,
                            "original_timestamp": existing.timestamp,
                        },
                    )
                    return True
                else:
                    # Expired, remove it and treat as new
                    del self._events[delivery_id]

            # Record this event
            self._events[delivery_id] = EventRecord(
                delivery_id=delivery_id, timestamp=time.time()
            )

            # Save to persistent storage if configured
            if self.storage_file:
                self._save_to_storage()

            logger.debug(
                "New webhook event recorded",
                extra={"delivery_id": delivery_id, "total_tracked": len(self._events)},
            )
            return False

    def cleanup_expired(self) -> int:
        """Remove expired event records.

        Returns:
            Number of records removed
        """
        with self._lock:
            return self._cleanup_expired_unlocked()

    def _cleanup_expired_unlocked(self) -> int:
        """Internal cleanup method that assumes lock is held."""
        current_time = time.time()
        expired_keys = [
            delivery_id
            for delivery_id, record in self._events.items()
            if record.is_expired(self.ttl_seconds)
        ]

        for key in expired_keys:
            del self._events[key]

        self._last_cleanup = current_time

        if expired_keys:
            logger.debug(
                "Cleaned up expired webhook events",
                extra={
                    "expired_count": len(expired_keys),
                    "remaining_count": len(self._events),
                },
            )

        return len(expired_keys)

    def _load_from_storage(self) -> None:
        """Load event records from persistent storage."""
        if not self.storage_file:
            return

        try:
            storage_path = Path(self.storage_file)
            if not storage_path.exists():
                return

            with open(storage_path) as f:
                data = json.load(f)

            # Convert back to EventRecord objects
            current_time = time.time()
            loaded_count = 0

            for delivery_id, record_data in data.items():
                record = EventRecord(
                    delivery_id=record_data["delivery_id"],
                    timestamp=record_data["timestamp"],
                    processed_at=record_data["processed_at"],
                )

                # Only load if not expired
                if not record.is_expired(self.ttl_seconds):
                    self._events[delivery_id] = record
                    loaded_count += 1

            logger.info(
                "Loaded webhook deduplication state from storage",
                extra={"storage_file": self.storage_file, "loaded_count": loaded_count},
            )

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(
                "Failed to load deduplication state, starting fresh",
                extra={"storage_file": self.storage_file, "error": str(e)},
            )
            self._events = {}

    def _save_to_storage(self) -> None:
        """Save current event records to persistent storage."""
        if not self.storage_file:
            return

        try:
            # Convert EventRecord objects to serializable dict
            data = {
                delivery_id: {
                    "delivery_id": record.delivery_id,
                    "timestamp": record.timestamp,
                    "processed_at": record.processed_at,
                }
                for delivery_id, record in self._events.items()
            }

            # Ensure directory exists
            storage_path = Path(self.storage_file)
            storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically using temporary file
            temp_file = storage_path.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic move
            temp_file.rename(storage_path)

        except OSError as e:
            logger.error(
                "Failed to save deduplication state",
                extra={"storage_file": self.storage_file, "error": str(e)},
            )


# Global deduplicator instance for easy module-level access
_global_deduplicator: Optional[WebhookDeduplicator] = None
_global_lock = threading.Lock()


def get_global_deduplicator() -> WebhookDeduplicator:
    """Get or create the global deduplicator instance."""
    global _global_deduplicator

    if _global_deduplicator is None:
        with _global_lock:
            if _global_deduplicator is None:
                # Use persistent storage in production
                storage_file = (
                    Path.home() / ".cache" / "autogen-review" / "webhook_dedup.json"
                )
                _global_deduplicator = WebhookDeduplicator(
                    ttl_seconds=3600,  # 1 hour TTL
                    cleanup_interval=300,  # 5 minute cleanup interval
                    storage_file=str(storage_file),
                )

    return _global_deduplicator


def is_duplicate_event(delivery_id: Optional[str]) -> bool:
    """Check if a webhook event is a duplicate using the global deduplicator.

    Args:
        delivery_id: GitHub delivery ID from X-GitHub-Delivery header

    Returns:
        True if this is a duplicate event, False if it's the first occurrence
    """
    return get_global_deduplicator().is_duplicate(delivery_id)


def cleanup_expired_events() -> int:
    """Clean up expired events from the global deduplicator.

    Returns:
        Number of expired events removed
    """
    return get_global_deduplicator().cleanup_expired()


def reset_global_deduplicator() -> None:
    """Reset the global deduplicator (mainly for testing)."""
    global _global_deduplicator
    with _global_lock:
        _global_deduplicator = None
