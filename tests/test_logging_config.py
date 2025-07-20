"""Tests for structured logging configuration."""

import json
import logging
import time
from io import StringIO
from unittest.mock import patch

import pytest

from autogen_code_review_bot.logging_config import (
    StructuredFormatter,
    configure_logging,
    get_logger,
    set_request_id,
    get_request_id,
    log_operation_start,
    log_operation_end,
    ContextLogger,
    request_id,
)


class TestStructuredFormatter:
    """Test the JSON log formatter."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter("test-service")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["level"] == "info"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert log_data["service"] == "test-service"
        assert "timestamp" in log_data
    
    def test_request_id_inclusion(self):
        """Test that request ID is included when set."""
        formatter = StructuredFormatter()
        test_id = "test-request-123"
        
        # Set request ID in context
        set_request_id(test_id)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["request_id"] == test_id
        
        # Clean up context
        request_id.set(None)
    
    def test_extra_fields(self):
        """Test that extra fields are included in log output."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        # Add extra fields
        record.operation = "test_operation"
        record.user_id = "user123"
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data["operation"] == "test_operation"
        assert log_data["user_id"] == "user123"
    
    def test_exception_formatting(self):
        """Test exception information is formatted correctly."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
        
        result = formatter.format(record)
        log_data = json.loads(result)
        
        assert "exception" in log_data
        assert "ValueError: Test exception" in log_data["exception"]


class TestLoggingConfiguration:
    """Test logging configuration functions."""
    
    def test_configure_logging(self):
        """Test that logging is configured with structured formatter."""
        configure_logging(level="DEBUG", service_name="test-service")
        
        logger = logging.getLogger("test")
        
        # Capture log output
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            logger.info("Test message")
            output = mock_stderr.getvalue()
        
        # Parse the JSON output
        log_data = json.loads(output.strip())
        assert log_data["message"] == "Test message"
        assert log_data["service"] == "test-service"
    
    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"


class TestRequestContext:
    """Test request ID context management."""
    
    def test_set_and_get_request_id(self):
        """Test setting and getting request ID."""
        test_id = "test-request-456"
        result_id = set_request_id(test_id)
        
        assert result_id == test_id
        assert get_request_id() == test_id
        
        # Clean up
        request_id.set(None)
    
    def test_auto_generate_request_id(self):
        """Test automatic request ID generation."""
        result_id = set_request_id()
        
        assert result_id is not None
        assert len(result_id) > 0
        assert get_request_id() == result_id
        
        # Clean up
        request_id.set(None)
    
    def test_request_id_isolation(self):
        """Test that request IDs are isolated between contexts."""
        # This test simulates context isolation
        # In real usage, each request would have its own context
        
        # Initial state
        assert get_request_id() is None
        
        # Set and verify
        set_request_id("request-1")
        assert get_request_id() == "request-1"
        
        # Reset and verify
        request_id.set(None)
        assert get_request_id() is None


class TestOperationLogging:
    """Test operation start/end logging utilities."""
    
    def test_log_operation_lifecycle(self):
        """Test complete operation logging lifecycle."""
        logger = get_logger("test.operations")
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            # Start operation
            context = log_operation_start(
                logger, 
                "test_operation", 
                repo="test/repo",
                pr_number=123
            )
            
            # Simulate some work
            time.sleep(0.01)
            
            # End operation successfully
            log_operation_end(
                logger, 
                context, 
                success=True,
                files_analyzed=5
            )
            
            output = mock_stderr.getvalue()
        
        # Parse log entries
        lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
        assert len(lines) == 2
        
        start_log = json.loads(lines[0])
        end_log = json.loads(lines[1])
        
        # Verify start log
        assert start_log["message"] == "Operation started"
        assert start_log["operation"] == "test_operation"
        assert start_log["repo"] == "test/repo"
        assert start_log["pr_number"] == 123
        assert "start_time" in start_log
        
        # Verify end log
        assert end_log["message"] == "Operation completed"
        assert end_log["operation"] == "test_operation"
        assert end_log["success"] is True
        assert end_log["files_analyzed"] == 5
        assert end_log["duration_seconds"] > 0
    
    def test_log_operation_failure(self):
        """Test operation failure logging."""
        logger = get_logger("test.operations")
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            context = log_operation_start(logger, "failing_operation")
            log_operation_end(
                logger, 
                context, 
                success=False, 
                error="Something went wrong"
            )
            
            output = mock_stderr.getvalue()
        
        lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
        end_log = json.loads(lines[1])
        
        assert end_log["message"] == "Operation failed"
        assert end_log["success"] is False
        assert end_log["error"] == "Something went wrong"


class TestContextLogger:
    """Test the context-aware logger wrapper."""
    
    def test_context_inclusion(self):
        """Test that context is included in all log messages."""
        base_logger = get_logger("test.context")
        context_logger = ContextLogger(
            base_logger,
            service="test-service",
            version="1.0.0"
        )
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            context_logger.info("Test message", additional_field="extra_value")
            output = mock_stderr.getvalue()
        
        log_data = json.loads(output.strip())
        
        assert log_data["message"] == "Test message"
        assert log_data["service"] == "test-service"
        assert log_data["version"] == "1.0.0"
        assert log_data["additional_field"] == "extra_value"
    
    def test_different_log_levels(self):
        """Test that context logger supports all log levels."""
        base_logger = get_logger("test.levels")
        context_logger = ContextLogger(base_logger, component="test")
        
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            context_logger.debug("Debug message")
            context_logger.info("Info message")
            context_logger.warning("Warning message")
            context_logger.error("Error message")
            context_logger.critical("Critical message")
            
            output = mock_stderr.getvalue()
        
        # Should have log entries (number depends on log level configured)
        lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
        
        # All should have the context
        for line in lines:
            log_data = json.loads(line)
            assert log_data["component"] == "test"