#!/usr/bin/env python3
"""
Comprehensive test suite for AutoGen Code Review Bot enterprise features.

Tests all Generation 1, 2, and 3 implementations including API gateway,
real-time collaboration, distributed processing, and intelligent caching.
"""

import pytest
import asyncio
import json
import time
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

import aiohttp
import websockets
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

# Import our modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autogen_code_review_bot.api_gateway import (
    create_api_app, AuthenticationManager, RateLimitManager, APIUser
)
from autogen_code_review_bot.real_time_collaboration import (
    RealTimeCollaborationManager, CollaborationSession, WebSocketHandler
)
from autogen_code_review_bot.distributed_processing import (
    DistributedTaskManager, DistributedTask, WorkerNode, TaskPriority, TaskStatus
)
from autogen_code_review_bot.intelligent_cache import (
    AdaptiveLRU, DistributedCache, PredictiveCache, CacheWarmer
)
from autogen_code_review_bot.resilience import (
    ResilienceOrchestrator, RetryManager, BulkheadManager, HealthMonitor
)
from autogen_code_review_bot.validation import (
    InputValidator, SchemaValidator, SecurityValidator
)
from autogen_code_review_bot.pr_analysis import analyze_pr, LinterConfig


class TestAPIGateway:
    """Test suite for the Enterprise API Gateway."""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        return create_api_app({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key'
        })
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    @pytest.fixture
    def auth_manager(self):
        """Create test authentication manager."""
        return AuthenticationManager('test-secret-key')
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['service'] == 'autogen-code-review-api'
    
    def test_authentication_manager_token_generation(self, auth_manager):
        """Test JWT token generation."""
        token = auth_manager.generate_token('enterprise_user_1')
        assert token is not None
        assert isinstance(token, str)
        
        # Test invalid user
        invalid_token = auth_manager.generate_token('nonexistent_user')
        assert invalid_token is None
    
    def test_authentication_manager_token_validation(self, auth_manager):
        """Test JWT token validation."""
        # Generate valid token
        token = auth_manager.generate_token('enterprise_user_1')
        user = auth_manager.authenticate_request(token)
        
        assert user is not None
        assert user.user_id == 'enterprise_user_1'
        assert user.email == 'admin@company.com'
        
        # Test invalid token
        invalid_user = auth_manager.authenticate_request('invalid-token')
        assert invalid_user is None
    
    def test_rate_limit_manager(self):
        """Test rate limiting functionality."""
        rate_manager = RateLimitManager()
        
        # Create test user
        user = APIUser(
            user_id='test_user',
            email='test@test.com',
            organization='Test Org',
            permissions=['read'],
            daily_quota=10,
            monthly_quota=300,
            created_at=datetime.now(timezone.utc)
        )
        
        # Test within limits
        assert rate_manager.check_rate_limit(user) is True
        
        # Record usage up to limit
        for i in range(10):
            rate_manager.record_usage(user)
        
        # Test exceeded limit
        assert rate_manager.check_rate_limit(user) is False
    
    def test_token_endpoint(self, client):
        """Test token generation endpoint."""
        response = client.post('/api/v1/auth/token', 
                             json={'user_id': 'enterprise_user_1'})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'token' in data
        assert 'expires_at' in data
        assert 'user' in data
        assert data['user']['user_id'] == 'enterprise_user_1'
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get('/api/v1/user/profile')
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token."""
        response = client.get('/api/v1/user/profile',
                            headers={'Authorization': 'Bearer invalid-token'})
        assert response.status_code == 401
    
    @patch('autogen_code_review_bot.api_gateway.analyze_pr')
    def test_analyze_repository_endpoint(self, mock_analyze_pr, client):
        """Test repository analysis endpoint."""
        # Mock the analysis result
        from autogen_code_review_bot.models import PRAnalysisResult, AnalysisSection
        
        mock_result = PRAnalysisResult(
            security=AnalysisSection(tool="test", output="secure"),
            style=AnalysisSection(tool="test", output="good style"),
            performance=AnalysisSection(tool="test", output="optimized"),
            metadata={"test": True}
        )
        mock_analyze_pr.return_value = mock_result
        
        # Get token first
        token_response = client.post('/api/v1/auth/token', 
                                   json={'user_id': 'enterprise_user_1'})
        token = json.loads(token_response.data)['token']
        
        # Test repository analysis
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = client.post('/api/v1/analyze/repository',
                                 headers={'Authorization': f'Bearer {token}'},
                                 json={'repository_path': tmp_dir})
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['status'] == 'completed'
            assert 'analysis_result' in data
            assert data['analysis_result']['security']['tool'] == 'test'


class TestRealTimeCollaboration:
    """Test suite for real-time collaboration features."""
    
    @pytest.fixture
    def redis_mock(self):
        """Create mock Redis instance."""
        return AsyncMock()
    
    @pytest.fixture
    def collaboration_manager(self, redis_mock):
        """Create collaboration manager with mocked Redis."""
        manager = RealTimeCollaborationManager()
        manager.redis = redis_mock
        return manager
    
    @pytest.mark.asyncio
    async def test_create_session(self, collaboration_manager):
        """Test collaboration session creation."""
        session_id = await collaboration_manager.create_session(
            repository="test/repo",
            pr_number=123,
            creator_id="user1"
        )
        
        assert session_id is not None
        assert session_id in collaboration_manager.sessions
        
        session = collaboration_manager.sessions[session_id]
        assert session.repository == "test/repo"
        assert session.pr_number == 123
        assert "user1" in session.participants
    
    @pytest.mark.asyncio
    async def test_join_session(self, collaboration_manager):
        """Test joining a collaboration session."""
        # Create session first
        session_id = await collaboration_manager.create_session("test/repo")
        
        # Mock WebSocket
        websocket_mock = AsyncMock()
        websocket_mock.send = AsyncMock()
        
        # Join session
        success = await collaboration_manager.join_session(
            session_id, "user2", websocket_mock
        )
        
        assert success is True
        assert "user2" in collaboration_manager.sessions[session_id].participants
    
    @pytest.mark.asyncio
    async def test_agent_message_handling(self, collaboration_manager):
        """Test agent message processing."""
        # Create session and join
        session_id = await collaboration_manager.create_session("test/repo")
        websocket_mock = AsyncMock()
        await collaboration_manager.join_session(session_id, "user1", websocket_mock)
        
        # Find connection ID
        connection_id = None
        for cid, conn in collaboration_manager.connections.items():
            if conn.websocket == websocket_mock:
                connection_id = cid
                break
        
        assert connection_id is not None
        
        # Handle agent message
        message = {
            'agent_type': 'coder',
            'message': 'Test message'
        }
        
        await collaboration_manager.handle_agent_message(connection_id, message)
        
        # Check that message was added to conversation history
        session = collaboration_manager.sessions[session_id]
        assert len(session.conversation_history) >= 1
    
    @pytest.mark.asyncio
    async def test_websocket_handler(self, collaboration_manager):
        """Test WebSocket message handling."""
        handler = WebSocketHandler(collaboration_manager)
        websocket_mock = AsyncMock()
        
        # Mock receiving JOIN_SESSION message
        websocket_mock.__aiter__.return_value = [
            json.dumps({
                'type': 'join_session',
                'session_id': 'test-session',
                'user_id': 'test-user'
            })
        ]
        
        # Mock session creation
        collaboration_manager.sessions['test-session'] = CollaborationSession(
            session_id='test-session',
            repository='test/repo',
            pr_number=None,
            created_at=datetime.now(timezone.utc),
            participants=set()
        )
        
        # Handle connection (should not raise exception)
        try:
            await handler.handle_connection(websocket_mock, '/ws')
        except StopAsyncIteration:
            pass  # Expected when mock iterator ends


class TestDistributedProcessing:
    """Test suite for distributed processing system."""
    
    @pytest.fixture
    def redis_mock(self):
        """Create mock Redis instance."""
        mock = AsyncMock()
        mock.keys.return_value = []
        mock.get.return_value = None
        mock.set.return_value = True
        mock.zadd.return_value = True
        mock.zpopmax.return_value = []
        mock.setex.return_value = True
        mock.zcard.return_value = 0
        return mock
    
    @pytest.fixture
    def task_manager(self, redis_mock):
        """Create distributed task manager."""
        manager = DistributedTaskManager(node_id="test-node", region="test")
        manager.redis = redis_mock
        return manager
    
    def test_distributed_task_creation(self):
        """Test DistributedTask creation and serialization."""
        task = DistributedTask(
            task_id="test-123",
            task_type="analysis",
            payload={"repo_path": "/test"},
            priority=TaskPriority.HIGH
        )
        
        assert task.task_id == "test-123"
        assert task.task_type == "analysis"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        
        # Test serialization
        task_dict = task.to_dict()
        assert task_dict['task_id'] == "test-123"
        assert task_dict['priority'] == TaskPriority.HIGH.value
        
        # Test deserialization
        restored_task = DistributedTask.from_dict(task_dict)
        assert restored_task.task_id == task.task_id
        assert restored_task.priority == task.priority
    
    def test_worker_node_creation(self):
        """Test WorkerNode creation and capabilities."""
        worker = WorkerNode(
            node_id="worker-1",
            hostname="host1",
            region="us-east-1",
            capabilities=["analysis", "security"],
            max_concurrent_tasks=10
        )
        
        assert worker.node_id == "worker-1"
        assert worker.get_load_percentage() == 0.0
        assert worker.can_accept_task() is True
        
        # Test load calculation
        worker.current_load = 5
        assert worker.get_load_percentage() == 50.0
        
        # Test overloaded
        worker.current_load = 10
        assert worker.can_accept_task() is False
    
    @pytest.mark.asyncio
    async def test_task_submission(self, task_manager):
        """Test task submission."""
        task_id = await task_manager.submit_task(
            task_type="test_task",
            payload={"data": "test"},
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Verify Redis calls
        task_manager.redis.set.assert_called()
        task_manager.redis.zadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_worker_registration(self, task_manager):
        """Test worker node registration."""
        await task_manager.start_worker(
            capabilities=["test"],
            max_concurrent_tasks=5
        )
        
        assert task_manager.worker_node is not None
        assert task_manager.worker_node.node_id == "test-node"
        assert task_manager.worker_node.capabilities == ["test"]
        assert task_manager.worker_node.max_concurrent_tasks == 5
        
        # Verify Redis registration
        task_manager.redis.set.assert_called()
    
    def test_task_handler_registration(self, task_manager):
        """Test task handler registration."""
        def test_handler(payload):
            return {"result": "success"}
        
        task_manager.register_task_handler("test_type", test_handler)
        assert "test_type" in task_manager.task_handlers
        assert task_manager.task_handlers["test_type"] == test_handler


class TestIntelligentCache:
    """Test suite for intelligent caching system."""
    
    @pytest.fixture
    def adaptive_lru(self):
        """Create AdaptiveLRU cache."""
        return AdaptiveLRU(max_size=100, max_memory_mb=1)
    
    @pytest.fixture
    def redis_mock(self):
        """Create mock Redis instance."""
        return AsyncMock()
    
    @pytest.fixture
    def distributed_cache(self, redis_mock):
        """Create distributed cache."""
        return DistributedCache(redis_mock)
    
    def test_adaptive_lru_basic_operations(self, adaptive_lru):
        """Test basic LRU cache operations."""
        # Test put and get
        success = adaptive_lru.put("key1", "value1", ttl_seconds=60)
        assert success is True
        
        value = adaptive_lru.get("key1")
        assert value == "value1"
        
        # Test miss
        value = adaptive_lru.get("nonexistent")
        assert value is None
        
        # Test stats
        stats = adaptive_lru.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entry_count == 1
        assert stats.hit_rate == 50.0
    
    def test_adaptive_lru_eviction(self, adaptive_lru):
        """Test LRU eviction behavior."""
        # Fill cache to capacity
        for i in range(adaptive_lru.max_size + 10):
            adaptive_lru.put(f"key{i}", f"value{i}")
        
        # Should have triggered evictions
        stats = adaptive_lru.get_stats()
        assert stats.entry_count <= adaptive_lru.max_size
        assert stats.evictions > 0
        
        # Oldest entries should be evicted
        assert adaptive_lru.get("key0") is None
        assert adaptive_lru.get("key1") is None
    
    def test_adaptive_lru_ttl_expiration(self, adaptive_lru):
        """Test TTL-based expiration."""
        # Put with very short TTL
        adaptive_lru.put("temp_key", "temp_value", ttl_seconds=0)
        
        # Should be expired immediately
        time.sleep(0.1)
        value = adaptive_lru.get("temp_key")
        assert value is None
    
    def test_cache_key_info(self, adaptive_lru):
        """Test cache key information retrieval."""
        adaptive_lru.put("info_key", "info_value", ttl_seconds=60, 
                        tags={"tag1", "tag2"})
        
        info = adaptive_lru.get_key_info("info_key")
        assert info is not None
        assert info['key'] == "info_key"
        assert info['size_bytes'] > 0
        assert info['access_count'] == 0
        assert "tag1" in info['tags']
        assert "tag2" in info['tags']
    
    def test_cache_tag_invalidation(self, adaptive_lru):
        """Test tag-based cache invalidation."""
        # Add entries with tags
        adaptive_lru.put("key1", "value1", tags={"group1", "common"})
        adaptive_lru.put("key2", "value2", tags={"group2", "common"})
        adaptive_lru.put("key3", "value3", tags={"group1"})
        
        # Invalidate by tag
        cleared = adaptive_lru.clear_by_tags({"group1"})
        assert cleared == 2  # key1 and key3
        
        # Verify correct entries were cleared
        assert adaptive_lru.get("key1") is None
        assert adaptive_lru.get("key2") == "value2"
        assert adaptive_lru.get("key3") is None
    
    @pytest.mark.asyncio
    async def test_distributed_cache_operations(self, distributed_cache, redis_mock):
        """Test distributed cache operations."""
        # Mock Redis responses
        redis_mock.get.return_value = json.dumps("cached_value")
        redis_mock.setex.return_value = True
        
        # Test get (cache hit in Redis)
        value = await distributed_cache.get("test_key")
        assert value == "cached_value"
        
        # Test put
        success = await distributed_cache.put("new_key", "new_value", ttl_seconds=300)
        assert success is True
        
        # Verify Redis calls
        redis_mock.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_distributed_cache_l1_l2_behavior(self, distributed_cache, redis_mock):
        """Test L1/L2 cache behavior."""
        # Mock Redis miss
        redis_mock.get.return_value = None
        
        # Should miss both L1 and L2
        value = await distributed_cache.get("missing_key")
        assert value is None
        
        # Put in cache
        await distributed_cache.put("test_key", "test_value")
        
        # Should hit L1 cache now
        value = distributed_cache.local_cache.get("test_key")
        assert value == "test_value"
    
    def test_predictive_cache_access_logging(self, distributed_cache):
        """Test predictive cache access pattern logging."""
        predictive_cache = PredictiveCache(distributed_cache)
        
        # Log some accesses
        predictive_cache.log_access("key1", "context1")
        predictive_cache.log_access("key2", "context1")
        predictive_cache.log_access("key1", "context2")
        
        # Check patterns
        patterns = predictive_cache.get_access_patterns()
        assert patterns['total_accesses'] == 3
        assert patterns['unique_keys'] == 2
        assert len(patterns['most_accessed_keys']) == 2
    
    def test_cache_warmer_strategy_registration(self, distributed_cache):
        """Test cache warmer strategy registration."""
        warmer = CacheWarmer(distributed_cache)
        
        def sample_strategy():
            return {"key1": "value1", "key2": "value2"}
        
        warmer.register_warming_strategy(
            name="test_strategy",
            loader_func=sample_strategy,
            schedule={"interval_seconds": 60}
        )
        
        assert "test_strategy" in warmer.warming_strategies
        assert "test_strategy" in warmer.warming_schedule
        assert warmer.warming_schedule["test_strategy"]["interval_seconds"] == 60


class TestResilience:
    """Test suite for resilience patterns."""
    
    @pytest.fixture
    def resilience_orchestrator(self):
        """Create resilience orchestrator."""
        return ResilienceOrchestrator()
    
    def test_retry_manager_success_on_first_attempt(self, resilience_orchestrator):
        """Test successful operation on first attempt."""
        retry_manager = resilience_orchestrator.retry_manager
        
        def always_succeeds():
            return "success"
        
        from autogen_code_review_bot.resilience import RetryConfig
        config = RetryConfig(max_attempts=3)
        
        decorated_func = retry_manager.retry(config)(always_succeeds)
        result = decorated_func()
        
        assert result == "success"
    
    def test_retry_manager_success_after_failures(self, resilience_orchestrator):
        """Test successful operation after some failures."""
        retry_manager = resilience_orchestrator.retry_manager
        
        attempt_count = 0
        def succeeds_on_third():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        from autogen_code_review_bot.resilience import RetryConfig, RetryStrategy
        config = RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=0.01  # Fast for testing
        )
        
        decorated_func = retry_manager.retry(config)(succeeds_on_third)
        result = decorated_func()
        
        assert result == "success"
        assert attempt_count == 3
    
    def test_retry_manager_max_attempts_exceeded(self, resilience_orchestrator):
        """Test max attempts exceeded."""
        retry_manager = resilience_orchestrator.retry_manager
        
        def always_fails():
            raise ConnectionError("Persistent failure")
        
        from autogen_code_review_bot.resilience import RetryConfig
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        
        decorated_func = retry_manager.retry(config)(always_fails)
        
        with pytest.raises(ConnectionError):
            decorated_func()
    
    @pytest.mark.asyncio
    async def test_bulkhead_manager_resource_isolation(self, resilience_orchestrator):
        """Test bulkhead resource isolation."""
        from autogen_code_review_bot.resilience import BulkheadConfig
        
        bulkhead_config = BulkheadConfig(max_concurrent_requests=2, timeout_seconds=1.0)
        bulkhead_manager = BulkheadManager(bulkhead_config)
        
        # Test normal acquisition
        async with bulkhead_manager.acquire_resource("test_bulkhead", priority=1):
            # Should succeed
            pass
        
        # Test concurrent limit
        async def long_running_task():
            async with bulkhead_manager.acquire_resource("test_bulkhead"):
                await asyncio.sleep(0.5)
                return "done"
        
        # Start tasks that will exceed bulkhead capacity
        tasks = [long_running_task() for _ in range(3)]
        
        # Should complete without timeout errors for first 2 tasks
        results = await asyncio.gather(*tasks[:2], return_exceptions=True)
        assert all(result == "done" for result in results)
    
    def test_health_monitor_registration(self, resilience_orchestrator):
        """Test health check registration."""
        health_monitor = resilience_orchestrator.health_monitor
        
        def sample_health_check():
            return {"healthy": True, "message": "OK"}
        
        health_monitor.register_health_check(
            name="test_check",
            check_func=sample_health_check,
            interval_seconds=30
        )
        
        assert "test_check" in health_monitor.health_checks
        assert "test_check" in health_monitor.health_status
        assert health_monitor.health_status["test_check"]["status"] == "unknown"
    
    def test_circuit_breaker_integration(self, resilience_orchestrator):
        """Test circuit breaker integration."""
        circuit_breaker = resilience_orchestrator.get_circuit_breaker("test_service")
        assert circuit_breaker is not None
        
        # Test successful call
        def working_service():
            return "success"
        
        result = circuit_breaker.call(working_service)
        assert result == "success"
        
        stats = circuit_breaker.get_stats()
        assert stats["success_count"] == 1
        assert stats["state"] == "CLOSED"


class TestValidation:
    """Test suite for input validation and security."""
    
    @pytest.fixture
    def input_validator(self):
        """Create input validator."""
        return InputValidator()
    
    @pytest.fixture
    def schema_validator(self, input_validator):
        """Create schema validator."""
        return SchemaValidator(input_validator)
    
    @pytest.fixture
    def security_validator(self):
        """Create security validator."""
        return SecurityValidator()
    
    def test_string_validation_success(self, input_validator):
        """Test successful string validation."""
        result = input_validator.validate("Hello World", "string")
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert len(result['warnings']) == 0
    
    def test_string_validation_sql_injection_detection(self, input_validator):
        """Test SQL injection detection."""
        malicious_input = "'; DROP TABLE users; --"
        result = input_validator.validate(malicious_input, "string")
        
        assert result['valid'] is False
        assert any("SQL injection" in error for error in result['errors'])
    
    def test_string_validation_xss_detection(self, input_validator):
        """Test XSS detection."""
        malicious_input = "<script>alert('xss')</script>"
        result = input_validator.validate(malicious_input, "string")
        
        assert result['valid'] is False
        assert any("XSS" in error for error in result['errors'])
    
    def test_path_validation_traversal_detection(self, input_validator):
        """Test path traversal detection."""
        malicious_path = "../../../etc/passwd"
        result = input_validator.validate(malicious_path, "path")
        
        assert result['valid'] is False
        assert any("traversal" in error for error in result['errors'])
    
    def test_email_validation(self, input_validator):
        """Test email format validation."""
        # Valid email
        result = input_validator.validate("user@example.com", "email")
        assert result['valid'] is True
        
        # Invalid email
        result = input_validator.validate("invalid-email", "email")
        assert result['valid'] is False
        assert any("email" in error for error in result['errors'])
    
    def test_json_validation(self, input_validator):
        """Test JSON validation."""
        # Valid JSON
        valid_json = '{"key": "value", "number": 123}'
        result = input_validator.validate(valid_json, "json")
        assert result['valid'] is True
        
        # Invalid JSON
        invalid_json = '{"key": "value", "number":}'
        result = input_validator.validate(invalid_json, "json")
        assert result['valid'] is False
    
    def test_sanitization(self, input_validator):
        """Test input sanitization."""
        # HTML sanitization
        html_input = '<script>alert("test")</script>Hello'
        sanitized = input_validator.sanitize(html_input, "string")
        assert '<script>' not in sanitized
        assert '&lt;script&gt;' in sanitized
        
        # Whitespace trimming
        whitespace_input = "  hello world  "
        sanitized = input_validator.sanitize(whitespace_input, "string")
        assert sanitized == "hello world"
    
    def test_schema_validation_success(self, schema_validator):
        """Test successful schema validation."""
        schema = {
            "required": ["name", "email"],
            "fields": {
                "name": {"type": "string", "min_length": 1, "max_length": 100},
                "email": {"type": "email"},
                "age": {"type": "string", "pattern": r"^\d+$"}
            }
        }
        
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "age": "30"
        }
        
        result = schema_validator.validate_schema(data, schema)
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_schema_validation_missing_required_field(self, schema_validator):
        """Test schema validation with missing required field."""
        schema = {
            "required": ["name", "email"],
            "fields": {
                "name": {"type": "string"},
                "email": {"type": "email"}
            }
        }
        
        data = {"name": "John Doe"}  # Missing email
        
        result = schema_validator.validate_schema(data, schema)
        assert result['valid'] is False
        assert any("email" in error and "missing" in error for error in result['errors'])
    
    def test_security_validator_ip_validation(self, security_validator):
        """Test IP address security validation."""
        # Valid public IP
        result = security_validator.validate_ip_address("8.8.8.8", allow_private=False)
        assert result['valid'] is True
        assert result['is_private'] is False
        
        # Private IP (should fail when not allowed)
        result = security_validator.validate_ip_address("192.168.1.1", allow_private=False)
        assert result['valid'] is False
        assert result['is_private'] is True
        
        # Private IP (should succeed when allowed)
        result = security_validator.validate_ip_address("192.168.1.1", allow_private=True)
        assert result['valid'] is True
    
    def test_security_validator_user_agent_analysis(self, security_validator):
        """Test User-Agent string analysis."""
        # Normal user agent
        normal_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        result = security_validator.validate_user_agent(normal_ua)
        assert result['valid'] is True
        
        # Bot user agent
        bot_ua = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        result = security_validator.validate_user_agent(bot_ua)
        assert result['valid'] is True  # Still valid, but should have warnings
        assert len(result['warnings']) > 0


class TestPRAnalysis:
    """Test suite for PR analysis functionality."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create some test files
        (Path(temp_dir) / "test.py").write_text('''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    
if __name__ == "__main__":
    hello_world()
''')
        
        (Path(temp_dir) / "README.md").write_text("# Test Repository\n\nThis is a test.")
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_linter_config_creation(self):
        """Test linter configuration creation."""
        config = LinterConfig()
        
        assert config.python == "ruff"
        assert config.javascript == "eslint"
        assert config.go == "golangci-lint"
    
    def test_linter_config_from_yaml(self):
        """Test loading linter config from YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('''
linters:
  python: pylint
  javascript: jshint
  go: gofmt
''')
            yaml_path = f.name
        
        try:
            config = LinterConfig.from_yaml(yaml_path)
            assert config.python == "pylint"
            assert config.javascript == "jshint"
            assert config.go == "gofmt"
        finally:
            Path(yaml_path).unlink()
    
    def test_analyze_pr_basic_functionality(self, temp_repo):
        """Test basic PR analysis functionality."""
        result = analyze_pr(temp_repo, use_cache=False, use_parallel=False)
        
        assert result is not None
        assert result.security is not None
        assert result.style is not None
        assert result.performance is not None
        assert result.metadata is not None
        
        # Check that analysis completed
        assert result.security.tool == "security-scanner"
        assert result.style.tool == "style-analyzer"
        assert result.performance.tool == "performance-analyzer"
        
        # Check metadata
        assert "analysis_timestamp" in result.metadata
        assert "repo_path" in result.metadata
        assert result.metadata["repo_path"] == temp_repo


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create a more complex test repository
        (Path(temp_dir) / "src").mkdir()
        (Path(temp_dir) / "tests").mkdir()
        
        # Python files
        (Path(temp_dir) / "src" / "__init__.py").write_text("")
        (Path(temp_dir) / "src" / "main.py").write_text('''
import os
import sys

def process_data(data):
    """Process some data."""
    if not data:
        return None
    
    results = []
    for item in data:
        if isinstance(item, str):
            results.append(item.upper())
        elif isinstance(item, int):
            results.append(item * 2)
    
    return results

def main():
    """Main entry point."""
    test_data = ["hello", "world", 42, "python"]
    result = process_data(test_data)
    print(f"Processed: {result}")

if __name__ == "__main__":
    main()
''')
        
        # Test files
        (Path(temp_dir) / "tests" / "test_main.py").write_text('''
import unittest
from src.main import process_data

class TestProcessData(unittest.TestCase):
    
    def test_empty_data(self):
        result = process_data([])
        self.assertIsNone(result)
    
    def test_mixed_data(self):
        data = ["hello", 42, "world"]
        result = process_data(data)
        expected = ["HELLO", 84, "WORLD"]
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
''')
        
        # Configuration files
        (Path(temp_dir) / "requirements.txt").write_text("requests==2.28.0\nnumpy>=1.20.0")
        (Path(temp_dir) / "setup.py").write_text('''
from setuptools import setup, find_packages

setup(
    name="test-package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["requests", "numpy"]
)
''')
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_complete_analysis_workflow(self, temp_repo):
        """Test complete analysis workflow from start to finish."""
        # Run analysis
        result = analyze_pr(
            repo_path=temp_repo,
            use_cache=True,
            use_parallel=True
        )
        
        # Verify all components completed
        assert result.security.tool == "security-scanner"
        assert result.style.tool == "style-analyzer"  
        assert result.performance.tool == "performance-analyzer"
        
        # Verify metadata
        assert "analysis_duration" in result.metadata
        assert result.metadata["cache_used"] is True
        assert result.metadata["parallel_execution"] is True
        
        # Verify outputs are strings
        assert isinstance(result.security.output, str)
        assert isinstance(result.style.output, str)
        assert isinstance(result.performance.output, str)
    
    @pytest.mark.asyncio
    async def test_distributed_analysis_workflow(self, temp_repo):
        """Test distributed analysis workflow."""
        # Mock Redis for testing
        redis_mock = AsyncMock()
        redis_mock.keys.return_value = []
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.zadd.return_value = True
        redis_mock.zpopmax.return_value = []
        redis_mock.setex.return_value = True
        redis_mock.zcard.return_value = 0
        
        # Create distributed task manager
        from autogen_code_review_bot.distributed_processing import create_distributed_manager
        manager = create_distributed_manager()
        manager.redis = redis_mock
        
        # Submit analysis task
        task_id = await manager.submit_task(
            task_type="analyze_repository",
            payload={"repo_path": temp_repo},
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Verify task was submitted to Redis
        manager.redis.set.assert_called()
        manager.redis.zadd.assert_called()


# Performance Tests
class TestPerformance:
    """Performance and load testing."""
    
    def test_cache_performance_under_load(self):
        """Test cache performance under high load."""
        cache = AdaptiveLRU(max_size=1000, max_memory_mb=10)
        
        # Measure time for bulk operations
        start_time = time.time()
        
        # Bulk insert
        for i in range(1000):
            cache.put(f"key{i}", f"value{i}" * 100)  # ~500 bytes each
        
        insert_time = time.time() - start_time
        
        # Bulk retrieve
        start_time = time.time()
        hit_count = 0
        for i in range(1000):
            if cache.get(f"key{i}") is not None:
                hit_count += 1
        
        retrieve_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 1.0  # Should complete within 1 second
        assert retrieve_time < 0.5  # Retrieval should be faster
        assert hit_count == 1000  # All items should be retrievable
        
        # Cache should be properly sized
        stats = cache.get_stats()
        assert stats.entry_count <= cache.max_size
    
    def test_validation_performance(self):
        """Test validation performance with large inputs."""
        validator = InputValidator()
        
        # Large string validation
        large_string = "x" * 10000  # 10KB
        
        start_time = time.time()
        result = validator.validate(large_string, "string")
        validation_time = time.time() - start_time
        
        assert result['valid'] is True
        assert validation_time < 0.1  # Should validate within 100ms
        
        # Bulk validation
        test_inputs = [f"test_string_{i}" for i in range(1000)]
        
        start_time = time.time()
        for test_input in test_inputs:
            validator.validate(test_input, "string")
        bulk_time = time.time() - start_time
        
        assert bulk_time < 1.0  # Should process 1000 validations within 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])