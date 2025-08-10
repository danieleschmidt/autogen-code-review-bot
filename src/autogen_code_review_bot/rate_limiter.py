"""Rate limiting and throttling implementation for API endpoints."""

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Union

import redis

from .exceptions import ValidationError
from .logging_config import get_logger
from .metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds
    burst: Optional[int] = None  # Burst limit (defaults to requests)

    def __post_init__(self):
        if self.burst is None:
            self.burst = self.requests


class RateLimitResult:
    """Result of a rate limit check."""

    def __init__(self, allowed: bool, remaining: int, reset_time: int, retry_after: Optional[int] = None):
        self.allowed = allowed
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after


class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window algorithm."""

    def __init__(self):
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def check_limit(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check if request is within rate limit.
        
        Args:
            key: Unique identifier for the rate limit (e.g., IP address, user ID)
            config: Rate limiting configuration
            
        Returns:
            RateLimitResult indicating if request should be allowed
        """
        current_time = time.time()
        window_start = current_time - config.window

        with self._lock:
            # Clean old requests outside the window
            request_times = self._requests[key]
            while request_times and request_times[0] < window_start:
                request_times.popleft()

            # Check if we're within limits
            if len(request_times) >= config.requests:
                # Find when the oldest request in window will expire
                oldest_request = request_times[0]
                retry_after = int(oldest_request + config.window - current_time) + 1

                metrics.record_counter("rate_limit_exceeded_total", 1, tags={"key": key[:10]})

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(oldest_request + config.window),
                    retry_after=retry_after
                )

            # Allow request and record it
            request_times.append(current_time)
            remaining = config.requests - len(request_times)

            # Calculate reset time (when oldest request expires)
            reset_time = int(request_times[0] + config.window) if request_times else int(current_time + config.window)

            metrics.record_counter("rate_limit_requests_total", 1, tags={"key": key[:10], "status": "allowed"})

            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                reset_time=reset_time
            )


class RedisRateLimiter:
    """Redis-based distributed rate limiter using sliding window algorithm."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._script_sha: Optional[str] = None
        self._load_lua_script()

    def _load_lua_script(self):
        """Load Lua script for atomic rate limiting operations."""
        script = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        -- Remove old entries outside the window
        redis.call('ZREMRANGEBYSCORE', key, '-inf', current_time - window)
        
        -- Count current entries
        local current_count = redis.call('ZCARD', key)
        
        -- Check if limit exceeded
        if current_count >= limit then
            -- Get oldest entry to calculate retry after
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if next(oldest) ~= nil then
                retry_after = math.ceil((tonumber(oldest[2]) + window) - current_time)
            end
            return {0, 0, current_time + window, retry_after}
        else
            -- Add current request
            redis.call('ZADD', key, current_time, current_time)
            redis.call('EXPIRE', key, window + 1)
            
            local remaining = limit - current_count - 1
            local reset_time = current_time + window
            
            return {1, remaining, reset_time, 0}
        end
        """

        try:
            self._script_sha = self.redis.script_load(script)
        except Exception as e:
            logger.error("Failed to load rate limiting Lua script", error=str(e))
            self._script_sha = None

    def check_limit(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check if request is within rate limit using Redis.
        
        Args:
            key: Unique identifier for the rate limit
            config: Rate limiting configuration
            
        Returns:
            RateLimitResult indicating if request should be allowed
        """
        try:
            current_time = time.time()
            redis_key = f"rate_limit:{key}"

            if self._script_sha:
                # Use Lua script for atomic operation
                result = self.redis.evalsha(
                    self._script_sha,
                    1,  # Number of keys
                    redis_key,
                    config.window,
                    config.requests,
                    current_time
                )

                allowed = bool(result[0])
                remaining = int(result[1])
                reset_time = int(result[2])
                retry_after = int(result[3]) if result[3] > 0 else None

            else:
                # Fallback to individual Redis commands (less atomic)
                with self.redis.pipeline() as pipe:
                    pipe.zremrangebyscore(redis_key, '-inf', current_time - config.window)
                    pipe.zcard(redis_key)
                    pipe.execute()

                    current_count = pipe.zcard(redis_key)

                    if current_count >= config.requests:
                        # Get oldest entry for retry calculation
                        oldest = pipe.zrange(redis_key, 0, 0, withscores=True)
                        retry_after = None
                        if oldest:
                            retry_after = int((oldest[0][1] + config.window) - current_time) + 1

                        allowed = False
                        remaining = 0
                        reset_time = int(current_time + config.window)
                    else:
                        pipe.zadd(redis_key, {current_time: current_time})
                        pipe.expire(redis_key, config.window + 1)
                        pipe.execute()

                        allowed = True
                        remaining = config.requests - current_count - 1
                        reset_time = int(current_time + config.window)
                        retry_after = None

            # Record metrics
            status = "allowed" if allowed else "denied"
            metrics.record_counter("rate_limit_requests_total", 1, tags={"key": key[:10], "status": status})

            if not allowed:
                metrics.record_counter("rate_limit_exceeded_total", 1, tags={"key": key[:10]})

            return RateLimitResult(allowed, remaining, reset_time, retry_after)

        except Exception as e:
            logger.error("Redis rate limiter error, allowing request", error=str(e))
            # Fail open - allow request if Redis is unavailable
            return RateLimitResult(
                allowed=True,
                remaining=config.requests,
                reset_time=int(time.time() + config.window)
            )


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts limits based on system load."""

    def __init__(self, base_limiter: Union[InMemoryRateLimiter, RedisRateLimiter]):
        self.base_limiter = base_limiter
        self._load_metrics = deque(maxlen=60)  # Track load for last 60 checks
        self._last_adjustment = 0

    def check_limit(self, key: str, config: RateLimitConfig, system_load: Optional[float] = None) -> RateLimitResult:
        """Check rate limit with adaptive adjustment based on system load.
        
        Args:
            key: Unique identifier for the rate limit
            config: Base rate limiting configuration
            system_load: Current system load (0.0 to 1.0), auto-detected if None
            
        Returns:
            RateLimitResult with potentially adjusted limits
        """
        # Detect system load if not provided
        if system_load is None:
            try:
                import psutil
                system_load = psutil.cpu_percent(interval=0.1) / 100.0
            except ImportError:
                system_load = 0.5  # Default to moderate load

        # Track load metrics
        self._load_metrics.append(system_load)

        # Calculate adjustment factor based on recent load
        if len(self._load_metrics) >= 10:  # Need sufficient data
            avg_load = sum(self._load_metrics) / len(self._load_metrics)

            # Adjust limits based on load (reduce under high load)
            if avg_load > 0.8:
                adjustment = 0.5  # Reduce to 50% under high load
            elif avg_load > 0.6:
                adjustment = 0.75  # Reduce to 75% under moderate load
            else:
                adjustment = 1.0  # No adjustment under low load

            adjusted_requests = max(1, int(config.requests * adjustment))
            adjusted_config = RateLimitConfig(
                requests=adjusted_requests,
                window=config.window,
                burst=config.burst
            )

            # Log adjustments
            if adjusted_requests != config.requests:
                logger.info(
                    "Adaptive rate limit adjustment",
                    original_limit=config.requests,
                    adjusted_limit=adjusted_requests,
                    system_load=avg_load,
                    adjustment_factor=adjustment
                )
                metrics.record_gauge("rate_limit_adjustment_factor", adjustment, tags={"key": key[:10]})
        else:
            adjusted_config = config

        return self.base_limiter.check_limit(key, adjusted_config)


class RateLimitMiddleware:
    """Middleware for applying rate limits to requests."""

    def __init__(self, limiter: Union[InMemoryRateLimiter, RedisRateLimiter, AdaptiveRateLimiter],
                 default_config: RateLimitConfig):
        self.limiter = limiter
        self.default_config = default_config
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}

    def configure_endpoint(self, endpoint: str, config: RateLimitConfig):
        """Configure rate limiting for a specific endpoint.
        
        Args:
            endpoint: Endpoint identifier
            config: Rate limiting configuration for this endpoint
        """
        self.endpoint_configs[endpoint] = config

    def get_client_key(self, request_info: Dict[str, str]) -> str:
        """Generate a unique key for rate limiting based on request info.
        
        Args:
            request_info: Dictionary containing client information
            
        Returns:
            Unique rate limiting key
        """
        # Use IP address as primary identifier
        client_ip = request_info.get("client_ip", "unknown")

        # Include user agent hash for additional uniqueness
        user_agent = request_info.get("user_agent", "")
        user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]

        return f"{client_ip}:{user_agent_hash}"

    def check_rate_limit(self, request_info: Dict[str, str], endpoint: Optional[str] = None) -> RateLimitResult:
        """Check rate limit for a request.
        
        Args:
            request_info: Dictionary containing request information
            endpoint: Optional endpoint identifier for endpoint-specific limits
            
        Returns:
            RateLimitResult indicating if request should be allowed
        """
        key = self.get_client_key(request_info)
        config = self.endpoint_configs.get(endpoint, self.default_config) if endpoint else self.default_config

        return self.limiter.check_limit(key, config)


def create_rate_limiter(backend: str = "memory", redis_client: Optional[redis.Redis] = None) -> Union[InMemoryRateLimiter, RedisRateLimiter]:
    """Factory function to create rate limiter based on backend type.
    
    Args:
        backend: Backend type ("memory" or "redis")
        redis_client: Redis client instance (required for Redis backend)
        
    Returns:
        Rate limiter instance
        
    Raises:
        ValidationError: If invalid backend or missing Redis client
    """
    if backend == "memory":
        return InMemoryRateLimiter()
    elif backend == "redis":
        if redis_client is None:
            raise ValidationError("Redis client required for Redis rate limiter")
        return RedisRateLimiter(redis_client)
    else:
        raise ValidationError(f"Invalid rate limiter backend: {backend}")


def create_adaptive_rate_limiter(backend: str = "memory", redis_client: Optional[redis.Redis] = None) -> AdaptiveRateLimiter:
    """Factory function to create adaptive rate limiter.
    
    Args:
        backend: Backend type ("memory" or "redis") 
        redis_client: Redis client instance (required for Redis backend)
        
    Returns:
        Adaptive rate limiter instance
    """
    base_limiter = create_rate_limiter(backend, redis_client)
    return AdaptiveRateLimiter(base_limiter)
