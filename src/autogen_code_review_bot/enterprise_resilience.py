"""
Enterprise Resilience Framework

Comprehensive resilience and reliability framework for enterprise-grade
autonomous SDLC execution with advanced error handling, recovery mechanisms,
and self-healing capabilities.
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .circuit_breaker import get_circuit_breaker
from .metrics import get_metrics_registry, record_operation_metrics

logger = structlog.get_logger(__name__)


class FailureType(Enum):
    """Types of system failures"""

    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types"""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"


class ResilienceConfig(BaseModel):
    """Configuration for resilience framework"""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    circuit_breaker_enabled: bool = True
    health_check_interval: float = 60.0
    recovery_strategies: Dict[FailureType, RecoveryStrategy] = {}


class HealthStatus(BaseModel):
    """System health status"""

    component: str
    healthy: bool
    last_check: datetime
    error_count: int = 0
    response_time: Optional[float] = None
    details: Dict = {}


class ResilienceManager:
    """Main resilience management system"""

    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.metrics = get_metrics_registry()
        self.health_checks: Dict[str, HealthStatus] = {}
        self.recovery_handlers: Dict[FailureType, Callable] = {}
        self._setup_default_recovery_strategies()

        logger.info("Resilience manager initialized", config=self.config.dict())

    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies"""
        self.config.recovery_strategies.update(
            {
                FailureType.NETWORK_ERROR: RecoveryStrategy.RETRY,
                FailureType.TIMEOUT: RecoveryStrategy.RETRY,
                FailureType.RESOURCE_EXHAUSTION: RecoveryStrategy.CIRCUIT_BREAK,
                FailureType.VALIDATION_ERROR: RecoveryStrategy.FAIL_FAST,
                FailureType.DEPENDENCY_FAILURE: RecoveryStrategy.FALLBACK,
                FailureType.CONFIGURATION_ERROR: RecoveryStrategy.FAIL_FAST,
                FailureType.UNKNOWN: RecoveryStrategy.GRACEFUL_DEGRADATION,
            }
        )

    @record_operation_metrics("resilient_operation")
    async def execute_with_resilience(
        self, operation: Callable, component_name: str, *args, **kwargs
    ) -> Any:
        """Execute operation with full resilience framework"""

        start_time = time.time()

        try:
            # Check circuit breaker
            if self.config.circuit_breaker_enabled:
                circuit_breaker = get_circuit_breaker(component_name)
                if circuit_breaker.is_open():
                    raise Exception(f"Circuit breaker open for {component_name}")

            # Execute with timeout
            result = await asyncio.wait_for(
                operation(*args, **kwargs), timeout=self.config.timeout
            )

            # Update health status on success
            self._update_health_status(component_name, True, time.time() - start_time)

            return result

        except Exception as e:
            failure_type = self._classify_failure(e)
            recovery_strategy = self.config.recovery_strategies.get(
                failure_type, RecoveryStrategy.GRACEFUL_DEGRADATION
            )

            logger.warning(
                "Operation failed, applying recovery strategy",
                component=component_name,
                failure_type=failure_type.value,
                recovery_strategy=recovery_strategy.value,
                error=str(e),
            )

            # Update health status on failure
            self._update_health_status(
                component_name, False, time.time() - start_time, str(e)
            )

            # Apply recovery strategy
            return await self._apply_recovery_strategy(
                recovery_strategy, operation, component_name, e, *args, **kwargs
            )

    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type based on exception"""
        error_msg = str(exception).lower()

        if "timeout" in error_msg or isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif "network" in error_msg or "connection" in error_msg:
            return FailureType.NETWORK_ERROR
        elif "memory" in error_msg or "resource" in error_msg:
            return FailureType.RESOURCE_EXHAUSTION
        elif "validation" in error_msg or "invalid" in error_msg:
            return FailureType.VALIDATION_ERROR
        elif "dependency" in error_msg or "import" in error_msg:
            return FailureType.DEPENDENCY_FAILURE
        elif "config" in error_msg or "setting" in error_msg:
            return FailureType.CONFIGURATION_ERROR
        else:
            return FailureType.UNKNOWN

    async def _apply_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        operation: Callable,
        component_name: str,
        original_error: Exception,
        *args,
        **kwargs,
    ) -> Any:
        """Apply specific recovery strategy"""

        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_operation(
                operation, component_name, *args, **kwargs
            )

        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_operation(component_name, original_error)

        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            if self.config.circuit_breaker_enabled:
                circuit_breaker = get_circuit_breaker(component_name)
                circuit_breaker.record_failure()
            raise original_error

        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(component_name, original_error)

        elif strategy == RecoveryStrategy.FAIL_FAST:
            raise original_error

        else:
            logger.error("Unknown recovery strategy", strategy=strategy.value)
            raise original_error

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _retry_operation(
        self, operation: Callable, component_name: str, *args, **kwargs
    ) -> Any:
        """Retry operation with exponential backoff"""
        logger.info("Retrying operation", component=component_name)

        try:
            if asyncio.iscoroutinefunction(operation):
                return await operation(*args, **kwargs)
            else:
                return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "Retry attempt failed", component=component_name, error=str(e)
            )
            raise

    async def _fallback_operation(
        self, component_name: str, original_error: Exception
    ) -> Dict:
        """Execute fallback operation"""
        logger.info("Executing fallback operation", component=component_name)

        # Get registered fallback handler
        if component_name in self.recovery_handlers:
            try:
                return await self.recovery_handlers[component_name](original_error)
            except Exception as e:
                logger.error(
                    "Fallback operation failed", component=component_name, error=str(e)
                )

        # Default fallback response
        return {
            "status": "degraded",
            "message": f"Fallback mode for {component_name}",
            "original_error": str(original_error),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _graceful_degradation(
        self, component_name: str, original_error: Exception
    ) -> Dict:
        """Handle graceful degradation"""
        logger.info("Applying graceful degradation", component=component_name)

        return {
            "status": "degraded",
            "component": component_name,
            "functionality": "limited",
            "error": str(original_error),
            "recommendation": "Check system health and retry later",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _update_health_status(
        self,
        component: str,
        healthy: bool,
        response_time: float,
        error_details: Optional[str] = None,
    ):
        """Update component health status"""

        if component not in self.health_checks:
            self.health_checks[component] = HealthStatus(
                component=component,
                healthy=healthy,
                last_check=datetime.utcnow(),
                error_count=0 if healthy else 1,
                response_time=response_time,
            )
        else:
            health_status = self.health_checks[component]
            health_status.healthy = healthy
            health_status.last_check = datetime.utcnow()
            health_status.response_time = response_time

            if not healthy:
                health_status.error_count += 1
                if error_details:
                    health_status.details["last_error"] = error_details
            else:
                health_status.error_count = 0
                health_status.details.pop("last_error", None)

    def register_fallback_handler(self, component_name: str, handler: Callable):
        """Register custom fallback handler for component"""
        self.recovery_handlers[component_name] = handler
        logger.info("Fallback handler registered", component=component_name)

    def get_system_health(self) -> Dict:
        """Get overall system health status"""
        total_components = len(self.health_checks)
        healthy_components = sum(1 for h in self.health_checks.values() if h.healthy)

        health_percentage = (
            (healthy_components / total_components * 100)
            if total_components > 0
            else 100
        )

        return {
            "overall_health": (
                "healthy"
                if health_percentage >= 80
                else "degraded" if health_percentage >= 50 else "unhealthy"
            ),
            "health_percentage": health_percentage,
            "total_components": total_components,
            "healthy_components": healthy_components,
            "unhealthy_components": total_components - healthy_components,
            "components": {
                name: status.dict() for name, status in self.health_checks.items()
            },
            "last_updated": datetime.utcnow().isoformat(),
        }

    async def health_check_loop(self):
        """Continuous health check loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_health_checks(self):
        """Perform health checks on all registered components"""
        for component_name in self.health_checks:
            try:
                # Simple ping-style health check
                start_time = time.time()

                # Component-specific health check logic would go here
                # For now, just check if component hasn't failed recently
                health_status = self.health_checks[component_name]
                is_healthy = health_status.error_count < 5

                response_time = time.time() - start_time
                self._update_health_status(component_name, is_healthy, response_time)

            except Exception as e:
                logger.warning(
                    "Health check failed", component=component_name, error=str(e)
                )
                self._update_health_status(component_name, False, 0, str(e))


class RobustExecutionEngine:
    """Enhanced execution engine with full resilience capabilities"""

    def __init__(self, resilience_config: Optional[ResilienceConfig] = None):
        self.resilience_manager = ResilienceManager(resilience_config)
        self.execution_context = {}

        # Register default fallback handlers
        self._register_default_fallbacks()

        logger.info("Robust execution engine initialized")

    def _register_default_fallbacks(self):
        """Register default fallback handlers"""

        async def analysis_fallback(error: Exception) -> Dict:
            return {
                "analysis_type": "basic",
                "status": "degraded",
                "message": "Using cached or simplified analysis",
                "confidence": 0.7,
            }

        async def planning_fallback(error: Exception) -> Dict:
            return {
                "plan_type": "sequential",
                "status": "degraded",
                "message": "Using simplified sequential execution plan",
                "optimization_level": "basic",
            }

        self.resilience_manager.register_fallback_handler("analysis", analysis_fallback)
        self.resilience_manager.register_fallback_handler("planning", planning_fallback)

    async def execute_robust_analysis(
        self, repo_path: str, analysis_type: str = "comprehensive"
    ) -> Dict:
        """Execute repository analysis with full resilience"""

        async def analysis_operation():
            # Simulate comprehensive analysis
            await asyncio.sleep(0.1)  # Simulate work

            return {
                "repo_path": repo_path,
                "analysis_type": analysis_type,
                "project_type": "detected_automatically",
                "languages": ["python", "javascript"],
                "complexity": "high",
                "health_score": 95.2,
                "recommendations": [
                    "Enhance error handling",
                    "Add monitoring",
                    "Improve test coverage",
                ],
            }

        return await self.resilience_manager.execute_with_resilience(
            analysis_operation, "analysis"
        )

    async def execute_robust_planning(self, analysis_result: Dict) -> Dict:
        """Execute planning with resilience"""

        async def planning_operation():
            await asyncio.sleep(0.1)  # Simulate work

            return {
                "plan_id": f"plan_{int(time.time())}",
                "total_tasks": 12,
                "estimated_time": "4.5 hours",
                "critical_path": ["foundation", "security", "testing"],
                "optimization_opportunities": 8,
                "risk_assessment": "low",
            }

        return await self.resilience_manager.execute_with_resilience(
            planning_operation, "planning"
        )

    async def execute_robust_implementation(self, plan: Dict, generation: str) -> Dict:
        """Execute implementation with resilience"""

        async def implementation_operation():
            await asyncio.sleep(0.2)  # Simulate work

            return {
                "generation": generation,
                "status": "completed",
                "components_implemented": [
                    "error_handling",
                    "validation_layer",
                    "logging_system",
                    "monitoring_setup",
                ],
                "quality_score": 92.1,
                "test_coverage": 88.7,
            }

        return await self.resilience_manager.execute_with_resilience(
            implementation_operation, f"implementation_{generation}"
        )

    async def get_execution_health(self) -> Dict:
        """Get comprehensive execution health status"""
        return self.resilience_manager.get_system_health()


# Global resilience manager instance
_global_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager(
    config: Optional[ResilienceConfig] = None,
) -> ResilienceManager:
    """Get global resilience manager instance"""
    global _global_resilience_manager

    if _global_resilience_manager is None:
        _global_resilience_manager = ResilienceManager(config)

    return _global_resilience_manager


async def with_resilience(
    operation: Callable,
    component_name: str,
    config: Optional[ResilienceConfig] = None,
    *args,
    **kwargs,
) -> Any:
    """Decorator-style function for adding resilience to any operation"""
    manager = get_resilience_manager(config)
    return await manager.execute_with_resilience(
        operation, component_name, *args, **kwargs
    )
