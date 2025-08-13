"""
Auto-Scaling Engine

Intelligent auto-scaling system with predictive analytics, resource optimization,
and quantum-inspired load balancing for autonomous SDLC execution.
"""

import asyncio
import math
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import structlog
from pydantic import BaseModel

from .metrics import get_metrics_registry, record_operation_metrics
from .enterprise_monitoring import get_enterprise_monitor
from .quantum_performance_optimizer import get_quantum_optimizer

logger = structlog.get_logger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Types of scaling triggers"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"
    PREDICTIVE = "predictive"


@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    name: str
    trigger: ScalingTrigger
    metric_threshold: float
    scale_direction: ScalingDirection
    scale_factor: float
    cooldown_period: int  # seconds
    min_instances: int = 1
    max_instances: int = 100
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: datetime
    trigger: ScalingTrigger
    direction: ScalingDirection
    from_instances: int
    to_instances: int
    reason: str
    success: bool
    metadata: Dict = None


class PredictiveModel:
    """Predictive analytics for proactive scaling"""
    
    def __init__(self):
        self.historical_data: List[Dict] = []
        self.trend_window = 300  # 5 minutes
        self.prediction_horizon = 60  # 1 minute ahead
        
    def add_data_point(self, timestamp: datetime, metrics: Dict):
        """Add historical data point"""
        data_point = {
            "timestamp": timestamp,
            "metrics": metrics.copy()
        }
        self.historical_data.append(data_point)
        
        # Keep only recent data
        cutoff_time = timestamp - timedelta(seconds=self.trend_window * 4)
        self.historical_data = [
            dp for dp in self.historical_data 
            if dp["timestamp"] > cutoff_time
        ]
    
    def predict_metric_value(self, metric_name: str, horizon_seconds: int = None) -> Optional[float]:
        """Predict future metric value using trend analysis"""
        if len(self.historical_data) < 5:
            return None
        
        horizon = horizon_seconds or self.prediction_horizon
        
        # Extract metric values with timestamps
        values = []
        timestamps = []
        
        for dp in self.historical_data:
            if metric_name in dp["metrics"]:
                values.append(dp["metrics"][metric_name])
                timestamps.append(dp["timestamp"].timestamp())
        
        if len(values) < 3:
            return None
        
        # Simple linear regression for trend prediction
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return values[-1]  # No trend, return last value
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict value at future timestamp
        future_timestamp = timestamps[-1] + horizon
        predicted_value = slope * future_timestamp + intercept
        
        return predicted_value
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> bool:
        """Detect if current value is anomalous"""
        if len(self.historical_data) < 10:
            return False
        
        # Get recent values
        recent_values = [
            dp["metrics"].get(metric_name, 0) 
            for dp in self.historical_data[-10:]
            if metric_name in dp["metrics"]
        ]
        
        if len(recent_values) < 5:
            return False
        
        # Calculate statistics
        mean_value = sum(recent_values) / len(recent_values)
        variance = sum((x - mean_value) ** 2 for x in recent_values) / len(recent_values)
        std_dev = math.sqrt(variance)
        
        # Check if current value is more than 2 standard deviations away
        z_score = abs(current_value - mean_value) / (std_dev + 1e-10)
        return z_score > 2.0


class AutoScalingEngine:
    """Intelligent auto-scaling engine with predictive capabilities"""
    
    def __init__(self):
        self.metrics = get_metrics_registry()
        self.monitor = get_enterprise_monitor()
        self.optimizer = get_quantum_optimizer()
        self.predictive_model = PredictiveModel()
        
        # Scaling configuration
        self.scaling_rules: List[ScalingRule] = []
        self.scaling_history: List[ScalingEvent] = []
        self.current_instances = 1
        self.last_scaling_time = datetime.utcnow()
        
        # State tracking
        self.is_scaling = False
        self.scaling_tasks: List[asyncio.Task] = []
        
        self._setup_default_scaling_rules()
        
        logger.info("Auto-scaling engine initialized")
    
    def _setup_default_scaling_rules(self):
        """Setup default scaling rules"""
        
        # CPU-based scaling
        self.scaling_rules.extend([
            ScalingRule(
                name="cpu_scale_up",
                trigger=ScalingTrigger.CPU_USAGE,
                metric_threshold=80.0,
                scale_direction=ScalingDirection.UP,
                scale_factor=1.5,
                cooldown_period=300,
                max_instances=20
            ),
            ScalingRule(
                name="cpu_scale_down",
                trigger=ScalingTrigger.CPU_USAGE,
                metric_threshold=30.0,
                scale_direction=ScalingDirection.DOWN,
                scale_factor=0.8,
                cooldown_period=600,
                min_instances=1
            )
        ])
        
        # Memory-based scaling
        self.scaling_rules.extend([
            ScalingRule(
                name="memory_scale_up",
                trigger=ScalingTrigger.MEMORY_USAGE,
                metric_threshold=85.0,
                scale_direction=ScalingDirection.UP,
                scale_factor=1.3,
                cooldown_period=300,
                max_instances=15
            )
        ])
        
        # Response time-based scaling
        self.scaling_rules.extend([
            ScalingRule(
                name="response_time_scale_up",
                trigger=ScalingTrigger.RESPONSE_TIME,
                metric_threshold=500.0,  # 500ms
                scale_direction=ScalingDirection.UP,
                scale_factor=1.4,
                cooldown_period=180,
                max_instances=25
            )
        ])
        
        # Error rate-based scaling
        self.scaling_rules.extend([
            ScalingRule(
                name="error_rate_scale_up",
                trigger=ScalingTrigger.ERROR_RATE,
                metric_threshold=5.0,  # 5%
                scale_direction=ScalingDirection.UP,
                scale_factor=1.6,
                cooldown_period=120,
                max_instances=30
            )
        ])
        
        # Predictive scaling
        self.scaling_rules.append(
            ScalingRule(
                name="predictive_scale_up",
                trigger=ScalingTrigger.PREDICTIVE,
                metric_threshold=70.0,  # Predicted CPU usage
                scale_direction=ScalingDirection.UP,
                scale_factor=1.2,
                cooldown_period=240,
                max_instances=10
            )
        )
    
    async def start_auto_scaling(self):
        """Start auto-scaling monitoring and decision loop"""
        self.scaling_tasks = [
            asyncio.create_task(self._scaling_decision_loop()),
            asyncio.create_task(self._predictive_scaling_loop()),
            asyncio.create_task(self._health_monitoring_loop())
        ]
        
        logger.info("Auto-scaling monitoring started")
    
    async def stop_auto_scaling(self):
        """Stop auto-scaling tasks"""
        for task in self.scaling_tasks:
            task.cancel()
        
        await asyncio.gather(*self.scaling_tasks, return_exceptions=True)
        logger.info("Auto-scaling monitoring stopped")
    
    async def _scaling_decision_loop(self):
        """Main scaling decision loop"""
        while True:
            try:
                # Get current metrics
                current_metrics = await self._collect_current_metrics()
                
                # Add to predictive model
                self.predictive_model.add_data_point(datetime.utcnow(), current_metrics)
                
                # Check scaling rules
                scaling_decision = await self._evaluate_scaling_rules(current_metrics)
                
                if scaling_decision and not self.is_scaling:
                    await self._execute_scaling_decision(scaling_decision)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Scaling decision loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _predictive_scaling_loop(self):
        """Predictive scaling based on trend analysis"""
        while True:
            try:
                # Predict CPU usage 1 minute ahead
                predicted_cpu = self.predictive_model.predict_metric_value("cpu_usage", 60)
                
                if predicted_cpu and predicted_cpu > 75.0:
                    # Proactively scale up if high CPU predicted
                    current_metrics = {"predicted_cpu_usage": predicted_cpu}
                    
                    predictive_rule = next(
                        (rule for rule in self.scaling_rules 
                         if rule.trigger == ScalingTrigger.PREDICTIVE), 
                        None
                    )
                    
                    if predictive_rule and await self._should_scale(predictive_rule):
                        scaling_decision = {
                            "rule": predictive_rule,
                            "trigger_value": predicted_cpu,
                            "reason": f"Predictive scaling: CPU usage predicted to reach {predicted_cpu:.1f}%"
                        }
                        
                        if not self.is_scaling:
                            await self._execute_scaling_decision(scaling_decision)
                
                await asyncio.sleep(60)  # Predict every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Predictive scaling loop error", error=str(e))
                await asyncio.sleep(120)
    
    async def _health_monitoring_loop(self):
        """Monitor system health and trigger emergency scaling"""
        while True:
            try:
                health_status = self.monitor.get_system_health()
                
                # Emergency scaling for critical health issues
                if health_status.get("overall_health") == "unhealthy":
                    critical_components = [
                        name for name, status in health_status.get("components", {}).items()
                        if not status.get("healthy", True)
                    ]
                    
                    if critical_components and not self.is_scaling:
                        emergency_scaling = {
                            "rule": ScalingRule(
                                name="emergency_scale_up",
                                trigger=ScalingTrigger.CUSTOM_METRIC,
                                metric_threshold=0,
                                scale_direction=ScalingDirection.UP,
                                scale_factor=2.0,
                                cooldown_period=60,
                                max_instances=50
                            ),
                            "trigger_value": len(critical_components),
                            "reason": f"Emergency scaling due to unhealthy components: {', '.join(critical_components)}"
                        }
                        
                        await self._execute_scaling_decision(emergency_scaling)
                
                await asyncio.sleep(45)  # Check health every 45 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring loop error", error=str(e))
                await asyncio.sleep(90)
    
    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        # In real implementation, this would collect from various sources
        # For now, return simulated metrics
        
        import random
        base_time = time.time()
        
        # Simulate realistic metrics with some variability
        metrics = {
            "cpu_usage": 50.0 + random.uniform(-20, 30),
            "memory_usage": 60.0 + random.uniform(-15, 25),
            "response_time": 200.0 + random.uniform(-50, 200),
            "queue_depth": max(0, 5 + random.uniform(-3, 10)),
            "error_rate": max(0, 1.0 + random.uniform(-0.8, 3)),
            "throughput": 1000 + random.uniform(-200, 400),
            "timestamp": base_time
        }
        
        return metrics
    
    async def _evaluate_scaling_rules(self, metrics: Dict[str, float]) -> Optional[Dict]:
        """Evaluate all scaling rules against current metrics"""
        
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check if rule should trigger
            trigger_value = None
            should_trigger = False
            
            if rule.trigger == ScalingTrigger.CPU_USAGE:
                trigger_value = metrics.get("cpu_usage", 0)
                should_trigger = (
                    (rule.scale_direction == ScalingDirection.UP and trigger_value > rule.metric_threshold) or
                    (rule.scale_direction == ScalingDirection.DOWN and trigger_value < rule.metric_threshold)
                )
            
            elif rule.trigger == ScalingTrigger.MEMORY_USAGE:
                trigger_value = metrics.get("memory_usage", 0)
                should_trigger = (
                    (rule.scale_direction == ScalingDirection.UP and trigger_value > rule.metric_threshold) or
                    (rule.scale_direction == ScalingDirection.DOWN and trigger_value < rule.metric_threshold)
                )
            
            elif rule.trigger == ScalingTrigger.RESPONSE_TIME:
                trigger_value = metrics.get("response_time", 0)
                should_trigger = trigger_value > rule.metric_threshold
            
            elif rule.trigger == ScalingTrigger.ERROR_RATE:
                trigger_value = metrics.get("error_rate", 0)
                should_trigger = trigger_value > rule.metric_threshold
            
            elif rule.trigger == ScalingTrigger.QUEUE_DEPTH:
                trigger_value = metrics.get("queue_depth", 0)
                should_trigger = trigger_value > rule.metric_threshold
            
            if should_trigger and await self._should_scale(rule):
                return {
                    "rule": rule,
                    "trigger_value": trigger_value,
                    "reason": f"{rule.trigger.value} {trigger_value} {'>' if rule.scale_direction == ScalingDirection.UP else '<'} {rule.metric_threshold}"
                }
        
        return None
    
    async def _should_scale(self, rule: ScalingRule) -> bool:
        """Check if scaling should proceed based on cooldown and constraints"""
        
        # Check cooldown period
        time_since_last_scaling = (datetime.utcnow() - self.last_scaling_time).total_seconds()
        if time_since_last_scaling < rule.cooldown_period:
            return False
        
        # Check instance limits
        if rule.scale_direction == ScalingDirection.UP:
            if self.current_instances >= rule.max_instances:
                return False
        elif rule.scale_direction == ScalingDirection.DOWN:
            if self.current_instances <= rule.min_instances:
                return False
        
        return True
    
    @record_operation_metrics("auto_scaling")
    async def _execute_scaling_decision(self, decision: Dict):
        """Execute scaling decision"""
        
        rule = decision["rule"]
        trigger_value = decision["trigger_value"]
        reason = decision["reason"]
        
        logger.info("Executing scaling decision",
                   rule_name=rule.name,
                   direction=rule.scale_direction.value,
                   current_instances=self.current_instances,
                   trigger_value=trigger_value)
        
        self.is_scaling = True
        scaling_start = datetime.utcnow()
        
        try:
            # Calculate new instance count
            if rule.scale_direction == ScalingDirection.UP:
                new_instances = min(
                    rule.max_instances,
                    max(1, int(self.current_instances * rule.scale_factor))
                )
            elif rule.scale_direction == ScalingDirection.DOWN:
                new_instances = max(
                    rule.min_instances,
                    max(1, int(self.current_instances * rule.scale_factor))
                )
            else:  # MAINTAIN
                new_instances = self.current_instances
            
            if new_instances == self.current_instances:
                logger.info("No scaling needed", current_instances=self.current_instances)
                return
            
            # Record scaling event
            scaling_event = ScalingEvent(
                timestamp=scaling_start,
                trigger=rule.trigger,
                direction=rule.scale_direction,
                from_instances=self.current_instances,
                to_instances=new_instances,
                reason=reason,
                success=False,  # Will be updated
                metadata={"rule_name": rule.name, "trigger_value": trigger_value}
            )
            
            # Execute scaling operation
            success = await self._perform_scaling_operation(
                self.current_instances, 
                new_instances, 
                rule.scale_direction
            )
            
            if success:
                self.current_instances = new_instances
                self.last_scaling_time = datetime.utcnow()
                scaling_event.success = True
                
                logger.info("Scaling completed successfully",
                           from_instances=scaling_event.from_instances,
                           to_instances=scaling_event.to_instances,
                           duration=(datetime.utcnow() - scaling_start).total_seconds())
            else:
                logger.error("Scaling operation failed",
                           rule_name=rule.name,
                           from_instances=scaling_event.from_instances,
                           to_instances=scaling_event.to_instances)
            
            self.scaling_history.append(scaling_event)
            
        except Exception as e:
            logger.error("Scaling execution error", error=str(e))
        finally:
            self.is_scaling = False
    
    async def _perform_scaling_operation(self, from_instances: int, to_instances: int, 
                                       direction: ScalingDirection) -> bool:
        """Perform the actual scaling operation"""
        
        # Simulate scaling operation
        scaling_delay = abs(to_instances - from_instances) * 2  # 2 seconds per instance
        
        logger.info("Performing scaling operation",
                   from_instances=from_instances,
                   to_instances=to_instances,
                   estimated_duration=scaling_delay)
        
        # Simulate scaling time
        await asyncio.sleep(min(scaling_delay, 30))  # Cap at 30 seconds
        
        # In real implementation, this would:
        # - Update load balancer configuration
        # - Start/stop worker processes
        # - Update Kubernetes deployments
        # - Configure auto-scaling groups
        # - Update service discovery
        
        # Simulate 95% success rate
        import random
        return random.random() < 0.95
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule"""
        self.scaling_rules.append(rule)
        logger.info("Scaling rule added", rule_name=rule.name)
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove scaling rule by name"""
        self.scaling_rules = [r for r in self.scaling_rules if r.name != rule_name]
        logger.info("Scaling rule removed", rule_name=rule_name)
    
    def get_scaling_status(self) -> Dict:
        """Get current scaling status and statistics"""
        recent_events = [
            event for event in self.scaling_history 
            if (datetime.utcnow() - event.timestamp).total_seconds() < 3600
        ]
        
        successful_scalings = sum(1 for event in recent_events if event.success)
        
        return {
            "current_instances": self.current_instances,
            "is_scaling": self.is_scaling,
            "last_scaling_time": self.last_scaling_time.isoformat(),
            "active_rules": len([r for r in self.scaling_rules if r.enabled]),
            "recent_scaling_events": len(recent_events),
            "successful_scalings": successful_scalings,
            "scaling_success_rate": (successful_scalings / max(len(recent_events), 1)) * 100,
            "scaling_rules": [
                {
                    "name": rule.name,
                    "trigger": rule.trigger.value,
                    "threshold": rule.metric_threshold,
                    "direction": rule.scale_direction.value,
                    "enabled": rule.enabled
                }
                for rule in self.scaling_rules
            ]
        }
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict]:
        """Get scaling history for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "trigger": event.trigger.value,
                "direction": event.direction.value,
                "from_instances": event.from_instances,
                "to_instances": event.to_instances,
                "reason": event.reason,
                "success": event.success,
                "metadata": event.metadata
            }
            for event in self.scaling_history
            if event.timestamp > cutoff_time
        ]
    
    async def force_scaling(self, target_instances: int, reason: str = "Manual scaling"):
        """Force scaling to specific instance count"""
        if target_instances == self.current_instances:
            logger.info("No scaling needed for forced scaling", target_instances=target_instances)
            return
        
        direction = ScalingDirection.UP if target_instances > self.current_instances else ScalingDirection.DOWN
        
        # Create temporary rule for forced scaling
        force_rule = ScalingRule(
            name="force_scaling",
            trigger=ScalingTrigger.CUSTOM_METRIC,
            metric_threshold=0,
            scale_direction=direction,
            scale_factor=1.0,
            cooldown_period=0,
            min_instances=1,
            max_instances=100
        )
        
        decision = {
            "rule": force_rule,
            "trigger_value": target_instances,
            "reason": reason
        }
        
        await self._execute_scaling_decision(decision)


# Global auto-scaling engine instance
_global_autoscaler: Optional[AutoScalingEngine] = None


def get_auto_scaling_engine() -> AutoScalingEngine:
    """Get global auto-scaling engine instance"""
    global _global_autoscaler
    
    if _global_autoscaler is None:
        _global_autoscaler = AutoScalingEngine()
    
    return _global_autoscaler