"""
Simple Metrics System
"""

import functools
import time
from typing import Dict, Any, Optional

class Counter:
    """Simple counter metric"""
    def __init__(self, name: str, description: str = "", labels: Optional[list] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.value = 0
    
    def inc(self, amount: int = 1):
        """Increment counter"""
        self.value += amount

class Gauge:
    """Simple gauge metric"""
    def __init__(self, name: str, description: str = "", labels: Optional[list] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.value = 0
    
    def set(self, value: float):
        """Set gauge value"""
        self.value = value

class Histogram:
    """Simple histogram metric"""
    def __init__(self, name: str, description: str = "", labels: Optional[list] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.observations = []
    
    def observe(self, value: float):
        """Record observation"""
        self.observations.append(value)

class MetricsRegistry:
    """Simple metrics registry"""
    def __init__(self):
        self._metrics = {}
    
    def counter(self, name: str, description: str = "", labels: Optional[list] = None):
        """Create or get counter"""
        if name not in self._metrics:
            self._metrics[name] = Counter(name, description, labels)
        return self._metrics[name]
    
    def gauge(self, name: str, description: str = "", labels: Optional[list] = None):
        """Create or get gauge"""
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, description, labels)
        return self._metrics[name]
    
    def histogram(self, name: str, description: str = "", labels: Optional[list] = None):
        """Create or get histogram"""
        if name not in self._metrics:
            self._metrics[name] = Histogram(name, description, labels)
        return self._metrics[name]
    
    def __setitem__(self, key: str, value: Any):
        """Support item assignment for backward compatibility"""
        self._metrics[key] = value
    
    def __getitem__(self, key: str):
        """Support item access"""
        return self._metrics[key]

# Global registry
_metrics_registry = MetricsRegistry()

def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry"""
    return _metrics_registry

def record_operation_metrics(operation_name: str):
    """Decorator to record operation metrics"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                _metrics_registry[operation_name] = {
                    "last_duration": duration,
                    "success": True,
                    "timestamp": start_time
                }
                return result
            except Exception as e:
                duration = time.time() - start_time
                _metrics_registry[operation_name] = {
                    "last_duration": duration,
                    "success": False,
                    "error": str(e),
                    "timestamp": start_time
                }
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                _metrics_registry[operation_name] = {
                    "last_duration": duration,
                    "success": True,
                    "timestamp": start_time
                }
                return result
            except Exception as e:
                duration = time.time() - start_time
                _metrics_registry[operation_name] = {
                    "last_duration": duration,
                    "success": False,
                    "error": str(e),
                    "timestamp": start_time
                }
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

def with_metrics(operation: str):
    """Decorator to record operation metrics - alias for record_operation_metrics"""
    return record_operation_metrics(operation)
