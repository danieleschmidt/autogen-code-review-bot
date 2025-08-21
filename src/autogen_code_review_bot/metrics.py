"""
Simple Metrics System
"""

import functools
import time
from typing import Dict, Any

# Simple global metrics registry
_metrics_registry = {}

def get_metrics_registry() -> Dict[str, Any]:
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

# Alias for compatibility 
with_metrics = record_operation_metrics
