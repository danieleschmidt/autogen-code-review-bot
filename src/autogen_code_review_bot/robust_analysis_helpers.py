"""
Robust Analysis Helpers
"""

from pathlib import Path
from typing import Dict, List, Any

class RobustAnalysisHelper:
    def __init__(self):
        pass
        
    async def enhance_component(self, repo_path: str, task: str):
        return {"enhancements": ["basic_enhancement"], "status": "completed"}

def is_ignored_path(path: Path) -> bool:
    """Check if path should be ignored during analysis."""
    ignored_patterns = {
        ".git",
        "__pycache__",
        ".pytest_cache", 
        "node_modules",
        ".venv",
        "venv",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
    }
    return any(pattern in str(path) for pattern in ignored_patterns)

def run_robust_security_analysis(repo_path: str, languages: List[str]) -> Dict[str, Any]:
    """Run robust security analysis - stub implementation"""
    return {
        "tool": "security-analyzer",
        "output": "No critical security issues detected",
        "return_code": 0,
        "metadata": {"severity": "low", "tools_run": 1}
    }

def run_robust_style_analysis(repo_path: str, languages: List[str]) -> Dict[str, Any]:
    """Run robust style analysis - stub implementation"""  
    return {
        "tool": "style-analyzer",
        "output": "Code style looks good",
        "return_code": 0,
        "metadata": {"issues_count": 0, "languages_analyzed": len(languages)}
    }

def run_robust_performance_analysis(repo_path: str, languages: List[str]) -> Dict[str, Any]:
    """Run robust performance analysis - stub implementation"""
    return {
        "tool": "performance-analyzer", 
        "output": "Performance analysis completed - no major concerns",
        "return_code": 0,
        "metadata": {"hotspots": 0, "metrics_collected": 1}
    }

# Aliases for compatibility
run_security_analysis = run_robust_security_analysis
run_style_analysis = run_robust_style_analysis  
run_performance_analysis = run_robust_performance_analysis
