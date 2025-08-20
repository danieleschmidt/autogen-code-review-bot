#!/usr/bin/env python3
"""
Analysis helper functions for the AutoGen Code Review Bot.

This module contains the actual implementation of security, style, and performance analysis.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Any
import structlog

from .linter_config import LinterConfig
from .models import AnalysisSection
from .robust_error_handling import ErrorSeverity, robust_operation

logger = structlog.get_logger(__name__)


class RepositoryAnalyzer:
    """Repository analysis utilities for autonomous SDLC execution"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    async def analyze_repository_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and patterns"""
        repo_path = Path(repo_path)
        
        analysis = {
            "file_counts": self._count_files_by_type(repo_path),
            "directory_structure": self._analyze_directory_structure(repo_path),
            "languages": self._detect_languages(repo_path),
            "frameworks": self._detect_frameworks(repo_path),
            "patterns": self._identify_patterns(repo_path)
        }
        
        return analysis
    
    def _count_files_by_type(self, repo_path: Path) -> Dict[str, int]:
        """Count files by type"""
        counts = {
            "python": 0,
            "javascript": 0,
            "typescript": 0,
            "go": 0,
            "java": 0,
            "config": 0,
            "test": 0,
            "docs": 0,
            "total": 0
        }
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not is_ignored_path(file_path):
                counts["total"] += 1
                
                suffix = file_path.suffix.lower()
                name = file_path.name.lower()
                
                if suffix == ".py":
                    counts["python"] += 1
                elif suffix in [".js", ".jsx"]:
                    counts["javascript"] += 1
                elif suffix in [".ts", ".tsx"]:
                    counts["typescript"] += 1
                elif suffix == ".go":
                    counts["go"] += 1
                elif suffix == ".java":
                    counts["java"] += 1
                elif suffix in [".yaml", ".yml", ".json", ".toml", ".ini"]:
                    counts["config"] += 1
                elif "test" in name or "spec" in name:
                    counts["test"] += 1
                elif suffix in [".md", ".rst", ".txt"]:
                    counts["docs"] += 1
        
        return counts
    
    def _analyze_directory_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze directory structure"""
        structure = {
            "has_src": (repo_path / "src").exists(),
            "has_tests": any((repo_path / d).exists() for d in ["tests", "test"]),
            "has_docs": (repo_path / "docs").exists(),
            "has_config": any((repo_path / f).exists() for f in ["pyproject.toml", "setup.py", "package.json"]),
            "has_docker": (repo_path / "Dockerfile").exists(),
            "has_ci": (repo_path / ".github").exists(),
            "depth": self._calculate_max_depth(repo_path)
        }
        
        return structure
    
    def _detect_languages(self, repo_path: Path) -> List[str]:
        """Detect programming languages in use"""
        languages = set()
        
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not is_ignored_path(file_path):
                suffix = file_path.suffix.lower()
                
                if suffix == ".py":
                    languages.add("python")
                elif suffix in [".js", ".jsx"]:
                    languages.add("javascript")
                elif suffix in [".ts", ".tsx"]:
                    languages.add("typescript")
                elif suffix == ".go":
                    languages.add("go")
                elif suffix == ".java":
                    languages.add("java")
                elif suffix in [".cpp", ".cc", ".cxx"]:
                    languages.add("cpp")
                elif suffix == ".rs":
                    languages.add("rust")
        
        return list(languages)
    
    def _detect_frameworks(self, repo_path: Path) -> List[str]:
        """Detect frameworks and libraries in use"""
        frameworks = set()
        
        # Check package files
        package_files = [
            ("requirements.txt", "python"),
            ("pyproject.toml", "python"),
            ("package.json", "javascript"),
            ("go.mod", "go"),
            ("Cargo.toml", "rust")
        ]
        
        for file_name, lang in package_files:
            file_path = repo_path / file_name
            if file_path.exists():
                try:
                    content = file_path.read_text().lower()
                    
                    # Python frameworks
                    if lang == "python":
                        if "flask" in content:
                            frameworks.add("flask")
                        if "django" in content:
                            frameworks.add("django")
                        if "fastapi" in content:
                            frameworks.add("fastapi")
                        if "pytest" in content:
                            frameworks.add("pytest")
                        if "autogen" in content:
                            frameworks.add("autogen")
                    
                    # JavaScript frameworks
                    if lang == "javascript":
                        if "react" in content:
                            frameworks.add("react")
                        if "vue" in content:
                            frameworks.add("vue")
                        if "express" in content:
                            frameworks.add("express")
                        if "next" in content:
                            frameworks.add("nextjs")
                        
                except:
                    continue
        
        return list(frameworks)
    
    def _identify_patterns(self, repo_path: Path) -> List[str]:
        """Identify architectural and code patterns"""
        patterns = set()
        
        structure = self._analyze_directory_structure(repo_path)
        
        if structure["has_tests"]:
            patterns.add("test_driven_development")
        
        if structure["has_docker"]:
            patterns.add("containerization")
        
        if structure["has_ci"]:
            patterns.add("continuous_integration")
        
        if structure["has_src"]:
            patterns.add("source_separation")
        
        # Check for specific patterns in Python files
        for py_file in repo_path.rglob("*.py"):
            if is_ignored_path(py_file):
                continue
                
            try:
                content = py_file.read_text().lower()
                
                if "async def" in content:
                    patterns.add("asynchronous_programming")
                
                if "class.*factory" in content:
                    patterns.add("factory_pattern")
                
                if "abstractmethod" in content:
                    patterns.add("abstract_classes")
                
                if "@dataclass" in content:
                    patterns.add("dataclasses")
                
                if "pydantic" in content:
                    patterns.add("data_validation")
                    
            except:
                continue
        
        return list(patterns)
    
    def _calculate_max_depth(self, repo_path: Path) -> int:
        """Calculate maximum directory depth"""
        max_depth = 0
        
        for file_path in repo_path.rglob("*"):
            if not is_ignored_path(file_path):
                depth = len(file_path.relative_to(repo_path).parts) - 1
                max_depth = max(max_depth, depth)
        
        return max_depth


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


@robust_operation(
    component="security_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.HIGH,
    retry_count=1,
)
def run_security_analysis(repo_path: str, languages: List[str]) -> AnalysisSection:
    """Run security analysis using available tools."""
    try:
        results = []

        # Python security analysis
        if "python" in languages:
            try:
                result = subprocess.run(
                    ["bandit", "-r", repo_path, "-f", "txt"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"Bandit (Python Security): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("Bandit: Tool not available or timed out")

        # Generic secret scanning
        try:
            result = subprocess.run(
                ["detect-secrets", "scan", "--all-files", repo_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=repo_path,
            )
            if result.stdout and "No secrets were detected" not in result.stdout:
                results.append(f"Secret Detection: {result.stdout[:500]}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results.append("Secret Detection: Tool not available or timed out")

        # Safety check for Python dependencies
        if "python" in languages:
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=repo_path,
                )
                if result.stdout and result.stdout != "[]":
                    results.append(f"Safety (Dependencies): {result.stdout[:500]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("Safety: Tool not available or timed out")

        output = "\n\n".join(results) if results else "No security issues detected"
        severity = (
            "high" if any("CRITICAL" in r or "HIGH" in r for r in results) else "low"
        )

        return AnalysisSection(
            tool="security-analyzer",
            output=output,
            return_code=0,
            metadata={"severity": severity, "tools_run": len(results)},
        )

    except Exception as e:
        return AnalysisSection(
            tool="security-analyzer",
            output=f"Security analysis failed: {e}",
            return_code=1,
            metadata={"error": str(e)},
        )


@robust_operation(
    component="style_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.MEDIUM,
    retry_count=2,
)
def run_style_analysis(
    repo_path: str, languages: List[str], linter_config: LinterConfig
) -> AnalysisSection:
    """Run style analysis using language-specific linters."""
    try:
        results = []

        # Python style analysis
        if "python" in languages:
            linter = getattr(linter_config, "python", "ruff")
            try:
                if linter == "ruff":
                    result = subprocess.run(
                        ["ruff", "check", repo_path, "--output-format", "text"],
                        capture_output=True,
                        text=True,
                        timeout=180,
                        cwd=repo_path,
                    )
                    if result.stdout:
                        results.append(f"Ruff (Python): {result.stdout[:1000]}")
                elif linter == "flake8":
                    result = subprocess.run(
                        ["flake8", repo_path],
                        capture_output=True,
                        text=True,
                        timeout=180,
                        cwd=repo_path,
                    )
                    if result.stdout:
                        results.append(f"Flake8 (Python): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append(
                    f"Python linter ({linter}): Tool not available or timed out"
                )

        # JavaScript/TypeScript style analysis
        if "javascript" in languages or "typescript" in languages:
            try:
                result = subprocess.run(
                    ["eslint", "--ext", ".js,.ts,.jsx,.tsx", repo_path],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"ESLint (JS/TS): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("ESLint: Tool not available or timed out")

        # Go style analysis
        if "go" in languages:
            try:
                result = subprocess.run(
                    ["golangci-lint", "run", repo_path],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"golangci-lint (Go): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("golangci-lint: Tool not available or timed out")

        output = "\n\n".join(results) if results else "No style issues detected"
        issues_count = len(
            [r for r in results if "error" in r.lower() or "warning" in r.lower()]
        )

        return AnalysisSection(
            tool="style-analyzer",
            output=output,
            return_code=0,
            metadata={
                "issues_count": issues_count,
                "languages_analyzed": len(languages),
            },
        )

    except Exception as e:
        return AnalysisSection(
            tool="style-analyzer",
            output=f"Style analysis failed: {e}",
            return_code=1,
            metadata={"error": str(e)},
        )


@robust_operation(
    component="performance_analysis",
    operation="run_analysis",
    severity=ErrorSeverity.LOW,
    retry_count=1,
)
def run_performance_analysis(repo_path: str, languages: List[str]) -> AnalysisSection:
    """Run performance analysis and complexity metrics."""
    try:
        results = []

        # Python complexity analysis
        if "python" in languages:
            try:
                result = subprocess.run(
                    ["radon", "cc", repo_path, "-s"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=repo_path,
                )
                if result.stdout:
                    results.append(f"Radon Complexity (Python): {result.stdout[:1000]}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                results.append("Radon: Tool not available or timed out")

        # Generic file size analysis
        try:
            large_files = []
            for file_path in Path(repo_path).rglob("*"):
                if file_path.is_file() and not is_ignored_path(file_path):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb > 1.0:  # Files larger than 1MB
                        large_files.append(f"{file_path.name}: {size_mb:.1f}MB")

            if large_files:
                results.append("Large files detected:\n" + "\n".join(large_files[:10]))
        except Exception:
            pass

        # Simple code metrics
        try:
            total_lines = 0
            total_files = 0

            for file_path in Path(repo_path).rglob("*"):
                if (
                    file_path.is_file()
                    and not is_ignored_path(file_path)
                    and file_path.suffix
                    in {".py", ".js", ".ts", ".go", ".java", ".cpp", ".c"}
                ):

                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            lines = len(f.readlines())
                            total_lines += lines
                            total_files += 1
                    except Exception:
                        continue

            if total_files > 0:
                avg_lines = total_lines / total_files
                results.append(
                    f"Code metrics: {total_files} files, {total_lines} lines, avg {avg_lines:.1f} lines/file"
                )
        except Exception:
            pass

        output = (
            "\n\n".join(results)
            if results
            else "Performance analysis completed - no major concerns"
        )
        hotspots = len(
            [r for r in results if "complex" in r.lower() or "large" in r.lower()]
        )

        return AnalysisSection(
            tool="performance-analyzer",
            output=output,
            return_code=0,
            metadata={"hotspots": hotspots, "metrics_collected": len(results)},
        )

    except Exception as e:
        return AnalysisSection(
            tool="performance-analyzer",
            output=f"Performance analysis failed: {e}",
            return_code=1,
            metadata={"error": str(e)},
        )
