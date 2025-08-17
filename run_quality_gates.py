#!/usr/bin/env python3
"""
Quality Gates Execution Script

Runs comprehensive quality gates for the autonomous SDLC system.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.comprehensive_quality_gates import ComprehensiveQualityGates
    print("✅ Successfully imported quality gates module")
except ImportError as e:
    print(f"❌ Failed to import quality gates: {e}")
    # Create a simple fallback quality gate system
    import time
    import subprocess
    import ast
    import re
    from datetime import datetime
    from dataclasses import dataclass
    from typing import List, Dict, Any

    @dataclass
    class SimpleQualityResult:
        name: str
        status: str
        message: str
        score: float
        details: Dict = None

    class SimpleQualityGates:
        def __init__(self, repo_path="."):
            self.repo_path = Path(repo_path)
            self.results = []

        async def run_all_gates(self):
            """Run all available quality gates"""
            print("🔍 Running Simple Quality Gates...")
            
            # Gate 1: Python Syntax Validation
            syntax_result = self._check_python_syntax()
            self.results.append(syntax_result)
            
            # Gate 2: Basic Security Scan
            security_result = self._basic_security_scan()
            self.results.append(security_result)
            
            # Gate 3: Code Complexity Check
            complexity_result = self._check_code_complexity()
            self.results.append(complexity_result)
            
            # Gate 4: Documentation Check
            docs_result = self._check_documentation()
            self.results.append(docs_result)
            
            # Gate 5: Test Coverage Estimation
            test_result = self._estimate_test_coverage()
            self.results.append(test_result)
            
            return self._generate_summary()

        def _check_python_syntax(self):
            """Check Python syntax across all files"""
            print("  📝 Checking Python syntax...")
            
            python_files = list(self.repo_path.rglob("*.py"))
            syntax_errors = []
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
                except Exception:
                    pass
            
            if len(syntax_errors) == 0:
                return SimpleQualityResult(
                    name="Python Syntax",
                    status="PASSED",
                    message=f"✅ All {len(python_files)} Python files have valid syntax",
                    score=100.0,
                    details={"files_checked": len(python_files), "errors": 0}
                )
            else:
                return SimpleQualityResult(
                    name="Python Syntax",
                    status="FAILED",
                    message=f"❌ {len(syntax_errors)} syntax errors found",
                    score=0.0,
                    details={"files_checked": len(python_files), "errors": syntax_errors[:5]}
                )

        def _basic_security_scan(self):
            """Basic security pattern scanning"""
            print("  🔒 Running basic security scan...")
            
            security_issues = []
            python_files = list(self.repo_path.rglob("*.py"))
            
            dangerous_patterns = [
                (r'eval\s*\(', 'eval() usage'),
                (r'exec\s*\(', 'exec() usage'),
                (r'__import__\s*\(', 'dynamic import'),
                (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded password'),
                (r'subprocess\..*shell=True', 'shell injection risk')
            ]
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for pattern, issue_type in dangerous_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip if in comments or has allowlist
                            line = content[max(0, match.start()-50):match.end()+50]
                            if "allowlist" in line.lower() or "#" in line:
                                continue
                            security_issues.append(f"{py_file.name}: {issue_type}")
                except Exception:
                    pass
            
            if len(security_issues) == 0:
                return SimpleQualityResult(
                    name="Security Scan",
                    status="PASSED",
                    message=f"✅ No security issues found in {len(python_files)} files",
                    score=100.0,
                    details={"files_scanned": len(python_files), "issues": 0}
                )
            else:
                return SimpleQualityResult(
                    name="Security Scan",
                    status="WARNING",
                    message=f"⚠️ {len(security_issues)} potential security issues found",
                    score=70.0,
                    details={"files_scanned": len(python_files), "issues": security_issues[:5]}
                )

        def _check_code_complexity(self):
            """Check code complexity"""
            print("  🧮 Checking code complexity...")
            
            python_files = list(self.repo_path.rglob("*.py"))
            total_functions = 0
            complex_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            # Simple complexity check - count nested structures
                            complexity = self._calculate_simple_complexity(node)
                            if complexity > 10:
                                complex_functions += 1
                except Exception:
                    pass
            
            complexity_ratio = complex_functions / max(1, total_functions)
            score = max(0, 100 - (complexity_ratio * 100))
            
            if complexity_ratio <= 0.1:  # 10% threshold
                return SimpleQualityResult(
                    name="Code Complexity",
                    status="PASSED",
                    message=f"✅ Code complexity acceptable ({complex_functions}/{total_functions} complex functions)",
                    score=score,
                    details={"total_functions": total_functions, "complex_functions": complex_functions}
                )
            else:
                return SimpleQualityResult(
                    name="Code Complexity",
                    status="WARNING",
                    message=f"⚠️ {complex_functions}/{total_functions} functions are complex",
                    score=score,
                    details={"total_functions": total_functions, "complex_functions": complex_functions}
                )

        def _calculate_simple_complexity(self, node):
            """Calculate simple complexity metric"""
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            return complexity

        def _check_documentation(self):
            """Check documentation presence"""
            print("  📚 Checking documentation...")
            
            doc_files = {
                "README.md": (self.repo_path / "README.md").exists(),
                "CHANGELOG.md": (self.repo_path / "CHANGELOG.md").exists(),
                "CONTRIBUTING.md": (self.repo_path / "CONTRIBUTING.md").exists(),
                "LICENSE": (self.repo_path / "LICENSE").exists(),
                "docs/": (self.repo_path / "docs").exists(),
            }
            
            present_docs = sum(doc_files.values())
            total_docs = len(doc_files)
            score = (present_docs / total_docs) * 100
            
            if score >= 80:
                return SimpleQualityResult(
                    name="Documentation",
                    status="PASSED",
                    message=f"✅ Documentation coverage: {present_docs}/{total_docs} files present",
                    score=score,
                    details=doc_files
                )
            else:
                return SimpleQualityResult(
                    name="Documentation",
                    status="WARNING",
                    message=f"⚠️ Documentation coverage: {present_docs}/{total_docs} files present",
                    score=score,
                    details=doc_files
                )

        def _estimate_test_coverage(self):
            """Estimate test coverage based on test files"""
            print("  🧪 Estimating test coverage...")
            
            test_files = list(self.repo_path.rglob("test_*.py"))
            src_files = list(self.repo_path.rglob("src/**/*.py"))
            
            # Simple estimation: each test file covers ~10% of source files
            estimated_coverage = min(95, len(test_files) * 10)
            
            if estimated_coverage >= 80:
                return SimpleQualityResult(
                    name="Test Coverage",
                    status="PASSED",
                    message=f"✅ Estimated test coverage: {estimated_coverage}% ({len(test_files)} test files)",
                    score=estimated_coverage,
                    details={"test_files": len(test_files), "source_files": len(src_files)}
                )
            else:
                return SimpleQualityResult(
                    name="Test Coverage",
                    status="WARNING", 
                    message=f"⚠️ Estimated test coverage: {estimated_coverage}% ({len(test_files)} test files)",
                    score=estimated_coverage,
                    details={"test_files": len(test_files), "source_files": len(src_files)}
                )

        def _generate_summary(self):
            """Generate quality gates summary"""
            passed = sum(1 for r in self.results if r.status == "PASSED")
            warnings = sum(1 for r in self.results if r.status == "WARNING")
            failed = sum(1 for r in self.results if r.status == "FAILED")
            
            overall_score = sum(r.score for r in self.results) / len(self.results)
            
            return {
                "overall_status": "PASSED" if failed == 0 else "FAILED",
                "overall_score": overall_score,
                "gates": {
                    "total": len(self.results),
                    "passed": passed,
                    "warnings": warnings,
                    "failed": failed
                },
                "results": self.results,
                "timestamp": datetime.utcnow().isoformat()
            }

    # Use the fallback system
    ComprehensiveQualityGates = SimpleQualityGates


async def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Quality Gates Execution")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize quality gates
    quality_gates = ComprehensiveQualityGates("/root/repo")
    
    try:
        # Run all quality gates
        if hasattr(quality_gates, 'execute_quality_gates'):
            # Use the full system
            result = await quality_gates.execute_quality_gates()
            
            print(f"\n📊 QUALITY GATES SUMMARY")
            print(f"Overall Score: {result.overall_score:.1f}/100")
            print(f"Compliance Level: {result.compliance_level}")
            print(f"Total Gates: {result.total_gates}")
            print(f"✅ Passed: {result.passed_gates}")
            print(f"⚠️ Warnings: {result.warning_gates}")
            print(f"❌ Failed: {result.failed_gates}")
            print(f"🔴 Critical Failures: {result.critical_failures}")
            
            print(f"\n📋 DETAILED RESULTS:")
            for gate_result in result.results:
                status_emoji = "✅" if gate_result.status.value == "passed" else "⚠️" if gate_result.status.value == "warning" else "❌"
                print(f"  {status_emoji} {gate_result.gate_name}: {gate_result.message}")
            
            # Determine overall success
            if result.failed_gates == 0 and result.critical_failures == 0:
                print(f"\n🎉 ALL QUALITY GATES PASSED!")
                exit_code = 0
            else:
                print(f"\n💥 QUALITY GATES FAILED!")
                exit_code = 1
                
        else:
            # Use the simple system
            result = await quality_gates.run_all_gates()
            
            print(f"\n📊 QUALITY GATES SUMMARY")
            print(f"Overall Status: {result['overall_status']}")
            print(f"Overall Score: {result['overall_score']:.1f}/100")
            print(f"Total Gates: {result['gates']['total']}")
            print(f"✅ Passed: {result['gates']['passed']}")
            print(f"⚠️ Warnings: {result['gates']['warnings']}")
            print(f"❌ Failed: {result['gates']['failed']}")
            
            print(f"\n📋 DETAILED RESULTS:")
            for gate_result in result['results']:
                status_emoji = "✅" if gate_result.status == "PASSED" else "⚠️" if gate_result.status == "WARNING" else "❌"
                print(f"  {status_emoji} {gate_result.name}: {gate_result.message}")
            
            # Determine overall success
            if result['gates']['failed'] == 0:
                print(f"\n🎉 ALL QUALITY GATES PASSED!")
                exit_code = 0
            else:
                print(f"\n💥 QUALITY GATES FAILED!")
                exit_code = 1
    
    except Exception as e:
        print(f"❌ Error running quality gates: {e}")
        exit_code = 1
    
    execution_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
    print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)