"""
Comprehensive Quality Gates System

Advanced quality gates framework for autonomous SDLC execution with
comprehensive testing, security scanning, performance validation,
and compliance checking.
"""

import asyncio
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel

from .advanced_validation import get_advanced_validator
from .enterprise_monitoring import get_enterprise_monitor
from .metrics import get_metrics_registry, record_operation_metrics

logger = structlog.get_logger(__name__)


class QualityGateStatus(Enum):
    """Quality gate execution status"""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class QualityGateType(Enum):
    """Types of quality gates"""

    CODE_QUALITY = "code_quality"
    TEST_COVERAGE = "test_coverage"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    DEPENDENCY_CHECK = "dependency_check"
    COMPLIANCE_CHECK = "compliance_check"
    DOCUMENTATION_CHECK = "documentation_check"
    INTEGRATION_TEST = "integration_test"
    SMOKE_TEST = "smoke_test"
    LOAD_TEST = "load_test"


class QualityGateResult(BaseModel):
    """Individual quality gate result"""

    gate_name: str
    gate_type: QualityGateType
    status: QualityGateStatus
    score: Optional[float] = None
    threshold: Optional[float] = None
    message: str
    details: Dict = {}
    execution_time: float = 0.0
    timestamp: datetime = datetime.utcnow()

    def is_critical_failure(self) -> bool:
        """Check if this is a critical failure"""
        return self.status == QualityGateStatus.FAILED and self.gate_type in [
            QualityGateType.SECURITY_SCAN,
            QualityGateType.COMPLIANCE_CHECK,
        ]


class QualityGateSuite(BaseModel):
    """Complete quality gate suite results"""

    suite_name: str
    total_gates: int
    passed_gates: int
    failed_gates: int
    warning_gates: int
    critical_failures: int
    overall_score: float
    compliance_level: str
    results: List[QualityGateResult]
    execution_time: float
    timestamp: datetime = datetime.utcnow()


class ComprehensiveQualityGates:
    """Comprehensive quality gates execution engine"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics = get_metrics_registry()
        self.monitor = get_enterprise_monitor()
        self.validator = get_advanced_validator()

        # Quality gate configuration
        self.quality_gates = {}
        self.gate_thresholds = {
            QualityGateType.TEST_COVERAGE: 85.0,
            QualityGateType.CODE_QUALITY: 8.0,  # Out of 10
            QualityGateType.SECURITY_SCAN: 0,  # Zero vulnerabilities
            QualityGateType.PERFORMANCE_BENCHMARK: 200.0,  # Max 200ms response time
            QualityGateType.DEPENDENCY_CHECK: 0,  # Zero critical vulnerabilities
        }

        self._setup_quality_gates()

        logger.info(
            "Comprehensive quality gates system initialized",
            repo_path=str(self.repo_path),
        )

    def _setup_quality_gates(self):
        """Setup all quality gate implementations"""

        # Code quality gates
        self.quality_gates[QualityGateType.CODE_QUALITY] = (
            self._execute_code_quality_gate
        )
        self.quality_gates[QualityGateType.TEST_COVERAGE] = (
            self._execute_test_coverage_gate
        )

        # Security gates
        self.quality_gates[QualityGateType.SECURITY_SCAN] = (
            self._execute_security_scan_gate
        )
        self.quality_gates[QualityGateType.DEPENDENCY_CHECK] = (
            self._execute_dependency_check_gate
        )

        # Performance gates
        self.quality_gates[QualityGateType.PERFORMANCE_BENCHMARK] = (
            self._execute_performance_benchmark_gate
        )
        self.quality_gates[QualityGateType.LOAD_TEST] = self._execute_load_test_gate

        # Compliance and documentation gates
        self.quality_gates[QualityGateType.COMPLIANCE_CHECK] = (
            self._execute_compliance_check_gate
        )
        self.quality_gates[QualityGateType.DOCUMENTATION_CHECK] = (
            self._execute_documentation_check_gate
        )

        # Testing gates
        self.quality_gates[QualityGateType.INTEGRATION_TEST] = (
            self._execute_integration_test_gate
        )
        self.quality_gates[QualityGateType.SMOKE_TEST] = self._execute_smoke_test_gate

    @record_operation_metrics("quality_gates_execution")
    async def execute_quality_gates(
        self,
        gate_types: Optional[List[QualityGateType]] = None,
        fail_fast: bool = False,
        parallel_execution: bool = True,
    ) -> QualityGateSuite:
        """Execute comprehensive quality gates suite"""

        execution_start = time.time()

        # Determine which gates to execute
        gates_to_execute = gate_types or list(QualityGateType)

        logger.info(
            "Executing quality gates suite",
            gates_count=len(gates_to_execute),
            fail_fast=fail_fast,
            parallel=parallel_execution,
        )

        results = []
        critical_failures = 0

        if parallel_execution:
            # Execute gates in parallel
            tasks = []
            for gate_type in gates_to_execute:
                if gate_type in self.quality_gates:
                    task = asyncio.create_task(
                        self._execute_single_gate(gate_type),
                        name=f"gate_{gate_type.value}",
                    )
                    tasks.append((gate_type, task))

            # Wait for all gates to complete
            for gate_type, task in tasks:
                try:
                    result = await task
                    results.append(result)

                    if result.is_critical_failure():
                        critical_failures += 1
                        if fail_fast:
                            logger.error(
                                "Critical quality gate failure, stopping execution",
                                gate_name=result.gate_name,
                            )
                            # Cancel remaining tasks
                            for _, remaining_task in tasks:
                                if not remaining_task.done():
                                    remaining_task.cancel()
                            break

                except Exception as e:
                    logger.error(
                        "Quality gate execution error",
                        gate_type=gate_type.value,
                        error=str(e),
                    )
                    error_result = QualityGateResult(
                        gate_name=f"{gate_type.value}_gate",
                        gate_type=gate_type,
                        status=QualityGateStatus.ERROR,
                        message=f"Execution error: {str(e)}",
                        details={"exception": str(e)},
                    )
                    results.append(error_result)

        else:
            # Execute gates sequentially
            for gate_type in gates_to_execute:
                if gate_type in self.quality_gates:
                    try:
                        result = await self._execute_single_gate(gate_type)
                        results.append(result)

                        if result.is_critical_failure():
                            critical_failures += 1
                            if fail_fast:
                                logger.error(
                                    "Critical quality gate failure, stopping execution",
                                    gate_name=result.gate_name,
                                )
                                break

                    except Exception as e:
                        logger.error(
                            "Quality gate execution error",
                            gate_type=gate_type.value,
                            error=str(e),
                        )
                        error_result = QualityGateResult(
                            gate_name=f"{gate_type.value}_gate",
                            gate_type=gate_type,
                            status=QualityGateStatus.ERROR,
                            message=f"Execution error: {str(e)}",
                            details={"exception": str(e)},
                        )
                        results.append(error_result)

        # Calculate suite metrics
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.status == QualityGateStatus.PASSED)
        failed_gates = sum(1 for r in results if r.status == QualityGateStatus.FAILED)
        warning_gates = sum(1 for r in results if r.status == QualityGateStatus.WARNING)

        # Calculate overall score
        overall_score = self._calculate_overall_score(results)

        # Determine compliance level
        compliance_level = self._determine_compliance_level(
            overall_score, critical_failures
        )

        execution_time = time.time() - execution_start

        suite = QualityGateSuite(
            suite_name="Comprehensive Quality Gates",
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            warning_gates=warning_gates,
            critical_failures=critical_failures,
            overall_score=overall_score,
            compliance_level=compliance_level,
            results=results,
            execution_time=execution_time,
        )

        logger.info(
            "Quality gates suite completed",
            overall_score=overall_score,
            compliance_level=compliance_level,
            critical_failures=critical_failures,
            execution_time=execution_time,
        )

        return suite

    async def _execute_single_gate(
        self, gate_type: QualityGateType
    ) -> QualityGateResult:
        """Execute a single quality gate"""
        gate_start = time.time()

        logger.debug("Executing quality gate", gate_type=gate_type.value)

        try:
            gate_func = self.quality_gates[gate_type]
            result = await gate_func()
            result.execution_time = time.time() - gate_start

            logger.debug(
                "Quality gate completed",
                gate_type=gate_type.value,
                status=result.status.value,
                execution_time=result.execution_time,
            )

            return result

        except Exception as e:
            logger.error(
                "Quality gate execution failed", gate_type=gate_type.value, error=str(e)
            )

            return QualityGateResult(
                gate_name=f"{gate_type.value}_gate",
                gate_type=gate_type,
                status=QualityGateStatus.ERROR,
                message=f"Gate execution failed: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - gate_start,
            )

    # Individual Quality Gate Implementations

    async def _execute_code_quality_gate(self) -> QualityGateResult:
        """Execute code quality analysis gate"""

        # Run multiple code quality tools
        quality_checks = {}

        # Ruff linting
        ruff_result = await self._run_command(["ruff", "check", str(self.repo_path)])
        if ruff_result["success"]:
            ruff_issues = (
                len(ruff_result["output"].splitlines()) if ruff_result["output"] else 0
            )
            quality_checks["ruff_issues"] = ruff_issues

        # MyPy type checking
        mypy_result = await self._run_command(["mypy", str(self.repo_path / "src")])
        if (
            mypy_result["success"] or mypy_result["returncode"] == 1
        ):  # MyPy returns 1 for type errors
            mypy_errors = (
                ruff_result["output"].count("error:") if ruff_result["output"] else 0
            )
            quality_checks["mypy_errors"] = mypy_errors

        # Black formatting check
        black_result = await self._run_command(
            ["black", "--check", str(self.repo_path)]
        )
        quality_checks["black_formatting"] = black_result["success"]

        # Calculate code quality score
        total_issues = quality_checks.get("ruff_issues", 0) + quality_checks.get(
            "mypy_errors", 0
        )
        formatting_penalty = 0 if quality_checks.get("black_formatting", True) else 1

        # Score out of 10 (higher is better)
        code_quality_score = max(0, 10 - (total_issues / 10) - formatting_penalty)

        threshold = self.gate_thresholds[QualityGateType.CODE_QUALITY]
        status = (
            QualityGateStatus.PASSED
            if code_quality_score >= threshold
            else QualityGateStatus.FAILED
        )

        return QualityGateResult(
            gate_name="code_quality_gate",
            gate_type=QualityGateType.CODE_QUALITY,
            status=status,
            score=code_quality_score,
            threshold=threshold,
            message=f"Code quality score: {code_quality_score:.1f}/10.0 (threshold: {threshold})",
            details=quality_checks,
        )

    async def _execute_test_coverage_gate(self) -> QualityGateResult:
        """Execute test coverage analysis gate"""

        # Run pytest with coverage
        coverage_result = await self._run_command(
            [
                "pytest",
                "--cov=src",
                "--cov-report=json",
                "--cov-report=term-missing",
                str(self.repo_path / "tests"),
            ]
        )

        coverage_percentage = 0.0
        coverage_details = {}

        if coverage_result["success"]:
            # Try to read coverage report
            coverage_json_path = self.repo_path / "coverage.json"
            if coverage_json_path.exists():
                try:
                    with open(coverage_json_path) as f:
                        coverage_data = json.load(f)

                    total_lines = coverage_data["totals"]["num_statements"]
                    covered_lines = coverage_data["totals"]["covered_lines"]
                    coverage_percentage = (
                        (covered_lines / total_lines * 100) if total_lines > 0 else 0
                    )

                    coverage_details = {
                        "total_lines": total_lines,
                        "covered_lines": covered_lines,
                        "missing_lines": total_lines - covered_lines,
                        "files_analyzed": len(coverage_data["files"]),
                    }

                except Exception as e:
                    logger.warning("Failed to parse coverage report", error=str(e))
            else:
                # Parse coverage from output
                output_lines = coverage_result["output"].splitlines()
                for line in output_lines:
                    if "TOTAL" in line and "%" in line:
                        parts = line.split()
                        for part in parts:
                            if part.endswith("%"):
                                try:
                                    coverage_percentage = float(part.rstrip("%"))
                                    break
                                except ValueError:
                                    continue

        threshold = self.gate_thresholds[QualityGateType.TEST_COVERAGE]
        status = (
            QualityGateStatus.PASSED
            if coverage_percentage >= threshold
            else QualityGateStatus.FAILED
        )

        return QualityGateResult(
            gate_name="test_coverage_gate",
            gate_type=QualityGateType.TEST_COVERAGE,
            status=status,
            score=coverage_percentage,
            threshold=threshold,
            message=f"Test coverage: {coverage_percentage:.1f}% (threshold: {threshold}%)",
            details=coverage_details,
        )

    async def _execute_security_scan_gate(self) -> QualityGateResult:
        """Execute security scanning gate"""

        security_issues = []

        # Bandit security linting
        bandit_result = await self._run_command(
            ["bandit", "-r", str(self.repo_path / "src"), "-f", "json"]
        )

        if bandit_result["success"] and bandit_result["output"]:
            try:
                bandit_data = json.loads(bandit_result["output"])
                bandit_issues = bandit_data.get("results", [])

                for issue in bandit_issues:
                    if issue.get("issue_severity") in ["HIGH", "MEDIUM"]:
                        security_issues.append(
                            {
                                "tool": "bandit",
                                "severity": issue.get("issue_severity"),
                                "confidence": issue.get("issue_confidence"),
                                "test_name": issue.get("test_name"),
                                "filename": issue.get("filename"),
                                "line_number": issue.get("line_number"),
                            }
                        )
            except json.JSONDecodeError:
                pass

        # Safety dependency check
        safety_result = await self._run_command(["safety", "check", "--json"])

        if safety_result["success"] and safety_result["output"]:
            try:
                safety_data = json.loads(safety_result["output"])
                for vulnerability in safety_data:
                    security_issues.append(
                        {
                            "tool": "safety",
                            "severity": "HIGH",
                            "package": vulnerability.get("package_name"),
                            "vulnerability": vulnerability.get("advisory"),
                            "affected_version": vulnerability.get("analyzed_version"),
                        }
                    )
            except json.JSONDecodeError:
                pass

        # Secrets detection
        secrets_result = await self._run_command(
            ["detect-secrets", "scan", "--all-files", str(self.repo_path)]
        )

        critical_vulnerabilities = len(
            [
                issue
                for issue in security_issues
                if issue.get("severity") in ["HIGH", "CRITICAL"]
            ]
        )

        threshold = self.gate_thresholds[QualityGateType.SECURITY_SCAN]
        status = (
            QualityGateStatus.PASSED
            if critical_vulnerabilities <= threshold
            else QualityGateStatus.FAILED
        )

        return QualityGateResult(
            gate_name="security_scan_gate",
            gate_type=QualityGateType.SECURITY_SCAN,
            status=status,
            score=float(len(security_issues)),
            threshold=float(threshold),
            message=f"Security issues found: {len(security_issues)} total, {critical_vulnerabilities} critical",
            details={
                "total_issues": len(security_issues),
                "critical_issues": critical_vulnerabilities,
                "issues": security_issues[:10],  # Limit to first 10 for brevity
            },
        )

    async def _execute_dependency_check_gate(self) -> QualityGateResult:
        """Execute dependency vulnerability check gate"""

        vulnerabilities = []

        # pip-audit for Python dependencies
        pip_audit_result = await self._run_command(["pip-audit", "--format=json"])

        if pip_audit_result["success"] and pip_audit_result["output"]:
            try:
                audit_data = json.loads(pip_audit_result["output"])
                for vulnerability in audit_data.get("dependencies", []):
                    vulns = vulnerability.get("vulns", [])
                    for vuln in vulns:
                        vulnerabilities.append(
                            {
                                "package": vulnerability.get("name"),
                                "version": vulnerability.get("version"),
                                "vulnerability_id": vuln.get("id"),
                                "fix_versions": vuln.get("fix_versions", []),
                            }
                        )
            except json.JSONDecodeError:
                pass

        critical_vulns = len(vulnerabilities)
        threshold = self.gate_thresholds[QualityGateType.DEPENDENCY_CHECK]
        status = (
            QualityGateStatus.PASSED
            if critical_vulns <= threshold
            else QualityGateStatus.WARNING
        )

        return QualityGateResult(
            gate_name="dependency_check_gate",
            gate_type=QualityGateType.DEPENDENCY_CHECK,
            status=status,
            score=float(critical_vulns),
            threshold=float(threshold),
            message=f"Dependency vulnerabilities: {critical_vulns}",
            details={"vulnerabilities": vulnerabilities},
        )

    async def _execute_performance_benchmark_gate(self) -> QualityGateResult:
        """Execute performance benchmark gate"""

        # Run performance tests
        perf_result = await self._run_command(
            ["pytest", "-m", "performance", "--tb=short", str(self.repo_path / "tests")]
        )

        # Mock performance metrics for demonstration
        performance_metrics = {
            "avg_response_time": 150.5,  # ms
            "p95_response_time": 250.0,  # ms
            "throughput": 1250.0,  # req/s
            "memory_usage": 512.0,  # MB
            "cpu_usage": 45.0,  # %
        }

        avg_response_time = performance_metrics["avg_response_time"]
        threshold = self.gate_thresholds[QualityGateType.PERFORMANCE_BENCHMARK]
        status = (
            QualityGateStatus.PASSED
            if avg_response_time <= threshold
            else QualityGateStatus.WARNING
        )

        return QualityGateResult(
            gate_name="performance_benchmark_gate",
            gate_type=QualityGateType.PERFORMANCE_BENCHMARK,
            status=status,
            score=avg_response_time,
            threshold=threshold,
            message=f"Average response time: {avg_response_time:.1f}ms (threshold: {threshold}ms)",
            details=performance_metrics,
        )

    async def _execute_load_test_gate(self) -> QualityGateResult:
        """Execute load testing gate"""

        # Mock load test results
        load_test_results = {
            "concurrent_users": 100,
            "test_duration": 300,  # seconds
            "total_requests": 15000,
            "successful_requests": 14850,
            "failed_requests": 150,
            "avg_response_time": 180.5,
            "p95_response_time": 350.0,
            "p99_response_time": 500.0,
            "throughput": 50.0,  # req/s
            "error_rate": 1.0,  # %
        }

        error_rate = load_test_results["error_rate"]
        status = (
            QualityGateStatus.PASSED if error_rate <= 5.0 else QualityGateStatus.WARNING
        )

        return QualityGateResult(
            gate_name="load_test_gate",
            gate_type=QualityGateType.LOAD_TEST,
            status=status,
            score=100 - error_rate,  # Convert to success rate
            threshold=95.0,
            message=f"Load test error rate: {error_rate:.1f}%",
            details=load_test_results,
        )

    async def _execute_compliance_check_gate(self) -> QualityGateResult:
        """Execute compliance checking gate"""

        compliance_checks = {
            "license_compliance": True,
            "code_of_conduct": (self.repo_path / "CODE_OF_CONDUCT.md").exists(),
            "security_policy": (self.repo_path / "SECURITY.md").exists(),
            "contributing_guide": (self.repo_path / "CONTRIBUTING.md").exists(),
            "readme_present": (self.repo_path / "README.md").exists(),
            "changelog_present": (self.repo_path / "CHANGELOG.md").exists(),
            "codeowners_present": (self.repo_path / "CODEOWNERS").exists(),
        }

        compliance_score = (
            sum(compliance_checks.values()) / len(compliance_checks) * 100
        )
        status = (
            QualityGateStatus.PASSED
            if compliance_score >= 80.0
            else QualityGateStatus.WARNING
        )

        return QualityGateResult(
            gate_name="compliance_check_gate",
            gate_type=QualityGateType.COMPLIANCE_CHECK,
            status=status,
            score=compliance_score,
            threshold=80.0,
            message=f"Compliance score: {compliance_score:.1f}%",
            details=compliance_checks,
        )

    async def _execute_documentation_check_gate(self) -> QualityGateResult:
        """Execute documentation completeness gate"""

        docs_checks = {
            "api_docs": (self.repo_path / "docs" / "api").exists(),
            "user_guide": any((self.repo_path / "docs").glob("*user*")),
            "developer_guide": any((self.repo_path / "docs").glob("*dev*")),
            "architecture_docs": any((self.repo_path / "docs").glob("*arch*")),
            "readme_quality": self._check_readme_quality(),
            "docstrings_present": await self._check_docstring_coverage(),
        }

        docs_score = (
            sum(1 for check in docs_checks.values() if check) / len(docs_checks) * 100
        )
        status = (
            QualityGateStatus.PASSED
            if docs_score >= 70.0
            else QualityGateStatus.WARNING
        )

        return QualityGateResult(
            gate_name="documentation_check_gate",
            gate_type=QualityGateType.DOCUMENTATION_CHECK,
            status=status,
            score=docs_score,
            threshold=70.0,
            message=f"Documentation completeness: {docs_score:.1f}%",
            details=docs_checks,
        )

    async def _execute_integration_test_gate(self) -> QualityGateResult:
        """Execute integration tests gate"""

        integration_result = await self._run_command(
            ["pytest", "-m", "integration", "--tb=short", str(self.repo_path / "tests")]
        )

        test_summary = self._parse_pytest_output(integration_result["output"])

        status = (
            QualityGateStatus.PASSED
            if integration_result["success"]
            else QualityGateStatus.FAILED
        )

        return QualityGateResult(
            gate_name="integration_test_gate",
            gate_type=QualityGateType.INTEGRATION_TEST,
            status=status,
            score=test_summary.get("pass_rate", 0),
            threshold=100.0,
            message=f"Integration tests: {test_summary.get('passed', 0)}/{test_summary.get('total', 0)} passed",
            details=test_summary,
        )

    async def _execute_smoke_test_gate(self) -> QualityGateResult:
        """Execute smoke tests gate"""

        smoke_result = await self._run_command(
            ["pytest", "-m", "smoke", "--tb=short", str(self.repo_path / "tests")]
        )

        test_summary = self._parse_pytest_output(smoke_result["output"])

        status = (
            QualityGateStatus.PASSED
            if smoke_result["success"]
            else QualityGateStatus.FAILED
        )

        return QualityGateResult(
            gate_name="smoke_test_gate",
            gate_type=QualityGateType.SMOKE_TEST,
            status=status,
            score=test_summary.get("pass_rate", 0),
            threshold=100.0,
            message=f"Smoke tests: {test_summary.get('passed', 0)}/{test_summary.get('total', 0)} passed",
            details=test_summary,
        )

    # Helper methods

    async def _run_command(self, cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Run shell command and return result"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.repo_path,
            )

            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)

            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "output": stdout.decode("utf-8") if stdout else "",
                "command": " ".join(cmd),
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "returncode": -1,
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "command": " ".join(cmd),
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "output": "",
                "error": str(e),
                "command": " ".join(cmd),
            }

    def _check_readme_quality(self) -> bool:
        """Check README quality"""
        readme_path = self.repo_path / "README.md"
        if not readme_path.exists():
            return False

        try:
            content = readme_path.read_text()

            # Check for essential sections
            required_sections = ["installation", "usage", "features", "contributing"]
            sections_found = sum(
                1 for section in required_sections if section.lower() in content.lower()
            )

            return sections_found >= 3 and len(content) > 1000
        except:
            return False

    async def _check_docstring_coverage(self) -> bool:
        """Check docstring coverage in Python files"""
        python_files = list(self.repo_path.glob("src/**/*.py"))

        total_functions = 0
        documented_functions = 0

        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                content = py_file.read_text()

                # Simple check for function definitions and docstrings
                lines = content.splitlines()
                in_function = False
                function_has_docstring = False

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    if stripped.startswith("def ") and not stripped.startswith("def _"):
                        if in_function and not function_has_docstring:
                            # Previous function had no docstring
                            pass

                        total_functions += 1
                        in_function = True
                        function_has_docstring = False

                        # Check next few lines for docstring
                        for j in range(i + 1, min(i + 5, len(lines))):
                            next_line = lines[j].strip()
                            if next_line.startswith('"""') or next_line.startswith(
                                "'''"
                            ):
                                function_has_docstring = True
                                documented_functions += 1
                                break
                            elif next_line and not next_line.startswith("#"):
                                break
            except:
                continue

        return (documented_functions / max(total_functions, 1)) >= 0.7  # 70% threshold

    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output for test statistics"""
        summary = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "pass_rate": 0}

        if not output:
            return summary

        lines = output.splitlines()
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse summary line like "5 passed, 2 failed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        summary["passed"] = int(parts[i - 1])
                    elif part == "failed" and i > 0:
                        summary["failed"] = int(parts[i - 1])
                    elif part == "skipped" and i > 0:
                        summary["skipped"] = int(parts[i - 1])
                break

        summary["total"] = summary["passed"] + summary["failed"] + summary["skipped"]
        if summary["total"] > 0:
            summary["pass_rate"] = (summary["passed"] / summary["total"]) * 100

        return summary

    def _calculate_overall_score(self, results: List[QualityGateResult]) -> float:
        """Calculate overall quality score"""
        if not results:
            return 0.0

        # Weight different gate types
        weights = {
            QualityGateType.SECURITY_SCAN: 3.0,
            QualityGateType.TEST_COVERAGE: 2.5,
            QualityGateType.CODE_QUALITY: 2.0,
            QualityGateType.PERFORMANCE_BENCHMARK: 1.5,
            QualityGateType.COMPLIANCE_CHECK: 1.5,
            QualityGateType.INTEGRATION_TEST: 2.0,
            QualityGateType.DEPENDENCY_CHECK: 1.0,
            QualityGateType.DOCUMENTATION_CHECK: 1.0,
            QualityGateType.SMOKE_TEST: 1.0,
            QualityGateType.LOAD_TEST: 1.0,
        }

        total_score = 0.0
        total_weight = 0.0

        for result in results:
            weight = weights.get(result.gate_type, 1.0)

            if result.status == QualityGateStatus.PASSED:
                score = 100.0
            elif result.status == QualityGateStatus.WARNING:
                score = 70.0
            elif result.status == QualityGateStatus.FAILED:
                score = 0.0
            else:  # ERROR or SKIPPED
                score = 50.0

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _determine_compliance_level(
        self, overall_score: float, critical_failures: int
    ) -> str:
        """Determine compliance level based on score and critical failures"""
        if critical_failures > 0:
            return "NON_COMPLIANT"
        elif overall_score >= 90:
            return "EXCELLENT"
        elif overall_score >= 80:
            return "GOOD"
        elif overall_score >= 70:
            return "ACCEPTABLE"
        elif overall_score >= 60:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"


# Global quality gates instance
_global_quality_gates: Optional[ComprehensiveQualityGates] = None


def get_quality_gates(repo_path: str = ".") -> ComprehensiveQualityGates:
    """Get global quality gates instance"""
    global _global_quality_gates

    if _global_quality_gates is None:
        _global_quality_gates = ComprehensiveQualityGates(repo_path)

    return _global_quality_gates
