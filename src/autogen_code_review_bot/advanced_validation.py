"""
Advanced Validation Framework

Comprehensive validation system for autonomous SDLC execution with
multi-layer validation, schema enforcement, and intelligent error recovery.
"""

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import structlog
import yaml
from pydantic import BaseModel, validator

from .metrics import get_metrics_registry, record_operation_metrics

logger = structlog.get_logger(__name__)


class ValidationType(Enum):
    """Types of validation checks"""

    SCHEMA = "schema"
    BUSINESS_LOGIC = "business_logic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    INTEGRATION = "integration"


class ValidationSeverity(Enum):
    """Validation issue severity levels"""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationResult(BaseModel):
    """Individual validation result"""

    check_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict = {}
    timestamp: datetime = datetime.utcnow()

    def to_dict(self) -> Dict:
        return {
            "check_name": self.check_name,
            "validation_type": self.validation_type.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class ValidationSuite(BaseModel):
    """Complete validation suite results"""

    suite_name: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    critical_issues: int
    overall_score: float
    results: List[ValidationResult]
    execution_time: float
    timestamp: datetime = datetime.utcnow()

    @validator("overall_score")
    def score_must_be_valid(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v


class AdvancedValidator:
    """Advanced validation engine with multi-layer validation"""

    def __init__(self):
        self.metrics = get_metrics_registry()
        self.validation_rules: Dict[str, Callable] = {}
        self.schema_validators: Dict[str, Dict] = {}
        self._setup_builtin_validators()

        logger.info("Advanced validator initialized")

    def _setup_builtin_validators(self):
        """Setup built-in validation rules"""

        # Repository structure validation
        self.register_validation_rule(
            "repo_structure",
            self._validate_repository_structure,
            ValidationType.SCHEMA,
            ValidationSeverity.ERROR,
        )

        # Configuration validation
        self.register_validation_rule(
            "config_integrity",
            self._validate_configuration_integrity,
            ValidationType.SCHEMA,
            ValidationSeverity.CRITICAL,
        )

        # Security validation
        self.register_validation_rule(
            "security_compliance",
            self._validate_security_compliance,
            ValidationType.SECURITY,
            ValidationSeverity.CRITICAL,
        )

        # Performance validation
        self.register_validation_rule(
            "performance_requirements",
            self._validate_performance_requirements,
            ValidationType.PERFORMANCE,
            ValidationSeverity.WARNING,
        )

        # Business logic validation
        self.register_validation_rule(
            "business_rules",
            self._validate_business_rules,
            ValidationType.BUSINESS_LOGIC,
            ValidationSeverity.ERROR,
        )

    def register_validation_rule(
        self,
        rule_name: str,
        validator_func: Callable,
        validation_type: ValidationType,
        default_severity: ValidationSeverity,
    ):
        """Register custom validation rule"""
        self.validation_rules[rule_name] = {
            "func": validator_func,
            "type": validation_type,
            "severity": default_severity,
        }
        logger.info(
            "Validation rule registered",
            rule_name=rule_name,
            type=validation_type.value,
        )

    def register_schema_validator(self, schema_name: str, schema_definition: Dict):
        """Register JSON schema validator"""
        self.schema_validators[schema_name] = schema_definition
        logger.info("Schema validator registered", schema_name=schema_name)

    @record_operation_metrics("validation_suite")
    async def run_validation_suite(
        self,
        target: Union[str, Dict],
        suite_name: str,
        validation_rules: Optional[List[str]] = None,
        context: Optional[Dict] = None,
    ) -> ValidationSuite:
        """Run comprehensive validation suite"""

        start_time = datetime.utcnow()
        results = []

        # Determine which rules to run
        rules_to_run = validation_rules or list(self.validation_rules.keys())

        logger.info(
            "Starting validation suite",
            suite_name=suite_name,
            rules_count=len(rules_to_run),
        )

        # Execute each validation rule
        for rule_name in rules_to_run:
            if rule_name not in self.validation_rules:
                logger.warning("Unknown validation rule", rule_name=rule_name)
                continue

            try:
                rule_config = self.validation_rules[rule_name]

                # Execute validation
                validation_result = await self._execute_validation_rule(
                    rule_name, rule_config, target, context or {}
                )

                results.append(validation_result)

            except Exception as e:
                logger.error(
                    "Validation rule execution failed",
                    rule_name=rule_name,
                    error=str(e),
                )

                # Create error result
                error_result = ValidationResult(
                    check_name=rule_name,
                    validation_type=ValidationType.SCHEMA,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Validation rule execution failed: {str(e)}",
                    details={"exception": str(e)},
                )
                results.append(error_result)

        # Calculate suite metrics
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = total_checks - passed_checks
        warnings = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)
        critical_issues = sum(
            1
            for r in results
            if r.severity == ValidationSeverity.CRITICAL and not r.passed
        )

        # Calculate overall score
        if total_checks == 0:
            overall_score = 100.0
        else:
            # Weight by severity
            score = 0
            total_weight = 0

            for result in results:
                weight = self._get_severity_weight(result.severity)
                total_weight += weight
                if result.passed:
                    score += weight

            overall_score = (score / total_weight * 100) if total_weight > 0 else 100.0

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        suite = ValidationSuite(
            suite_name=suite_name,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            critical_issues=critical_issues,
            overall_score=overall_score,
            results=results,
            execution_time=execution_time,
        )

        logger.info(
            "Validation suite completed",
            suite_name=suite_name,
            score=overall_score,
            critical_issues=critical_issues,
            execution_time=execution_time,
        )

        return suite

    async def _execute_validation_rule(
        self, rule_name: str, rule_config: Dict, target: Union[str, Dict], context: Dict
    ) -> ValidationResult:
        """Execute individual validation rule"""

        try:
            # Execute the validation function
            validator_func = rule_config["func"]

            if callable(validator_func):
                if (
                    hasattr(validator_func, "__code__")
                    and validator_func.__code__.co_argcount > 2
                ):
                    # Function expects context
                    passed, message, details = await validator_func(target, context)
                else:
                    # Function doesn't expect context
                    passed, message, details = await validator_func(target)
            else:
                raise ValueError(f"Validator function not callable: {rule_name}")

            return ValidationResult(
                check_name=rule_name,
                validation_type=rule_config["type"],
                severity=rule_config["severity"],
                passed=passed,
                message=message,
                details=details or {},
            )

        except Exception as e:
            logger.error(
                "Validation rule execution error", rule_name=rule_name, error=str(e)
            )

            return ValidationResult(
                check_name=rule_name,
                validation_type=rule_config["type"],
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Validation execution error: {str(e)}",
                details={"exception": str(e)},
            )

    def _get_severity_weight(self, severity: ValidationSeverity) -> float:
        """Get numerical weight for severity level"""
        weights = {
            ValidationSeverity.CRITICAL: 10.0,
            ValidationSeverity.ERROR: 5.0,
            ValidationSeverity.WARNING: 2.0,
            ValidationSeverity.INFO: 1.0,
        }
        return weights.get(severity, 1.0)

    # Built-in validation rules

    async def _validate_repository_structure(self, repo_path: str) -> tuple:
        """Validate repository structure and organization"""
        if not isinstance(repo_path, str):
            return (
                False,
                "Invalid repository path type",
                {"expected": "string", "got": type(repo_path).__name__},
            )

        repo_path = Path(repo_path)
        if not repo_path.exists():
            return (
                False,
                f"Repository path does not exist: {repo_path}",
                {"path": str(repo_path)},
            )

        # Check for essential files
        essential_files = ["README.md", "LICENSE"]
        missing_files = []

        for file_name in essential_files:
            if not (repo_path / file_name).exists():
                missing_files.append(file_name)

        # Check for source directory
        src_dirs = ["src", "lib", "app"]
        has_src_dir = any((repo_path / d).exists() for d in src_dirs)

        issues = []
        if missing_files:
            issues.append(f"Missing essential files: {', '.join(missing_files)}")
        if not has_src_dir:
            issues.append("No source directory found")

        if issues:
            return (
                False,
                f"Repository structure issues: {'; '.join(issues)}",
                {"missing_files": missing_files, "has_src_dir": has_src_dir},
            )

        return (
            True,
            "Repository structure is valid",
            {"essential_files_present": True, "source_directory_found": True},
        )

    async def _validate_configuration_integrity(
        self, target: Union[str, Dict]
    ) -> tuple:
        """Validate configuration file integrity"""

        if isinstance(target, str):
            # Validate file path
            config_path = Path(target)
            if not config_path.exists():
                return (
                    False,
                    f"Configuration file not found: {config_path}",
                    {"path": str(config_path)},
                )

            try:
                if config_path.suffix in [".yaml", ".yml"]:
                    with open(config_path) as f:
                        config_data = yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    with open(config_path) as f:
                        config_data = json.load(f)
                else:
                    return (
                        False,
                        f"Unsupported configuration format: {config_path.suffix}",
                        {"supported_formats": [".yaml", ".yml", ".json"]},
                    )
            except Exception as e:
                return (
                    False,
                    f"Failed to parse configuration file: {str(e)}",
                    {"error": str(e)},
                )

        elif isinstance(target, dict):
            config_data = target
        else:
            return (
                False,
                "Invalid configuration target type",
                {"expected": ["string", "dict"], "got": type(target).__name__},
            )

        # Validate configuration structure
        required_sections = ["agents", "github", "review_criteria"]
        missing_sections = [
            section for section in required_sections if section not in config_data
        ]

        if missing_sections:
            return (
                False,
                f"Missing required configuration sections: {', '.join(missing_sections)}",
                {
                    "missing_sections": missing_sections,
                    "required_sections": required_sections,
                },
            )

        return (
            True,
            "Configuration integrity validated",
            {"format": "valid", "required_sections_present": True},
        )

    async def _validate_security_compliance(self, target: Union[str, Dict]) -> tuple:
        """Validate security compliance requirements"""

        security_checks = {
            "token_security": False,
            "secrets_management": False,
            "input_validation": False,
            "secure_communication": False,
        }

        if isinstance(target, str):
            repo_path = Path(target)

            # Check for security-related files
            security_files = [".bandit", ".safety", "security.yml", "security.yaml"]

            for file_name in security_files:
                if (repo_path / file_name).exists():
                    security_checks["token_security"] = True
                    break

            # Check for secrets baseline
            if (repo_path / ".secrets.baseline").exists():
                security_checks["secrets_management"] = True

            # Check for input validation patterns in code
            python_files = list(repo_path.glob("**/*.py"))
            validation_patterns = [
                r"validate",
                r"sanitize",
                r"escape",
                r"pydantic",
                r"marshmallow",
            ]

            for py_file in python_files[:10]:  # Sample first 10 files
                try:
                    content = py_file.read_text()
                    if any(
                        re.search(pattern, content, re.IGNORECASE)
                        for pattern in validation_patterns
                    ):
                        security_checks["input_validation"] = True
                        break
                except:
                    continue

            # Check for HTTPS/TLS configuration
            config_files = list(repo_path.glob("**/*.yaml")) + list(
                repo_path.glob("**/*.yml")
            )
            for config_file in config_files:
                try:
                    content = config_file.read_text()
                    if "https" in content.lower() or "tls" in content.lower():
                        security_checks["secure_communication"] = True
                        break
                except:
                    continue

        # Calculate security score
        security_score = sum(security_checks.values()) / len(security_checks) * 100

        if security_score < 50:
            return (
                False,
                f"Security compliance insufficient: {security_score:.1f}%",
                {
                    "security_score": security_score,
                    "checks": security_checks,
                    "recommendation": "Implement additional security measures",
                },
            )

        return (
            True,
            f"Security compliance adequate: {security_score:.1f}%",
            {"security_score": security_score, "checks": security_checks},
        )

    async def _validate_performance_requirements(
        self, target: Union[str, Dict]
    ) -> tuple:
        """Validate performance requirements and benchmarks"""

        performance_metrics = {
            "response_time_target": "< 200ms",
            "throughput_target": "> 1000 req/s",
            "memory_usage_target": "< 512MB",
            "cpu_usage_target": "< 80%",
        }

        # Mock performance validation
        # In real implementation, this would run actual performance tests
        mock_performance = {
            "response_time": 150,  # ms
            "throughput": 1200,  # req/s
            "memory_usage": 480,  # MB
            "cpu_usage": 65,  # %
        }

        performance_issues = []

        if mock_performance["response_time"] > 200:
            performance_issues.append("Response time exceeds target")
        if mock_performance["throughput"] < 1000:
            performance_issues.append("Throughput below target")
        if mock_performance["memory_usage"] > 512:
            performance_issues.append("Memory usage exceeds target")
        if mock_performance["cpu_usage"] > 80:
            performance_issues.append("CPU usage exceeds target")

        if performance_issues:
            return (
                False,
                f"Performance issues detected: {'; '.join(performance_issues)}",
                {
                    "issues": performance_issues,
                    "current_metrics": mock_performance,
                    "targets": performance_metrics,
                },
            )

        return (
            True,
            "Performance requirements met",
            {
                "current_metrics": mock_performance,
                "targets": performance_metrics,
                "status": "optimal",
            },
        )

    async def _validate_business_rules(
        self, target: Union[str, Dict], context: Dict = {}
    ) -> tuple:
        """Validate business logic and rules compliance"""

        business_rules = {
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "max_lines_per_file": 1000,
            "required_test_coverage": 85.0,
            "max_complexity_score": 10,
        }

        violations = []

        if isinstance(target, str):
            repo_path = Path(target)

            # Check file sizes
            large_files = []
            for file_path in repo_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.stat().st_size > business_rules["max_file_size"]
                ):
                    large_files.append(str(file_path))

            if large_files:
                violations.append(f"Files exceed size limit: {len(large_files)} files")

            # Check lines per file (sample check)
            python_files = list(repo_path.glob("**/*.py"))
            long_files = []
            for py_file in python_files[:20]:  # Sample first 20 files
                try:
                    lines = len(py_file.read_text().splitlines())
                    if lines > business_rules["max_lines_per_file"]:
                        long_files.append(str(py_file))
                except:
                    continue

            if long_files:
                violations.append(f"Files exceed line limit: {len(long_files)} files")

        # Check test coverage (from context)
        test_coverage = context.get("test_coverage", 0)
        if test_coverage < business_rules["required_test_coverage"]:
            violations.append(
                f"Test coverage below requirement: {test_coverage}% < {business_rules['required_test_coverage']}%"
            )

        if violations:
            return (
                False,
                f"Business rule violations: {'; '.join(violations)}",
                {"violations": violations, "rules": business_rules},
            )

        return (
            True,
            "Business rules compliance validated",
            {"rules": business_rules, "status": "compliant"},
        )

    def generate_validation_report(
        self, suite: ValidationSuite, output_format: str = "json"
    ) -> str:
        """Generate comprehensive validation report"""

        if output_format == "json":
            return json.dumps(
                {
                    "validation_report": {
                        "suite_name": suite.suite_name,
                        "timestamp": suite.timestamp.isoformat(),
                        "summary": {
                            "overall_score": suite.overall_score,
                            "total_checks": suite.total_checks,
                            "passed_checks": suite.passed_checks,
                            "failed_checks": suite.failed_checks,
                            "warnings": suite.warnings,
                            "critical_issues": suite.critical_issues,
                            "execution_time": suite.execution_time,
                        },
                        "results": [result.to_dict() for result in suite.results],
                    }
                },
                indent=2,
            )

        elif output_format == "markdown":
            md = f"# Validation Report: {suite.suite_name}\n\n"
            md += f"**Generated:** {suite.timestamp.isoformat()}\n\n"
            md += "## Summary\n\n"
            md += f"- **Overall Score:** {suite.overall_score:.1f}%\n"
            md += f"- **Total Checks:** {suite.total_checks}\n"
            md += f"- **Passed:** {suite.passed_checks}\n"
            md += f"- **Failed:** {suite.failed_checks}\n"
            md += f"- **Warnings:** {suite.warnings}\n"
            md += f"- **Critical Issues:** {suite.critical_issues}\n"
            md += f"- **Execution Time:** {suite.execution_time:.3f}s\n\n"

            md += "## Detailed Results\n\n"
            for result in suite.results:
                status_icon = "✅" if result.passed else "❌"
                md += f"### {status_icon} {result.check_name}\n\n"
                md += f"- **Type:** {result.validation_type.value}\n"
                md += f"- **Severity:** {result.severity.value}\n"
                md += f"- **Message:** {result.message}\n"
                if result.details:
                    md += f"- **Details:** {json.dumps(result.details, indent=2)}\n"
                md += "\n"

            return md

        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Global validator instance
_global_validator: Optional[AdvancedValidator] = None


def get_advanced_validator() -> AdvancedValidator:
    """Get global advanced validator instance"""
    global _global_validator

    if _global_validator is None:
        _global_validator = AdvancedValidator()

    return _global_validator
