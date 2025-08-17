#!/usr/bin/env python3
"""
Production Readiness Validation

Comprehensive validation of production readiness including infrastructure,
security, compliance, performance, and operational capabilities.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    """Validation result for a specific check"""
    check_name: str
    category: str
    status: str  # PASS, WARN, FAIL
    score: float
    message: str
    details: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class ProductionReadinessValidator:
    """Comprehensive production readiness validator"""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results: List[ValidationResult] = []
        
    async def validate_production_readiness(self) -> Dict:
        """Run comprehensive production readiness validation"""
        print("🔍 Starting Production Readiness Validation")
        print("=" * 60)
        
        start_time = time.time()
        
        validation_categories = [
            ("Infrastructure", self._validate_infrastructure),
            ("Security", self._validate_security),
            ("Compliance", self._validate_compliance), 
            ("Performance", self._validate_performance),
            ("Monitoring", self._validate_monitoring),
            ("Documentation", self._validate_documentation),
            ("Operational", self._validate_operational),
            ("Global Deployment", self._validate_global_deployment)
        ]
        
        for category, validator in validation_categories:
            print(f"\n📋 Validating {category}...")
            try:
                await validator()
            except Exception as e:
                self.results.append(ValidationResult(
                    check_name=f"{category.lower()}_validation",
                    category=category,
                    status="FAIL",
                    score=0.0,
                    message=f"Validation failed: {str(e)}",
                    details={"error": str(e)}
                ))
        
        # Calculate overall readiness
        overall_result = self._calculate_overall_readiness()
        execution_time = time.time() - start_time
        
        return {
            "overall_readiness": overall_result,
            "category_results": self._get_category_summaries(),
            "detailed_results": [self._result_to_dict(r) for r in self.results],
            "validation_time": execution_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_infrastructure(self):
        """Validate infrastructure readiness"""
        
        # Check Docker configuration
        dockerfile_exists = (self.repo_path / "Dockerfile").exists()
        docker_compose_exists = (self.repo_path / "docker-compose.prod.yml").exists()
        
        if dockerfile_exists and docker_compose_exists:
            self.results.append(ValidationResult(
                check_name="docker_configuration",
                category="Infrastructure",
                status="PASS",
                score=100.0,
                message="✅ Docker configuration complete",
                details={
                    "dockerfile": dockerfile_exists,
                    "docker_compose": docker_compose_exists
                }
            ))
        else:
            self.results.append(ValidationResult(
                check_name="docker_configuration",
                category="Infrastructure",
                status="FAIL",
                score=0.0,
                message="❌ Missing Docker configuration files",
                recommendations=["Add Dockerfile and docker-compose.prod.yml"]
            ))
        
        # Check Kubernetes configuration
        k8s_config_exists = (self.repo_path / "global-deployment-config.yml").exists()
        
        if k8s_config_exists:
            self.results.append(ValidationResult(
                check_name="kubernetes_configuration",
                category="Infrastructure",
                status="PASS",
                score=100.0,
                message="✅ Kubernetes deployment configuration ready",
                details={"k8s_config": True}
            ))
        else:
            self.results.append(ValidationResult(
                check_name="kubernetes_configuration",
                category="Infrastructure",
                status="WARN",
                score=50.0,
                message="⚠️ Kubernetes configuration not found",
                recommendations=["Add Kubernetes deployment manifests"]
            ))
        
        # Check infrastructure as code
        terraform_files = list(self.repo_path.rglob("*.tf"))
        helm_charts = list(self.repo_path.rglob("Chart.yaml"))
        
        if terraform_files or helm_charts:
            self.results.append(ValidationResult(
                check_name="infrastructure_as_code",
                category="Infrastructure",
                status="PASS",
                score=100.0,
                message="✅ Infrastructure as Code present",
                details={
                    "terraform_files": len(terraform_files),
                    "helm_charts": len(helm_charts)
                }
            ))
        else:
            self.results.append(ValidationResult(
                check_name="infrastructure_as_code",
                category="Infrastructure",
                status="WARN",
                score=70.0,
                message="⚠️ No Infrastructure as Code detected",
                recommendations=["Consider adding Terraform or Helm charts"]
            ))
    
    async def _validate_security(self):
        """Validate security readiness"""
        
        # Check security documentation
        security_md = (self.repo_path / "SECURITY.md").exists()
        
        if security_md:
            self.results.append(ValidationResult(
                check_name="security_documentation",
                category="Security",
                status="PASS",
                score=100.0,
                message="✅ Security documentation present"
            ))
        else:
            self.results.append(ValidationResult(
                check_name="security_documentation",
                category="Security",
                status="WARN",
                score=70.0,
                message="⚠️ Missing SECURITY.md file",
                recommendations=["Add security policy documentation"]
            ))
        
        # Check for security scanning configuration
        security_config_files = [
            ".bandit",
            ".safety",
            ".secrets.baseline",
            "security_scan_results.json"
        ]
        
        security_files_present = sum(1 for f in security_config_files if (self.repo_path / f).exists())
        
        if security_files_present >= 2:
            self.results.append(ValidationResult(
                check_name="security_scanning",
                category="Security",
                status="PASS",
                score=100.0,
                message=f"✅ Security scanning configured ({security_files_present} tools)",
                details={"security_tools": security_files_present}
            ))
        else:
            self.results.append(ValidationResult(
                check_name="security_scanning",
                category="Security",
                status="WARN",
                score=50.0,
                message="⚠️ Limited security scanning configuration",
                recommendations=["Add more security scanning tools"]
            ))
        
        # Check SSL/TLS configuration
        ssl_config = self._check_ssl_configuration()
        
        if ssl_config["configured"]:
            self.results.append(ValidationResult(
                check_name="ssl_tls_configuration",
                category="Security",
                status="PASS",
                score=100.0,
                message="✅ SSL/TLS configuration ready",
                details=ssl_config
            ))
        else:
            self.results.append(ValidationResult(
                check_name="ssl_tls_configuration",
                category="Security",
                status="FAIL",
                score=0.0,
                message="❌ SSL/TLS configuration missing",
                recommendations=["Configure SSL certificates and TLS settings"]
            ))
    
    def _check_ssl_configuration(self) -> Dict:
        """Check SSL/TLS configuration"""
        nginx_config = (self.repo_path / "nginx").exists()
        ssl_dir = (self.repo_path / "ssl").exists()
        cert_manager = any("cert-manager" in str(f) for f in self.repo_path.rglob("*.yml"))
        
        return {
            "configured": nginx_config or cert_manager,
            "nginx_config": nginx_config,
            "ssl_directory": ssl_dir,
            "cert_manager": cert_manager
        }
    
    async def _validate_compliance(self):
        """Validate compliance readiness"""
        
        compliance_items = {
            "license": (self.repo_path / "LICENSE").exists(),
            "privacy_policy": any(f.name.lower().startswith("privacy") for f in self.repo_path.rglob("*.md")),
            "code_of_conduct": (self.repo_path / "CODE_OF_CONDUCT.md").exists(),
            "codeowners": (self.repo_path / "CODEOWNERS").exists() or (self.repo_path / ".github" / "CODEOWNERS").exists(),
            "contributing": (self.repo_path / "CONTRIBUTING.md").exists(),
            "changelog": (self.repo_path / "CHANGELOG.md").exists(),
        }
        
        compliance_score = (sum(compliance_items.values()) / len(compliance_items)) * 100
        
        if compliance_score >= 80:
            status = "PASS"
            message = f"✅ Compliance documentation complete ({compliance_score:.0f}%)"
        elif compliance_score >= 60:
            status = "WARN"
            message = f"⚠️ Compliance documentation needs improvement ({compliance_score:.0f}%)"
        else:
            status = "FAIL"
            message = f"❌ Compliance documentation incomplete ({compliance_score:.0f}%)"
        
        self.results.append(ValidationResult(
            check_name="compliance_documentation",
            category="Compliance",
            status=status,
            score=compliance_score,
            message=message,
            details=compliance_items,
            recommendations=self._get_compliance_recommendations(compliance_items)
        ))
        
        # Check GDPR compliance readiness
        gdpr_indicators = [
            "gdpr" in str(f).lower() for f in self.repo_path.rglob("*")
        ]
        
        gdpr_ready = any(gdpr_indicators) or compliance_items["privacy_policy"]
        
        self.results.append(ValidationResult(
            check_name="gdpr_compliance",
            category="Compliance",
            status="PASS" if gdpr_ready else "WARN",
            score=100.0 if gdpr_ready else 60.0,
            message="✅ GDPR compliance indicators present" if gdpr_ready else "⚠️ GDPR compliance needs verification",
            details={"gdpr_indicators": sum(gdpr_indicators), "privacy_policy": compliance_items["privacy_policy"]}
        ))
    
    def _get_compliance_recommendations(self, compliance_items: Dict) -> List[str]:
        """Get compliance recommendations"""
        recommendations = []
        
        if not compliance_items["license"]:
            recommendations.append("Add LICENSE file")
        if not compliance_items["privacy_policy"]:
            recommendations.append("Add privacy policy documentation")
        if not compliance_items["code_of_conduct"]:
            recommendations.append("Add CODE_OF_CONDUCT.md")
        if not compliance_items["codeowners"]:
            recommendations.append("Add CODEOWNERS file")
        if not compliance_items["contributing"]:
            recommendations.append("Add CONTRIBUTING.md")
        if not compliance_items["changelog"]:
            recommendations.append("Add CHANGELOG.md")
        
        return recommendations
    
    async def _validate_performance(self):
        """Validate performance readiness"""
        
        # Check performance testing
        perf_test_files = list(self.repo_path.rglob("*performance*"))
        load_test_files = list(self.repo_path.rglob("*load*test*"))
        benchmark_files = list(self.repo_path.rglob("*benchmark*"))
        
        perf_testing_score = min(100.0, (len(perf_test_files) + len(load_test_files) + len(benchmark_files)) * 20)
        
        if perf_testing_score >= 80:
            self.results.append(ValidationResult(
                check_name="performance_testing",
                category="Performance",
                status="PASS",
                score=perf_testing_score,
                message=f"✅ Performance testing configured ({len(perf_test_files + load_test_files + benchmark_files)} files)",
                details={
                    "performance_tests": len(perf_test_files),
                    "load_tests": len(load_test_files),
                    "benchmarks": len(benchmark_files)
                }
            ))
        else:
            self.results.append(ValidationResult(
                check_name="performance_testing",
                category="Performance",
                status="WARN",
                score=perf_testing_score,
                message="⚠️ Limited performance testing configuration",
                recommendations=["Add performance tests, load tests, and benchmarks"]
            ))
        
        # Check caching configuration
        caching_indicators = [
            "redis" in str(f).lower() for f in self.repo_path.rglob("*")
        ]
        
        caching_configured = any(caching_indicators)
        
        self.results.append(ValidationResult(
            check_name="caching_configuration",
            category="Performance",
            status="PASS" if caching_configured else "WARN",
            score=100.0 if caching_configured else 50.0,
            message="✅ Caching configured" if caching_configured else "⚠️ Caching not configured",
            details={"caching_indicators": sum(caching_indicators)}
        ))
    
    async def _validate_monitoring(self):
        """Validate monitoring readiness"""
        
        monitoring_dir = (self.repo_path / "monitoring").exists()
        prometheus_config = any("prometheus" in str(f).lower() for f in self.repo_path.rglob("*.yml"))
        grafana_config = any("grafana" in str(f).lower() for f in self.repo_path.rglob("*"))
        
        monitoring_components = {
            "monitoring_directory": monitoring_dir,
            "prometheus": prometheus_config,
            "grafana": grafana_config,
            "health_checks": self._check_health_endpoints(),
            "metrics_endpoints": self._check_metrics_endpoints()
        }
        
        monitoring_score = (sum(monitoring_components.values()) / len(monitoring_components)) * 100
        
        if monitoring_score >= 80:
            status = "PASS"
            message = f"✅ Monitoring infrastructure ready ({monitoring_score:.0f}%)"
        elif monitoring_score >= 60:
            status = "WARN"
            message = f"⚠️ Monitoring needs improvement ({monitoring_score:.0f}%)"
        else:
            status = "FAIL"
            message = f"❌ Monitoring infrastructure incomplete ({monitoring_score:.0f}%)"
        
        self.results.append(ValidationResult(
            check_name="monitoring_infrastructure",
            category="Monitoring",
            status=status,
            score=monitoring_score,
            message=message,
            details=monitoring_components
        ))
    
    def _check_health_endpoints(self) -> bool:
        """Check for health check endpoints"""
        # Look for health check patterns in code
        health_patterns = ["health", "/health", "healthz", "ready", "/ready"]
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if any(pattern in content.lower() for pattern in health_patterns):
                    return True
            except:
                continue
        
        return False
    
    def _check_metrics_endpoints(self) -> bool:
        """Check for metrics endpoints"""
        # Look for metrics patterns in code
        metrics_patterns = ["metrics", "/metrics", "prometheus", "monitoring"]
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                if any(pattern in content.lower() for pattern in metrics_patterns):
                    return True
            except:
                continue
        
        return False
    
    async def _validate_documentation(self):
        """Validate documentation readiness"""
        
        docs_items = {
            "readme": (self.repo_path / "README.md").exists(),
            "api_docs": (self.repo_path / "docs").exists(),
            "architecture": any("arch" in str(f).lower() for f in self.repo_path.rglob("*.md")),
            "deployment_guide": any("deploy" in str(f).lower() for f in self.repo_path.rglob("*.md")),
            "user_guide": any("user" in str(f).lower() or "guide" in str(f).lower() for f in self.repo_path.rglob("*.md")),
            "troubleshooting": any("troubleshoot" in str(f).lower() for f in self.repo_path.rglob("*.md"))
        }
        
        docs_score = (sum(docs_items.values()) / len(docs_items)) * 100
        
        if docs_score >= 80:
            status = "PASS"
            message = f"✅ Documentation complete ({docs_score:.0f}%)"
        elif docs_score >= 60:
            status = "WARN"
            message = f"⚠️ Documentation needs improvement ({docs_score:.0f}%)"
        else:
            status = "FAIL"
            message = f"❌ Documentation incomplete ({docs_score:.0f}%)"
        
        self.results.append(ValidationResult(
            check_name="documentation_completeness",
            category="Documentation",
            status=status,
            score=docs_score,
            message=message,
            details=docs_items
        ))
    
    async def _validate_operational(self):
        """Validate operational readiness"""
        
        operational_items = {
            "ci_cd_pipeline": self._check_ci_cd(),
            "automated_testing": self._check_automated_testing(),
            "dependency_management": self._check_dependency_management(),
            "secret_management": self._check_secret_management(),
            "backup_strategy": self._check_backup_strategy(),
            "incident_response": self._check_incident_response()
        }
        
        operational_score = (sum(operational_items.values()) / len(operational_items)) * 100
        
        if operational_score >= 80:
            status = "PASS"
            message = f"✅ Operational readiness achieved ({operational_score:.0f}%)"
        elif operational_score >= 60:
            status = "WARN"
            message = f"⚠️ Operational readiness needs improvement ({operational_score:.0f}%)"
        else:
            status = "FAIL"
            message = f"❌ Operational readiness insufficient ({operational_score:.0f}%)"
        
        self.results.append(ValidationResult(
            check_name="operational_readiness",
            category="Operational",
            status=status,
            score=operational_score,
            message=message,
            details=operational_items
        ))
    
    def _check_ci_cd(self) -> bool:
        """Check for CI/CD configuration"""
        github_actions = (self.repo_path / ".github" / "workflows").exists()
        gitlab_ci = (self.repo_path / ".gitlab-ci.yml").exists()
        jenkins_file = (self.repo_path / "Jenkinsfile").exists()
        
        return github_actions or gitlab_ci or jenkins_file
    
    def _check_automated_testing(self) -> bool:
        """Check for automated testing"""
        test_files = list(self.repo_path.rglob("test_*.py"))
        pytest_config = (self.repo_path / "pytest.ini").exists() or (self.repo_path / "pyproject.toml").exists()
        
        return len(test_files) > 0 and pytest_config
    
    def _check_dependency_management(self) -> bool:
        """Check dependency management"""
        requirements = (self.repo_path / "requirements.txt").exists()
        pyproject = (self.repo_path / "pyproject.toml").exists()
        pipfile = (self.repo_path / "Pipfile").exists()
        
        return requirements or pyproject or pipfile
    
    def _check_secret_management(self) -> bool:
        """Check secret management"""
        env_example = (self.repo_path / ".env.example").exists()
        secrets_in_compose = any("secrets:" in str(f) for f in self.repo_path.rglob("docker-compose*.yml"))
        k8s_secrets = any("kind: Secret" in str(f) for f in self.repo_path.rglob("*.yml"))
        
        return env_example or secrets_in_compose or k8s_secrets
    
    def _check_backup_strategy(self) -> bool:
        """Check backup strategy"""
        backup_scripts = list(self.repo_path.rglob("*backup*"))
        docker_volumes = any("volumes:" in str(f) for f in self.repo_path.rglob("docker-compose*.yml"))
        
        return len(backup_scripts) > 0 or docker_volumes
    
    def _check_incident_response(self) -> bool:
        """Check incident response procedures"""
        runbooks = list(self.repo_path.rglob("*runbook*"))
        incident_docs = any("incident" in str(f).lower() for f in self.repo_path.rglob("*.md"))
        
        return len(runbooks) > 0 or incident_docs
    
    async def _validate_global_deployment(self):
        """Validate global deployment readiness"""
        
        global_config = (self.repo_path / "global-deployment-config.yml").exists()
        multi_region_support = self._check_multi_region_support()
        internationalization = self._check_internationalization()
        
        global_items = {
            "global_config": global_config,
            "multi_region": multi_region_support,
            "internationalization": internationalization,
            "compliance_ready": self._check_compliance_frameworks(),
            "cdn_ready": self._check_cdn_configuration()
        }
        
        global_score = (sum(global_items.values()) / len(global_items)) * 100
        
        if global_score >= 80:
            status = "PASS"
            message = f"✅ Global deployment ready ({global_score:.0f}%)"
        elif global_score >= 60:
            status = "WARN"
            message = f"⚠️ Global deployment needs improvement ({global_score:.0f}%)"
        else:
            status = "FAIL"
            message = f"❌ Global deployment not ready ({global_score:.0f}%)"
        
        self.results.append(ValidationResult(
            check_name="global_deployment_readiness",
            category="Global Deployment",
            status=status,
            score=global_score,
            message=message,
            details=global_items
        ))
    
    def _check_multi_region_support(self) -> bool:
        """Check multi-region deployment support"""
        region_configs = any("region" in str(f).lower() for f in self.repo_path.rglob("*.yml"))
        load_balancer_config = any("load" in str(f).lower() for f in self.repo_path.rglob("*.yml"))
        
        return region_configs and load_balancer_config
    
    def _check_internationalization(self) -> bool:
        """Check internationalization support"""
        i18n_patterns = ["i18n", "locale", "translation", "lang"]
        
        for pattern in i18n_patterns:
            if any(pattern in str(f).lower() for f in self.repo_path.rglob("*")):
                return True
        
        return False
    
    def _check_compliance_frameworks(self) -> bool:
        """Check compliance framework support"""
        compliance_patterns = ["gdpr", "ccpa", "pdpa", "compliance"]
        
        for pattern in compliance_patterns:
            if any(pattern in str(f).lower() for f in self.repo_path.rglob("*")):
                return True
        
        return False
    
    def _check_cdn_configuration(self) -> bool:
        """Check CDN configuration"""
        cdn_patterns = ["cdn", "cloudfront", "cloudflare"]
        
        for pattern in cdn_patterns:
            if any(pattern in str(f).lower() for f in self.repo_path.rglob("*")):
                return True
        
        return False
    
    def _calculate_overall_readiness(self) -> Dict:
        """Calculate overall production readiness"""
        if not self.results:
            return {"score": 0.0, "status": "FAIL", "message": "No validation results"}
        
        # Calculate weighted scores by category
        category_weights = {
            "Infrastructure": 0.20,
            "Security": 0.25,
            "Compliance": 0.15,
            "Performance": 0.15,
            "Monitoring": 0.10,
            "Documentation": 0.05,
            "Operational": 0.10,
            "Global Deployment": 0.00  # Optional for basic deployment
        }
        
        category_scores = {}
        for category in category_weights.keys():
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                avg_score = sum(r.score for r in category_results) / len(category_results)
                category_scores[category] = avg_score
            else:
                category_scores[category] = 100.0  # No issues if no checks
        
        # Calculate weighted overall score
        weighted_score = 0.0
        total_weight = 0.0
        
        for category, weight in category_weights.items():
            if category in category_scores and weight > 0:
                weighted_score += category_scores[category] * weight
                total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine status
        if overall_score >= 90:
            status = "PRODUCTION_READY"
            message = "🎉 Production deployment ready!"
        elif overall_score >= 80:
            status = "MOSTLY_READY"
            message = "✅ Production ready with minor improvements needed"
        elif overall_score >= 70:
            status = "NEEDS_IMPROVEMENT"
            message = "⚠️ Significant improvements needed before production"
        else:
            status = "NOT_READY"
            message = "❌ Not ready for production deployment"
        
        # Count critical failures
        critical_failures = len([r for r in self.results if r.status == "FAIL"])
        
        return {
            "score": overall_score,
            "status": status,
            "message": message,
            "critical_failures": critical_failures,
            "category_scores": category_scores,
            "recommendations": self._get_top_recommendations()
        }
    
    def _get_category_summaries(self) -> Dict:
        """Get summary by category"""
        categories = {}
        
        for result in self.results:
            category = result.category
            if category not in categories:
                categories[category] = {
                    "total_checks": 0,
                    "passed": 0,
                    "warnings": 0,
                    "failed": 0,
                    "avg_score": 0.0
                }
            
            categories[category]["total_checks"] += 1
            
            if result.status == "PASS":
                categories[category]["passed"] += 1
            elif result.status == "WARN":
                categories[category]["warnings"] += 1
            elif result.status == "FAIL":
                categories[category]["failed"] += 1
        
        # Calculate average scores
        for category in categories:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                categories[category]["avg_score"] = sum(r.score for r in category_results) / len(category_results)
        
        return categories
    
    def _get_top_recommendations(self) -> List[str]:
        """Get top recommendations for improvement"""
        recommendations = []
        
        for result in self.results:
            if result.status in ["FAIL", "WARN"] and result.recommendations:
                recommendations.extend(result.recommendations)
        
        # Return top 10 unique recommendations
        return list(dict.fromkeys(recommendations))[:10]
    
    def _result_to_dict(self, result: ValidationResult) -> Dict:
        """Convert ValidationResult to dictionary"""
        return {
            "check_name": result.check_name,
            "category": result.category,
            "status": result.status,
            "score": result.score,
            "message": result.message,
            "details": result.details,
            "recommendations": result.recommendations
        }

async def main():
    """Main execution function"""
    validator = ProductionReadinessValidator()
    
    try:
        results = await validator.validate_production_readiness()
        
        # Print summary
        overall = results["overall_readiness"]
        print(f"\n🎯 PRODUCTION READINESS SUMMARY")
        print("=" * 60)
        print(f"Overall Score: {overall['score']:.1f}/100")
        print(f"Status: {overall['status']}")
        print(f"Message: {overall['message']}")
        print(f"Critical Failures: {overall['critical_failures']}")
        
        # Print category breakdown
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, summary in results["category_results"].items():
            print(f"  {category}: {summary['avg_score']:.1f}/100 "
                  f"(✅{summary['passed']} ⚠️{summary['warnings']} ❌{summary['failed']})")
        
        # Print top recommendations
        if overall["recommendations"]:
            print(f"\n🔧 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(overall["recommendations"][:5], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n⏱️ Validation completed in {results['validation_time']:.2f} seconds")
        
        # Return appropriate exit code
        if overall["status"] in ["PRODUCTION_READY", "MOSTLY_READY"]:
            return 0
        else:
            return 1
    
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)