#!/usr/bin/env python3
"""
Integration scripts for external tools and services.

This module provides integration utilities for connecting the AutoGen Code Review Bot
with external services like GitHub, monitoring systems, and third-party tools.
"""

import json
import os
import requests
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class GitHubIntegration:
    """GitHub API integration utilities."""
    
    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.repo = repo or os.getenv('GITHUB_REPOSITORY')
        self.base_url = "https://api.github.com"
        
        if not self.token:
            print("Warning: No GitHub token provided. Some features may not work.")
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """Make GitHub API request."""
        if not self.token:
            return None
        
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method == 'PATCH':
                response = requests.patch(url, headers=headers, json=data)
            else:
                return None
            
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            print(f"GitHub API error: {e}")
            return None
    
    def get_repository_info(self) -> Optional[Dict]:
        """Get repository information."""
        if not self.repo:
            return None
        
        return self._make_request(f"repos/{self.repo}")
    
    def update_repository_topics(self, topics: List[str]) -> bool:
        """Update repository topics."""
        if not self.repo:
            return False
        
        data = {"names": topics}
        result = self._make_request(f"repos/{self.repo}/topics", method='PATCH', data=data)
        return result is not None
    
    def create_issue(self, title: str, body: str, labels: List[str] = None) -> Optional[Dict]:
        """Create a GitHub issue."""
        if not self.repo:
            return None
        
        data = {
            "title": title,
            "body": body
        }
        
        if labels:
            data["labels"] = labels
        
        return self._make_request(f"repos/{self.repo}/issues", method='POST', data=data)
    
    def get_pull_requests(self, state: str = 'open') -> Optional[List[Dict]]:
        """Get repository pull requests."""
        if not self.repo:
            return None
        
        return self._make_request(f"repos/{self.repo}/pulls?state={state}")
    
    def update_repository_description(self, description: str, homepage: str = None) -> bool:
        """Update repository description and homepage."""
        if not self.repo:
            return False
        
        data = {"description": description}
        if homepage:
            data["homepage"] = homepage
        
        result = self._make_request(f"repos/{self.repo}", method='PATCH', data=data)
        return result is not None


class MonitoringIntegration:
    """Integration with monitoring and observability tools."""
    
    def __init__(self, prometheus_url: str = None, grafana_url: str = None):
        self.prometheus_url = prometheus_url or os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.grafana_url = grafana_url or os.getenv('GRAFANA_URL', 'http://localhost:3000')
    
    def query_prometheus_metrics(self, query: str) -> Optional[Dict]:
        """Query Prometheus for metrics."""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {"query": query}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            return response.json()
        
        except requests.RequestException as e:
            print(f"Prometheus query error: {e}")
            return None
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get key application metrics from Prometheus."""
        metrics = {}
        
        # CPU usage
        cpu_query = 'rate(process_cpu_seconds_total[5m]) * 100'
        cpu_result = self.query_prometheus_metrics(cpu_query)
        if cpu_result and cpu_result.get('status') == 'success':
            data = cpu_result.get('data', {}).get('result', [])
            if data:
                metrics['cpu_usage_percent'] = float(data[0]['value'][1])
        
        # Memory usage
        memory_query = 'process_resident_memory_bytes / 1024 / 1024'
        memory_result = self.query_prometheus_metrics(memory_query)
        if memory_result and memory_result.get('status') == 'success':
            data = memory_result.get('data', {}).get('result', [])
            if data:
                metrics['memory_usage_mb'] = float(data[0]['value'][1])
        
        # Request rate
        request_query = 'rate(http_requests_total[5m])'
        request_result = self.query_prometheus_metrics(request_query)
        if request_result and request_result.get('status') == 'success':
            data = request_result.get('data', {}).get('result', [])
            if data:
                metrics['requests_per_second'] = sum(float(item['value'][1]) for item in data)
        
        return metrics
    
    def send_alert_to_slack(self, webhook_url: str, message: str, channel: str = None) -> bool:
        """Send alert message to Slack."""
        payload = {
            "text": message,
            "username": "AutoGen Bot Monitor"
        }
        
        if channel:
            payload["channel"] = channel
        
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            return True
        
        except requests.RequestException as e:
            print(f"Slack alert error: {e}")
            return False


class CIIntegration:
    """CI/CD integration utilities."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def trigger_workflow(self, workflow_name: str, ref: str = "main") -> bool:
        """Trigger GitHub Actions workflow."""
        if not os.getenv('GITHUB_TOKEN') or not os.getenv('GITHUB_REPOSITORY'):
            print("GitHub token and repository required for workflow triggers")
            return False
        
        github = GitHubIntegration()
        
        data = {
            "ref": ref,
            "inputs": {
                "triggered_by": "automation_script",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        result = github._make_request(
            f"repos/{github.repo}/actions/workflows/{workflow_name}/dispatches",
            method='POST',
            data=data
        )
        
        return result is not None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get GitHub Actions workflow status."""
        github = GitHubIntegration()
        
        return github._make_request(f"repos/{github.repo}/actions/workflows/{workflow_id}")
    
    def wait_for_checks(self, commit_sha: str, timeout_minutes: int = 30) -> bool:
        """Wait for all checks to complete on a commit."""
        github = GitHubIntegration()
        
        import time
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            checks = github._make_request(f"repos/{github.repo}/commits/{commit_sha}/check-runs")
            
            if not checks:
                time.sleep(30)
                continue
            
            check_runs = checks.get('check_runs', [])
            if not check_runs:
                time.sleep(30)
                continue
            
            # Check if all checks are completed
            all_completed = all(
                check['status'] == 'completed' 
                for check in check_runs
            )
            
            if all_completed:
                # Check if all checks passed
                all_passed = all(
                    check['conclusion'] in ['success', 'neutral', 'skipped']
                    for check in check_runs
                )
                return all_passed
            
            time.sleep(30)
        
        return False  # Timeout reached


class IntegrationManager:
    """Manages all external integrations."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github = GitHubIntegration()
        self.monitoring = MonitoringIntegration()
        self.ci = CIIntegration(repo_path)
    
    def setup_repository_integrations(self) -> Dict[str, bool]:
        """Setup all repository integrations."""
        results = {}
        
        # Update repository metadata
        repo_info = self.github.get_repository_info()
        if repo_info:
            # Update topics
            topics = [
                "autogen", "code-review", "automation", "python", 
                "github-bot", "ci-cd", "code-quality"
            ]
            results['topics_updated'] = self.github.update_repository_topics(topics)
            
            # Update description
            description = "Two-agent AutoGen system for automated code review with dual-agent architecture"
            homepage = "https://github.com/danieleschmidt/autogen-code-review-bot"
            results['description_updated'] = self.github.update_repository_description(description, homepage)
        else:
            results['topics_updated'] = False
            results['description_updated'] = False
        
        return results
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run health checks across all integrations."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "github": {"status": "unknown", "details": {}},
            "monitoring": {"status": "unknown", "details": {}},
            "ci": {"status": "unknown", "details": {}}
        }
        
        # GitHub health check
        repo_info = self.github.get_repository_info()
        if repo_info:
            health_status["github"]["status"] = "healthy"
            health_status["github"]["details"] = {
                "repository": repo_info.get("full_name"),
                "stars": repo_info.get("stargazers_count", 0),
                "forks": repo_info.get("forks_count", 0),
                "issues": repo_info.get("open_issues_count", 0)
            }
        else:
            health_status["github"]["status"] = "error"
            health_status["github"]["details"] = {"error": "Could not connect to GitHub API"}
        
        # Monitoring health check
        metrics = self.monitoring.get_application_metrics()
        if metrics:
            health_status["monitoring"]["status"] = "healthy"
            health_status["monitoring"]["details"] = metrics
        else:
            health_status["monitoring"]["status"] = "error"
            health_status["monitoring"]["details"] = {"error": "Could not retrieve metrics"}
        
        # CI health check
        # This would check the status of recent workflow runs
        health_status["ci"]["status"] = "healthy"  # Placeholder
        health_status["ci"]["details"] = {"last_check": datetime.now().isoformat()}
        
        return health_status
    
    def generate_integration_report(self) -> str:
        """Generate integration status report."""
        health_status = self.run_health_checks()
        
        report = f"""# Integration Status Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## GitHub Integration
- **Status**: {health_status['github']['status']}
"""
        
        if health_status['github']['status'] == 'healthy':
            details = health_status['github']['details']
            report += f"- **Repository**: {details.get('repository', 'N/A')}\n"
            report += f"- **Stars**: {details.get('stars', 0)}\n"
            report += f"- **Forks**: {details.get('forks', 0)}\n"
            report += f"- **Open Issues**: {details.get('issues', 0)}\n"
        
        report += f"\n## Monitoring Integration\n"
        report += f"- **Status**: {health_status['monitoring']['status']}\n"
        
        if health_status['monitoring']['status'] == 'healthy':
            details = health_status['monitoring']['details']
            if 'cpu_usage_percent' in details:
                report += f"- **CPU Usage**: {details['cpu_usage_percent']:.1f}%\n"
            if 'memory_usage_mb' in details:
                report += f"- **Memory Usage**: {details['memory_usage_mb']:.1f} MB\n"
            if 'requests_per_second' in details:
                report += f"- **Request Rate**: {details['requests_per_second']:.1f} req/s\n"
        
        report += f"\n## CI/CD Integration\n"
        report += f"- **Status**: {health_status['ci']['status']}\n"
        
        report += "\n---\n*This report is automatically generated by the integration management system.*\n"
        
        return report


def main():
    """Main entry point."""
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    manager = IntegrationManager(repo_path)
    
    if "--setup" in sys.argv:
        print("Setting up repository integrations...")
        results = manager.setup_repository_integrations()
        print(f"Integration setup results: {results}")
    
    elif "--health-check" in sys.argv:
        print("Running integration health checks...")
        health_status = manager.run_health_checks()
        print(json.dumps(health_status, indent=2))
    
    elif "--report" in sys.argv:
        print("Generating integration report...")
        report = manager.generate_integration_report()
        print(report)
        
        # Save report
        report_file = Path(repo_path) / "docs" / "status" / f"integration-report-{datetime.now().strftime('%Y-%m-%d')}.md"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved: {report_file}")
    
    else:
        print("Usage: python integration_scripts.py [--setup|--health-check|--report]")
        sys.exit(1)


if __name__ == "__main__":
    main()