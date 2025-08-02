#!/usr/bin/env python3
"""
Automated metrics collection script for the AutoGen Code Review Bot.

This script collects various metrics about the project including code quality,
security, performance, and development metrics. It updates the project-metrics.json
file and can generate reports for stakeholders.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import requests


class MetricsCollector:
    """Collects and aggregates project metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.metrics_data = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics data."""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metrics(self):
        """Save metrics data to file."""
        self.metrics_data["project"]["updated"] = datetime.now().isoformat()
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
    
    def _run_command(self, command: str) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            print(f"Error running command '{command}': {e}")
        return None
    
    def collect_code_quality_metrics(self):
        """Collect code quality metrics."""
        print("Collecting code quality metrics...")
        
        # Test coverage
        coverage_output = self._run_command("python -m pytest --cov=src --cov-report=json")
        if coverage_output and Path("coverage.json").exists():
            with open("coverage.json", 'r') as f:
                coverage_data = json.load(f)
                coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
                self.metrics_data["metrics"]["code_quality"]["test_coverage"]["current"] = round(coverage_percent, 2)
        
        # Code complexity with radon
        complexity_output = self._run_command("radon cc src/ -a -s")
        if complexity_output:
            # Parse average complexity from radon output
            lines = complexity_output.split('\n')
            for line in lines:
                if "Average complexity:" in line:
                    try:
                        complexity = float(line.split(":")[1].strip().split()[0])
                        self.metrics_data["metrics"]["code_quality"]["code_complexity"]["current"] = complexity
                    except (IndexError, ValueError):
                        pass
        
        # Maintainability index
        maintainability_output = self._run_command("radon mi src/ -s")
        if maintainability_output:
            # Parse maintainability index
            lines = maintainability_output.split('\n')
            scores = []
            for line in lines:
                if " - " in line and "(" in line:
                    try:
                        score = float(line.split("(")[1].split(")")[0])
                        scores.append(score)
                    except (IndexError, ValueError):
                        pass
            if scores:
                avg_score = sum(scores) / len(scores)
                self.metrics_data["metrics"]["code_quality"]["maintainability_index"]["current"] = round(avg_score, 2)
    
    def collect_security_metrics(self):
        """Collect security metrics."""
        print("Collecting security metrics...")
        
        # Bandit security scan
        bandit_output = self._run_command("bandit -r src/ -f json -o bandit-report.json")
        if Path("bandit-report.json").exists():
            with open("bandit-report.json", 'r') as f:
                try:
                    bandit_data = json.load(f)
                    results = bandit_data.get("results", [])
                    
                    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                    for result in results:
                        severity = result.get("issue_severity", "low").lower()
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                    
                    self.metrics_data["metrics"]["security"]["vulnerability_count"].update(severity_counts)
                    self.metrics_data["metrics"]["security"]["vulnerability_count"]["last_scan"] = datetime.now().isoformat()
                except json.JSONDecodeError:
                    pass
        
        # Safety dependency scan
        safety_output = self._run_command("safety check --json")
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                vuln_count = len(safety_data)
                self.metrics_data["metrics"]["security"]["dependency_vulnerabilities"]["count"] = vuln_count
                self.metrics_data["metrics"]["security"]["dependency_vulnerabilities"]["last_scan"] = datetime.now().isoformat()
            except json.JSONDecodeError:
                pass
    
    def collect_development_metrics(self):
        """Collect development and git metrics."""
        print("Collecting development metrics...")
        
        # Commit frequency (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        commit_count_output = self._run_command(f"git rev-list --count --since='{thirty_days_ago}' HEAD")
        if commit_count_output:
            try:
                commit_count = int(commit_count_output)
                daily_average = round(commit_count / 30, 2)
                self.metrics_data["metrics"]["development"]["commit_frequency"]["daily_average"] = daily_average
                self.metrics_data["metrics"]["development"]["commit_frequency"]["weekly_total"] = commit_count
            except ValueError:
                pass
        
        # Recent commits for trend analysis
        recent_commits_output = self._run_command("git log --oneline -10")
        if recent_commits_output:
            commits = recent_commits_output.split('\n')
            self.metrics_data["recent_commits"] = commits[:5]  # Store last 5 commits
    
    def collect_performance_metrics(self):
        """Collect performance metrics from benchmarks."""
        print("Collecting performance metrics...")
        
        # Run performance benchmarks if available
        benchmark_output = self._run_command("python -m pytest benchmarks/ -v")
        if benchmark_output:
            # Basic performance metrics collection
            # This would be enhanced with actual benchmark parsing
            self.metrics_data["metrics"]["performance"]["last_benchmark"] = datetime.now().isoformat()
    
    def generate_report(self) -> str:
        """Generate a metrics report in markdown format."""
        report = f"""# Project Metrics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Code Quality
- **Test Coverage**: {self.metrics_data.get('metrics', {}).get('code_quality', {}).get('test_coverage', {}).get('current', 'N/A')}%
- **Code Complexity**: {self.metrics_data.get('metrics', {}).get('code_quality', {}).get('code_complexity', {}).get('current', 'N/A')}
- **Maintainability Index**: {self.metrics_data.get('metrics', {}).get('code_quality', {}).get('maintainability_index', {}).get('current', 'N/A')}

## Security
- **Critical Vulnerabilities**: {self.metrics_data.get('metrics', {}).get('security', {}).get('vulnerability_count', {}).get('critical', 0)}
- **High Vulnerabilities**: {self.metrics_data.get('metrics', {}).get('security', {}).get('vulnerability_count', {}).get('high', 0)}
- **Dependency Vulnerabilities**: {self.metrics_data.get('metrics', {}).get('security', {}).get('dependency_vulnerabilities', {}).get('count', 0)}

## Development
- **Daily Commit Average**: {self.metrics_data.get('metrics', {}).get('development', {}).get('commit_frequency', {}).get('daily_average', 'N/A')}
- **Recent Activity**: {len(self.metrics_data.get('recent_commits', []))} recent commits

## Trends
{'📈 Improving' if self._is_trending_up() else '📉 Needs attention' if self._has_issues() else '✅ Stable'}

---
*This report is automatically generated. See `.github/project-metrics.json` for detailed data.*
"""
        return report
    
    def _is_trending_up(self) -> bool:
        """Check if metrics are trending upward."""
        # Simple heuristic - would be enhanced with historical data
        coverage = self.metrics_data.get('metrics', {}).get('code_quality', {}).get('test_coverage', {}).get('current', 0)
        return coverage > 70
    
    def _has_issues(self) -> bool:
        """Check if there are critical issues."""
        critical_vulns = self.metrics_data.get('metrics', {}).get('security', {}).get('vulnerability_count', {}).get('critical', 0)
        high_vulns = self.metrics_data.get('metrics', {}).get('security', {}).get('vulnerability_count', {}).get('high', 0)
        return critical_vulns > 0 or high_vulns > 5
    
    def run_full_collection(self):
        """Run all metrics collection."""
        print("Starting comprehensive metrics collection...")
        
        self.collect_code_quality_metrics()
        self.collect_security_metrics()
        self.collect_development_metrics()
        self.collect_performance_metrics()
        
        self._save_metrics()
        print("Metrics collection completed!")
        
        # Generate and save report
        report = self.generate_report()
        report_file = self.repo_path / "docs" / "status" / f"metrics-report-{datetime.now().strftime('%Y-%m-%d')}.md"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Report generated: {report_file}")
        return report


def main():
    """Main entry point."""
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    collector = MetricsCollector(repo_path)
    
    if "--report-only" in sys.argv:
        report = collector.generate_report()
        print(report)
    else:
        collector.run_full_collection()


if __name__ == "__main__":
    main()