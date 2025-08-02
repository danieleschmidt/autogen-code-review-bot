#!/usr/bin/env python3
"""
Code quality monitoring script for the AutoGen Code Review Bot.

This script continuously monitors code quality metrics and provides
automated reporting and alerting when quality thresholds are breached.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class CodeQualityMonitor:
    """Monitors and reports on code quality metrics."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_file = self.repo_path / ".github" / "project-metrics.json"
        self.quality_history_file = self.repo_path / "docs" / "status" / "quality-history.json"
        self.config = self._load_config()
        self.history = self._load_history()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load project metrics configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load quality metrics history."""
        if self.quality_history_file.exists():
            with open(self.quality_history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save quality metrics history."""
        self.quality_history_file.parent.mkdir(exist_ok=True)
        with open(self.quality_history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _run_command(self, command: str) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def measure_test_coverage(self) -> Optional[float]:
        """Measure current test coverage."""
        print("Measuring test coverage...")
        
        exit_code, stdout, stderr = self._run_command("python -m pytest --cov=src --cov-report=json --quiet")
        
        coverage_file = self.repo_path / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    return coverage_data.get("totals", {}).get("percent_covered", 0)
            except (json.JSONDecodeError, KeyError):
                pass
        
        return None
    
    def measure_code_complexity(self) -> Optional[Dict[str, float]]:
        """Measure code complexity using radon."""
        print("Measuring code complexity...")
        
        exit_code, stdout, stderr = self._run_command("radon cc src/ -a -s")
        
        if exit_code == 0 and stdout:
            complexity_data = {"average": 0, "files": {}}
            
            lines = stdout.split('\n')
            for line in lines:
                if "Average complexity:" in line:
                    try:
                        avg_complexity = float(line.split(":")[1].strip().split()[0])
                        complexity_data["average"] = avg_complexity
                    except (IndexError, ValueError):
                        pass
                elif " - " in line and "(" in line:
                    # Parse individual file complexity
                    try:
                        parts = line.split(" - ")
                        if len(parts) >= 2:
                            file_path = parts[0].strip()
                            complexity_str = parts[1].split("(")[1].split(")")[0]
                            complexity = float(complexity_str)
                            complexity_data["files"][file_path] = complexity
                    except (IndexError, ValueError):
                        pass
            
            return complexity_data
        
        return None
    
    def measure_maintainability(self) -> Optional[Dict[str, float]]:
        """Measure maintainability index using radon."""
        print("Measuring maintainability index...")
        
        exit_code, stdout, stderr = self._run_command("radon mi src/ -s")
        
        if exit_code == 0 and stdout:
            maintainability_data = {"average": 0, "files": {}}
            scores = []
            
            lines = stdout.split('\n')
            for line in lines:
                if " - " in line and "(" in line:
                    try:
                        parts = line.split(" - ")
                        if len(parts) >= 2:
                            file_path = parts[0].strip()
                            score_str = parts[1].split("(")[1].split(")")[0]
                            score = float(score_str)
                            maintainability_data["files"][file_path] = score
                            scores.append(score)
                    except (IndexError, ValueError):
                        pass
            
            if scores:
                maintainability_data["average"] = sum(scores) / len(scores)
            
            return maintainability_data
        
        return None
    
    def count_lines_of_code(self) -> Optional[Dict[str, int]]:
        """Count lines of code using cloc or wc."""
        print("Counting lines of code...")
        
        # Try cloc first
        exit_code, stdout, stderr = self._run_command("cloc src/ --json")
        
        if exit_code == 0 and stdout:
            try:
                cloc_data = json.loads(stdout)
                if "SUM" in cloc_data:
                    return {
                        "total": cloc_data["SUM"]["code"],
                        "comments": cloc_data["SUM"]["comment"],
                        "blank": cloc_data["SUM"]["blank"]
                    }
            except json.JSONDecodeError:
                pass
        
        # Fallback to simple line counting
        exit_code, stdout, stderr = self._run_command("find src/ -name '*.py' -exec wc -l {} +")
        
        if exit_code == 0 and stdout:
            lines = stdout.strip().split('\n')
            if lines and "total" in lines[-1]:
                try:
                    total_lines = int(lines[-1].split()[0])
                    return {"total": total_lines, "comments": 0, "blank": 0}
                except (IndexError, ValueError):
                    pass
        
        return None
    
    def detect_code_smells(self) -> Dict[str, List[str]]:
        """Detect code smells and anti-patterns."""
        print("Detecting code smells...")
        
        smells = {
            "long_functions": [],
            "complex_functions": [],
            "duplicate_code": [],
            "large_classes": []
        }
        
        # Use radon to find complex functions
        exit_code, stdout, stderr = self._run_command("radon cc src/ -n C")
        
        if exit_code == 0 and stdout:
            lines = stdout.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('**'):
                    smells["complex_functions"].append(line.strip())
        
        # Use radon to find long functions (raw metrics)
        exit_code, stdout, stderr = self._run_command("radon raw src/ -s")
        
        if exit_code == 0 and stdout:
            lines = stdout.split('\n')
            current_file = None
            
            for line in lines:
                if line.endswith('.py'):
                    current_file = line
                elif 'LOC:' in line and current_file:
                    try:
                        loc = int(line.split('LOC:')[1].split()[0])
                        if loc > 100:  # Functions/files longer than 100 lines
                            smells["long_functions"].append(f"{current_file}: {loc} lines")
                    except (IndexError, ValueError):
                        pass
        
        return smells
    
    def analyze_quality_trends(self) -> Dict[str, str]:
        """Analyze quality trends from historical data."""
        if len(self.history) < 2:
            return {"overall": "insufficient_data"}
        
        latest = self.history[-1]
        previous = self.history[-2]
        
        trends = {}
        
        # Compare coverage
        if "coverage" in latest and "coverage" in previous:
            if latest["coverage"] > previous["coverage"]:
                trends["coverage"] = "improving"
            elif latest["coverage"] < previous["coverage"]:
                trends["coverage"] = "declining"
            else:
                trends["coverage"] = "stable"
        
        # Compare complexity
        if "complexity" in latest and "complexity" in previous:
            if latest["complexity"]["average"] < previous["complexity"]["average"]:
                trends["complexity"] = "improving"
            elif latest["complexity"]["average"] > previous["complexity"]["average"]:
                trends["complexity"] = "declining"
            else:
                trends["complexity"] = "stable"
        
        # Overall trend
        improving_count = sum(1 for trend in trends.values() if trend == "improving")
        declining_count = sum(1 for trend in trends.values() if trend == "declining")
        
        if improving_count > declining_count:
            trends["overall"] = "improving"
        elif declining_count > improving_count:
            trends["overall"] = "declining"
        else:
            trends["overall"] = "stable"
        
        return trends
    
    def check_quality_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check if metrics meet quality thresholds."""
        alerts = []
        thresholds = self.config.get("thresholds", {}).get("code_quality", {})
        
        # Check test coverage
        if "coverage" in metrics and "test_coverage_min" in thresholds:
            if metrics["coverage"] < thresholds["test_coverage_min"]:
                alerts.append(f"⚠️ Test coverage ({metrics['coverage']:.1f}%) below threshold ({thresholds['test_coverage_min']}%)")
        
        # Check complexity
        if "complexity" in metrics and "complexity_max" in thresholds:
            avg_complexity = metrics["complexity"].get("average", 0)
            if avg_complexity > thresholds["complexity_max"]:
                alerts.append(f"⚠️ Code complexity ({avg_complexity:.1f}) above threshold ({thresholds['complexity_max']})")
        
        # Check maintainability
        if "maintainability" in metrics and "maintainability_min" in thresholds:
            avg_maintainability = metrics["maintainability"].get("average", 100)
            if avg_maintainability < thresholds["maintainability_min"]:
                alerts.append(f"⚠️ Maintainability index ({avg_maintainability:.1f}) below threshold ({thresholds['maintainability_min']})")
        
        return alerts
    
    def generate_quality_report(self, metrics: Dict[str, Any], trends: Dict[str, str], alerts: List[str]) -> str:
        """Generate comprehensive quality report."""
        report = f"""# Code Quality Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Current Metrics
"""
        
        if "coverage" in metrics:
            report += f"- **Test Coverage**: {metrics['coverage']:.1f}%\n"
        
        if "complexity" in metrics:
            report += f"- **Average Complexity**: {metrics['complexity']['average']:.1f}\n"
        
        if "maintainability" in metrics:
            report += f"- **Maintainability Index**: {metrics['maintainability']['average']:.1f}\n"
        
        if "loc" in metrics:
            report += f"- **Lines of Code**: {metrics['loc']['total']:,}\n"
        
        report += f"\n## Quality Trends\n"
        
        trend_icons = {
            "improving": "📈",
            "declining": "📉",
            "stable": "➡️",
            "insufficient_data": "❓"
        }
        
        for metric, trend in trends.items():
            icon = trend_icons.get(trend, "❓")
            report += f"- **{metric.title()}**: {icon} {trend.replace('_', ' ').title()}\n"
        
        report += f"\n## Quality Alerts\n"
        
        if alerts:
            for alert in alerts:
                report += f"- {alert}\n"
        else:
            report += "- ✅ All quality thresholds met\n"
        
        if "code_smells" in metrics:
            report += f"\n## Code Smells Detected\n"
            
            for smell_type, smells in metrics["code_smells"].items():
                if smells:
                    report += f"### {smell_type.replace('_', ' ').title()}\n"
                    for smell in smells[:5]:  # Limit to top 5
                        report += f"- {smell}\n"
                    if len(smells) > 5:
                        report += f"- ... and {len(smells) - 5} more\n"
                    report += "\n"
        
        report += f"\n## Recommendations\n"
        
        if alerts:
            report += "- Address quality threshold violations\n"
        
        if "code_smells" in metrics:
            total_smells = sum(len(smells) for smells in metrics["code_smells"].values())
            if total_smells > 0:
                report += f"- Refactor {total_smells} detected code smells\n"
        
        if trends.get("overall") == "declining":
            report += "- Focus on improving declining quality metrics\n"
        elif trends.get("overall") == "stable":
            report += "- Consider setting more ambitious quality targets\n"
        else:
            report += "- Continue current development practices\n"
        
        report += "\n---\n*This report is automatically generated by the code quality monitoring system.*\n"
        
        return report
    
    def run_quality_monitoring(self) -> str:
        """Run complete quality monitoring process."""
        print("Starting code quality monitoring...")
        
        # Collect current metrics
        current_metrics = {
            "timestamp": datetime.now().isoformat(),
            "git_hash": self._get_current_commit_hash()
        }
        
        # Measure all quality metrics
        coverage = self.measure_test_coverage()
        if coverage is not None:
            current_metrics["coverage"] = coverage
        
        complexity = self.measure_code_complexity()
        if complexity is not None:
            current_metrics["complexity"] = complexity
        
        maintainability = self.measure_maintainability()
        if maintainability is not None:
            current_metrics["maintainability"] = maintainability
        
        loc = self.count_lines_of_code()
        if loc is not None:
            current_metrics["loc"] = loc
        
        code_smells = self.detect_code_smells()
        current_metrics["code_smells"] = code_smells
        
        # Add to history
        self.history.append(current_metrics)
        
        # Keep only last 30 entries
        if len(self.history) > 30:
            self.history = self.history[-30:]
        
        self._save_history()
        
        # Analyze trends
        trends = self.analyze_quality_trends()
        
        # Check thresholds
        alerts = self.check_quality_thresholds(current_metrics)
        
        # Generate report
        report = self.generate_quality_report(current_metrics, trends, alerts)
        
        # Save report
        report_file = self.repo_path / "docs" / "status" / f"quality-report-{datetime.now().strftime('%Y-%m-%d')}.md"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Quality report saved: {report_file}")
        
        # Print alerts to console
        if alerts:
            print("\n🚨 Quality Alerts:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("\n✅ All quality thresholds met")
        
        print("Code quality monitoring completed!")
        return report
    
    def _get_current_commit_hash(self) -> str:
        """Get current git commit hash."""
        exit_code, stdout, stderr = self._run_command("git rev-parse HEAD")
        if exit_code == 0:
            return stdout.strip()
        return "unknown"


def main():
    """Main entry point."""
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    monitor = CodeQualityMonitor(repo_path)
    
    if "--report-only" in sys.argv:
        # Generate report from existing data
        if monitor.history:
            latest_metrics = monitor.history[-1]
            trends = monitor.analyze_quality_trends()
            alerts = monitor.check_quality_thresholds(latest_metrics)
            report = monitor.generate_quality_report(latest_metrics, trends, alerts)
            print(report)
        else:
            print("No historical data available. Run without --report-only first.")
    else:
        monitor.run_quality_monitoring()


if __name__ == "__main__":
    main()