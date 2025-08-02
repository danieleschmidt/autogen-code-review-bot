#!/usr/bin/env python3
"""
Repository maintenance automation script.

This script performs regular maintenance tasks including:
- Cleaning up old branches
- Updating documentation
- Managing releases
- Repository health checks
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class RepositoryMaintainer:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.maintenance_log = []
        self.dry_run = False
    
    def _run_command(self, command: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr."""
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {command}")
            return 0, "", ""
        
        try:
            result = subprocess.run(
                command.split() if isinstance(command, str) else command,
                capture_output=capture_output,
                text=True,
                cwd=self.repo_path
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def _log_action(self, action: str, details: str = ""):
        """Log maintenance action."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        self.maintenance_log.append(log_entry)
        print(f"✓ {action}: {details}")
    
    def cleanup_old_branches(self, days_old: int = 30) -> List[str]:
        """Clean up merged and stale branches."""
        print(f"Cleaning up branches older than {days_old} days...")
        
        # Get list of merged branches
        exit_code, stdout, stderr = self._run_command("git branch --merged main")
        
        merged_branches = []
        if exit_code == 0:
            branches = stdout.strip().split('\n')
            for branch in branches:
                branch = branch.strip()
                if branch and not branch.startswith('*') and branch != 'main':
                    merged_branches.append(branch)
        
        # Get branch last commit dates
        old_branches = []
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for branch in merged_branches:
            exit_code, stdout, stderr = self._run_command(f"git log -1 --format=%ci {branch}")
            
            if exit_code == 0 and stdout.strip():
                try:
                    commit_date = datetime.fromisoformat(stdout.strip().replace(' ', 'T', 1))
                    if commit_date < cutoff_date:
                        old_branches.append(branch)
                except ValueError:
                    continue
        
        # Delete old branches
        deleted_branches = []
        for branch in old_branches:
            exit_code, stdout, stderr = self._run_command(f"git branch -d {branch}")
            
            if exit_code == 0:
                deleted_branches.append(branch)
                self._log_action("Branch deleted", branch)
            else:
                self._log_action("Branch deletion failed", f"{branch}: {stderr}")
        
        return deleted_branches
    
    def cleanup_remote_tracking_branches(self) -> List[str]:
        """Clean up remote tracking branches for deleted remotes."""
        print("Cleaning up stale remote tracking branches...")
        
        # Prune remote branches
        exit_code, stdout, stderr = self._run_command("git remote prune origin")
        
        pruned_branches = []
        if exit_code == 0:
            lines = stdout.split('\n')
            for line in lines:
                if 'deleted' in line.lower() and 'origin/' in line:
                    branch_name = line.split('origin/')[-1].strip()
                    pruned_branches.append(branch_name)
                    self._log_action("Remote branch pruned", branch_name)
        
        return pruned_branches
    
    def update_dependencies_cache(self) -> bool:
        """Update and clean dependency caches."""
        print("Updating dependency caches...")
        
        # Update pip cache
        if (self.repo_path / "requirements.txt").exists() or (self.repo_path / "pyproject.toml").exists():
            exit_code, stdout, stderr = self._run_command("pip cache purge")
            if exit_code == 0:
                self._log_action("Python cache purged", "pip cache cleared")
        
        # Update npm cache if package.json exists
        if (self.repo_path / "package.json").exists():
            exit_code, stdout, stderr = self._run_command("npm cache clean --force")
            if exit_code == 0:
                self._log_action("Node.js cache cleaned", "npm cache cleared")
        
        return True
    
    def analyze_repository_health(self) -> Dict[str, Any]:
        """Analyze overall repository health."""
        print("Analyzing repository health...")
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        # Check repository size
        exit_code, stdout, stderr = self._run_command("du -sh .")
        if exit_code == 0:
            size_str = stdout.split()[0]
            health_report["metrics"]["repository_size"] = size_str
            
            # Flag large repositories
            if size_str.endswith('G') and float(size_str[:-1]) > 1:
                health_report["issues"].append(f"Large repository size: {size_str}")
                health_report["recommendations"].append("Consider using Git LFS for large files")
        
        # Check number of files
        exit_code, stdout, stderr = self._run_command("find . -type f | wc -l")
        if exit_code == 0:
            file_count = int(stdout.strip())
            health_report["metrics"]["file_count"] = file_count
            
            if file_count > 10000:
                health_report["issues"].append(f"Large number of files: {file_count}")
                health_report["recommendations"].append("Review if all files are necessary")
        
        # Check for large files
        exit_code, stdout, stderr = self._run_command("find . -type f -size +10M")
        if exit_code == 0 and stdout.strip():
            large_files = stdout.strip().split('\n')
            health_report["metrics"]["large_files"] = len(large_files)
            health_report["issues"].append(f"Found {len(large_files)} files larger than 10MB")
            health_report["recommendations"].append("Consider using Git LFS for large binary files")
        
        # Check commit history health
        exit_code, stdout, stderr = self._run_command("git log --oneline -100")
        if exit_code == 0:
            commits = stdout.strip().split('\n')
            health_report["metrics"]["recent_commits"] = len(commits)
            
            # Check for merge commits ratio
            merge_commits = [c for c in commits if 'merge' in c.lower()]
            if len(merge_commits) / len(commits) > 0.3:
                health_report["issues"].append("High ratio of merge commits")
                health_report["recommendations"].append("Consider using squash and merge or rebase workflows")
        
        # Check for security files
        security_files = [".gitignore", "SECURITY.md", "LICENSE"]
        missing_security_files = []
        
        for file in security_files:
            if not (self.repo_path / file).exists():
                missing_security_files.append(file)
        
        if missing_security_files:
            health_report["issues"].append(f"Missing security files: {', '.join(missing_security_files)}")
            health_report["recommendations"].append("Add missing security and governance files")
        
        # Check for documentation
        doc_files = ["README.md", "CONTRIBUTING.md", "docs/"]
        missing_docs = []
        
        for doc in doc_files:
            if not (self.repo_path / doc).exists():
                missing_docs.append(doc)
        
        if missing_docs:
            health_report["issues"].append(f"Missing documentation: {', '.join(missing_docs)}")
            health_report["recommendations"].append("Improve project documentation")
        
        return health_report
    
    def generate_maintenance_summary(self) -> Dict[str, Any]:
        """Generate summary of all maintenance activities."""
        print("Generating maintenance summary...")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "actions_performed": len(self.maintenance_log),
            "log": self.maintenance_log,
            "next_maintenance": (datetime.now() + timedelta(weeks=1)).isoformat(),
            "recommendations": []
        }
        
        # Analyze maintenance patterns
        action_types = {}
        for log_entry in self.maintenance_log:
            action = log_entry["action"]
            action_types[action] = action_types.get(action, 0) + 1
        
        summary["action_summary"] = action_types
        
        # Generate recommendations based on maintenance history
        if action_types.get("Branch deleted", 0) > 5:
            summary["recommendations"].append("Consider implementing branch protection rules")
        
        if action_types.get("Remote branch pruned", 0) > 3:
            summary["recommendations"].append("Educate team on proper branch cleanup")
        
        if not action_types:
            summary["recommendations"].append("Repository appears well-maintained")
        
        return summary
    
    def update_repository_metrics(self):
        """Update repository metrics in project-metrics.json."""
        print("Updating repository metrics...")
        
        metrics_file = self.repo_path / ".github" / "project-metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Update maintenance information
            if "maintenance" not in metrics_data:
                metrics_data["maintenance"] = {}
            
            metrics_data["maintenance"]["last_run"] = datetime.now().isoformat()
            metrics_data["maintenance"]["actions_performed"] = len(self.maintenance_log)
            metrics_data["maintenance"]["next_scheduled"] = (datetime.now() + timedelta(weeks=1)).isoformat()
            
            # Update repository health metrics
            health_report = self.analyze_repository_health()
            metrics_data["repository_health"] = health_report
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self._log_action("Metrics updated", str(metrics_file))
    
    def generate_maintenance_report(self, summary: Dict[str, Any], health_report: Dict[str, Any]) -> str:
        """Generate comprehensive maintenance report."""
        report = f"""# Repository Maintenance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Maintenance Summary
- **Actions Performed**: {summary['actions_performed']}
- **Next Maintenance**: {datetime.fromisoformat(summary['next_maintenance']).strftime('%Y-%m-%d')}

## Actions Breakdown
"""
        
        if summary["action_summary"]:
            for action, count in summary["action_summary"].items():
                report += f"- **{action}**: {count}\n"
        else:
            report += "- No maintenance actions required\n"
        
        report += f"\n## Repository Health\n"
        
        if health_report["metrics"]:
            for metric, value in health_report["metrics"].items():
                report += f"- **{metric.replace('_', ' ').title()}**: {value}\n"
        
        report += f"\n## Issues Identified\n"
        
        if health_report["issues"]:
            for issue in health_report["issues"]:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "- ✅ No issues identified\n"
        
        report += f"\n## Recommendations\n"
        
        all_recommendations = health_report["recommendations"] + summary["recommendations"]
        if all_recommendations:
            for rec in set(all_recommendations):  # Remove duplicates
                report += f"- 💡 {rec}\n"
        else:
            report += "- ✅ Repository is well-maintained\n"
        
        report += f"\n## Detailed Log\n"
        
        for log_entry in self.maintenance_log:
            timestamp = datetime.fromisoformat(log_entry["timestamp"]).strftime('%H:%M:%S')
            report += f"- `{timestamp}` **{log_entry['action']}**: {log_entry['details']}\n"
        
        if not self.maintenance_log:
            report += "- No actions performed during this maintenance cycle\n"
        
        report += "\n---\n*This report is automatically generated by the repository maintenance system.*\n"
        
        return report
    
    def run_maintenance_cycle(self, cleanup_branches: bool = True, update_caches: bool = True) -> str:
        """Run complete maintenance cycle."""
        print("Starting repository maintenance cycle...")
        
        if "--dry-run" in sys.argv:
            self.dry_run = True
            print("🔍 Running in DRY RUN mode - no changes will be made")
        
        # Cleanup tasks
        if cleanup_branches:
            deleted_branches = self.cleanup_old_branches()
            pruned_branches = self.cleanup_remote_tracking_branches()
        
        if update_caches:
            self.update_dependencies_cache()
        
        # Analysis tasks
        health_report = self.analyze_repository_health()
        summary = self.generate_maintenance_summary()
        
        # Update metrics
        if not self.dry_run:
            self.update_repository_metrics()
        
        # Generate report
        report = self.generate_maintenance_report(summary, health_report)
        
        # Save report
        if not self.dry_run:
            report_file = self.repo_path / "docs" / "status" / f"maintenance-report-{datetime.now().strftime('%Y-%m-%d')}.md"
            report_file.parent.mkdir(exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"Maintenance report saved: {report_file}")
        
        print("Repository maintenance cycle completed!")
        return report


def main():
    """Main entry point."""
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    maintainer = RepositoryMaintainer(repo_path)
    
    # Parse command line options
    cleanup_branches = "--no-branch-cleanup" not in sys.argv
    update_caches = "--no-cache-update" not in sys.argv
    
    if "--health-only" in sys.argv:
        health_report = maintainer.analyze_repository_health()
        print(json.dumps(health_report, indent=2))
    else:
        maintainer.run_maintenance_cycle(
            cleanup_branches=cleanup_branches,
            update_caches=update_caches
        )


if __name__ == "__main__":
    main()