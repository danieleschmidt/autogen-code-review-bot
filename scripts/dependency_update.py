#!/usr/bin/env python3
"""
Automated dependency update script for the AutoGen Code Review Bot.

This script checks for outdated dependencies, updates them safely,
and creates pull requests for review.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DependencyUpdater:
    """Handles automated dependency updates."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.update_log = []
    
    def _run_command(self, command: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr."""
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
    
    def check_python_dependencies(self) -> List[Dict[str, str]]:
        """Check for outdated Python dependencies."""
        print("Checking Python dependencies...")
        
        # Use pip-outdated or pip list --outdated
        exit_code, stdout, stderr = self._run_command("pip list --outdated --format=json")
        
        outdated_packages = []
        if exit_code == 0:
            try:
                packages = json.loads(stdout)
                for package in packages:
                    outdated_packages.append({
                        "name": package["name"],
                        "current": package["version"],
                        "latest": package["latest_version"],
                        "type": "python"
                    })
            except json.JSONDecodeError:
                print("Error parsing pip outdated output")
        
        return outdated_packages
    
    def check_node_dependencies(self) -> List[Dict[str, str]]:
        """Check for outdated Node.js dependencies."""
        package_json = self.repo_path / "package.json"
        if not package_json.exists():
            return []
        
        print("Checking Node.js dependencies...")
        
        # Use npm outdated
        exit_code, stdout, stderr = self._run_command("npm outdated --json")
        
        outdated_packages = []
        if stdout:  # npm outdated returns exit code 1 when there are outdated packages
            try:
                packages = json.loads(stdout)
                for name, info in packages.items():
                    outdated_packages.append({
                        "name": name,
                        "current": info.get("current", "unknown"),
                        "latest": info.get("latest", "unknown"),
                        "type": "nodejs"
                    })
            except json.JSONDecodeError:
                print("Error parsing npm outdated output")
        
        return outdated_packages
    
    def categorize_updates(self, packages: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Categorize updates by severity (major, minor, patch)."""
        categories = {
            "major": [],
            "minor": [],
            "patch": [],
            "unknown": []
        }
        
        for package in packages:
            current = package["current"]
            latest = package["latest"]
            
            # Simple semantic version parsing
            try:
                current_parts = current.split(".")
                latest_parts = latest.split(".")
                
                if len(current_parts) >= 3 and len(latest_parts) >= 3:
                    if current_parts[0] != latest_parts[0]:
                        categories["major"].append(package)
                    elif current_parts[1] != latest_parts[1]:
                        categories["minor"].append(package)
                    else:
                        categories["patch"].append(package)
                else:
                    categories["unknown"].append(package)
            except (IndexError, ValueError):
                categories["unknown"].append(package)
        
        return categories
    
    def create_update_strategy(self, categorized_updates: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
        """Create update strategy based on package categories."""
        strategy = {
            "immediate": [],  # Patch updates
            "review": [],     # Minor updates
            "manual": []      # Major updates
        }
        
        # Patch updates can be applied immediately
        for package in categorized_updates["patch"]:
            strategy["immediate"].append(f"{package['name']}: {package['current']} → {package['latest']}")
        
        # Minor updates need review
        for package in categorized_updates["minor"]:
            strategy["review"].append(f"{package['name']}: {package['current']} → {package['latest']}")
        
        # Major updates need manual intervention
        for package in categorized_updates["major"]:
            strategy["manual"].append(f"{package['name']}: {package['current']} → {package['latest']}")
        
        return strategy
    
    def update_python_dependencies(self, packages: List[str]) -> bool:
        """Update Python dependencies."""
        if not packages:
            return True
        
        print(f"Updating Python packages: {', '.join(packages)}")
        
        # Update requirements files if they exist
        requirements_files = ["requirements.txt", "requirements.in", "pyproject.toml"]
        
        for req_file in requirements_files:
            file_path = self.repo_path / req_file
            if file_path.exists():
                print(f"Found {req_file}")
                # This would implement specific update logic for each file type
                # For now, just log the action
                self.update_log.append(f"Would update {req_file} with: {', '.join(packages)}")
        
        return True
    
    def update_node_dependencies(self, packages: List[str]) -> bool:
        """Update Node.js dependencies."""
        if not packages:
            return True
        
        package_json = self.repo_path / "package.json"
        if not package_json.exists():
            return True
        
        print(f"Updating Node.js packages: {', '.join(packages)}")
        
        for package in packages:
            package_name = package.split(":")[0]
            exit_code, stdout, stderr = self._run_command(f"npm update {package_name}")
            
            if exit_code == 0:
                self.update_log.append(f"Updated {package}")
            else:
                self.update_log.append(f"Failed to update {package}: {stderr}")
        
        return True
    
    def run_tests_after_update(self) -> bool:
        """Run tests to ensure updates don't break functionality."""
        print("Running tests after dependency updates...")
        
        # Run Python tests
        exit_code, stdout, stderr = self._run_command("python -m pytest tests/ -v")
        if exit_code != 0:
            print("Python tests failed after update")
            self.update_log.append("Python tests failed after dependency update")
            return False
        
        # Run linting
        exit_code, stdout, stderr = self._run_command("python -m flake8 src/")
        if exit_code != 0:
            print("Linting failed after update")
            self.update_log.append("Linting failed after dependency update")
            return False
        
        self.update_log.append("All tests passed after dependency update")
        return True
    
    def generate_update_report(self, strategy: Dict[str, List[str]]) -> str:
        """Generate dependency update report."""
        report = f"""# Dependency Update Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Immediate Updates (Patch)**: {len(strategy['immediate'])} packages
- **Review Required (Minor)**: {len(strategy['review'])} packages  
- **Manual Updates (Major)**: {len(strategy['manual'])} packages

## Immediate Updates Applied
"""
        
        if strategy["immediate"]:
            for update in strategy["immediate"]:
                report += f"- ✅ {update}\n"
        else:
            report += "- None\n"
        
        report += "\n## Updates Requiring Review\n"
        if strategy["review"]:
            for update in strategy["review"]:
                report += f"- 🔍 {update}\n"
        else:
            report += "- None\n"
        
        report += "\n## Manual Updates Required\n"
        if strategy["manual"]:
            for update in strategy["manual"]:
                report += f"- ⚠️ {update}\n"
        else:
            report += "- None\n"
        
        report += f"\n## Update Log\n"
        for log_entry in self.update_log:
            report += f"- {log_entry}\n"
        
        report += "\n---\n*This report is automatically generated by the dependency update automation.*\n"
        
        return report
    
    def create_update_pr(self, report: str) -> bool:
        """Create a pull request for dependency updates."""
        # Check if there are changes to commit
        exit_code, stdout, stderr = self._run_command("git status --porcelain")
        
        if not stdout.strip():
            print("No changes to commit")
            return False
        
        # Create branch for updates
        branch_name = f"automated/dependency-updates-{datetime.now().strftime('%Y%m%d')}"
        
        commands = [
            f"git checkout -b {branch_name}",
            "git add .",
            f"git commit -m 'chore: automated dependency updates\n\n{report[:500]}...'",
            f"git push -u origin {branch_name}"
        ]
        
        for command in commands:
            exit_code, stdout, stderr = self._run_command(command)
            if exit_code != 0:
                print(f"Failed to execute: {command}")
                print(f"Error: {stderr}")
                return False
        
        self.update_log.append(f"Created PR branch: {branch_name}")
        return True
    
    def run_dependency_update(self, auto_apply_patches: bool = True, create_pr: bool = False):
        """Run the complete dependency update process."""
        print("Starting dependency update process...")
        
        # Check for outdated dependencies
        python_outdated = self.check_python_dependencies()
        node_outdated = self.check_node_dependencies()
        
        all_outdated = python_outdated + node_outdated
        
        if not all_outdated:
            print("All dependencies are up to date!")
            return
        
        # Categorize updates
        categorized = self.categorize_updates(all_outdated)
        strategy = self.create_update_strategy(categorized)
        
        print(f"Found {len(all_outdated)} outdated dependencies")
        print(f"- Patch updates: {len(strategy['immediate'])}")
        print(f"- Minor updates: {len(strategy['review'])}")
        print(f"- Major updates: {len(strategy['manual'])}")
        
        # Apply patch updates automatically if enabled
        if auto_apply_patches and strategy["immediate"]:
            print("Applying patch updates...")
            
            # Separate by package type
            python_patches = [p for p in categorized["patch"] if p["type"] == "python"]
            node_patches = [p for p in categorized["patch"] if p["type"] == "nodejs"]
            
            success = True
            
            if python_patches:
                package_names = [p["name"] for p in python_patches]
                success &= self.update_python_dependencies(package_names)
            
            if node_patches:
                package_names = [p["name"] for p in node_patches]
                success &= self.update_node_dependencies(package_names)
            
            # Run tests after updates
            if success:
                success = self.run_tests_after_update()
            
            if not success:
                print("Updates failed validation, rolling back...")
                self._run_command("git checkout .")
                return
        
        # Generate report
        report = self.generate_update_report(strategy)
        
        # Save report
        report_file = self.repo_path / "docs" / "status" / f"dependency-update-{datetime.now().strftime('%Y-%m-%d')}.md"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Update report saved: {report_file}")
        
        # Create PR if requested
        if create_pr and (strategy["immediate"] or strategy["review"]):
            if self.create_update_pr(report):
                print("Created pull request for dependency updates")
            else:
                print("Failed to create pull request")
        
        print("Dependency update process completed!")


def main():
    """Main entry point."""
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    updater = DependencyUpdater(repo_path)
    
    auto_apply = "--auto-apply" in sys.argv
    create_pr = "--create-pr" in sys.argv
    
    updater.run_dependency_update(auto_apply_patches=auto_apply, create_pr=create_pr)


if __name__ == "__main__":
    main()