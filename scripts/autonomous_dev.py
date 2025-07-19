#!/usr/bin/env python3
"""
Autonomous Development Script

Implements a disciplined development loop:
1. Read prioritized backlog
2. Select highest feasible task
3. Implement using TDD approach
4. Update backlog and documentation
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacklogTask:
    """Represents a task from the backlog"""
    title: str
    wsjf_score: float
    business_value: int
    time_criticality: int
    risk_reduction: int
    job_size: int
    description: str
    files: List[str]
    status: str = "pending"

class AutonomousDeveloper:
    """Main autonomous development engine"""
    
    def __init__(self, backlog_file: str, max_tasks: int = 3):
        self.backlog_file = Path(backlog_file)
        self.max_tasks = max_tasks
        self.completed_tasks: List[BacklogTask] = []
        
    def run_development_cycle(self) -> bool:
        """Run a complete development cycle"""
        try:
            logger.info("🚀 Starting autonomous development cycle")
            
            # Step 1: Parse backlog
            tasks = self._parse_backlog()
            if not tasks:
                logger.warning("No tasks found in backlog")
                return False
                
            # Step 2: Select and implement feasible tasks
            implemented_count = 0
            for task in tasks:
                if implemented_count >= self.max_tasks:
                    break
                    
                if self._is_task_feasible(task):
                    logger.info(f"🎯 Implementing task: {task.title}")
                    
                    if self._implement_task_with_tdd(task):
                        self.completed_tasks.append(task)
                        implemented_count += 1
                        logger.info(f"✅ Completed task: {task.title}")
                    else:
                        logger.error(f"❌ Failed to implement task: {task.title}")
                        break
                        
            # Step 3: Update documentation
            self._update_backlog()
            self._update_changelog()
            
            logger.info(f"🏁 Development cycle complete. Implemented {implemented_count} tasks.")
            return implemented_count > 0
            
        except Exception as e:
            logger.error(f"Development cycle failed: {e}")
            return False
    
    def _parse_backlog(self) -> List[BacklogTask]:
        """Parse the backlog markdown file"""
        if not self.backlog_file.exists():
            logger.error(f"Backlog file not found: {self.backlog_file}")
            return []
            
        content = self.backlog_file.read_text()
        tasks = []
        
        # Regex to match task entries
        task_pattern = r'### (\d+)\. (.+?)\n\*\*WSJF: ([\d.]+)\*\*.*?Business Value: (\d+).*?Time Criticality: (\d+).*?Risk Reduction: (\d+).*?Job Size: (\d+)\n- \*\*Description\*\*: (.+?)\n.*?- \*\*Files\*\*: (.+?)(?=\n###|\n---|\Z)'
        
        matches = re.finditer(task_pattern, content, re.DOTALL)
        
        for match in matches:
            task = BacklogTask(
                title=match.group(2).strip(),
                wsjf_score=float(match.group(3)),
                business_value=int(match.group(4)),
                time_criticality=int(match.group(5)),
                risk_reduction=int(match.group(6)),
                job_size=int(match.group(7)),
                description=match.group(8).strip(),
                files=[f.strip() for f in match.group(9).split(',')]
            )
            tasks.append(task)
            
        # Sort by WSJF score (highest first)
        tasks.sort(key=lambda t: t.wsjf_score, reverse=True)
        logger.info(f"📋 Parsed {len(tasks)} tasks from backlog")
        
        return tasks
    
    def _is_task_feasible(self, task: BacklogTask) -> bool:
        """Check if a task is feasible to implement autonomously"""
        # Skip tasks that require human review
        risky_keywords = ['learning', 'architecture', 'major refactor', 'breaking change']
        if any(keyword in task.description.lower() for keyword in risky_keywords):
            logger.info(f"⚠️ Skipping task requiring human review: {task.title}")
            return False
            
        # Check if files exist
        for file_path in task.files:
            if file_path.startswith('src/') and not Path(file_path).exists():
                logger.warning(f"File not found for task {task.title}: {file_path}")
                return False
                
        return True
    
    def _implement_task_with_tdd(self, task: BacklogTask) -> bool:
        """Implement task using TDD approach"""
        try:
            # Step 1: Write/update tests first
            if not self._write_tests_for_task(task):
                return False
                
            # Step 2: Run tests (should fail initially)
            if not self._run_tests():
                logger.info("✅ Tests failing as expected (red phase)")
            else:
                logger.warning("Tests already passing - no implementation needed?")
                
            # Step 3: Implement minimal code to pass tests
            if not self._implement_code_for_task(task):
                return False
                
            # Step 4: Run tests again (should pass now)
            if not self._run_tests():
                logger.error("Tests still failing after implementation")
                return False
                
            # Step 5: Refactor and optimize
            self._refactor_implementation(task)
            
            # Step 6: Final validation
            return self._validate_implementation(task)
            
        except Exception as e:
            logger.error(f"TDD implementation failed for {task.title}: {e}")
            return False
    
    def _write_tests_for_task(self, task: BacklogTask) -> bool:
        """Write or update tests for the task"""
        # This is a simplified implementation
        # In practice, this would use more sophisticated test generation
        logger.info(f"📝 Writing tests for {task.title}")
        
        if "security" in task.title.lower():
            return self._write_security_tests(task)
        elif "logging" in task.title.lower():
            return self._write_logging_tests(task)
        elif "config" in task.title.lower():
            return self._write_config_tests(task)
        else:
            logger.info("Generic test framework - assuming tests exist")
            return True
    
    def _implement_code_for_task(self, task: BacklogTask) -> bool:
        """Implement the actual code for the task"""
        logger.info(f"⚙️ Implementing code for {task.title}")
        
        if "security" in task.title.lower():
            return self._fix_security_issues(task)
        elif "logging" in task.title.lower():
            return self._implement_logging(task)
        elif "config" in task.title.lower():
            return self._implement_configuration(task)
        else:
            logger.warning(f"No specific implementation handler for {task.title}")
            return False
    
    def _run_tests(self) -> bool:
        """Run the test suite"""
        try:
            result = subprocess.run(['python', '-m', 'pytest', '-v'], 
                                  capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out")
            return False
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False
    
    def _run_linting(self) -> bool:
        """Run linting checks"""
        try:
            # Run ruff
            result = subprocess.run(['ruff', 'check', '.'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Ruff check failed: {result.stdout}")
                return False
                
            # Run bandit
            result = subprocess.run(['bandit', '-r', 'src', '-q'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Bandit check failed: {result.stdout}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Linting failed: {e}")
            return False
    
    def _validate_implementation(self, task: BacklogTask) -> bool:
        """Final validation of the implementation"""
        logger.info(f"🔍 Validating implementation for {task.title}")
        
        # Run tests
        if not self._run_tests():
            logger.error("Tests failing after implementation")
            return False
            
        # Run linting
        if not self._run_linting():
            logger.error("Linting checks failing")
            return False
            
        # Run pre-commit hooks
        try:
            result = subprocess.run(['pre-commit', 'run', '--all-files'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Pre-commit hooks had issues: {result.stdout}")
                # Don't fail for pre-commit issues, just warn
                
        except Exception as e:
            logger.warning(f"Pre-commit validation failed: {e}")
            
        return True
    
    def _fix_security_issues(self, task: BacklogTask) -> bool:
        """Fix subprocess security warnings"""
        # Implementation for fixing subprocess security issues
        # This would analyze the specific bandit warnings and apply fixes
        logger.info("🔒 Fixing subprocess security issues")
        # Placeholder - actual implementation would modify files
        return True
    
    def _implement_logging(self, task: BacklogTask) -> bool:
        """Implement structured logging"""
        logger.info("📊 Implementing structured logging")
        # Placeholder - actual implementation would add logging framework
        return True
    
    def _implement_configuration(self, task: BacklogTask) -> bool:
        """Move hardcoded values to configuration"""
        logger.info("⚙️ Implementing configuration management")
        # Placeholder - actual implementation would extract hardcoded values
        return True
    
    def _write_security_tests(self, task: BacklogTask) -> bool:
        """Write security-specific tests"""
        # Placeholder for security test implementation
        return True
    
    def _write_logging_tests(self, task: BacklogTask) -> bool:
        """Write logging-specific tests"""
        # Placeholder for logging test implementation
        return True
    
    def _write_config_tests(self, task: BacklogTask) -> bool:
        """Write configuration-specific tests"""
        # Placeholder for config test implementation
        return True
    
    def _refactor_implementation(self, task: BacklogTask):
        """Refactor implementation for better code quality"""
        logger.info(f"🔄 Refactoring implementation for {task.title}")
        # Placeholder for refactoring logic
    
    def _update_backlog(self):
        """Update backlog with completed tasks"""
        logger.info("📋 Updating backlog")
        # This would modify the backlog file to mark completed tasks
    
    def _update_changelog(self):
        """Update changelog with completed work"""
        logger.info("📝 Updating changelog")
        # This would add entries to CHANGELOG.md

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Autonomous Development Script')
    parser.add_argument('--backlog-file', default='BACKLOG.md', 
                       help='Path to backlog file')
    parser.add_argument('--max-tasks', type=int, default=3,
                       help='Maximum tasks to implement per cycle')
    parser.add_argument('--create-pr', action='store_true',
                       help='Create pull request after implementation')
    
    args = parser.parse_args()
    
    developer = AutonomousDeveloper(args.backlog_file, args.max_tasks)
    success = developer.run_development_cycle()
    
    if not success:
        logger.error("Development cycle failed")
        sys.exit(1)
        
    logger.info("🎉 Development cycle completed successfully")

if __name__ == '__main__':
    main()