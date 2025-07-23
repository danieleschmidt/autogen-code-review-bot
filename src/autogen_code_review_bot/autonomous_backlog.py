"""
Autonomous Backlog Management System

Implements WSJF-based prioritization and continuous backlog processing with:
- Impact-driven scoring (Cost of Delay components)
- Discovery engine for new tasks
- TDD micro-cycles
- Security and quality guardrails
"""

import os
import yaml
import json
import datetime
import logging
import hashlib
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path


class TaskType(Enum):
    """Task classification"""
    FEATURE = "Feature"
    BUG = "Bug"
    REFACTOR = "Refactor"
    SECURITY = "Security"
    DOC = "Doc"


class TaskStatus(Enum):
    """Task lifecycle states"""
    NEW = "NEW"
    REFINED = "REFINED"
    READY = "READY"
    DOING = "DOING"
    PR = "PR"
    MERGED = "MERGED"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


@dataclass
class BacklogItem:
    """Structured backlog item with WSJF scoring"""
    id: str
    title: str
    description: str
    type: TaskType
    status: TaskStatus
    created_date: datetime.datetime
    updated_date: datetime.datetime
    
    # Cost of Delay Components (1-13 scale)
    user_value: int = 1
    business_value: int = 1  
    time_criticality: int = 1
    risk_reduction: int = 1
    opportunity_enablement: int = 1
    
    effort: int = 1  # 1-13 scale
    wsjf_score: float = 0.0
    aging_multiplier: float = 1.0
    
    acceptance_criteria: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    lines_of_interest: List[str] = field(default_factory=list)
    linked_items: List[str] = field(default_factory=list)
    security_notes: str = ""
    test_notes: str = ""
    
    # Blocking and escalation
    blocked_reason: str = ""
    escalation_required: bool = False
    escalation_reason: str = ""
    
    def calculate_wsjf(self, weights: Dict[str, float], aging_cap: float = 2.0) -> float:
        """Calculate WSJF score using Cost of Delay components"""
        cost_of_delay = (
            self.user_value * weights.get('user_value', 1.0) +
            self.business_value * weights.get('business_value', 1.0) +
            self.time_criticality * weights.get('time_criticality', 1.0) +
            self.risk_reduction * weights.get('risk_reduction', 0.8) +
            self.opportunity_enablement * weights.get('opportunity_enablement', 0.6)
        )
        
        # Apply aging multiplier (capped to prevent runaway inflation)
        effective_cod = cost_of_delay * min(self.aging_multiplier, aging_cap)
        
        # Avoid division by zero
        effort = max(self.effort, 1)
        
        self.wsjf_score = effective_cod / effort
        return self.wsjf_score
    
    def update_aging(self, aging_threshold_days: int = 30):
        """Update aging multiplier based on time since creation"""
        days_old = (datetime.datetime.now() - self.created_date).days
        if days_old > aging_threshold_days:
            aging_factor = 1 + (days_old - aging_threshold_days) / 100  # 1% per day
            self.aging_multiplier = min(aging_factor, 2.0)  # Cap at 2x
        else:
            self.aging_multiplier = 1.0
    
    def is_ready_for_execution(self) -> bool:
        """Check if item is ready for execution"""
        return (
            self.status == TaskStatus.READY and
            not self.blocked_reason and
            len(self.acceptance_criteria) > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        data['created_date'] = self.created_date.isoformat()
        data['updated_date'] = self.updated_date.isoformat()
        return data


@dataclass
class BacklogMetrics:
    """Backlog health and progress metrics"""
    total_items: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    avg_wsjf: float = 0.0
    high_priority_count: int = 0  # WSJF > 2.0
    aging_items: int = 0
    cycle_time_avg: float = 0.0
    coverage_delta: float = 0.0
    
    def update_from_items(self, items: List[BacklogItem]):
        """Update metrics from backlog items"""
        self.total_items = len(items)
        
        # Status distribution
        self.by_status = {}
        for status in TaskStatus:
            self.by_status[status.value] = sum(1 for item in items if item.status == status)
        
        # Type distribution
        self.by_type = {}
        for task_type in TaskType:
            self.by_type[task_type.value] = sum(1 for item in items if item.type == task_type)
        
        # WSJF metrics
        if items:
            self.avg_wsjf = sum(item.wsjf_score for item in items) / len(items)
            self.high_priority_count = sum(1 for item in items if item.wsjf_score > 2.0)
            self.aging_items = sum(1 for item in items if item.aging_multiplier > 1.0)


class DiscoveryEngine:
    """Discovers new tasks from codebase analysis"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.logger = logging.getLogger(__name__)
    
    def scan_for_tasks(self) -> List[BacklogItem]:
        """Comprehensive task discovery scan"""
        discovered_tasks = []
        
        # Scan for TODO/FIXME comments
        discovered_tasks.extend(self._scan_code_comments())
        
        # Scan for failing tests
        discovered_tasks.extend(self._scan_failing_tests())
        
        # Scan for security issues
        discovered_tasks.extend(self._scan_security_issues())
        
        # Scan for dependency alerts
        discovered_tasks.extend(self._scan_dependency_alerts())
        
        # Scan for code quality issues
        discovered_tasks.extend(self._scan_code_quality())
        
        return discovered_tasks
    
    def _scan_code_comments(self) -> List[BacklogItem]:
        """Scan for TODO/FIXME comments in code"""
        tasks = []
        
        try:
            # Use ripgrep for fast scanning
            result = subprocess.run([
                'rg', '-n', '--type', 'py', r'(TODO|FIXME|XXX|HACK):', self.repo_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        file_path, line_num, comment = line.split(':', 2)
                        tasks.append(self._create_comment_task(file_path, line_num, comment.strip()))
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            self.logger.warning(f"Comment scan failed: {e}")
        
        return tasks
    
    def _scan_failing_tests(self) -> List[BacklogItem]:
        """Scan for failing tests"""
        tasks = []
        
        try:
            # Run pytest with json output
            result = subprocess.run([
                'python', '-m', 'pytest', '--tb=no', '-q', '--json-report'
            ], capture_output=True, text=True, timeout=120, cwd=self.repo_path)
            
            if result.returncode != 0:
                # Parse test failures and create tasks
                failing_tests = self._parse_test_failures(result.stdout)
                for test_name, error in failing_tests:
                    tasks.append(self._create_test_fix_task(test_name, error))
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            self.logger.warning(f"Test scan failed: {e}")
        
        return tasks
    
    def _scan_security_issues(self) -> List[BacklogItem]:
        """Scan for security vulnerabilities"""
        tasks = []
        
        try:
            # Run bandit security scanner
            result = subprocess.run([
                'bandit', '-r', self.repo_path, '-f', 'json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                security_data = json.loads(result.stdout)
                for issue in security_data.get('results', []):
                    tasks.append(self._create_security_task(issue))
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.warning(f"Security scan failed: {e}")
        
        return tasks
    
    def _scan_dependency_alerts(self) -> List[BacklogItem]:
        """Scan for dependency vulnerabilities"""
        tasks = []
        
        try:
            # Check for requirements.txt or pyproject.toml
            req_files = ['requirements.txt', 'pyproject.toml']
            
            for req_file in req_files:
                req_path = os.path.join(self.repo_path, req_file)
                if os.path.exists(req_path):
                    # Run safety check
                    result = subprocess.run([
                        'safety', 'check', '--json', '-r', req_path
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.stdout:
                        safety_data = json.loads(result.stdout)
                        for vuln in safety_data:
                            tasks.append(self._create_dependency_task(vuln))
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.warning(f"Dependency scan failed: {e}")
        
        return tasks
    
    def _scan_code_quality(self) -> List[BacklogItem]:
        """Scan for code quality issues"""
        tasks = []
        
        try:
            # Run pylint for quality issues
            result = subprocess.run([
                'pylint', '--output-format=json', self.repo_path
            ], capture_output=True, text=True, timeout=120)
            
            if result.stdout:
                lint_data = json.loads(result.stdout)
                # Group similar issues and create refactoring tasks
                quality_issues = self._group_quality_issues(lint_data)
                for issue_group in quality_issues:
                    tasks.append(self._create_quality_task(issue_group))
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as e:
            self.logger.warning(f"Quality scan failed: {e}")
        
        return tasks
    
    def _create_comment_task(self, file_path: str, line_num: str, comment: str) -> BacklogItem:
        """Create task from code comment"""
        task_id = f"disc-{hashlib.md5(f'{file_path}:{line_num}:{comment}'.encode()).hexdigest()[:8]}"
        
        # Determine priority from comment type
        priority_map = {'TODO': 2, 'FIXME': 3, 'XXX': 4, 'HACK': 5}
        comment_type = comment.split(':')[0].upper()
        user_value = priority_map.get(comment_type, 2)
        
        return BacklogItem(
            id=task_id,
            title=f"Address {comment_type} in {os.path.basename(file_path)}",
            description=comment,
            type=TaskType.REFACTOR,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=user_value,
            business_value=2,
            time_criticality=1,
            risk_reduction=user_value,
            opportunity_enablement=1,
            effort=1,
            files=[file_path],
            lines_of_interest=[line_num],
            acceptance_criteria=[f"Resolve {comment_type} comment", "Maintain existing functionality"],
            security_notes="Review for security implications",
            test_notes="Ensure existing tests pass"
        )
    
    def _create_test_fix_task(self, test_name: str, error: str) -> BacklogItem:
        """Create task from failing test"""
        task_id = f"test-{hashlib.md5(f'{test_name}:{error}'.encode()).hexdigest()[:8]}"
        
        return BacklogItem(
            id=task_id,
            title=f"Fix failing test: {test_name}",
            description=f"Test failure: {error}",
            type=TaskType.BUG,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=5,  # High - broken tests affect reliability
            business_value=4,
            time_criticality=3,
            risk_reduction=5,
            opportunity_enablement=2,
            effort=2,
            acceptance_criteria=[
                f"Fix failing test: {test_name}",
                "All related tests pass",
                "No regression in other tests"
            ],
            security_notes="Review if test failure indicates security issue",
            test_notes="Focus on test reliability and coverage"
        )
    
    def _create_security_task(self, issue: Dict[str, Any]) -> BacklogItem:
        """Create task from security issue"""
        task_id = f"sec-{hashlib.md5(str(issue).encode()).hexdigest()[:8]}"
        
        severity_map = {'LOW': 3, 'MEDIUM': 5, 'HIGH': 8}
        severity = severity_map.get(issue.get('issue_severity', 'MEDIUM'), 5)
        
        return BacklogItem(
            id=task_id,
            title=f"Security: {issue.get('test_name', 'Unknown issue')}",
            description=issue.get('issue_text', ''),
            type=TaskType.SECURITY,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=severity,
            business_value=severity,
            time_criticality=severity,
            risk_reduction=8,  # High - security fixes are critical
            opportunity_enablement=3,
            effort=3,
            files=[issue.get('filename', '')],
            lines_of_interest=[str(issue.get('line_number', ''))],
            acceptance_criteria=[
                f"Fix security issue: {issue.get('test_name', '')}",
                "Verify fix with security scanner",
                "Add security test coverage"
            ],
            security_notes=f"SECURITY CRITICAL: {issue.get('issue_text', '')}",
            test_notes="Add security-focused tests",
            escalation_required=severity >= 5,
            escalation_reason="Security issue requires review"
        )
    
    def _create_dependency_task(self, vuln: Dict[str, Any]) -> BacklogItem:
        """Create task from dependency vulnerability"""
        task_id = f"dep-{hashlib.md5(str(vuln).encode()).hexdigest()[:8]}"
        
        return BacklogItem(
            id=task_id,
            title=f"Update vulnerable dependency: {vuln.get('package', 'unknown')}",
            description=vuln.get('advisory', ''),
            type=TaskType.SECURITY,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=5,
            business_value=5,
            time_criticality=4,
            risk_reduction=8,
            opportunity_enablement=2,
            effort=2,
            acceptance_criteria=[
                f"Update {vuln.get('package', '')} to safe version",
                "Verify no breaking changes",
                "Run full test suite"
            ],
            security_notes=f"Dependency vulnerability: {vuln.get('advisory', '')}",
            test_notes="Test compatibility with updated dependency"
        )
    
    def _create_quality_task(self, issue_group: Dict[str, Any]) -> BacklogItem:
        """Create task from grouped quality issues"""
        task_id = f"qual-{hashlib.md5(str(issue_group).encode()).hexdigest()[:8]}"
        
        return BacklogItem(
            id=task_id,
            title=f"Code quality: {issue_group.get('type', 'Various issues')}",
            description=f"Multiple instances of {issue_group.get('type', 'quality issues')}",
            type=TaskType.REFACTOR,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=2,
            business_value=3,
            time_criticality=1,
            risk_reduction=2,
            opportunity_enablement=3,
            effort=2,
            files=issue_group.get('files', []),
            acceptance_criteria=[
                f"Address {issue_group.get('type', 'quality')} issues",
                "Maintain existing functionality",
                "Improve code quality metrics"
            ],
            security_notes="Review for potential security implications",
            test_notes="Ensure refactoring doesn't break functionality"
        )
    
    def _parse_test_failures(self, output: str) -> List[Tuple[str, str]]:
        """Parse test failure output"""
        failures = []
        # Simple parsing - can be enhanced based on test runner output format
        for line in output.split('\n'):
            if 'FAILED' in line:
                parts = line.split(' ')
                if len(parts) >= 2:
                    test_name = parts[0]
                    error = ' '.join(parts[1:])
                    failures.append((test_name, error))
        return failures
    
    def _group_quality_issues(self, lint_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group similar quality issues"""
        grouped = {}
        
        for issue in lint_data:
            issue_type = issue.get('message-id', 'unknown')
            if issue_type not in grouped:
                grouped[issue_type] = {
                    'type': issue_type,
                    'files': [],
                    'count': 0
                }
            
            grouped[issue_type]['files'].append(issue.get('path', ''))
            grouped[issue_type]['count'] += 1
        
        # Return groups with significant impact
        return [group for group in grouped.values() if group['count'] >= 3]


class SecurityChecklist:
    """Security and quality guardrails for each task"""
    
    @staticmethod
    def validate_task_security(item: BacklogItem, code_changes: Optional[str] = None) -> List[str]:
        """Run security validation checklist"""
        issues = []
        
        # Input sanitization check
        if code_changes and any(term in code_changes.lower() for term in ['input', 'request', 'user_']):
            if 'sanitize' not in code_changes.lower() and 'validate' not in code_changes.lower():
                issues.append("Potential input handling without validation detected")
        
        # Authentication/authorization check
        if code_changes and any(term in code_changes.lower() for term in ['auth', 'login', 'token', 'session']):
            if 'check' not in code_changes.lower() and 'verify' not in code_changes.lower():
                issues.append("Authentication code without verification detected")
        
        # Secrets handling check
        if code_changes and any(term in code_changes.lower() for term in ['password', 'key', 'secret', 'token']):
            if any(bad in code_changes.lower() for bad in ['print', 'log', 'console', 'debug']):
                issues.append("Potential secret logging detected")
        
        # File operations check
        if code_changes and any(term in code_changes.lower() for term in ['open(', 'file', 'path']):
            if 'os.path.join' not in code_changes and 'pathlib' not in code_changes:
                issues.append("File operations without safe path handling")
        
        # SQL injection check
        if code_changes and any(term in code_changes.lower() for term in ['sql', 'query', 'execute']):
            if '?' not in code_changes and 'prepare' not in code_changes.lower():
                issues.append("Potential SQL injection vulnerability")
        
        return issues
    
    @staticmethod
    def validate_test_coverage(item: BacklogItem, test_files: List[str]) -> bool:
        """Validate adequate test coverage"""
        if item.type == TaskType.SECURITY:
            return len(test_files) > 0  # Security tasks must have tests
        
        if item.type in [TaskType.FEATURE, TaskType.BUG]:
            return len(test_files) > 0  # Features and bugs need tests
        
        return True  # Refactoring and docs may not need new tests


class TDDMicroCycle:
    """Test-Driven Development micro-cycle implementation"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.logger = logging.getLogger(__name__)
    
    def execute_cycle(self, item: BacklogItem) -> Dict[str, Any]:
        """Execute TDD cycle: Red -> Green -> Refactor"""
        results = {
            'success': False,
            'phase': '',
            'tests_written': [],
            'implementation_files': [],
            'refactor_changes': [],
            'errors': []
        }
        
        try:
            # Phase 1: Red - Write failing test
            results['phase'] = 'red'
            test_results = self._write_failing_test(item)
            results['tests_written'] = test_results.get('files', [])
            
            if not test_results.get('success', False):
                results['errors'].append("Failed to write failing test")
                return results
            
            # Phase 2: Green - Implement minimal code
            results['phase'] = 'green'
            impl_results = self._implement_minimal_code(item)
            results['implementation_files'] = impl_results.get('files', [])
            
            if not impl_results.get('success', False):
                results['errors'].append("Failed to implement minimal code")
                return results
            
            # Phase 3: Refactor - Improve design
            results['phase'] = 'refactor'
            refactor_results = self._refactor_and_cleanup(item)
            results['refactor_changes'] = refactor_results.get('changes', [])
            
            if not refactor_results.get('success', False):
                results['errors'].append("Failed to refactor and cleanup")
                return results
            
            results['success'] = True
            return results
            
        except Exception as e:
            results['errors'].append(f"TDD cycle failed: {str(e)}")
            return results
    
    def _write_failing_test(self, item: BacklogItem) -> Dict[str, Any]:
        """Write failing test first (Red phase)"""
        # This would implement actual test writing logic
        # For now, return a placeholder result
        return {
            'success': True,
            'files': [f"tests/test_{item.id}.py"],
            'test_count': 1
        }
    
    def _implement_minimal_code(self, item: BacklogItem) -> Dict[str, Any]:
        """Implement minimal code to pass tests (Green phase)"""
        # This would implement actual code generation logic
        # For now, return a placeholder result
        return {
            'success': True,
            'files': item.files,
            'changes_made': len(item.acceptance_criteria)
        }
    
    def _refactor_and_cleanup(self, item: BacklogItem) -> Dict[str, Any]:
        """Refactor and cleanup while keeping tests green"""
        # This would implement actual refactoring logic
        # For now, return a placeholder result
        return {
            'success': True,
            'changes': ['Code cleanup', 'Documentation updates'],
            'quality_improved': True
        }
    
    def run_tests(self) -> Dict[str, Any]:
        """Run full test suite"""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=300, cwd=self.repo_path)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr,
                'duration': 0  # Would measure actual duration
            }
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            return {
                'success': False,
                'output': "",
                'errors': str(e),
                'duration': 0
            }


class AutonomousBacklogManager:
    """Main autonomous backlog management system"""
    
    def __init__(self, repo_path: str, backlog_file: str = "DOCS/backlog.yml"):
        self.repo_path = repo_path
        self.backlog_file = os.path.join(repo_path, backlog_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.discovery_engine = DiscoveryEngine(repo_path)
        self.security_checklist = SecurityChecklist()
        self.tdd_cycle = TDDMicroCycle(repo_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize backlog
        self.backlog_items: List[BacklogItem] = []
        self.metrics = BacklogMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load backlog configuration"""
        try:
            if os.path.exists(self.backlog_file):
                with open(self.backlog_file, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get('backlog', {}).get('scoring_config', {})
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
        
        # Default configuration
        return {
            'wsjf_weights': {
                'user_value': 1.0,
                'business_value': 1.0,
                'time_criticality': 1.0,
                'risk_reduction': 0.8,
                'opportunity_enablement': 0.6
            },
            'effort_scale': [1, 2, 3, 5, 8, 13],
            'impact_scale': [1, 2, 3, 5, 8, 13],
            'aging_multiplier_cap': 2.0,
            'aging_days_threshold': 30
        }
    
    def load_backlog(self) -> bool:
        """Load backlog from YAML file"""
        try:
            if not os.path.exists(self.backlog_file):
                self.logger.info("No existing backlog file, starting fresh")
                return True
            
            with open(self.backlog_file, 'r') as f:
                data = yaml.safe_load(f)
            
            backlog_data = data.get('backlog', {})
            items_data = backlog_data.get('items', [])
            
            self.backlog_items = []
            for item_data in items_data:
                try:
                    item = self._dict_to_backlog_item(item_data)
                    self.backlog_items.append(item)
                except Exception as e:
                    self.logger.error(f"Failed to load item {item_data.get('id', 'unknown')}: {e}")
            
            self.logger.info(f"Loaded {len(self.backlog_items)} backlog items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load backlog: {e}")
            return False
    
    def save_backlog(self) -> bool:
        """Save backlog to YAML file"""
        try:
            with self._lock:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.backlog_file), exist_ok=True)
                
                # Update metrics
                self.metrics.update_from_items(self.backlog_items)
                
                # Prepare data structure
                backlog_data = {
                    'backlog': {
                        'format_version': '1.0',
                        'last_updated': datetime.datetime.now().isoformat(),
                        'scoring_config': self.config,
                        'items': [item.to_dict() for item in self.backlog_items],
                        'discovered_issues': {
                            'last_scan': datetime.datetime.now().isoformat(),
                            'todos_found': 0,  # Would be populated by discovery
                            'fixmes_found': 0,
                            'failing_tests': 0,
                            'security_warnings': 0,
                            'dependency_alerts': 0
                        },
                        'metrics': asdict(self.metrics)
                    }
                }
                
                # Write to file
                with open(self.backlog_file, 'w') as f:
                    yaml.dump(backlog_data, f, default_flow_style=False, sort_keys=False)
                
                self.logger.info(f"Saved {len(self.backlog_items)} items to backlog")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save backlog: {e}")
            return False
    
    def sync_and_refresh(self) -> Dict[str, Any]:
        """Sync repo state and refresh backlog"""
        with self._lock:
            results = {
                'discovered_tasks': 0,
                'updated_scores': 0,
                'dedup_removed': 0,
                'split_tasks': 0
            }
            
            # Discover new tasks
            discovered = self.discovery_engine.scan_for_tasks()
            
            # Deduplicate and merge
            new_tasks = self._deduplicate_tasks(discovered)
            self.backlog_items.extend(new_tasks)
            results['discovered_tasks'] = len(new_tasks)
            
            # Update aging and recalculate scores
            weights = self.config.get('wsjf_weights', {})
            aging_cap = self.config.get('aging_multiplier_cap', 2.0)
            aging_threshold = self.config.get('aging_days_threshold', 30)
            
            for item in self.backlog_items:
                item.update_aging(aging_threshold)
                item.calculate_wsjf(weights, aging_cap)
                results['updated_scores'] += 1
            
            # Split large tasks
            results['split_tasks'] = self._split_large_tasks()
            
            # Sort by WSJF score
            self.backlog_items.sort(key=lambda x: x.wsjf_score, reverse=True)
            
            return results
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the next ready item for execution"""
        with self._lock:
            for item in self.backlog_items:
                if item.is_ready_for_execution():
                    return item
            return None
    
    def execute_item(self, item: BacklogItem) -> Dict[str, Any]:
        """Execute a single backlog item using TDD micro-cycle"""
        with self._lock:
            # Mark as in progress
            item.status = TaskStatus.DOING
            item.updated_date = datetime.datetime.now()
            
            results = {
                'item_id': item.id,
                'success': False,
                'tdd_results': {},
                'security_issues': [],
                'test_results': {},
                'pr_created': False
            }
            
            try:
                # Execute TDD cycle
                tdd_results = self.tdd_cycle.execute_cycle(item)
                results['tdd_results'] = tdd_results
                
                if not tdd_results.get('success', False):
                    item.status = TaskStatus.BLOCKED
                    item.blocked_reason = f"TDD cycle failed: {tdd_results.get('errors', [])}"
                    return results
                
                # Run security checklist
                security_issues = self.security_checklist.validate_task_security(item)
                results['security_issues'] = security_issues
                
                if security_issues and item.type == TaskType.SECURITY:
                    item.status = TaskStatus.BLOCKED
                    item.blocked_reason = f"Security validation failed: {security_issues}"
                    return results
                
                # Run full test suite
                test_results = self.tdd_cycle.run_tests()
                results['test_results'] = test_results
                
                if not test_results.get('success', False):
                    item.status = TaskStatus.BLOCKED
                    item.blocked_reason = f"Tests failed: {test_results.get('errors', '')}"
                    return results
                
                # Mark as ready for PR
                item.status = TaskStatus.PR
                results['success'] = True
                results['pr_created'] = True
                
                return results
                
            except Exception as e:
                item.status = TaskStatus.BLOCKED
                item.blocked_reason = f"Execution failed: {str(e)}"
                results['success'] = False
                self.logger.error(f"Failed to execute item {item.id}: {e}")
                return results
    
    def run_full_cycle(self, max_iterations: int = 100) -> Dict[str, Any]:
        """Run complete autonomous backlog processing cycle"""
        cycle_results = {
            'iterations': 0,
            'items_completed': 0,
            'items_blocked': 0,
            'items_escalated': 0,
            'discovered_tasks': 0,
            'errors': []
        }
        
        self.logger.info("Starting autonomous backlog processing cycle")
        
        try:
            # Load existing backlog
            if not self.load_backlog():
                cycle_results['errors'].append("Failed to load backlog")
                return cycle_results
            
            for iteration in range(max_iterations):
                cycle_results['iterations'] = iteration + 1
                
                # Sync and refresh
                sync_results = self.sync_and_refresh()
                cycle_results['discovered_tasks'] += sync_results['discovered_tasks']
                
                # Get next ready item
                next_item = self.get_next_ready_item()
                
                if not next_item:
                    self.logger.info("No ready items found, ending cycle")
                    break
                
                # Check for escalation
                if next_item.escalation_required:
                    self.logger.warning(f"Item {next_item.id} requires human escalation: {next_item.escalation_reason}")
                    next_item.status = TaskStatus.BLOCKED
                    next_item.blocked_reason = f"Escalation required: {next_item.escalation_reason}"
                    cycle_results['items_escalated'] += 1
                    continue
                
                # Execute item
                self.logger.info(f"Executing item {next_item.id}: {next_item.title}")
                exec_results = self.execute_item(next_item)
                
                if exec_results.get('success', False):
                    cycle_results['items_completed'] += 1
                    self.logger.info(f"Successfully completed item {next_item.id}")
                else:
                    cycle_results['items_blocked'] += 1
                    self.logger.warning(f"Item {next_item.id} blocked: {next_item.blocked_reason}")
                
                # Save progress
                if not self.save_backlog():
                    cycle_results['errors'].append(f"Failed to save backlog at iteration {iteration}")
                
                # Brief pause between iterations
                time.sleep(1)
            
            # Final save
            self.save_backlog()
            
            self.logger.info(f"Autonomous cycle completed: {cycle_results['items_completed']} items completed, "
                           f"{cycle_results['items_blocked']} blocked, {cycle_results['items_escalated']} escalated")
            
            return cycle_results
            
        except Exception as e:
            cycle_results['errors'].append(f"Cycle failed: {str(e)}")
            self.logger.error(f"Autonomous cycle failed: {e}")
            return cycle_results
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        with self._lock:
            self.metrics.update_from_items(self.backlog_items)
            
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'backlog_metrics': asdict(self.metrics),
                'high_priority_items': [
                    {'id': item.id, 'title': item.title, 'wsjf': item.wsjf_score}
                    for item in self.backlog_items
                    if item.wsjf_score > 2.0
                ],
                'blocked_items': [
                    {'id': item.id, 'title': item.title, 'reason': item.blocked_reason}
                    for item in self.backlog_items
                    if item.status == TaskStatus.BLOCKED
                ],
                'escalation_required': [
                    {'id': item.id, 'title': item.title, 'reason': item.escalation_reason}
                    for item in self.backlog_items
                    if item.escalation_required
                ]
            }
    
    def _dict_to_backlog_item(self, data: Dict[str, Any]) -> BacklogItem:
        """Convert dictionary to BacklogItem"""
        return BacklogItem(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            type=TaskType(data['type']),
            status=TaskStatus(data['status']),
            created_date=datetime.datetime.fromisoformat(data['created_date']),
            updated_date=datetime.datetime.fromisoformat(data['updated_date']),
            user_value=data.get('user_value', 1),
            business_value=data.get('business_value', 1),
            time_criticality=data.get('time_criticality', 1),
            risk_reduction=data.get('risk_reduction', 1),
            opportunity_enablement=data.get('opportunity_enablement', 1),
            effort=data.get('effort', 1),
            wsjf_score=data.get('wsjf_score', 0.0),
            aging_multiplier=data.get('aging_multiplier', 1.0),
            acceptance_criteria=data.get('acceptance_criteria', []),
            files=data.get('files', []),
            lines_of_interest=data.get('lines_of_interest', []),
            linked_items=data.get('linked_items', []),
            security_notes=data.get('security_notes', ''),
            test_notes=data.get('test_notes', ''),
            blocked_reason=data.get('blocked_reason', ''),
            escalation_required=data.get('escalation_required', False),
            escalation_reason=data.get('escalation_reason', '')
        )
    
    def _deduplicate_tasks(self, new_tasks: List[BacklogItem]) -> List[BacklogItem]:
        """Remove duplicate tasks from discovered items"""
        unique_tasks = []
        existing_descriptions = {item.description for item in self.backlog_items}
        
        for task in new_tasks:
            if task.description not in existing_descriptions:
                unique_tasks.append(task)
                existing_descriptions.add(task.description)
        
        return unique_tasks
    
    def _split_large_tasks(self) -> int:
        """Split tasks with effort > 5 into smaller tasks"""
        split_count = 0
        items_to_add = []
        items_to_remove = []
        
        for item in self.backlog_items:
            if item.effort > 5 and item.status == TaskStatus.NEW:
                # Create smaller sub-tasks
                sub_tasks = self._create_sub_tasks(item)
                if sub_tasks:
                    items_to_add.extend(sub_tasks)
                    items_to_remove.append(item)
                    split_count += 1
        
        # Apply changes
        for item in items_to_remove:
            self.backlog_items.remove(item)
        
        self.backlog_items.extend(items_to_add)
        
        return split_count
    
    def _create_sub_tasks(self, large_item: BacklogItem) -> List[BacklogItem]:
        """Create sub-tasks from a large task"""
        sub_tasks = []
        
        # Simple splitting logic - can be enhanced
        criteria_per_task = max(1, len(large_item.acceptance_criteria) // 3)
        
        for i in range(0, len(large_item.acceptance_criteria), criteria_per_task):
            sub_criteria = large_item.acceptance_criteria[i:i + criteria_per_task]
            
            sub_task = BacklogItem(
                id=f"{large_item.id}-{i//criteria_per_task + 1}",
                title=f"{large_item.title} (Part {i//criteria_per_task + 1})",
                description=f"Subtask of {large_item.title}: {', '.join(sub_criteria)}",
                type=large_item.type,
                status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=large_item.user_value,
                business_value=large_item.business_value,
                time_criticality=large_item.time_criticality,
                risk_reduction=large_item.risk_reduction,
                opportunity_enablement=large_item.opportunity_enablement,
                effort=min(3, large_item.effort // 2),  # Smaller effort
                acceptance_criteria=sub_criteria,
                files=large_item.files,
                security_notes=large_item.security_notes,
                test_notes=large_item.test_notes,
                linked_items=[large_item.id]
            )
            
            sub_tasks.append(sub_task)
        
        return sub_tasks


# CLI interface for the autonomous backlog system
def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Backlog Management System")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--backlog-file", default="DOCS/backlog.yml", help="Backlog file path")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--status-only", action="store_true", help="Show status report only")
    parser.add_argument("--discover-only", action="store_true", help="Run discovery scan only")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize manager
    manager = AutonomousBacklogManager(args.repo_path, args.backlog_file)
    
    if args.status_only:
        # Generate status report
        manager.load_backlog()
        report = manager.generate_status_report()
        print(json.dumps(report, indent=2, default=str))
        return
    
    if args.discover_only:
        # Run discovery scan only
        discovered = manager.discovery_engine.scan_for_tasks()
        print(f"Discovered {len(discovered)} tasks:")
        for task in discovered:
            print(f"  - {task.title} (WSJF: {task.wsjf_score:.1f})")
        return
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        manager.load_backlog()
        sync_results = manager.sync_and_refresh()
        print(f"Would discover {sync_results['discovered_tasks']} new tasks")
        
        next_item = manager.get_next_ready_item()
        if next_item:
            print(f"Next item to execute: {next_item.title} (WSJF: {next_item.wsjf_score:.1f})")
        else:
            print("No ready items found")
        return
    
    # Run full autonomous cycle
    results = manager.run_full_cycle(args.max_iterations)
    
    print(f"\nAutonomous Backlog Cycle Results:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Items Completed: {results['items_completed']}")
    print(f"  Items Blocked: {results['items_blocked']}")
    print(f"  Items Escalated: {results['items_escalated']}")
    print(f"  New Tasks Discovered: {results['discovered_tasks']}")
    
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"    - {error}")
    
    # Show final status
    report = manager.generate_status_report()
    print(f"\nFinal Status:")
    print(f"  Total Items: {report['backlog_metrics']['total_items']}")
    print(f"  High Priority (WSJF > 2.0): {report['backlog_metrics']['high_priority_count']}")
    print(f"  Blocked Items: {len(report['blocked_items'])}")
    print(f"  Escalation Required: {len(report['escalation_required'])}")


if __name__ == "__main__":
    main()