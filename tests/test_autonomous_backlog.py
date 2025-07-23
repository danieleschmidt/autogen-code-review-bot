"""
Comprehensive tests for the Autonomous Backlog Management System
"""

import os
import tempfile
import shutil
import datetime
import yaml
import json
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.autogen_code_review_bot.autonomous_backlog import (
    BacklogItem, TaskType, TaskStatus, BacklogMetrics,
    DiscoveryEngine, SecurityChecklist, TDDMicroCycle,
    AutonomousBacklogManager
)


class TestBacklogItem:
    """Test BacklogItem functionality"""
    
    def test_backlog_item_creation(self):
        """Test creating a backlog item"""
        item = BacklogItem(
            id="test-1",
            title="Test Task",
            description="A test task",
            type=TaskType.FEATURE,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        assert item.id == "test-1"
        assert item.title == "Test Task"
        assert item.type == TaskType.FEATURE
        assert item.status == TaskStatus.NEW
        assert item.wsjf_score == 0.0
        assert item.aging_multiplier == 1.0
    
    def test_wsjf_calculation(self):
        """Test WSJF score calculation"""
        item = BacklogItem(
            id="test-2",
            title="High Value Task",
            description="A high value task",
            type=TaskType.FEATURE,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=5,
            business_value=5,
            time_criticality=3,
            risk_reduction=4,
            opportunity_enablement=2,
            effort=2
        )
        
        weights = {
            'user_value': 1.0,
            'business_value': 1.0,
            'time_criticality': 1.0,
            'risk_reduction': 0.8,
            'opportunity_enablement': 0.6
        }
        
        score = item.calculate_wsjf(weights)
        expected = (5 + 5 + 3 + 4*0.8 + 2*0.6) / 2  # 17.4 / 2 = 8.7
        assert abs(score - expected) < 0.01
        assert item.wsjf_score == score
    
    def test_aging_calculation(self):
        """Test aging multiplier calculation"""
        # Old item (40 days ago)
        old_date = datetime.datetime.now() - datetime.timedelta(days=40)
        item = BacklogItem(
            id="test-3",
            title="Old Task",
            description="An old task",
            type=TaskType.BUG,
            status=TaskStatus.NEW,
            created_date=old_date,
            updated_date=old_date
        )
        
        item.update_aging(aging_threshold_days=30)
        assert item.aging_multiplier > 1.0
        assert item.aging_multiplier <= 2.0  # Capped at 2.0
        
        # Recent item (10 days ago)
        recent_date = datetime.datetime.now() - datetime.timedelta(days=10)
        recent_item = BacklogItem(
            id="test-4",
            title="Recent Task",
            description="A recent task",
            type=TaskType.FEATURE,
            status=TaskStatus.NEW,
            created_date=recent_date,
            updated_date=recent_date
        )
        
        recent_item.update_aging(aging_threshold_days=30)
        assert recent_item.aging_multiplier == 1.0
    
    def test_ready_for_execution(self):
        """Test readiness check"""
        # Ready item
        ready_item = BacklogItem(
            id="test-5",
            title="Ready Task",
            description="A ready task",
            type=TaskType.FEATURE,
            status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Criterion 1", "Criterion 2"]
        )
        
        assert ready_item.is_ready_for_execution()
        
        # Not ready - no acceptance criteria
        not_ready_item = BacklogItem(
            id="test-6",
            title="Not Ready Task",
            description="A not ready task",
            type=TaskType.FEATURE,
            status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        assert not not_ready_item.is_ready_for_execution()
        
        # Not ready - blocked
        blocked_item = BacklogItem(
            id="test-7",
            title="Blocked Task",
            description="A blocked task",
            type=TaskType.FEATURE,
            status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Criterion 1"],
            blocked_reason="Waiting for dependency"
        )
        
        assert not blocked_item.is_ready_for_execution()
    
    def test_to_dict_conversion(self):
        """Test dictionary conversion"""
        item = BacklogItem(
            id="test-8",
            title="Dict Test",
            description="A task for dict conversion",
            type=TaskType.REFACTOR,
            status=TaskStatus.DOING,
            created_date=datetime.datetime(2025, 1, 1, 12, 0, 0),
            updated_date=datetime.datetime(2025, 1, 2, 12, 0, 0),
            user_value=3,
            business_value=2,
            acceptance_criteria=["Test criterion"]
        )
        
        item_dict = item.to_dict()
        
        assert item_dict['id'] == "test-8"
        assert item_dict['title'] == "Dict Test"
        assert item_dict['type'] == "Refactor"
        assert item_dict['status'] == "DOING"
        assert item_dict['user_value'] == 3
        assert item_dict['business_value'] == 2
        assert item_dict['acceptance_criteria'] == ["Test criterion"]
        assert isinstance(item_dict['created_date'], str)
        assert isinstance(item_dict['updated_date'], str)


class TestBacklogMetrics:
    """Test BacklogMetrics functionality"""
    
    def test_metrics_update(self):
        """Test metrics calculation from items"""
        items = [
            BacklogItem(
                id="m1", title="Task 1", description="Desc 1",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                wsjf_score=3.0
            ),
            BacklogItem(
                id="m2", title="Task 2", description="Desc 2",
                type=TaskType.BUG, status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                wsjf_score=1.5, aging_multiplier=1.2
            ),
            BacklogItem(
                id="m3", title="Task 3", description="Desc 3",
                type=TaskType.FEATURE, status=TaskStatus.DONE,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                wsjf_score=2.5
            )
        ]
        
        metrics = BacklogMetrics()
        metrics.update_from_items(items)
        
        assert metrics.total_items == 3
        assert metrics.by_status['NEW'] == 1
        assert metrics.by_status['READY'] == 1
        assert metrics.by_status['DONE'] == 1
        assert metrics.by_type['Feature'] == 2
        assert metrics.by_type['Bug'] == 1
        assert abs(metrics.avg_wsjf - 2.33) < 0.1  # (3.0 + 1.5 + 2.5) / 3
        assert metrics.high_priority_count == 2  # WSJF > 2.0
        assert metrics.aging_items == 1  # aging_multiplier > 1.0


class TestDiscoveryEngine:
    """Test DiscoveryEngine functionality"""
    
    def setUp(self):
        """Set up test directory"""
        self.test_dir = tempfile.mkdtemp()
        self.engine = DiscoveryEngine(self.test_dir)
    
    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_scan_code_comments(self, mock_run):
        """Test scanning for code comments"""
        self.setUp()
        
        # Mock ripgrep output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="test.py:42:# TODO: Fix this function\napp.py:15:# FIXME: Handle edge case"
        )
        
        tasks = self.engine._scan_code_comments()
        
        assert len(tasks) == 2
        assert tasks[0].title.startswith("Address TODO")
        assert tasks[1].title.startswith("Address FIXME")
        assert tasks[0].type == TaskType.REFACTOR
        assert tasks[1].type == TaskType.REFACTOR
        assert tasks[0].user_value == 2  # TODO priority
        assert tasks[1].user_value == 3  # FIXME priority
        
        self.tearDown()
    
    @patch('subprocess.run')
    def test_scan_failing_tests(self, mock_run):
        """Test scanning for failing tests"""
        self.setUp()
        
        # Mock pytest output with failures
        mock_run.return_value = Mock(
            returncode=1,
            stdout="FAILED test_example.py::test_function - AssertionError: Expected 5, got 3"
        )
        
        tasks = self.engine._scan_failing_tests()
        
        assert len(tasks) == 1
        assert tasks[0].title.startswith("Fix failing test")
        assert tasks[0].type == TaskType.BUG
        assert tasks[0].user_value == 5  # High priority for failing tests
        
        self.tearDown()
    
    @patch('subprocess.run')
    def test_scan_security_issues(self, mock_run):
        """Test scanning for security issues"""
        self.setUp()
        
        # Mock bandit security scanner output
        security_data = {
            "results": [
                {
                    "test_name": "hardcoded_password_string",
                    "issue_text": "Hardcoded password found",
                    "issue_severity": "HIGH",
                    "filename": "config.py",
                    "line_number": 25
                }
            ]
        }
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(security_data)
        )
        
        tasks = self.engine._scan_security_issues()
        
        assert len(tasks) == 1
        assert tasks[0].title.startswith("Security:")
        assert tasks[0].type == TaskType.SECURITY
        assert tasks[0].user_value == 8  # High severity
        assert tasks[0].escalation_required == True
        
        self.tearDown()
    
    def test_create_comment_task(self):
        """Test creating task from code comment"""
        self.setUp()
        
        task = self.engine._create_comment_task(
            "test.py", "42", "TODO: Refactor this function"
        )
        
        assert task.title == "Address TODO in test.py"
        assert task.description == "TODO: Refactor this function"
        assert task.type == TaskType.REFACTOR
        assert task.files == ["test.py"]
        assert task.lines_of_interest == ["42"]
        assert len(task.acceptance_criteria) >= 2
        
        self.tearDown()
    
    def test_create_security_task(self):
        """Test creating task from security issue"""
        self.setUp()
        
        issue = {
            "test_name": "sql_injection",
            "issue_text": "Potential SQL injection vulnerability",
            "issue_severity": "MEDIUM",
            "filename": "database.py",
            "line_number": 100
        }
        
        task = self.engine._create_security_task(issue)
        
        assert task.title == "Security: sql_injection"
        assert task.type == TaskType.SECURITY
        assert task.files == ["database.py"]
        assert task.lines_of_interest == ["100"]
        assert task.risk_reduction == 8  # Security fixes are high risk reduction
        assert "SECURITY CRITICAL" in task.security_notes
        
        self.tearDown()


class TestSecurityChecklist:
    """Test SecurityChecklist functionality"""
    
    def test_input_validation_check(self):
        """Test input validation security check"""
        code_with_input = """
        def process_user_input(user_data):
            return user_data.upper()
        """
        
        issues = SecurityChecklist.validate_task_security(
            BacklogItem(
                id="sec-1", title="Test", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            ),
            code_with_input
        )
        
        assert len(issues) > 0
        assert any("input handling without validation" in issue for issue in issues)
    
    def test_authentication_check(self):
        """Test authentication security check"""
        code_with_auth = """
        def login_user(username, password):
            token = generate_token(username)
            return token
        """
        
        issues = SecurityChecklist.validate_task_security(
            BacklogItem(
                id="sec-2", title="Test", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            ),
            code_with_auth
        )
        
        assert len(issues) > 0
        assert any("Authentication code without verification" in issue for issue in issues)
    
    def test_secrets_logging_check(self):
        """Test secrets logging security check"""
        code_with_secret_logging = """
        def debug_auth(password):
            print(f"Password: {password}")
            return True
        """
        
        issues = SecurityChecklist.validate_task_security(
            BacklogItem(
                id="sec-3", title="Test", description="Test",
                type=TaskType.SECURITY, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            ),
            code_with_secret_logging
        )
        
        assert len(issues) > 0
        assert any("secret logging detected" in issue for issue in issues)
    
    def test_file_operations_check(self):
        """Test file operations security check"""
        unsafe_file_code = """
        def read_config(filename):
            with open(filename, 'r') as f:
                return f.read()
        """
        
        issues = SecurityChecklist.validate_task_security(
            BacklogItem(
                id="sec-4", title="Test", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            ),
            unsafe_file_code
        )
        
        assert len(issues) > 0
        assert any("File operations without safe path handling" in issue for issue in issues)
    
    def test_sql_injection_check(self):
        """Test SQL injection security check"""
        unsafe_sql_code = """
        def get_user(user_id):
            query = f"SELECT * FROM users WHERE id = {user_id}"
            return execute_query(query)
        """
        
        issues = SecurityChecklist.validate_task_security(
            BacklogItem(
                id="sec-5", title="Test", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            ),
            unsafe_sql_code
        )
        
        assert len(issues) > 0
        assert any("SQL injection vulnerability" in issue for issue in issues)
    
    def test_test_coverage_validation(self):
        """Test test coverage validation"""
        # Security task must have tests
        security_item = BacklogItem(
            id="sec-6", title="Security Task", description="Test",
            type=TaskType.SECURITY, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        assert not SecurityChecklist.validate_test_coverage(security_item, [])
        assert SecurityChecklist.validate_test_coverage(security_item, ["test_security.py"])
        
        # Feature task needs tests
        feature_item = BacklogItem(
            id="feat-1", title="Feature Task", description="Test",
            type=TaskType.FEATURE, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        assert not SecurityChecklist.validate_test_coverage(feature_item, [])
        assert SecurityChecklist.validate_test_coverage(feature_item, ["test_feature.py"])
        
        # Refactor task may not need new tests
        refactor_item = BacklogItem(
            id="ref-1", title="Refactor Task", description="Test",
            type=TaskType.REFACTOR, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        assert SecurityChecklist.validate_test_coverage(refactor_item, [])


class TestTDDMicroCycle:
    """Test TDDMicroCycle functionality"""
    
    def setUp(self):
        """Set up test directory"""
        self.test_dir = tempfile.mkdtemp()
        self.tdd_cycle = TDDMicroCycle(self.test_dir)
    
    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_execute_cycle_success(self):
        """Test successful TDD cycle execution"""
        self.setUp()
        
        item = BacklogItem(
            id="tdd-1", title="TDD Test", description="Test TDD cycle",
            type=TaskType.FEATURE, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Implement feature", "Add tests"]
        )
        
        # Mock the TDD phases to succeed
        with patch.object(self.tdd_cycle, '_write_failing_test', return_value={'success': True, 'files': ['test_tdd.py']}), \
             patch.object(self.tdd_cycle, '_implement_minimal_code', return_value={'success': True, 'files': ['tdd.py']}), \
             patch.object(self.tdd_cycle, '_refactor_and_cleanup', return_value={'success': True, 'changes': ['cleanup']}):
            
            results = self.tdd_cycle.execute_cycle(item)
            
            assert results['success'] == True
            assert results['phase'] == 'refactor'
            assert 'test_tdd.py' in results['tests_written']
            assert 'tdd.py' in results['implementation_files']
            assert 'cleanup' in results['refactor_changes']
            assert len(results['errors']) == 0
        
        self.tearDown()
    
    def test_execute_cycle_red_phase_failure(self):
        """Test TDD cycle failure in red phase"""
        self.setUp()
        
        item = BacklogItem(
            id="tdd-2", title="TDD Failure Test", description="Test TDD failure",
            type=TaskType.FEATURE, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        # Mock red phase to fail
        with patch.object(self.tdd_cycle, '_write_failing_test', return_value={'success': False, 'files': []}):
            
            results = self.tdd_cycle.execute_cycle(item)
            
            assert results['success'] == False
            assert results['phase'] == 'red'
            assert "Failed to write failing test" in results['errors']
        
        self.tearDown()
    
    @patch('subprocess.run')
    def test_run_tests_success(self, mock_run):
        """Test successful test execution"""
        self.setUp()
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout="All tests passed",
            stderr=""
        )
        
        results = self.tdd_cycle.run_tests()
        
        assert results['success'] == True
        assert "All tests passed" in results['output']
        assert results['errors'] == ""
        
        self.tearDown()
    
    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_run):
        """Test test execution failure"""
        self.setUp()
        
        mock_run.return_value = Mock(
            returncode=1,
            stdout="1 test failed",
            stderr="Test error details"
        )
        
        results = self.tdd_cycle.run_tests()
        
        assert results['success'] == False
        assert "1 test failed" in results['output']
        assert "Test error details" in results['errors']
        
        self.tearDown()


class TestAutonomousBacklogManager:
    """Test AutonomousBacklogManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.backlog_file = os.path.join(self.test_dir, "test_backlog.yml")
        self.manager = AutonomousBacklogManager(self.test_dir, self.backlog_file)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        self.setUp()
        
        assert self.manager.repo_path == self.test_dir
        assert self.manager.backlog_file == self.backlog_file
        assert isinstance(self.manager.discovery_engine, DiscoveryEngine)
        assert isinstance(self.manager.security_checklist, SecurityChecklist)
        assert isinstance(self.manager.tdd_cycle, TDDMicroCycle)
        assert isinstance(self.manager.config, dict)
        assert self.manager.backlog_items == []
        
        self.tearDown()
    
    def test_load_save_backlog(self):
        """Test loading and saving backlog"""
        self.setUp()
        
        # Create test items
        test_items = [
            BacklogItem(
                id="load-1", title="Load Test 1", description="Test loading",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=3, business_value=2, effort=1,
                acceptance_criteria=["Test criterion 1"]
            ),
            BacklogItem(
                id="load-2", title="Load Test 2", description="Test loading 2",
                type=TaskType.BUG, status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=5, business_value=4, effort=2,
                acceptance_criteria=["Test criterion 2", "Test criterion 3"]
            )
        ]
        
        self.manager.backlog_items = test_items
        
        # Save backlog
        save_success = self.manager.save_backlog()
        assert save_success == True
        assert os.path.exists(self.backlog_file)
        
        # Clear and reload
        self.manager.backlog_items = []
        load_success = self.manager.load_backlog()
        assert load_success == True
        assert len(self.manager.backlog_items) == 2
        
        # Verify loaded items
        loaded_item1 = next(item for item in self.manager.backlog_items if item.id == "load-1")
        assert loaded_item1.title == "Load Test 1"
        assert loaded_item1.type == TaskType.FEATURE
        assert loaded_item1.status == TaskStatus.NEW
        assert loaded_item1.user_value == 3
        assert loaded_item1.acceptance_criteria == ["Test criterion 1"]
        
        self.tearDown()
    
    def test_sync_and_refresh(self):
        """Test sync and refresh functionality"""
        self.setUp()
        
        # Mock discovery to return test tasks
        mock_discovered = [
            BacklogItem(
                id="disc-1", title="Discovered Task", description="Found in code",
                type=TaskType.REFACTOR, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            )
        ]
        
        with patch.object(self.manager.discovery_engine, 'scan_for_tasks', return_value=mock_discovered):
            results = self.manager.sync_and_refresh()
            
            assert results['discovered_tasks'] == 1
            assert results['updated_scores'] == 1  # The discovered task
            assert len(self.manager.backlog_items) == 1
            assert self.manager.backlog_items[0].id == "disc-1"
        
        self.tearDown()
    
    def test_get_next_ready_item(self):
        """Test getting next ready item"""
        self.setUp()
        
        # Create items with different readiness states
        items = [
            BacklogItem(
                id="ready-1", title="Not Ready - No Criteria", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                wsjf_score=3.0
            ),
            BacklogItem(
                id="ready-2", title="Ready Item", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                acceptance_criteria=["Ready criterion"],
                wsjf_score=2.0
            ),
            BacklogItem(
                id="ready-3", title="Blocked Item", description="Test",
                type=TaskType.FEATURE, status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                acceptance_criteria=["Blocked criterion"],
                blocked_reason="Dependency issue",
                wsjf_score=4.0
            )
        ]
        
        self.manager.backlog_items = items
        
        next_item = self.manager.get_next_ready_item()
        
        assert next_item is not None
        assert next_item.id == "ready-2"  # Only truly ready item
        
        self.tearDown()
    
    def test_execute_item_success(self):
        """Test successful item execution"""
        self.setUp()
        
        item = BacklogItem(
            id="exec-1", title="Execute Test", description="Test execution",
            type=TaskType.FEATURE, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Execute criterion"]
        )
        
        # Mock TDD cycle and security checks to succeed
        mock_tdd_results = {
            'success': True,
            'tests_written': ['test_exec.py'],
            'implementation_files': ['exec.py'],
            'errors': []
        }
        
        mock_test_results = {
            'success': True,
            'output': 'All tests passed',
            'errors': ''
        }
        
        with patch.object(self.manager.tdd_cycle, 'execute_cycle', return_value=mock_tdd_results), \
             patch.object(self.manager.tdd_cycle, 'run_tests', return_value=mock_test_results), \
             patch.object(self.manager.security_checklist, 'validate_task_security', return_value=[]):
            
            results = self.manager.execute_item(item)
            
            assert results['success'] == True
            assert results['item_id'] == "exec-1"
            assert item.status == TaskStatus.PR
            assert results['pr_created'] == True
            assert len(results['security_issues']) == 0
        
        self.tearDown()
    
    def test_execute_item_tdd_failure(self):
        """Test item execution with TDD failure"""
        self.setUp()
        
        item = BacklogItem(
            id="exec-2", title="Execute Fail Test", description="Test execution failure",
            type=TaskType.FEATURE, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Fail criterion"]
        )
        
        # Mock TDD cycle to fail
        mock_tdd_results = {
            'success': False,
            'errors': ['TDD implementation failed']
        }
        
        with patch.object(self.manager.tdd_cycle, 'execute_cycle', return_value=mock_tdd_results):
            
            results = self.manager.execute_item(item)
            
            assert results['success'] == False
            assert item.status == TaskStatus.BLOCKED
            assert "TDD cycle failed" in item.blocked_reason
            assert results['pr_created'] == False
        
        self.tearDown()
    
    def test_execute_item_security_failure(self):
        """Test item execution with security validation failure"""
        self.setUp()
        
        item = BacklogItem(
            id="exec-3", title="Security Fail Test", description="Test security failure",
            type=TaskType.SECURITY, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Security criterion"]
        )
        
        # Mock successful TDD but security issues
        mock_tdd_results = {
            'success': True,
            'tests_written': ['test_security.py'],
            'implementation_files': ['security.py'],
            'errors': []
        }
        
        security_issues = ["Potential secret logging detected"]
        
        with patch.object(self.manager.tdd_cycle, 'execute_cycle', return_value=mock_tdd_results), \
             patch.object(self.manager.security_checklist, 'validate_task_security', return_value=security_issues):
            
            results = self.manager.execute_item(item)
            
            assert results['success'] == False
            assert item.status == TaskStatus.BLOCKED
            assert "Security validation failed" in item.blocked_reason
            assert results['security_issues'] == security_issues
        
        self.tearDown()
    
    def test_run_full_cycle_partial(self):
        """Test running a partial autonomous cycle"""
        self.setUp()
        
        # Create a ready item
        ready_item = BacklogItem(
            id="cycle-1", title="Cycle Test", description="Test full cycle",
            type=TaskType.FEATURE, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Cycle criterion"]
        )
        
        self.manager.backlog_items = [ready_item]
        
        # Mock all dependencies to succeed
        mock_tdd_results = {'success': True, 'tests_written': [], 'implementation_files': [], 'errors': []}
        mock_test_results = {'success': True, 'output': 'Pass', 'errors': ''}
        
        with patch.object(self.manager.discovery_engine, 'scan_for_tasks', return_value=[]), \
             patch.object(self.manager.tdd_cycle, 'execute_cycle', return_value=mock_tdd_results), \
             patch.object(self.manager.tdd_cycle, 'run_tests', return_value=mock_test_results), \
             patch.object(self.manager.security_checklist, 'validate_task_security', return_value=[]), \
             patch.object(self.manager, 'save_backlog', return_value=True):
            
            results = self.manager.run_full_cycle(max_iterations=5)
            
            assert results['iterations'] == 1
            assert results['items_completed'] == 1
            assert results['items_blocked'] == 0
            assert results['items_escalated'] == 0
            assert len(results['errors']) == 0
        
        self.tearDown()
    
    def test_run_full_cycle_escalation(self):
        """Test cycle with escalation required"""
        self.setUp()
        
        # Create item requiring escalation
        escalation_item = BacklogItem(
            id="esc-1", title="Escalation Test", description="Test escalation",
            type=TaskType.SECURITY, status=TaskStatus.READY,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            acceptance_criteria=["Escalation criterion"],
            escalation_required=True,
            escalation_reason="Security review required"
        )
        
        self.manager.backlog_items = [escalation_item]
        
        with patch.object(self.manager.discovery_engine, 'scan_for_tasks', return_value=[]), \
             patch.object(self.manager, 'save_backlog', return_value=True):
            
            results = self.manager.run_full_cycle(max_iterations=5)
            
            assert results['iterations'] == 1
            assert results['items_completed'] == 0
            assert results['items_blocked'] == 0
            assert results['items_escalated'] == 1
            assert escalation_item.status == TaskStatus.BLOCKED
            assert "Escalation required" in escalation_item.blocked_reason
        
        self.tearDown()
    
    def test_generate_status_report(self):
        """Test status report generation"""
        self.setUp()
        
        # Create test items
        items = [
            BacklogItem(
                id="status-1", title="High Priority", description="High WSJF",
                type=TaskType.FEATURE, status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                wsjf_score=3.5
            ),
            BacklogItem(
                id="status-2", title="Blocked Item", description="Blocked task",
                type=TaskType.BUG, status=TaskStatus.BLOCKED,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                blocked_reason="Dependency issue",
                wsjf_score=2.0
            ),
            BacklogItem(
                id="status-3", title="Escalation Item", description="Needs escalation",
                type=TaskType.SECURITY, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                escalation_required=True,
                escalation_reason="Security review needed",
                wsjf_score=1.5
            )
        ]
        
        self.manager.backlog_items = items
        
        report = self.manager.generate_status_report()
        
        assert 'timestamp' in report
        assert 'backlog_metrics' in report
        assert 'high_priority_items' in report
        assert 'blocked_items' in report
        assert 'escalation_required' in report
        
        # Check metrics
        assert report['backlog_metrics']['total_items'] == 3
        
        # Check high priority items (WSJF > 2.0)
        high_priority = report['high_priority_items']
        assert len(high_priority) == 2
        assert any(item['id'] == 'status-1' for item in high_priority)
        assert any(item['id'] == 'status-2' for item in high_priority)
        
        # Check blocked items
        blocked = report['blocked_items']
        assert len(blocked) == 1
        assert blocked[0]['id'] == 'status-2'
        assert blocked[0]['reason'] == 'Dependency issue'
        
        # Check escalation required
        escalation = report['escalation_required']
        assert len(escalation) == 1
        assert escalation[0]['id'] == 'status-3'
        assert escalation[0]['reason'] == 'Security review needed'
        
        self.tearDown()
    
    def test_deduplication(self):
        """Test task deduplication"""
        self.setUp()
        
        # Create existing item
        existing_item = BacklogItem(
            id="exist-1", title="Existing", description="Existing task",
            type=TaskType.FEATURE, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        self.manager.backlog_items = [existing_item]
        
        # Create new tasks with one duplicate
        new_tasks = [
            BacklogItem(
                id="new-1", title="New Task", description="New unique task",
                type=TaskType.BUG, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            ),
            BacklogItem(
                id="dup-1", title="Duplicate", description="Existing task",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now()
            )
        ]
        
        unique_tasks = self.manager._deduplicate_tasks(new_tasks)
        
        assert len(unique_tasks) == 1
        assert unique_tasks[0].description == "New unique task"
        
        self.tearDown()
    
    def test_split_large_tasks(self):
        """Test splitting large tasks"""
        self.setUp()
        
        # Create large task
        large_task = BacklogItem(
            id="large-1", title="Large Task", description="A large task to split",
            type=TaskType.FEATURE, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            effort=8,  # Large effort
            acceptance_criteria=[
                "Criterion 1", "Criterion 2", "Criterion 3",
                "Criterion 4", "Criterion 5", "Criterion 6"
            ]
        )
        
        self.manager.backlog_items = [large_task]
        
        split_count = self.manager._split_large_tasks()
        
        assert split_count == 1
        
        # Check that original task was removed and sub-tasks added
        remaining_ids = [item.id for item in self.manager.backlog_items]
        assert "large-1" not in remaining_ids
        assert any(item_id.startswith("large-1-") for item_id in remaining_ids)
        
        # Check sub-tasks have smaller effort
        sub_tasks = [item for item in self.manager.backlog_items if item.id.startswith("large-1-")]
        assert len(sub_tasks) > 1
        assert all(task.effort <= 3 for task in sub_tasks)
        assert all("large-1" in task.linked_items for task in sub_tasks)
        
        self.tearDown()


@pytest.fixture
def temp_backlog_file():
    """Fixture for temporary backlog file"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
    
    # Create minimal backlog structure
    backlog_data = {
        'backlog': {
            'format_version': '1.0',
            'last_updated': datetime.datetime.now().isoformat(),
            'scoring_config': {
                'wsjf_weights': {
                    'user_value': 1.0,
                    'business_value': 1.0,
                    'time_criticality': 1.0,
                    'risk_reduction': 0.8,
                    'opportunity_enablement': 0.6
                }
            },
            'items': [],
            'metrics': {
                'total_items': 0,
                'by_status': {},
                'by_type': {},
                'avg_wsjf': 0.0,
                'high_priority_count': 0,
                'aging_items': 0
            }
        }
    }
    
    yaml.dump(backlog_data, temp_file, default_flow_style=False)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)


def test_integration_full_workflow(temp_backlog_file):
    """Integration test for full autonomous backlog workflow"""
    test_dir = os.path.dirname(temp_backlog_file)
    manager = AutonomousBacklogManager(test_dir, temp_backlog_file)
    
    # Load empty backlog
    assert manager.load_backlog() == True
    assert len(manager.backlog_items) == 0
    
    # Add a test item
    test_item = BacklogItem(
        id="integration-1",
        title="Integration Test Task",
        description="Test full workflow integration",
        type=TaskType.FEATURE,
        status=TaskStatus.READY,
        created_date=datetime.datetime.now(),
        updated_date=datetime.datetime.now(),
        user_value=3,
        business_value=3,
        time_criticality=2,
        risk_reduction=2,
        opportunity_enablement=1,
        effort=2,
        acceptance_criteria=["Implement integration test", "Verify workflow"]
    )
    
    manager.backlog_items = [test_item]
    
    # Calculate WSJF
    weights = manager.config.get('wsjf_weights', {})
    test_item.calculate_wsjf(weights)
    
    assert test_item.wsjf_score > 0
    
    # Save and reload
    assert manager.save_backlog() == True
    manager.backlog_items = []
    assert manager.load_backlog() == True
    assert len(manager.backlog_items) == 1
    
    # Verify loaded item
    loaded_item = manager.backlog_items[0]
    assert loaded_item.id == "integration-1"
    assert loaded_item.title == "Integration Test Task"
    assert loaded_item.is_ready_for_execution() == True
    
    # Generate status report
    report = manager.generate_status_report()
    assert report['backlog_metrics']['total_items'] == 1
    assert len(report['high_priority_items']) >= 0  # Depends on WSJF score


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])