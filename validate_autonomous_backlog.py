#!/usr/bin/env python3
"""
Validation script for the Autonomous Backlog Management System
"""

import os
import sys
import datetime
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from autogen_code_review_bot.autonomous_backlog import (
        BacklogItem, TaskType, TaskStatus, BacklogMetrics,
        DiscoveryEngine, SecurityChecklist, TDDMicroCycle,
        AutonomousBacklogManager
    )
    print("✅ Successfully imported autonomous backlog components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_backlog_item():
    """Test BacklogItem functionality"""
    print("\n🧪 Testing BacklogItem...")
    
    try:
        # Create a backlog item
        item = BacklogItem(
            id="test-1",
            title="Test Task",
            description="A test task for validation",
            type=TaskType.FEATURE,
            status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now(),
            user_value=5,
            business_value=4,
            time_criticality=3,
            risk_reduction=3,
            opportunity_enablement=2,
            effort=2,
            acceptance_criteria=["Implement feature", "Add tests", "Update docs"]
        )
        
        # Test WSJF calculation
        weights = {
            'user_value': 1.0,
            'business_value': 1.0,
            'time_criticality': 1.0,
            'risk_reduction': 0.8,
            'opportunity_enablement': 0.6
        }
        
        score = item.calculate_wsjf(weights)
        expected = (5 + 4 + 3 + 3*0.8 + 2*0.6) / 2  # 15.6 / 2 = 7.8
        
        assert abs(score - expected) < 0.01, f"WSJF calculation failed: {score} != {expected}"
        print(f"  ✅ WSJF calculation: {score:.2f}")
        
        # Test aging
        item.update_aging(aging_threshold_days=30)
        print(f"  ✅ Aging multiplier: {item.aging_multiplier}")
        
        # Test readiness
        assert item.is_ready_for_execution() == False, "Item should not be ready (status is NEW)"
        item.status = TaskStatus.READY
        assert item.is_ready_for_execution() == True, "Item should be ready now"
        print("  ✅ Readiness check passed")
        
        # Test dictionary conversion
        item_dict = item.to_dict()
        assert item_dict['id'] == "test-1"
        assert item_dict['type'] == "Feature"
        assert item_dict['status'] == "READY"
        print("  ✅ Dictionary conversion passed")
        
        print("✅ BacklogItem tests passed")
        return True
        
    except Exception as e:
        print(f"❌ BacklogItem test failed: {e}")
        return False


def test_backlog_metrics():
    """Test BacklogMetrics functionality"""
    print("\n🧪 Testing BacklogMetrics...")
    
    try:
        # Create test items
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
        
        print("  ✅ Metrics calculation passed")
        print(f"    Total items: {metrics.total_items}")
        print(f"    Average WSJF: {metrics.avg_wsjf:.2f}")
        print(f"    High priority: {metrics.high_priority_count}")
        
        print("✅ BacklogMetrics tests passed")
        return True
        
    except Exception as e:
        print(f"❌ BacklogMetrics test failed: {e}")
        return False


def test_security_checklist():
    """Test SecurityChecklist functionality"""
    print("\n🧪 Testing SecurityChecklist...")
    
    try:
        # Test input validation check
        code_with_input = """
        def process_user_input(user_data):
            return user_data.upper()
        """
        
        item = BacklogItem(
            id="sec-1", title="Security Test", description="Test security",
            type=TaskType.FEATURE, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        issues = SecurityChecklist.validate_task_security(item, code_with_input)
        assert len(issues) > 0, "Should detect input validation issue"
        assert any("input handling without validation" in issue for issue in issues)
        print("  ✅ Input validation check passed")
        
        # Test secrets logging check
        code_with_secret_logging = """
        def debug_auth(password):
            print(f"Password: {password}")
            return True
        """
        
        issues = SecurityChecklist.validate_task_security(item, code_with_secret_logging)
        assert len(issues) > 0, "Should detect secret logging"
        assert any("secret logging detected" in issue for issue in issues)
        print("  ✅ Secret logging check passed")
        
        # Test coverage validation
        security_item = BacklogItem(
            id="sec-2", title="Security Task", description="Security",
            type=TaskType.SECURITY, status=TaskStatus.NEW,
            created_date=datetime.datetime.now(),
            updated_date=datetime.datetime.now()
        )
        
        assert not SecurityChecklist.validate_test_coverage(security_item, [])
        assert SecurityChecklist.validate_test_coverage(security_item, ["test_security.py"])
        print("  ✅ Test coverage validation passed")
        
        print("✅ SecurityChecklist tests passed")
        return True
        
    except Exception as e:
        print(f"❌ SecurityChecklist test failed: {e}")
        return False


def test_discovery_engine():
    """Test DiscoveryEngine functionality"""
    print("\n🧪 Testing DiscoveryEngine...")
    
    try:
        # Create temporary test directory
        test_dir = tempfile.mkdtemp()
        
        try:
            engine = DiscoveryEngine(test_dir)
            
            # Test comment task creation
            task = engine._create_comment_task(
                "test.py", "42", "TODO: Refactor this function"
            )
            
            assert task.title == "Address TODO in test.py"
            assert task.description == "TODO: Refactor this function"
            assert task.type == TaskType.REFACTOR
            assert task.files == ["test.py"]
            assert task.lines_of_interest == ["42"]
            assert len(task.acceptance_criteria) >= 2
            print("  ✅ Comment task creation passed")
            
            # Test security task creation
            issue = {
                "test_name": "sql_injection",
                "issue_text": "Potential SQL injection vulnerability",
                "issue_severity": "HIGH",
                "filename": "database.py",
                "line_number": 100
            }
            
            security_task = engine._create_security_task(issue)
            assert security_task.title == "Security: sql_injection"
            assert security_task.type == TaskType.SECURITY
            assert security_task.files == ["database.py"]
            assert security_task.escalation_required == True
            print("  ✅ Security task creation passed")
            
            print("✅ DiscoveryEngine tests passed")
            return True
            
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ DiscoveryEngine test failed: {e}")
        return False


def test_autonomous_backlog_manager():
    """Test AutonomousBacklogManager functionality"""
    print("\n🧪 Testing AutonomousBacklogManager...")
    
    try:
        # Create temporary test environment
        test_dir = tempfile.mkdtemp()
        backlog_file = os.path.join(test_dir, "test_backlog.yml")
        
        try:
            manager = AutonomousBacklogManager(test_dir, backlog_file)
            
            # Test manager initialization
            assert manager.repo_path == test_dir
            assert manager.backlog_file == backlog_file
            assert isinstance(manager.discovery_engine, DiscoveryEngine)
            assert isinstance(manager.security_checklist, SecurityChecklist)
            assert isinstance(manager.tdd_cycle, TDDMicroCycle)
            assert isinstance(manager.config, dict)
            print("  ✅ Manager initialization passed")
            
            # Test adding and saving items
            test_item = BacklogItem(
                id="mgr-1", title="Manager Test", description="Test manager",
                type=TaskType.FEATURE, status=TaskStatus.NEW,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=3, business_value=2, effort=1,
                acceptance_criteria=["Test criterion"]
            )
            
            manager.backlog_items = [test_item]
            
            # Test save
            save_success = manager.save_backlog()
            assert save_success == True
            assert os.path.exists(backlog_file)
            print("  ✅ Backlog save passed")
            
            # Test load
            manager.backlog_items = []
            load_success = manager.load_backlog()
            assert load_success == True
            assert len(manager.backlog_items) == 1
            
            loaded_item = manager.backlog_items[0]
            assert loaded_item.id == "mgr-1"
            assert loaded_item.title == "Manager Test"
            print("  ✅ Backlog load passed")
            
            # Test status report
            report = manager.generate_status_report()
            assert 'timestamp' in report
            assert 'backlog_metrics' in report
            assert report['backlog_metrics']['total_items'] == 1
            print("  ✅ Status report generation passed")
            
            # Test deduplication
            duplicate_items = [
                BacklogItem(
                    id="dup-1", title="New", description="New task",
                    type=TaskType.BUG, status=TaskStatus.NEW,
                    created_date=datetime.datetime.now(),
                    updated_date=datetime.datetime.now()
                ),
                BacklogItem(
                    id="dup-2", title="Duplicate", description="Test manager",  # Same as existing
                    type=TaskType.FEATURE, status=TaskStatus.NEW,
                    created_date=datetime.datetime.now(),
                    updated_date=datetime.datetime.now()
                )
            ]
            
            unique_items = manager._deduplicate_tasks(duplicate_items)
            assert len(unique_items) == 1
            assert unique_items[0].description == "New task"
            print("  ✅ Deduplication passed")
            
            print("✅ AutonomousBacklogManager tests passed")
            return True
            
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ AutonomousBacklogManager test failed: {e}")
        return False


def test_yaml_backlog_format():
    """Test YAML backlog file format"""
    print("\n🧪 Testing YAML backlog format...")
    
    try:
        backlog_file = "DOCS/backlog.yml"
        
        if not os.path.exists(backlog_file):
            print("  ⚠️  Backlog file not found, skipping YAML format test")
            return True
        
        # Try to read the backlog file we created
        try:
            import yaml
            with open(backlog_file, 'r') as f:
                data = yaml.safe_load(f)
            
            # Validate structure
            assert 'backlog' in data
            backlog = data['backlog']
            
            assert 'format_version' in backlog
            assert 'scoring_config' in backlog
            assert 'items' in backlog
            assert 'metrics' in backlog
            
            print(f"  ✅ YAML structure valid")
            print(f"    Format version: {backlog['format_version']}")
            print(f"    Items count: {len(backlog['items'])}")
            
            # Validate scoring config
            scoring_config = backlog['scoring_config']
            assert 'wsjf_weights' in scoring_config
            assert 'effort_scale' in scoring_config
            assert 'impact_scale' in scoring_config
            
            print("  ✅ Scoring configuration valid")
            
        except ImportError:
            print("  ⚠️  PyYAML not available, skipping YAML validation")
            return True
        
        print("✅ YAML backlog format tests passed")
        return True
        
    except Exception as e:
        print(f"❌ YAML format test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🚀 Autonomous Backlog Management System Validation")
    print("=" * 60)
    
    tests = [
        test_backlog_item,
        test_backlog_metrics,
        test_security_checklist,
        test_discovery_engine,
        test_autonomous_backlog_manager,
        test_yaml_backlog_format
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The autonomous backlog system is ready.")
        
        # Show example usage
        print("\n📋 Example Usage:")
        print("  python3 validate_autonomous_backlog.py")
        print("  python3 -m src.autogen_code_review_bot.autonomous_backlog --help")
        print("  python3 -m src.autogen_code_review_bot.autonomous_backlog --status-only")
        print("  python3 -m src.autogen_code_review_bot.autonomous_backlog --discover-only")
        print("  python3 -m src.autogen_code_review_bot.autonomous_backlog --dry-run")
        
        return 0
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())