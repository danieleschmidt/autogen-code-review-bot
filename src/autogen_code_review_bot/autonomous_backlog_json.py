"""
Autonomous Backlog Management System - JSON Version

A lightweight version using only Python standard library modules.
Implements WSJF-based prioritization and continuous backlog processing.
"""

import os
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
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['status'] = self.status.value
        data['created_date'] = self.created_date.isoformat()
        data['updated_date'] = self.updated_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacklogItem':
        """Create BacklogItem from dictionary"""
        # Handle datetime conversion
        created_date = data['created_date']
        if isinstance(created_date, str):
            created_date = datetime.datetime.fromisoformat(created_date)
        
        updated_date = data['updated_date']
        if isinstance(updated_date, str):
            updated_date = datetime.datetime.fromisoformat(updated_date)
        
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            type=TaskType(data['type']),
            status=TaskStatus(data['status']),
            created_date=created_date,
            updated_date=updated_date,
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


class AutonomousBacklogManagerJSON:
    """JSON-based autonomous backlog management system"""
    
    def __init__(self, repo_path: str, backlog_file: str = "DOCS/backlog.json"):
        self.repo_path = repo_path
        self.backlog_file = os.path.join(repo_path, backlog_file)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_default_config()
        
        # Initialize backlog
        self.backlog_items: List[BacklogItem] = []
        self.metrics = BacklogMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
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
        """Load backlog from JSON file"""
        try:
            if not os.path.exists(self.backlog_file):
                self.logger.info("No existing backlog file, starting fresh")
                return True
            
            with open(self.backlog_file, 'r') as f:
                data = json.load(f)
            
            backlog_data = data.get('backlog', {})
            items_data = backlog_data.get('items', [])
            
            self.backlog_items = []
            for item_data in items_data:
                try:
                    item = BacklogItem.from_dict(item_data)
                    self.backlog_items.append(item)
                except Exception as e:
                    self.logger.error(f"Failed to load item {item_data.get('id', 'unknown')}: {e}")
            
            # Load config if present
            if 'scoring_config' in backlog_data:
                self.config.update(backlog_data['scoring_config'])
            
            self.logger.info(f"Loaded {len(self.backlog_items)} backlog items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load backlog: {e}")
            return False
    
    def save_backlog(self) -> bool:
        """Save backlog to JSON file"""
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
                            'todos_found': 0,
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
                    json.dump(backlog_data, f, indent=2, default=str)
                
                self.logger.info(f"Saved {len(self.backlog_items)} items to backlog")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save backlog: {e}")
            return False
    
    def add_item(self, item: BacklogItem) -> bool:
        """Add item to backlog"""
        try:
            with self._lock:
                # Calculate WSJF score
                weights = self.config.get('wsjf_weights', {})
                aging_cap = self.config.get('aging_multiplier_cap', 2.0)
                item.calculate_wsjf(weights, aging_cap)
                
                # Add to backlog
                self.backlog_items.append(item)
                
                # Sort by WSJF score
                self.backlog_items.sort(key=lambda x: x.wsjf_score, reverse=True)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add item {item.id}: {e}")
            return False
    
    def update_item_status(self, item_id: str, new_status: TaskStatus, 
                          blocked_reason: str = "") -> bool:
        """Update item status"""
        try:
            with self._lock:
                for item in self.backlog_items:
                    if item.id == item_id:
                        item.status = new_status
                        item.updated_date = datetime.datetime.now()
                        if blocked_reason:
                            item.blocked_reason = blocked_reason
                        return True
                
                self.logger.warning(f"Item {item_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update item {item_id}: {e}")
            return False
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the next ready item for execution"""
        with self._lock:
            for item in self.backlog_items:
                if item.is_ready_for_execution():
                    return item
            return None
    
    def get_high_priority_items(self, threshold: float = 2.0) -> List[BacklogItem]:
        """Get high priority items above WSJF threshold"""
        with self._lock:
            return [item for item in self.backlog_items if item.wsjf_score > threshold]
    
    def get_blocked_items(self) -> List[BacklogItem]:
        """Get blocked items"""
        with self._lock:
            return [item for item in self.backlog_items if item.status == TaskStatus.BLOCKED]
    
    def get_escalation_items(self) -> List[BacklogItem]:
        """Get items requiring escalation"""
        with self._lock:
            return [item for item in self.backlog_items if item.escalation_required]
    
    def refresh_scores(self) -> int:
        """Refresh WSJF scores for all items"""
        try:
            with self._lock:
                weights = self.config.get('wsjf_weights', {})
                aging_cap = self.config.get('aging_multiplier_cap', 2.0)
                aging_threshold = self.config.get('aging_days_threshold', 30)
                
                updated_count = 0
                for item in self.backlog_items:
                    item.update_aging(aging_threshold)
                    item.calculate_wsjf(weights, aging_cap)
                    updated_count += 1
                
                # Re-sort by WSJF score
                self.backlog_items.sort(key=lambda x: x.wsjf_score, reverse=True)
                
                return updated_count
                
        except Exception as e:
            self.logger.error(f"Failed to refresh scores: {e}")
            return 0
    
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
    
    def create_sample_backlog(self) -> None:
        """Create a sample backlog for demonstration"""
        sample_items = [
            BacklogItem(
                id="sample-001",
                title="Refactor Code Duplication in PR Analysis",
                description="Extract common error result creation into utility functions, consolidate sequential/parallel style check logic",
                type=TaskType.REFACTOR,
                status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=2,
                business_value=2,
                time_criticality=1,
                risk_reduction=3,
                opportunity_enablement=2,
                effort=1,
                acceptance_criteria=[
                    "Extract common error result creation patterns into utility functions",
                    "Consolidate sequential/parallel style check logic",
                    "Maintain existing test coverage",
                    "No functional behavior changes",
                    "Code duplication reduced by >50% in target areas"
                ],
                files=["src/autogen_code_review_bot/pr_analysis.py"],
                lines_of_interest=["436-492", "316-831"],
                security_notes="Low risk - isolated refactoring only",
                test_notes="Existing tests should pass without modification"
            ),
            BacklogItem(
                id="sample-002",
                title="Add File Count Limits in Language Detection",
                description="Implement early exit conditions and file count limits in _detect_repo_languages to prevent excessive memory usage",
                type=TaskType.FEATURE,
                status=TaskStatus.READY,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=5,
                business_value=3,
                time_criticality=3,
                risk_reduction=5,
                opportunity_enablement=2,
                effort=1,
                acceptance_criteria=[
                    "Add configurable file count limit (default: 10000 files)",
                    "Implement early exit when limit exceeded",
                    "Log when limits are triggered",
                    "Graceful degradation with partial analysis",
                    "Memory usage remains bounded on large repositories"
                ],
                files=["src/autogen_code_review_bot/pr_analysis.py"],
                lines_of_interest=["80-101"],
                security_notes="Prevents resource exhaustion attacks",
                test_notes="Add tests for large repository scenarios"
            ),
            BacklogItem(
                id="sample-003",
                title="Improve Agent Implementation with LLM Integration",
                description="Replace placeholder review methods with actual LLM calls for enhanced code analysis",
                type=TaskType.FEATURE,
                status=TaskStatus.BLOCKED,
                created_date=datetime.datetime.now(),
                updated_date=datetime.datetime.now(),
                user_value=8,
                business_value=5,
                time_criticality=3,
                risk_reduction=2,
                opportunity_enablement=8,
                effort=3,
                blocked_reason="Requires human review due to LLM API integration and security implications",
                escalation_required=True,
                escalation_reason="LLM API integration requires security review and architectural decisions",
                acceptance_criteria=[
                    "Integrate with LLM API (OpenAI, Claude, etc.)",
                    "Replace placeholder methods with real analysis",
                    "Add configuration for LLM settings",
                    "Implement rate limiting and error handling",
                    "Add comprehensive testing with mocked LLM responses"
                ],
                files=["src/autogen_code_review_bot/agents.py"],
                security_notes="HIGH RISK - Requires API key management, input sanitization, rate limiting",
                test_notes="Mock LLM responses for deterministic testing"
            )
        ]
        
        for item in sample_items:
            self.add_item(item)
        
        self.logger.info(f"Created sample backlog with {len(sample_items)} items")


# CLI interface for the autonomous backlog system
def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Backlog Management System (JSON Version)")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--backlog-file", default="DOCS/backlog.json", help="Backlog file path")
    parser.add_argument("--status-only", action="store_true", help="Show status report only")
    parser.add_argument("--create-sample", action="store_true", help="Create sample backlog")
    parser.add_argument("--refresh-scores", action="store_true", help="Refresh WSJF scores")
    parser.add_argument("--high-priority", action="store_true", help="Show high priority items")
    parser.add_argument("--blocked", action="store_true", help="Show blocked items")
    parser.add_argument("--escalation", action="store_true", help="Show items requiring escalation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize manager
    manager = AutonomousBacklogManagerJSON(args.repo_path, args.backlog_file)
    
    if args.create_sample:
        # Create sample backlog
        manager.create_sample_backlog()
        manager.save_backlog()
        print("✅ Sample backlog created")
        return
    
    # Load existing backlog
    if not manager.load_backlog():
        print("❌ Failed to load backlog")
        return
    
    if args.refresh_scores:
        # Refresh scores
        updated = manager.refresh_scores()
        manager.save_backlog()
        print(f"✅ Refreshed {updated} item scores")
        return
    
    if args.high_priority:
        # Show high priority items
        items = manager.get_high_priority_items()
        print(f"📈 High Priority Items (WSJF > 2.0): {len(items)}")
        for item in items:
            print(f"  - {item.id}: {item.title} (WSJF: {item.wsjf_score:.1f})")
        return
    
    if args.blocked:
        # Show blocked items
        items = manager.get_blocked_items()
        print(f"🚫 Blocked Items: {len(items)}")
        for item in items:
            print(f"  - {item.id}: {item.title}")
            print(f"    Reason: {item.blocked_reason}")
        return
    
    if args.escalation:
        # Show escalation items
        items = manager.get_escalation_items()
        print(f"⚠️  Items Requiring Escalation: {len(items)}")
        for item in items:
            print(f"  - {item.id}: {item.title}")
            print(f"    Reason: {item.escalation_reason}")
        return
    
    if args.status_only:
        # Generate status report
        report = manager.generate_status_report()
        print(json.dumps(report, indent=2, default=str))
        return
    
    # Default: show summary
    report = manager.generate_status_report()
    metrics = report['backlog_metrics']
    
    print("📋 Autonomous Backlog Status Summary")
    print("=" * 40)
    print(f"Total Items: {metrics['total_items']}")
    print(f"High Priority (WSJF > 2.0): {metrics['high_priority_count']}")
    print(f"Average WSJF: {metrics['avg_wsjf']:.2f}")
    print(f"Aging Items: {metrics['aging_items']}")
    
    print("\nBy Status:")
    for status, count in metrics['by_status'].items():
        if count > 0:
            print(f"  {status}: {count}")
    
    print("\nBy Type:")
    for task_type, count in metrics['by_type'].items():
        if count > 0:
            print(f"  {task_type}: {count}")
    
    # Show next ready item
    next_item = manager.get_next_ready_item()
    if next_item:
        print(f"\n🎯 Next Ready Item:")
        print(f"  ID: {next_item.id}")
        print(f"  Title: {next_item.title}")
        print(f"  WSJF: {next_item.wsjf_score:.1f}")
        print(f"  Type: {next_item.type.value}")
    else:
        print(f"\n⏸️  No ready items found")
    
    # Show blocked and escalation counts
    blocked_count = len(report['blocked_items'])
    escalation_count = len(report['escalation_required'])
    
    if blocked_count > 0:
        print(f"\n🚫 Blocked Items: {blocked_count}")
    
    if escalation_count > 0:
        print(f"⚠️  Escalation Required: {escalation_count}")


if __name__ == "__main__":
    main()