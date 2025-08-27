#!/usr/bin/env python3
"""
Comprehensive Autonomous SDLC Test Suite

Complete integration test for the enhanced autonomous SDLC system including:
- Progressive quality gates
- Research-driven development
- Breakthrough monitoring
- End-to-end workflow validation
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.autogen_code_review_bot.enhanced_progressive_quality_gates import (
    get_enhanced_quality_gates,
    QualityGateType,
    QualityGateStatus
)
from src.autogen_code_review_bot.research_driven_sdlc import (
    get_research_engine,
    ExperimentType,
    ResearchPhase,
    ValidationLevel
)
from src.autogen_code_review_bot.breakthrough_monitoring_engine import (
    get_monitoring_engine,
    MetricType,
    AlertSeverity
)


class TestCompleteAutonomousSDLC:
    """Comprehensive test suite for the complete autonomous SDLC system"""
    
    @pytest.fixture
    async def temp_repo(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Create mock project structure
        (repo_path / "src").mkdir()
        (repo_path / "tests").mkdir()
        (repo_path / "docs").mkdir()
        
        # Create mock Python files
        (repo_path / "src" / "__init__.py").write_text("")
        (repo_path / "src" / "main.py").write_text("""
def calculate_fibonacci(n):
    \"\"\"Calculate fibonacci number\"\"\"
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
""")
        
        # Create mock test files
        (repo_path / "tests" / "__init__.py").write_text("")
        (repo_path / "tests" / "test_main.py").write_text("""
import pytest
from src.main import calculate_fibonacci

def test_fibonacci():
    assert calculate_fibonacci(0) == 0
    assert calculate_fibonacci(1) == 1
    assert calculate_fibonacci(5) == 5
    assert calculate_fibonacci(10) == 55

def test_fibonacci_edge_cases():
    assert calculate_fibonacci(2) == 1
    assert calculate_fibonacci(3) == 2
""")
        
        # Create configuration files
        (repo_path / "README.md").write_text("""
# Test Project

A test project for autonomous SDLC validation.

## Installation

pip install -e .

## Usage

python src/main.py

## Features

- Fibonacci calculation
- Comprehensive testing
- Documentation
""")
        
        (repo_path / "LICENSE").write_text("MIT License")
        (repo_path / "CONTRIBUTING.md").write_text("# Contributing Guidelines")
        (repo_path / "SECURITY.md").write_text("# Security Policy")
        (repo_path / "CHANGELOG.md").write_text("# Changelog")
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_progressive_quality_gates_simple(self, temp_repo):
        """Test simple generation quality gates"""
        
        quality_gates = get_enhanced_quality_gates(str(temp_repo))
        
        # Execute simple generation quality gates
        results = await quality_gates.execute_progressive_quality_gates(
            generation_level="simple",
            research_mode=False,
            parallel_execution=True
        )
        
        # Validate results
        assert results.suite_name == "Progressive Quality Gates - Simple"
        assert results.generation_level == "simple"
        assert results.total_gates > 0
        assert results.overall_score >= 0.0
        assert results.execution_time > 0.0
        assert isinstance(results.results, list)
        assert len(results.results) == results.total_gates
        
        # Check that basic gates are present
        gate_types = {result.gate_type for result in results.results}
        expected_gates = {
            QualityGateType.CODE_QUALITY,
            QualityGateType.SIMPLE_VALIDATION,
            QualityGateType.TEST_COVERAGE,
            QualityGateType.DOCUMENTATION_QUALITY
        }
        
        assert expected_gates.issubset(gate_types)
        
        # Validate individual gate results
        for result in results.results:
            assert result.gate_name is not None
            assert result.gate_type is not None
            assert result.status in QualityGateStatus
            assert result.execution_time >= 0.0
            assert result.timestamp is not None
            assert isinstance(result.breakthrough_metrics, dict)
        
        print(f"✅ Simple Quality Gates: {results.passed_gates}/{results.total_gates} passed")
        print(f"   Overall Score: {results.overall_score:.1f}")
        print(f"   Next Generation Ready: {results.next_generation_ready}")
    
    @pytest.mark.asyncio
    async def test_progressive_quality_gates_robust(self, temp_repo):
        """Test robust generation quality gates"""
        
        quality_gates = get_enhanced_quality_gates(str(temp_repo))
        
        # Execute robust generation quality gates
        results = await quality_gates.execute_progressive_quality_gates(
            generation_level="robust",
            research_mode=False,
            parallel_execution=True
        )
        
        # Validate results
        assert results.generation_level == "robust"
        assert results.total_gates > 0
        
        # Check that robust gates are present
        gate_types = {result.gate_type for result in results.results}
        expected_robust_gates = {
            QualityGateType.SECURITY_SCAN,
            QualityGateType.ROBUST_VALIDATION,
            QualityGateType.COMPLIANCE_CHECK
        }
        
        # At least some robust gates should be present
        assert len(expected_robust_gates.intersection(gate_types)) > 0
        
        print(f"✅ Robust Quality Gates: {results.passed_gates}/{results.total_gates} passed")
        print(f"   Overall Score: {results.overall_score:.1f}")
        print(f"   Compliance Level: {results.compliance_level}")
    
    @pytest.mark.asyncio
    async def test_progressive_quality_gates_optimized(self, temp_repo):
        """Test optimized generation quality gates with research mode"""
        
        quality_gates = get_enhanced_quality_gates(str(temp_repo))
        
        # Execute optimized generation quality gates with research mode
        results = await quality_gates.execute_progressive_quality_gates(
            generation_level="optimized",
            research_mode=True,
            parallel_execution=True
        )
        
        # Validate results
        assert results.generation_level == "optimized"
        assert results.total_gates > 0
        assert results.research_metrics is not None
        
        # Check that optimized and research gates are present
        gate_types = {result.gate_type for result in results.results}
        expected_optimized_gates = {
            QualityGateType.PERFORMANCE_BENCHMARK,
            QualityGateType.DEPENDENCY_SECURITY,
            QualityGateType.OPTIMIZED_VALIDATION
        }
        
        expected_research_gates = {
            QualityGateType.REPRODUCIBILITY_CHECK,
            QualityGateType.STATISTICAL_SIGNIFICANCE,
            QualityGateType.BASELINE_COMPARISON
        }
        
        # Check for optimized gates
        assert len(expected_optimized_gates.intersection(gate_types)) > 0
        
        # Check for research gates
        assert len(expected_research_gates.intersection(gate_types)) > 0
        
        # Validate research metrics
        assert isinstance(results.research_metrics, dict)
        assert len(results.research_metrics) > 0
        
        print(f"✅ Optimized Quality Gates: {results.passed_gates}/{results.total_gates} passed")
        print(f"   Overall Score: {results.overall_score:.1f}")
        print(f"   Research Metrics: {len(results.research_metrics)} indicators")
    
    @pytest.mark.asyncio
    async def test_research_driven_development(self, temp_repo):
        """Test research-driven development capabilities"""
        
        research_engine = get_research_engine(str(temp_repo))
        
        # Test research session initiation
        session = await research_engine.initiate_research_session(
            title="Fibonacci Optimization Research",
            research_questions=[
                "Can memoization improve fibonacci calculation performance?",
                "What is the memory overhead of the optimized implementation?",
                "Does the optimization maintain calculation accuracy?"
            ],
            baseline_metrics={
                "calculation_time": 1000.0,  # ms
                "memory_usage": 10.0,  # MB
                "accuracy": 1.0
            }
        )
        
        # Validate session creation
        assert session.session_id is not None
        assert session.title == "Fibonacci Optimization Research"
        assert session.current_phase == ResearchPhase.HYPOTHESIS_FORMATION
        assert len(session.hypotheses) == 3
        
        # Test literature review
        literature_review = await research_engine.conduct_literature_review(
            session.session_id,
            search_terms=["fibonacci", "memoization", "performance"],
            focus_areas=["algorithmic optimization", "memory usage"]
        )
        
        assert session.current_phase == ResearchPhase.LITERATURE_REVIEW
        assert "relevant_papers" in literature_review
        assert "research_gaps" in literature_review
        assert isinstance(literature_review["baseline_comparisons"], dict)
        
        # Test experimental design
        experimental_designs = await research_engine.design_experiments(
            session.session_id,
            experiment_types=[
                ExperimentType.PERFORMANCE_COMPARISON,
                ExperimentType.RESOURCE_UTILIZATION
            ]
        )
        
        assert session.current_phase == ResearchPhase.EXPERIMENTAL_DESIGN
        assert len(experimental_designs) > 0
        
        # Validate experimental design structure
        for design in experimental_designs:
            assert "experiment_id" in design
            assert "experiment_type" in design
            assert "hypothesis_id" in design
            assert "sample_size" in design
            assert "statistical_tests" in design
        
        # Test experiment execution
        first_design = experimental_designs[0]
        experiment_result = await research_engine.execute_experiment(
            session.session_id,
            first_design,
            implementation_code="# Memoized fibonacci implementation"
        )
        
        assert session.current_phase == ResearchPhase.VALIDATION
        assert experiment_result.experiment_id == first_design["experiment_id"]
        assert experiment_result.p_value >= 0.0
        assert experiment_result.effect_size >= 0.0
        assert experiment_result.validation_level in ValidationLevel
        
        # Test reproducibility validation
        reproducibility_analysis = await research_engine.validate_reproducibility(
            session.session_id,
            experiment_result.experiment_id,
            num_replications=3
        )
        
        assert "overall_reproducibility" in reproducibility_analysis
        assert 0.0 <= reproducibility_analysis["overall_reproducibility"] <= 1.0
        assert reproducibility_analysis["num_replications"] == 3
        
        # Test breakthrough analysis
        breakthrough_analysis = await research_engine.analyze_breakthrough_potential(
            session.session_id
        )
        
        assert session.current_phase == ResearchPhase.OPTIMIZATION
        assert "innovation_score" in breakthrough_analysis
        assert "breakthrough_indicators" in breakthrough_analysis
        assert "publication_potential" in breakthrough_analysis
        assert 0.0 <= breakthrough_analysis["innovation_score"] <= 1.0
        
        # Test publication preparation
        publication_prep = await research_engine.prepare_for_publication(
            session.session_id,
            target_venue="conference"
        )
        
        assert session.current_phase == ResearchPhase.PUBLICATION_PREPARATION
        assert "publication_package" in publication_prep
        assert "readiness_score" in publication_prep
        assert "recommended_venue" in publication_prep
        assert 0.0 <= publication_prep["readiness_score"] <= 1.0
        
        # Test research report generation
        research_report = await research_engine.generate_research_report(session.session_id)
        
        assert "session_summary" in research_report
        assert "key_findings" in research_report
        assert "statistical_summary" in research_report
        assert "breakthrough_assessment" in research_report
        assert "publication_readiness" in research_report
        
        print(f"✅ Research Session: {session.session_id}")
        print(f"   Hypotheses: {len(session.hypotheses)}")
        print(f"   Experiments: {len(session.experiments)}")
        print(f"   Innovation Score: {breakthrough_analysis['innovation_score']:.3f}")
        print(f"   Publication Readiness: {publication_prep['readiness_score']:.3f}")
    
    @pytest.mark.asyncio
    async def test_breakthrough_monitoring(self, temp_repo):
        """Test breakthrough monitoring system"""
        
        monitoring_engine = get_monitoring_engine(str(temp_repo))
        
        # Test alert callback
        received_alerts = []
        
        def test_alert_handler(alert):
            received_alerts.append(alert)
        
        monitoring_engine.add_alert_callback(test_alert_handler)
        
        # Start monitoring in background
        monitoring_task = asyncio.create_task(monitoring_engine.start_monitoring())
        
        # Wait for monitoring to initialize
        await asyncio.sleep(0.5)
        
        # Test metric recording
        monitoring_engine.record_metric("test.performance", 95.0)
        monitoring_engine.record_metric("test.throughput", 1500.0)
        monitoring_engine.record_metric("test.error_rate", 0.01)
        
        # Test breakthrough metric (should trigger breakthrough alert)
        monitoring_engine.record_metric("application.response_time", 50.0)  # Breakthrough threshold
        monitoring_engine.record_metric("application.throughput", 1200.0)  # Breakthrough threshold
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Test metric retrieval
        performance_history = monitoring_engine.get_metric_history("test.performance")
        assert len(performance_history) > 0
        
        latest_point = performance_history[-1]
        assert latest_point.name == "test.performance"
        assert latest_point.value == 95.0
        assert latest_point.timestamp is not None
        
        # Test system status
        status = monitoring_engine.get_system_status()
        assert "monitoring_status" in status
        assert "uptime_seconds" in status
        assert "active_alerts" in status
        assert "registered_metrics" in status
        assert status["uptime_seconds"] > 0
        
        # Test breakthrough metrics
        breakthrough_metrics = monitoring_engine.get_breakthrough_metrics()
        assert isinstance(breakthrough_metrics, dict)
        
        # Test monitoring report
        report = monitoring_engine.generate_monitoring_report()
        assert "report_generated" in report
        assert "system_status" in report
        assert "metric_summary" in report
        assert "breakthrough_summary" in report
        
        # Check for alerts (might have breakthrough alerts)
        active_alerts = monitoring_engine.get_active_alerts()
        
        # Stop monitoring
        await monitoring_engine.stop_monitoring()
        await monitoring_task
        
        print(f"✅ Monitoring Engine: {len(performance_history)} data points collected")
        print(f"   Active Alerts: {len(active_alerts)}")
        print(f"   Received Alerts: {len(received_alerts)}")
        print(f"   System Status: {status['monitoring_status']}")
    
    @pytest.mark.asyncio
    async def test_integrated_autonomous_workflow(self, temp_repo):
        """Test complete integrated autonomous SDLC workflow"""
        
        print("\n🚀 Testing Complete Integrated Autonomous SDLC Workflow")
        print("=" * 70)
        
        # Initialize all systems
        quality_gates = get_enhanced_quality_gates(str(temp_repo))
        research_engine = get_research_engine(str(temp_repo))
        monitoring_engine = get_monitoring_engine(str(temp_repo))
        
        # Start monitoring
        monitoring_task = asyncio.create_task(monitoring_engine.start_monitoring())
        await asyncio.sleep(0.5)
        
        # Phase 1: Simple Generation with Monitoring
        print("\n📋 Phase 1: Simple Generation Quality Gates")
        
        start_time = time.time()
        
        simple_results = await quality_gates.execute_progressive_quality_gates(
            generation_level="simple",
            research_mode=False,
            parallel_execution=True
        )
        
        simple_duration = time.time() - start_time
        monitoring_engine.record_metric("sdlc.simple_generation_time", simple_duration)
        monitoring_engine.record_metric("sdlc.simple_score", simple_results.overall_score)
        
        assert simple_results.total_gates > 0
        print(f"   ✅ Simple Generation: {simple_results.passed_gates}/{simple_results.total_gates} passed")
        
        # Phase 2: Research Session Initiation
        if simple_results.overall_score >= 60.0:  # Proceed if reasonable quality
            print("\n🔬 Phase 2: Research-Driven Development")
            
            session = await research_engine.initiate_research_session(
                title="Autonomous SDLC Performance Research",
                research_questions=[
                    "Can autonomous quality gates improve development velocity?",
                    "What is the impact on code quality metrics?",
                    "How does monitoring affect system reliability?"
                ],
                baseline_metrics={
                    "development_velocity": 10.0,
                    "code_quality_score": simple_results.overall_score,
                    "system_reliability": 0.95
                }
            )
            
            monitoring_engine.record_metric("sdlc.research_sessions", 1.0)
            print(f"   ✅ Research Session: {len(session.hypotheses)} hypotheses")
            
            # Quick experimental design and execution
            designs = await research_engine.design_experiments(
                session.session_id,
                experiment_types=[ExperimentType.PERFORMANCE_COMPARISON]
            )
            
            if designs:
                experiment_result = await research_engine.execute_experiment(
                    session.session_id,
                    designs[0]
                )
                
                monitoring_engine.record_metric("sdlc.experiment_p_value", experiment_result.p_value)
                monitoring_engine.record_metric("sdlc.experiment_effect_size", experiment_result.effect_size)
                
                print(f"   ✅ Experiment: p={experiment_result.p_value:.4f}, effect={experiment_result.effect_size:.3f}")
        
        # Phase 3: Robust Generation (if simple passed well)
        if simple_results.next_generation_ready:
            print("\n🛡️ Phase 3: Robust Generation Quality Gates")
            
            robust_results = await quality_gates.execute_progressive_quality_gates(
                generation_level="robust",
                research_mode=False,
                parallel_execution=True
            )
            
            monitoring_engine.record_metric("sdlc.robust_score", robust_results.overall_score)
            print(f"   ✅ Robust Generation: {robust_results.passed_gates}/{robust_results.total_gates} passed")
            
            # Phase 4: Optimized Generation (if robust passed well)
            if robust_results.next_generation_ready:
                print("\n⚡ Phase 4: Optimized Generation with Research")
                
                optimized_results = await quality_gates.execute_progressive_quality_gates(
                    generation_level="optimized",
                    research_mode=True,
                    parallel_execution=True
                )
                
                monitoring_engine.record_metric("sdlc.optimized_score", optimized_results.overall_score)
                monitoring_engine.record_metric("sdlc.research_indicators", len(optimized_results.research_metrics))
                
                print(f"   ✅ Optimized Generation: {optimized_results.passed_gates}/{optimized_results.total_gates} passed")
                print(f"   🔬 Research Indicators: {len(optimized_results.research_metrics)}")
        
        # Phase 5: Final Analysis and Reporting
        print("\n📊 Phase 5: Final Analysis and Reporting")
        
        # Wait a bit for metrics to be processed
        await asyncio.sleep(1.0)
        
        # Generate comprehensive reports
        monitoring_report = monitoring_engine.generate_monitoring_report()
        active_alerts = monitoring_engine.get_active_alerts()
        breakthrough_metrics = monitoring_engine.get_breakthrough_metrics()
        
        # Final validations
        assert monitoring_report["system_status"]["monitoring_status"] == "running"
        assert len(monitoring_report["metric_summary"]) > 0
        
        print(f"   ✅ Monitoring Report: {len(monitoring_report['metric_summary'])} metrics analyzed")
        print(f"   🚨 Active Alerts: {len(active_alerts)}")
        print(f"   🚀 Breakthrough Metrics: {breakthrough_metrics.get('active_breakthroughs', 0)}")
        
        # Cleanup
        await monitoring_engine.stop_monitoring()
        await monitoring_task
        
        print(f"\n🎯 Integrated Autonomous SDLC Workflow Complete!")
        print(f"   Total Execution Time: {time.time() - start_time:.2f} seconds")
        
        # Final assertions for complete workflow
        assert simple_results.overall_score >= 0.0
        assert len(monitoring_report["metric_summary"]) >= 5  # Should have collected various metrics
        
        return {
            "simple_results": simple_results,
            "monitoring_report": monitoring_report,
            "workflow_duration": time.time() - start_time
        }
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, temp_repo):
        """Test error handling and system resilience"""
        
        quality_gates = get_enhanced_quality_gates(str(temp_repo))
        
        # Test with invalid generation level
        results = await quality_gates.execute_progressive_quality_gates(
            generation_level="invalid",
            research_mode=False,
            parallel_execution=True
        )
        
        # Should still work with fallback
        assert results.total_gates > 0
        
        # Test monitoring with invalid metrics
        monitoring_engine = get_monitoring_engine(str(temp_repo))
        
        # This should not crash the system
        monitoring_engine.record_metric("invalid.metric", float('nan'))
        monitoring_engine.record_metric("test.metric", 100.0)  # Valid metric
        
        # System should continue working
        status = monitoring_engine.get_system_status()
        assert status["monitoring_status"] in ["initializing", "running"]
        
        print("✅ Error handling and resilience tests passed")
    
    def test_performance_benchmarks(self, temp_repo):
        """Test performance benchmarks and requirements"""
        
        # Test quality gates performance
        start_time = time.time()
        
        quality_gates = get_enhanced_quality_gates(str(temp_repo))
        
        initialization_time = time.time() - start_time
        
        # Should initialize quickly
        assert initialization_time < 5.0, f"Initialization took {initialization_time:.2f}s, should be < 5.0s"
        
        # Test monitoring engine performance
        start_time = time.time()
        
        monitoring_engine = get_monitoring_engine(str(temp_repo))
        
        # Record many metrics quickly
        for i in range(100):
            monitoring_engine.record_metric(f"test.metric_{i % 10}", float(i))
        
        metric_recording_time = time.time() - start_time
        
        # Should record metrics efficiently
        assert metric_recording_time < 1.0, f"Recording 100 metrics took {metric_recording_time:.2f}s, should be < 1.0s"
        
        print(f"✅ Performance benchmarks:")
        print(f"   Initialization: {initialization_time:.3f}s")
        print(f"   Metric recording (100 points): {metric_recording_time:.3f}s")


@pytest.mark.asyncio
async def test_integration_demo():
    """Demo function to showcase complete integration"""
    
    print("🎯 Comprehensive Autonomous SDLC Integration Demo")
    print("=" * 60)
    
    # Create temporary test environment
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    
    try:
        # Setup mock project
        (repo_path / "src").mkdir()
        (repo_path / "src" / "main.py").write_text("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def optimized_fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = optimized_fibonacci(n-1, memo) + optimized_fibonacci(n-2, memo)
    return memo[n]
""")
        
        (repo_path / "README.md").write_text("# Demo Project")
        (repo_path / "LICENSE").write_text("MIT License")
        
        # Initialize systems
        quality_gates = get_enhanced_quality_gates(str(repo_path))
        research_engine = get_research_engine(str(repo_path))
        monitoring_engine = get_monitoring_engine(str(repo_path))
        
        # Start monitoring
        monitoring_task = asyncio.create_task(monitoring_engine.start_monitoring())
        await asyncio.sleep(0.5)
        
        print("\n✅ All systems initialized")
        
        # Run progressive quality gates
        print("\n📋 Running Progressive Quality Gates...")
        
        simple_results = await quality_gates.execute_progressive_quality_gates(
            generation_level="simple",
            parallel_execution=True
        )
        
        print(f"   Simple Generation: {simple_results.overall_score:.1f} score")
        monitoring_engine.record_metric("demo.quality_score", simple_results.overall_score)
        
        # Run research session
        print("\n🔬 Running Research Session...")
        
        session = await research_engine.initiate_research_session(
            title="Fibonacci Optimization Demo",
            research_questions=["Can optimization improve performance?"],
            baseline_metrics={"performance": 1.0}
        )
        
        designs = await research_engine.design_experiments(
            session.session_id,
            experiment_types=[ExperimentType.PERFORMANCE_COMPARISON]
        )
        
        if designs:
            result = await research_engine.execute_experiment(
                session.session_id,
                designs[0]
            )
            print(f"   Experiment result: p={result.p_value:.3f}")
            monitoring_engine.record_metric("demo.p_value", result.p_value)
        
        # Generate final reports
        print("\n📊 Generating Reports...")
        
        monitoring_report = monitoring_engine.generate_monitoring_report()
        research_report = await research_engine.generate_research_report(session.session_id)
        
        print(f"   Monitoring metrics: {len(monitoring_report['metric_summary'])}")
        print(f"   Research findings: {len(research_report['key_findings'])}")
        
        # Cleanup
        await monitoring_engine.stop_monitoring()
        await monitoring_task
        
        print("\n🎯 Integration Demo Complete!")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(test_integration_demo())
    
    # Run the full test suite
    pytest.main([__file__, "-v", "--tb=short"])