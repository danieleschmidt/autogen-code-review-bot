#!/usr/bin/env python3
"""
Test enhanced agent conversation system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.pr_analysis import analyze_pr, format_analysis_with_agents
    from autogen_code_review_bot.enhanced_agents import run_enhanced_dual_review
    
    print("✅ Enhanced agent imports successful")
    
    # Run analysis on current repository
    print("\n🔍 Running PR analysis...")
    result = analyze_pr(".", use_cache=False, use_parallel=False)
    print(f"✅ Analysis completed in {result.metadata['analysis_duration']:.2f} seconds")
    
    # Test enhanced agent conversation
    print("\n🤖 Running enhanced agent conversation...")
    formatted_result = format_analysis_with_agents(result, "config/agents.yaml")
    
    print("✅ Agent conversation completed")
    print(f"📝 Result length: {len(formatted_result)} characters")
    
    # Show first part of the conversation
    print("\n📋 Agent Conversation Preview:")
    print("=" * 60)
    print(formatted_result[:1500] + "..." if len(formatted_result) > 1500 else formatted_result)
    print("=" * 60)
    
    print("\n🎉 Enhanced agent test PASSED")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)