#!/usr/bin/env python3
"""
Enhanced test to verify the real analysis functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.pr_analysis import analyze_pr
    from autogen_code_review_bot.language_detection import detect_language
    
    print("✅ Enhanced module imports successful")
    
    # Test language detection on actual repository
    files = list(Path(".").rglob("*.py"))[:10]  # Get some Python files
    detected = detect_language([str(f) for f in files])
    print(f"✅ Language detection works: {detected}")
    
    # Test full analysis on current repository
    print("\n🔍 Running full PR analysis on current repository...")
    result = analyze_pr(".", use_cache=False, use_parallel=False)
    
    print(f"✅ Security analysis: {result.security.tool}")
    print(f"   Output length: {len(result.security.output)} chars")
    print(f"   Metadata: {result.security.metadata}")
    
    print(f"✅ Style analysis: {result.style.tool}")
    print(f"   Output length: {len(result.style.output)} chars")
    print(f"   Metadata: {result.style.metadata}")
    
    print(f"✅ Performance analysis: {result.performance.tool}")
    print(f"   Output length: {len(result.performance.output)} chars")
    print(f"   Metadata: {result.performance.metadata}")
    
    print(f"\n📊 Analysis duration: {result.metadata['analysis_duration']} seconds")
    
    print("\n🎉 Enhanced functionality test PASSED")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)