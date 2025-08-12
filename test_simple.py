#!/usr/bin/env python3
"""
Simple test to verify basic system functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot.language_detection import detect_language
    from autogen_code_review_bot.models import AnalysisSection, PRAnalysisResult
    print("✅ Module imports successful")
    
    # Test language detection
    files = ["main.py", "app.js", "script.ts"]
    detected = detect_language(files)
    print(f"✅ Language detection works: {detected}")
    
    # Test model creation
    section = AnalysisSection(
        tool="test",
        output="Test output",
        return_code=0
    )
    print("✅ Model creation works")
    
    print("\n🎉 Basic functionality test PASSED")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)