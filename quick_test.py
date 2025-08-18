#!/usr/bin/env python3
"""Quick functionality test."""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from autogen_code_review_bot import detect_language
    print("✅ Language detection imported")
    
    from autogen_code_review_bot.quantum_planner import QuantumTaskPlanner
    print("✅ Quantum planner imported")
    
    from autogen_code_review_bot.models import PRAnalysisResult, AnalysisSection
    print("✅ Models imported")
    
    # Test language detection
    files = ["test.py", "script.js", "app.ts"]
    langs = [detect_language(f) for f in files]
    print(f"✅ Language detection: {langs}")
    
    print("🎉 All basic functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()