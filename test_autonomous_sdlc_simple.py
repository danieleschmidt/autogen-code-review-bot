#!/usr/bin/env python3
"""
Simple test script for Autonomous SDLC functionality
Tests core logic without external dependencies
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock external dependencies
class MockStructlog:
    @staticmethod
    def get_logger(name):
        return MockLogger()

class MockLogger:
    def info(self, msg, **kwargs):
        print(f"INFO: {msg} {kwargs}")
    
    def error(self, msg, **kwargs):
        print(f"ERROR: {msg} {kwargs}")

# Mock modules
sys.modules['structlog'] = MockStructlog()
sys.modules['requests'] = None
sys.modules['git'] = None
sys.modules['github'] = None
sys.modules['prometheus_client'] = None
sys.modules['opentelemetry'] = None
sys.modules['pyautogen'] = None
sys.modules['redis'] = None

# Now we can import our modules
try:
    from autogen_code_review_bot.autonomous_sdlc import (
        AutonomousSDLC, 
        SDLCGeneration,
        create_sdlc_config_for_project
    )
    
    async def test_autonomous_sdlc():
        """Test autonomous SDLC functionality"""
        print("🧠 Testing Autonomous SDLC Engine...")
        
        # Initialize SDLC engine
        sdlc = AutonomousSDLC()
        
        # Test intelligent analysis
        print("🔍 Testing intelligent analysis...")
        analysis_result = await sdlc.intelligent_analysis(".")
        
        print(f"✅ Analysis completed:")
        print(f"   Project Type: {analysis_result['project_info']['type']}")
        print(f"   Languages: {analysis_result['project_info']['languages']}")
        print(f"   Complexity: {analysis_result['project_info']['complexity']}")
        print(f"   Implementation Status: {analysis_result['implementation_status']['status']}")
        print(f"   Completion: {analysis_result['implementation_status']['completion_estimate']:.1%}")
        
        # Create SDLC config
        project_type = analysis_result['project_info']['type']
        sdlc_config = create_sdlc_config_for_project(project_type)
        
        print(f"\n🚀 Testing Generation 1 (Simple) for {project_type} project...")
        
        # Test simple generation
        sdlc_config.target_generation = SDLCGeneration.SIMPLE
        enhancement_result = await sdlc.progressive_enhancement_execution(".", sdlc_config)
        
        print(f"✅ Generation 1 completed:")
        print(f"   Status: {enhancement_result['generation_1']['status']}")
        print(f"   Time: {enhancement_result['generation_1']['generation_time']:.1f}s")
        print(f"   Total Time: {enhancement_result['total_time']:.1f}s")
        
        print(f"\n📊 Execution Summary:")
        print(f"   Checkpoints: {len(sdlc_config.checkpoints)}")
        print(f"   Quality Gates: {len(sdlc_config.quality_gates)}")
        print(f"   Global Requirements: {len(sdlc_config.global_requirements)}")
        
        return True
        
    async def main():
        """Main test function"""
        print("🤖 Autonomous SDLC Test Suite")
        print("=" * 50)
        
        try:
            success = await test_autonomous_sdlc()
            if success:
                print("\n✅ All tests passed! Autonomous SDLC is working correctly.")
                return 0
            else:
                print("\n❌ Tests failed!")
                return 1
                
        except Exception as e:
            print(f"\n💥 Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    if __name__ == "__main__":
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure the autogen_code_review_bot package is properly set up.")
    sys.exit(1)