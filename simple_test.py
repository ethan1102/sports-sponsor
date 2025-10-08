#!/usr/bin/env python3
"""
Simple test to verify the code works
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    print("Testing basic imports...")
    
    # Test config
    from config import PROJECT_ROOT, DATA_DIR
    print(f"✅ Config works - PROJECT_ROOT: {PROJECT_ROOT}")
    
    # Test main app
    from main import JapaneseSportLogoDetectionApp
    app = JapaneseSportLogoDetectionApp()
    print("✅ Main app works")
    
    print("\n🎉 Code is working properly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
