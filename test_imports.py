#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        print("1. Testing config...")
        from config import PROJECT_ROOT, DATA_DIR, MODEL_CONFIG
        print(f"   ✅ Config loaded - PROJECT_ROOT: {PROJECT_ROOT}")
        
        print("2. Testing data collector...")
        from data_collector import JapaneseSportLogoCollector
        print("   ✅ Data collector imported")
        
        print("3. Testing data preprocessor...")
        from data_preprocessor import LogoDataPreprocessor
        print("   ✅ Data preprocessor imported")
        
        print("4. Testing model trainer...")
        from model_trainer import LogoDetectionTrainer
        print("   ✅ Model trainer imported")
        
        print("5. Testing inference engine...")
        from inference_engine import LogoDetectionInference
        print("   ✅ Inference engine imported")
        
        print("6. Testing evaluation metrics...")
        from evaluation_metrics import LogoDetectionEvaluator
        print("   ✅ Evaluation metrics imported")
        
        print("7. Testing main app...")
        from main import JapaneseSportLogoDetectionApp
        print("   ✅ Main app imported")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from main import JapaneseSportLogoDetectionApp
        
        # Create app instance
        app = JapaneseSportLogoDetectionApp()
        print("   ✅ App instance created")
        
        # Test config access
        from config import MODEL_CONFIG
        print(f"   ✅ Model config: {MODEL_CONFIG['model_name']}")
        
        print("   ✅ Basic functionality works!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Japanese Sport Logo Detection AI")
    print("=" * 45)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n✅ All tests passed! The code is working properly.")
        else:
            print("\n❌ Basic functionality test failed.")
    else:
        print("\n❌ Import tests failed.")
    
    print("\n" + "=" * 45)
