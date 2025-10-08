#!/usr/bin/env python3
"""
Comprehensive functionality test for Japanese Sport Logo Detection AI
"""
import sys
import os
from pathlib import Path
import traceback

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports without dependencies"""
    print("🔍 Testing basic imports...")
    
    try:
        # Test config import
        from config import PROJECT_ROOT, DATA_DIR, MODEL_CONFIG
        print(f"   ✅ Config imported - PROJECT_ROOT: {PROJECT_ROOT}")
        
        # Test that directories exist
        if PROJECT_ROOT.exists():
            print(f"   ✅ PROJECT_ROOT exists: {PROJECT_ROOT}")
        else:
            print(f"   ❌ PROJECT_ROOT missing: {PROJECT_ROOT}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return False

def test_dependency_imports():
    """Test if required dependencies are available"""
    print("\n🔍 Testing dependency imports...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLO"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("requests", "Requests"),
        ("selenium", "Selenium"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas")
    ]
    
    available = []
    missing = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name} ({module})")
            available.append(name)
        except ImportError:
            print(f"   ❌ {name} ({module}) - MISSING")
            missing.append(name)
    
    print(f"\n   📊 Available: {len(available)}/{len(dependencies)}")
    print(f"   📊 Missing: {len(missing)}/{len(dependencies)}")
    
    if missing:
        print(f"   ⚠️  Missing dependencies: {', '.join(missing)}")
        return False
    else:
        print("   🎉 All dependencies available!")
        return True

def test_core_modules():
    """Test core module imports"""
    print("\n🔍 Testing core module imports...")
    
    modules = [
        ("data_collector", "Data Collector"),
        ("data_preprocessor", "Data Preprocessor"),
        ("model_trainer", "Model Trainer"),
        ("inference_engine", "Inference Engine"),
        ("evaluation_metrics", "Evaluation Metrics"),
        ("annotation_tool", "Annotation Tool"),
        ("annotation_validator", "Annotation Validator"),
        ("main", "Main Application")
    ]
    
    success_count = 0
    
    for module_name, display_name in modules:
        try:
            module = __import__(module_name)
            print(f"   ✅ {display_name}")
            success_count += 1
        except Exception as e:
            print(f"   ❌ {display_name} - {str(e)[:50]}...")
    
    print(f"\n   📊 Modules working: {success_count}/{len(modules)}")
    return success_count == len(modules)

def test_application_creation():
    """Test if main application can be created"""
    print("\n🔍 Testing application creation...")
    
    try:
        from main import JapaneseSportLogoDetectionApp
        app = JapaneseSportLogoDetectionApp()
        print("   ✅ Main application created successfully")
        
        # Test that app has required methods
        required_methods = [
            'collect_data', 'preprocess_data', 'train_model', 
            'optimize_model', 'run_inference', 'batch_inference', 
            'evaluate_model', 'run_full_pipeline'
        ]
        
        for method in required_methods:
            if hasattr(app, method):
                print(f"   ✅ Method {method} exists")
            else:
                print(f"   ❌ Method {method} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Application creation failed: {e}")
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test if directory structure is correct"""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        "src",
        "data",
        "data/raw",
        "data/processed", 
        "data/annotations",
        "data/test_images",
        "models",
        "models/checkpoints",
        "models/exports",
        "tools",
        "tools/annotation",
        "tools/validation",
        "tools/evaluation",
        "scripts",
        "scripts/setup",
        "scripts/training",
        "scripts/inference",
        "scripts/evaluation",
        "docs",
        "docs/guides",
        "tests",
        "tests/unit",
        "tests/integration"
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n   ⚠️  Missing directories: {len(missing_dirs)}")
        return False
    else:
        print("\n   🎉 All directories exist!")
        return True

def test_configuration():
    """Test configuration values"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR,
            MODEL_CONFIG, TRAINING_CONFIG, AUGMENTATION_CONFIG
        )
        
        # Test that paths are correct
        if PROJECT_ROOT.name == "aI":
            print("   ✅ PROJECT_ROOT is correct")
        else:
            print(f"   ❌ PROJECT_ROOT incorrect: {PROJECT_ROOT}")
            return False
        
        # Test model config
        if "model_name" in MODEL_CONFIG:
            print(f"   ✅ Model config: {MODEL_CONFIG['model_name']}")
        else:
            print("   ❌ Model config missing model_name")
            return False
        
        # Test training config
        if "epochs" in TRAINING_CONFIG:
            print(f"   ✅ Training config: {TRAINING_CONFIG['epochs']} epochs")
        else:
            print("   ❌ Training config missing epochs")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_file_permissions():
    """Test file permissions and access"""
    print("\n🔍 Testing file permissions...")
    
    try:
        # Test if we can create files in output directory
        test_file = Path("output/test_write.txt")
        test_file.parent.mkdir(exist_ok=True)
        
        with open(test_file, 'w') as f:
            f.write("test")
        
        if test_file.exists():
            print("   ✅ Can write to output directory")
            test_file.unlink()  # Clean up
        else:
            print("   ❌ Cannot write to output directory")
            return False
        
        # Test if we can create logs
        log_file = Path("logs/test.log")
        log_file.parent.mkdir(exist_ok=True)
        
        with open(log_file, 'w') as f:
            f.write("test log")
        
        if log_file.exists():
            print("   ✅ Can write to logs directory")
            log_file.unlink()  # Clean up
        else:
            print("   ❌ Cannot write to logs directory")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ File permissions test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Dependency Imports", test_dependency_imports),
        ("Core Modules", test_core_modules),
        ("Application Creation", test_application_creation),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_configuration),
        ("File Permissions", test_file_permissions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The code is working properly!")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. The code needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
