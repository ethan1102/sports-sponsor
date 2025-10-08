#!/usr/bin/env python3
"""
Installation script for Japanese Sport Logo Detection AI
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/annotations", 
        "data/test_images",
        "models/checkpoints",
        "models/exports",
        "logs",
        "output",
        "output/detections",
        "output/evaluation"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  📁 Created: {directory}")
    
    print("✅ Directories created successfully!")

def verify_installation():
    """Verify installation"""
    print("Verifying installation...")
    
    try:
        import torch
        import cv2
        import ultralytics
        print("✅ Core dependencies verified!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Installing Japanese Sport Logo Detection AI")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Installation failed!")
        return False
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed!")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\nNext steps:")
    print("1. Run: python scripts/training/train_model.py")
    print("2. Or run: python src/main.py --mode full")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
