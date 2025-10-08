#!/usr/bin/env python3
"""
Training script for Japanese Sport Logo Detection AI
"""
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from main import JapaneseSportLogoDetectionApp
import logging

def main():
    """Main training function"""
    print("🏋️ Training Japanese Sport Logo Detection Model")
    print("=" * 50)
    
    # Initialize app
    app = JapaneseSportLogoDetectionApp()
    
    # Step 1: Collect data
    print("\n📊 Step 1: Collecting data...")
    if not app.collect_data(num_synthetic=200):
        print("❌ Data collection failed!")
        return False
    print("✅ Data collection completed!")
    
    # Step 2: Preprocess data
    print("\n🔧 Step 2: Preprocessing data...")
    if not app.preprocess_data():
        print("❌ Data preprocessing failed!")
        return False
    print("✅ Data preprocessing completed!")
    
    # Step 3: Train model
    print("\n🤖 Step 3: Training model...")
    if not app.train_model("yolov8n.pt"):
        print("❌ Model training failed!")
        return False
    print("✅ Model training completed!")
    
    # Step 4: Optimize model
    print("\n⚡ Step 4: Optimizing model...")
    if not app.optimize_model():
        print("⚠️ Model optimization failed, but continuing...")
    else:
        print("✅ Model optimization completed!")
    
    print("\n🎉 Training pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Run inference: python scripts/inference/run_inference.py")
    print("2. Evaluate model: python scripts/evaluation/evaluate_model.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
