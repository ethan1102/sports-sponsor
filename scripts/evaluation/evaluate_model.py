#!/usr/bin/env python3
"""
Evaluation script for Japanese Sport Logo Detection AI
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from main import JapaneseSportLogoDetectionApp
import logging

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate logo detection model")
    parser.add_argument("--test_data", type=str, default="data/test_images",
                       help="Path to test data directory")
    parser.add_argument("--output_dir", type=str, default="output/evaluation",
                       help="Output directory for evaluation results")
    parser.add_argument("--model", type=str, default="models/checkpoints/best.pt",
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    print("📊 Evaluating Japanese Sport Logo Detection Model")
    print("=" * 50)
    
    # Initialize app
    app = JapaneseSportLogoDetectionApp()
    
    # Run evaluation
    print(f"🧪 Running evaluation on: {args.test_data}")
    result = app.evaluate_model(args.test_data)
    
    if result:
        print("✅ Evaluation completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print key metrics if available
        if 'metrics' in result:
            metrics = result['metrics']
            print("\n📈 Key Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
    else:
        print("❌ Evaluation failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
