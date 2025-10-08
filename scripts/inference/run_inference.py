#!/usr/bin/env python3
"""
Inference script for Japanese Sport Logo Detection AI
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from main import JapaneseSportLogoDetectionApp
import logging

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Run logo detection inference")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images")
    parser.add_argument("--output_dir", type=str, default="output/detections", 
                       help="Output directory for results")
    parser.add_argument("--model", type=str, default="models/checkpoints/best.pt",
                       help="Path to trained model")
    
    args = parser.parse_args()
    
    print("🔍 Running Japanese Sport Logo Detection")
    print("=" * 40)
    
    # Initialize app
    app = JapaneseSportLogoDetectionApp()
    
    if args.image:
        # Single image inference
        print(f"📸 Processing single image: {args.image}")
        result = app.run_inference(args.image, args.output_dir)
        
        if result:
            print(f"✅ Detection completed!")
            print(f"   Found {result.get('total_detections', 0)} logos")
            print(f"   Average confidence: {result.get('average_confidence', 0):.3f}")
        else:
            print("❌ Detection failed!")
            return False
    
    elif args.image_dir:
        # Batch inference
        print(f"📁 Processing image directory: {args.image_dir}")
        results = app.batch_inference(args.image_dir, args.output_dir)
        
        if results:
            total_detections = sum(r.get("total_detections", 0) for r in results)
            print(f"✅ Batch processing completed!")
            print(f"   Processed {len(results)} images")
            print(f"   Total logos detected: {total_detections}")
        else:
            print("❌ Batch processing failed!")
            return False
    
    else:
        print("❌ Please provide either --image or --image_dir")
        return False
    
    print(f"\n📁 Results saved to: {args.output_dir}")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
