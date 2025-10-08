#!/usr/bin/env python3
"""
Main entry point for Japanese Sport Logo Detection AI
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import JapaneseSportLogoDetectionApp

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Japanese Sport Logo Detection AI - 90%+ Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start - full pipeline
  python run.py --mode full
  
  # Collect data only
  python run.py --mode collect --synthetic 200
  
  # Train model
  python run.py --mode train
  
  # Run inference
  python run.py --mode inference --image path/to/image.jpg
  
  # Batch processing
  python run.py --mode batch --image_dir path/to/images/
        """
    )
    
    parser.add_argument("--mode", 
                       choices=["collect", "preprocess", "train", "optimize", 
                               "inference", "batch", "evaluate", "full"],
                       default="full", 
                       help="Mode to run")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images for batch inference")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to use")
    parser.add_argument("--synthetic", type=int, default=100, help="Number of synthetic logos to generate")
    
    args = parser.parse_args()
    
    print("🏆 Japanese Sport Logo Detection AI")
    print("=" * 40)
    print("Target: 90%+ Detection Accuracy")
    print("=" * 40)
    
    # Create app instance
    app = JapaneseSportLogoDetectionApp()
    
    # Run based on mode
    if args.mode == "collect":
        success = app.collect_data(args.synthetic)
    elif args.mode == "preprocess":
        success = app.preprocess_data()
    elif args.mode == "train":
        success = app.train_model(args.model)
    elif args.mode == "optimize":
        success = app.optimize_model()
    elif args.mode == "inference":
        if not args.image:
            print("❌ Please provide --image argument for inference mode")
            sys.exit(1)
        result = app.run_inference(args.image, args.output_dir)
        success = bool(result)
    elif args.mode == "batch":
        if not args.image_dir:
            print("❌ Please provide --image_dir argument for batch mode")
            sys.exit(1)
        results = app.batch_inference(args.image_dir, args.output_dir)
        success = bool(results)
    elif args.mode == "evaluate":
        result = app.evaluate_model()
        success = bool(result)
    elif args.mode == "full":
        success = app.run_full_pipeline()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        success = False
    
    if success:
        print("\n✅ Operation completed successfully!")
        print("\nNext steps:")
        print("1. Check results in output/ directory")
        print("2. Run evaluation: python run.py --mode evaluate")
        print("3. See docs/guides/ for more information")
        sys.exit(0)
    else:
        print("\n❌ Operation failed!")
        print("Check logs/ directory for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
