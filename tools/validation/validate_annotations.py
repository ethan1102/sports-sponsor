#!/usr/bin/env python3
"""
Annotation Validation Tool
"""
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from annotation_validator import AnnotationValidator

def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description="Validate annotation quality")
    parser.add_argument("--annotation_dir", type=str, default="data/annotations",
                       help="Directory containing annotation files")
    parser.add_argument("--image_dir", type=str, default="data/raw",
                       help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="output/validation",
                       help="Output directory for validation reports")
    parser.add_argument("--fix_issues", action="store_true",
                       help="Automatically fix common issues")
    
    args = parser.parse_args()
    
    print("🔍 Validating Annotation Quality")
    print("=" * 35)
    
    # Initialize validator
    validator = AnnotationValidator(args.annotation_dir, args.image_dir)
    
    # Run validation
    print(f"📊 Validating annotations in: {args.annotation_dir}")
    results = validator.validate_all_annotations()
    
    # Print results
    print(f"\n📈 Validation Results:")
    print(f"   Valid files: {results['valid_files']}/{results['total_files']}")
    print(f"   Valid annotations: {results['valid_annotations']}/{results['total_annotations']}")
    print(f"   Success rate: {results['valid_files']/results['total_files']*100:.1f}%")
    
    # Generate report
    if args.output_dir:
        print(f"\n📝 Generating quality report...")
        report = validator.generate_quality_report(args.output_dir)
        print(f"   Report saved to: {args.output_dir}")
    
    # Fix issues if requested
    if args.fix_issues:
        print(f"\n🔧 Fixing common issues...")
        fix_results = validator.fix_common_issues(args.output_dir)
        print(f"   Fixed {fix_results['annotations_fixed']} annotations")
    
    print("\n✅ Validation completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
