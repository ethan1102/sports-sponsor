"""
Annotation Validation and Quality Control Tool
"""
import json
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnnotationValidator:
    """
    Comprehensive annotation validation and quality control
    """
    
    def __init__(self, annotation_dir: str, image_dir: str):
        self.annotation_dir = Path(annotation_dir)
        self.image_dir = Path(image_dir)
        self.validation_results = []
        self.quality_metrics = {}
        
    def validate_all_annotations(self) -> Dict:
        """Validate all annotations in the directory"""
        logger.info("Starting comprehensive annotation validation...")
        
        annotation_files = list(self.annotation_dir.glob("*.json"))
        
        if not annotation_files:
            logger.error("No annotation files found")
            return {}
        
        validation_results = {
            "total_files": len(annotation_files),
            "valid_files": 0,
            "invalid_files": 0,
            "total_annotations": 0,
            "valid_annotations": 0,
            "issues": [],
            "quality_metrics": {}
        }
        
        for annotation_file in annotation_files:
            result = self.validate_single_annotation(annotation_file)
            validation_results["issues"].extend(result.get("issues", []))
            
            if result["valid"]:
                validation_results["valid_files"] += 1
            else:
                validation_results["invalid_files"] += 1
            
            validation_results["total_annotations"] += result.get("annotation_count", 0)
            validation_results["valid_annotations"] += result.get("valid_annotations", 0)
        
        # Calculate quality metrics
        validation_results["quality_metrics"] = self.calculate_quality_metrics()
        
        self.validation_results = validation_results
        logger.info(f"Validation complete: {validation_results['valid_files']}/{validation_results['total_files']} files valid")
        
        return validation_results
    
    def validate_single_annotation(self, annotation_file: Path) -> Dict:
        """Validate a single annotation file"""
        result = {
            "file": str(annotation_file),
            "valid": True,
            "issues": [],
            "annotation_count": 0,
            "valid_annotations": 0
        }
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            result["annotation_count"] = len(annotations)
            
            # Validate file structure
            if not isinstance(annotations, list):
                result["valid"] = False
                result["issues"].append("Invalid annotations format - should be a list")
                return result
            
            # Validate each annotation
            for i, annotation in enumerate(annotations):
                annotation_result = self.validate_single_bbox(annotation, i)
                
                if annotation_result["valid"]:
                    result["valid_annotations"] += 1
                else:
                    result["issues"].extend(annotation_result["issues"])
            
            # Check if image exists
            image_path = data.get('image_path')
            if image_path and not Path(image_path).exists():
                result["issues"].append(f"Referenced image not found: {image_path}")
            
            # Check annotation count
            if len(annotations) == 0:
                result["issues"].append("No annotations found")
            
        except json.JSONDecodeError as e:
            result["valid"] = False
            result["issues"].append(f"Invalid JSON format: {e}")
        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Error reading file: {e}")
        
        return result
    
    def validate_single_bbox(self, annotation: Dict, index: int) -> Dict:
        """Validate a single bounding box annotation"""
        result = {
            "index": index,
            "valid": True,
            "issues": []
        }
        
        # Check required fields
        required_fields = ['bbox', 'class']
        for field in required_fields:
            if field not in annotation:
                result["valid"] = False
                result["issues"].append(f"Missing required field: {field}")
        
        if not result["valid"]:
            return result
        
        # Validate bbox format
        bbox = annotation['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            result["valid"] = False
            result["issues"].append(f"Invalid bbox format - should be [x1, y1, x2, y2]")
            return result
        
        x1, y1, x2, y2 = bbox
        
        # Validate bbox values
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            result["valid"] = False
            result["issues"].append(f"Bbox coordinates must be numbers")
            return result
        
        # Validate coordinate ranges (normalized coordinates)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            result["valid"] = False
            result["issues"].append(f"Bbox coordinates out of range [0,1]: ({x1}, {y1}, {x2}, {y2})")
            return result
        
        # Validate bbox size
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            result["valid"] = False
            result["issues"].append(f"Invalid bbox size: width={width}, height={height}")
            return result
        
        # Check minimum size
        min_size = 0.01  # 1% of image
        if width < min_size or height < min_size:
            result["issues"].append(f"Very small bbox: {width:.3f}x{height:.3f}")
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            result["issues"].append(f"Extreme aspect ratio: {aspect_ratio:.2f}")
        
        return result
    
    def calculate_quality_metrics(self) -> Dict:
        """Calculate quality metrics for the annotation dataset"""
        metrics = {
            "average_annotations_per_image": 0,
            "bbox_size_distribution": {},
            "aspect_ratio_distribution": {},
            "coverage_statistics": {},
            "consistency_metrics": {}
        }
        
        annotation_files = list(self.annotation_dir.glob("*.json"))
        if not annotation_files:
            return metrics
        
        all_bboxes = []
        all_sizes = []
        all_aspect_ratios = []
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                annotations = data.get('annotations', [])
                for annotation in annotations:
                    bbox = annotation.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        width = x2 - x1
                        height = y2 - y1
                        
                        all_bboxes.append(bbox)
                        all_sizes.append(width * height)
                        all_aspect_ratios.append(width / height)
                        
            except Exception as e:
                logger.warning(f"Error processing {annotation_file}: {e}")
                continue
        
        if all_sizes:
            metrics["average_annotations_per_image"] = len(all_bboxes) / len(annotation_files)
            metrics["bbox_size_distribution"] = {
                "min": min(all_sizes),
                "max": max(all_sizes),
                "mean": np.mean(all_sizes),
                "std": np.std(all_sizes)
            }
            metrics["aspect_ratio_distribution"] = {
                "min": min(all_aspect_ratios),
                "max": max(all_aspect_ratios),
                "mean": np.mean(all_aspect_ratios),
                "std": np.std(all_aspect_ratios)
            }
        
        return metrics
    
    def generate_quality_report(self, output_dir: str = None) -> str:
        """Generate comprehensive quality report"""
        if not self.validation_results:
            self.validate_all_annotations()
        
        report = self._create_quality_report()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save text report
            report_file = output_path / "annotation_quality_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Generate visualizations
            self._create_quality_visualizations(output_path)
            
            logger.info(f"Quality report saved to {output_path}")
        
        return report
    
    def _create_quality_report(self) -> str:
        """Create text quality report"""
        report = "=" * 60 + "\n"
        report += "ANNOTATION QUALITY REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Summary
        report += "SUMMARY\n"
        report += "-" * 20 + "\n"
        report += f"Total files: {self.validation_results['total_files']}\n"
        report += f"Valid files: {self.validation_results['valid_files']}\n"
        report += f"Invalid files: {self.validation_results['invalid_files']}\n"
        report += f"Success rate: {self.validation_results['valid_files']/self.validation_results['total_files']*100:.1f}%\n\n"
        
        report += f"Total annotations: {self.validation_results['total_annotations']}\n"
        report += f"Valid annotations: {self.validation_results['valid_annotations']}\n"
        report += f"Annotation success rate: {self.validation_results['valid_annotations']/max(self.validation_results['total_annotations'], 1)*100:.1f}%\n\n"
        
        # Quality metrics
        metrics = self.validation_results['quality_metrics']
        report += "QUALITY METRICS\n"
        report += "-" * 20 + "\n"
        report += f"Average annotations per image: {metrics.get('average_annotations_per_image', 0):.2f}\n\n"
        
        if 'bbox_size_distribution' in metrics:
            size_dist = metrics['bbox_size_distribution']
            report += "Bbox Size Distribution:\n"
            report += f"  Min: {size_dist.get('min', 0):.4f}\n"
            report += f"  Max: {size_dist.get('max', 0):.4f}\n"
            report += f"  Mean: {size_dist.get('mean', 0):.4f}\n"
            report += f"  Std: {size_dist.get('std', 0):.4f}\n\n"
        
        if 'aspect_ratio_distribution' in metrics:
            ar_dist = metrics['aspect_ratio_distribution']
            report += "Aspect Ratio Distribution:\n"
            report += f"  Min: {ar_dist.get('min', 0):.2f}\n"
            report += f"  Max: {ar_dist.get('max', 0):.2f}\n"
            report += f"  Mean: {ar_dist.get('mean', 0):.2f}\n"
            report += f"  Std: {ar_dist.get('std', 0):.2f}\n\n"
        
        # Issues
        if self.validation_results['issues']:
            report += "ISSUES FOUND\n"
            report += "-" * 20 + "\n"
            
            issue_counts = defaultdict(int)
            for issue in self.validation_results['issues']:
                issue_counts[issue] += 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                report += f"{count:3d}x {issue}\n"
        
        report += "\n" + "=" * 60 + "\n"
        return report
    
    def _create_quality_visualizations(self, output_dir: Path):
        """Create quality visualization plots"""
        try:
            # Bbox size distribution
            self._plot_bbox_size_distribution(output_dir)
            
            # Aspect ratio distribution
            self._plot_aspect_ratio_distribution(output_dir)
            
            # Annotation count per image
            self._plot_annotation_count_distribution(output_dir)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _plot_bbox_size_distribution(self, output_dir: Path):
        """Plot bbox size distribution"""
        # This would create actual plots
        # For now, just create a placeholder
        pass
    
    def _plot_aspect_ratio_distribution(self, output_dir: Path):
        """Plot aspect ratio distribution"""
        # This would create actual plots
        pass
    
    def _plot_annotation_count_distribution(self, output_dir: Path):
        """Plot annotation count per image distribution"""
        # This would create actual plots
        pass
    
    def fix_common_issues(self, output_dir: str = None) -> Dict:
        """Fix common annotation issues"""
        logger.info("Fixing common annotation issues...")
        
        fixed_count = 0
        issues_fixed = defaultdict(int)
        
        annotation_files = list(self.annotation_dir.glob("*.json"))
        
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                annotations = data.get('annotations', [])
                fixed_annotations = []
                
                for annotation in annotations:
                    fixed_annotation = self._fix_single_annotation(annotation)
                    if fixed_annotation != annotation:
                        fixed_count += 1
                        issues_fixed["bbox_fixed"] += 1
                    
                    fixed_annotations.append(fixed_annotation)
                
                # Save fixed annotations
                if output_dir:
                    data['annotations'] = fixed_annotations
                    output_file = Path(output_dir) / annotation_file.name
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error fixing {annotation_file}: {e}")
        
        result = {
            "files_processed": len(annotation_files),
            "annotations_fixed": fixed_count,
            "issues_fixed": dict(issues_fixed)
        }
        
        logger.info(f"Fixed {fixed_count} annotations in {len(annotation_files)} files")
        return result
    
    def _fix_single_annotation(self, annotation: Dict) -> Dict:
        """Fix a single annotation"""
        fixed = annotation.copy()
        
        bbox = fixed.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are in [0, 1] range
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            # Ensure x1 < x2 and y1 < y2
            if x1 >= x2:
                x1, x2 = min(x1, x2), max(x1, x2)
                if x1 == x2:
                    x2 = min(1, x1 + 0.01)
            
            if y1 >= y2:
                y1, y2 = min(y1, y2), max(y1, y2)
                if y1 == y2:
                    y2 = min(1, y1 + 0.01)
            
            fixed['bbox'] = [x1, y1, x2, y2]
        
        return fixed

def main():
    """Main function for annotation validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Annotation Validation Tool")
    parser.add_argument("--annotation_dir", required=True, help="Directory containing annotation files")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", help="Output directory for reports")
    parser.add_argument("--fix_issues", action="store_true", help="Fix common issues")
    
    args = parser.parse_args()
    
    validator = AnnotationValidator(args.annotation_dir, args.image_dir)
    
    # Validate annotations
    results = validator.validate_all_annotations()
    print("Validation Results:")
    print(f"Valid files: {results['valid_files']}/{results['total_files']}")
    print(f"Valid annotations: {results['valid_annotations']}/{results['total_annotations']}")
    
    # Generate report
    if args.output_dir:
        report = validator.generate_quality_report(args.output_dir)
        print(f"\nQuality report generated in {args.output_dir}")
    
    # Fix issues if requested
    if args.fix_issues:
        fix_results = validator.fix_common_issues(args.output_dir)
        print(f"\nFixed {fix_results['annotations_fixed']} annotations")

if __name__ == "__main__":
    main()
