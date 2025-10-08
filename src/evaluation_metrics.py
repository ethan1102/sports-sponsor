"""
Evaluation metrics and validation for Japanese sport sponsor logo detection
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import cv2
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDetectionEvaluator:
    """
    Comprehensive evaluation system for logo detection model
    """
    
    def __init__(self):
        self.metrics = {}
        self.detection_results = []
        
    def calculate_detection_metrics(self, predictions: List[Dict], 
                                  ground_truth: List[Dict],
                                  iou_threshold: float = 0.5) -> Dict:
        """
        Calculate comprehensive detection metrics
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries
            iou_threshold: IoU threshold for matching detections
            
        Returns:
            Dictionary containing all metrics
        """
        logger.info("Calculating detection metrics...")
        
        # Initialize metrics
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mAP": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "average_precision": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "detection_accuracy": 0.0
        }
        
        try:
            # Calculate precision, recall, F1
            precision, recall, f1 = self._calculate_precision_recall_f1(
                predictions, ground_truth, iou_threshold
            )
            
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1
            
            # Calculate mAP
            map_scores = self._calculate_map(predictions, ground_truth, iou_threshold)
            metrics["mAP"] = map_scores["mAP"]
            metrics["mAP50"] = map_scores["mAP50"]
            metrics["mAP75"] = map_scores["mAP75"]
            
            # Calculate additional metrics
            metrics["average_precision"] = self._calculate_average_precision(
                predictions, ground_truth, iou_threshold
            )
            
            # Calculate error rates
            fp_rate, fn_rate = self._calculate_error_rates(
                predictions, ground_truth, iou_threshold
            )
            metrics["false_positive_rate"] = fp_rate
            metrics["false_negative_rate"] = fn_rate
            
            # Calculate detection accuracy
            metrics["detection_accuracy"] = self._calculate_detection_accuracy(
                predictions, ground_truth, iou_threshold
            )
            
            self.metrics = metrics
            logger.info(f"Detection metrics calculated: {metrics}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _calculate_precision_recall_f1(self, predictions: List[Dict], 
                                      ground_truth: List[Dict],
                                      iou_threshold: float) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get("detections", [])
            gt_boxes = gt.get("detections", [])
            
            # Match predictions with ground truth
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            # Count unmatched ground truth as false negatives
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1["x1"], box1["y1"], box1["x2"], box1["y2"]
        x1_2, y1_2, x2_2, y2_2 = box2["x1"], box2["y1"], box2["x2"], box2["y2"]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_map(self, predictions: List[Dict], ground_truth: List[Dict],
                      iou_threshold: float) -> Dict:
        """Calculate mean Average Precision (mAP)"""
        # This is a simplified mAP calculation
        # In practice, you'd use more sophisticated methods
        
        precisions = []
        recalls = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get("detections", [])
            gt_boxes = gt.get("detections", [])
            
            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                continue
            
            # Calculate precision and recall for this image
            tp = 0
            fp = 0
            fn = len(gt_boxes)
            
            for pred_box in pred_boxes:
                matched = False
                for gt_box in gt_boxes:
                    iou = self._calculate_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou >= iou_threshold:
                        tp += 1
                        fn -= 1
                        matched = True
                        break
                
                if not matched:
                    fp += 1
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precisions.append(precision)
                recalls.append(recall)
        
        # Calculate average precision
        if len(precisions) > 0:
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
        else:
            avg_precision = 0.0
            avg_recall = 0.0
        
        return {
            "mAP": avg_precision,
            "mAP50": avg_precision,  # Simplified
            "mAP75": avg_precision   # Simplified
        }
    
    def _calculate_average_precision(self, predictions: List[Dict], 
                                   ground_truth: List[Dict],
                                   iou_threshold: float) -> float:
        """Calculate average precision using precision-recall curve"""
        all_precisions = []
        all_recalls = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get("detections", [])
            gt_boxes = gt.get("detections", [])
            
            if len(pred_boxes) == 0:
                continue
            
            # Sort predictions by confidence
            sorted_preds = sorted(pred_boxes, key=lambda x: x["confidence"], reverse=True)
            
            precisions = []
            recalls = []
            
            tp = 0
            fp = 0
            fn = len(gt_boxes)
            
            for pred_box in sorted_preds:
                matched = False
                for gt_box in gt_boxes:
                    iou = self._calculate_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou >= iou_threshold:
                        tp += 1
                        fn -= 1
                        matched = True
                        break
                
                if not matched:
                    fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            all_precisions.extend(precisions)
            all_recalls.extend(recalls)
        
        if len(all_precisions) > 0:
            # Calculate area under precision-recall curve
            return np.trapz(all_precisions, all_recalls)
        else:
            return 0.0
    
    def _calculate_error_rates(self, predictions: List[Dict], 
                             ground_truth: List[Dict],
                             iou_threshold: float) -> Tuple[float, float]:
        """Calculate false positive and false negative rates"""
        total_predictions = sum(len(p.get("detections", [])) for p in predictions)
        total_ground_truth = sum(len(gt.get("detections", [])) for gt in ground_truth)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get("detections", [])
            gt_boxes = gt.get("detections", [])
            
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        fp_rate = false_positives / total_predictions if total_predictions > 0 else 0
        fn_rate = false_negatives / total_ground_truth if total_ground_truth > 0 else 0
        
        return fp_rate, fn_rate
    
    def _calculate_detection_accuracy(self, predictions: List[Dict], 
                                    ground_truth: List[Dict],
                                    iou_threshold: float) -> float:
        """Calculate overall detection accuracy"""
        correct_detections = 0
        total_detections = 0
        
        for pred, gt in zip(predictions, ground_truth):
            pred_boxes = pred.get("detections", [])
            gt_boxes = gt.get("detections", [])
            
            total_detections += len(gt_boxes)
            
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box["bbox"], gt_box["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    correct_detections += 1
                    matched_gt.add(best_gt_idx)
        
        return correct_detections / total_detections if total_detections > 0 else 0
    
    def plot_precision_recall_curve(self, predictions: List[Dict], 
                                  ground_truth: List[Dict],
                                  save_path: str = None) -> None:
        """Plot precision-recall curve"""
        try:
            # Calculate precision and recall at different confidence thresholds
            confidence_thresholds = np.arange(0.1, 1.0, 0.05)
            precisions = []
            recalls = []
            
            for threshold in confidence_thresholds:
                # Filter predictions by confidence threshold
                filtered_predictions = []
                for pred in predictions:
                    filtered_detections = [
                        det for det in pred.get("detections", [])
                        if det["confidence"] >= threshold
                    ]
                    filtered_predictions.append({
                        "detections": filtered_detections
                    })
                
                # Calculate precision and recall
                precision, recall, _ = self._calculate_precision_recall_f1(
                    filtered_predictions, ground_truth, 0.5
                )
                precisions.append(precision)
                recalls.append(recall)
            
            # Plot curve
            plt.figure(figsize=(10, 6))
            plt.plot(recalls, precisions, 'b-', linewidth=2, label='Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Logo Detection')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add AUC
            auc_score = auc(recalls, precisions)
            plt.text(0.6, 0.2, f'AUC: {auc_score:.3f}', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Precision-recall curve saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curve: {e}")
    
    def plot_confusion_matrix(self, predictions: List[Dict], 
                            ground_truth: List[Dict],
                            save_path: str = None) -> None:
        """Plot confusion matrix for detection results"""
        try:
            # Convert to binary classification (detection vs no detection)
            pred_binary = []
            gt_binary = []
            
            for pred, gt in zip(predictions, ground_truth):
                pred_binary.append(1 if len(pred.get("detections", [])) > 0 else 0)
                gt_binary.append(1 if len(gt.get("detections", [])) > 0 else 0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(gt_binary, pred_binary)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Logo', 'Logo'],
                       yticklabels=['No Logo', 'Logo'])
            plt.title('Confusion Matrix - Logo Detection')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def generate_evaluation_report(self, predictions: List[Dict], 
                                 ground_truth: List[Dict],
                                 output_dir: str = None) -> Dict:
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Calculate all metrics
        metrics = self.calculate_detection_metrics(predictions, ground_truth)
        
        # Create report
        report = {
            "evaluation_summary": {
                "total_images": len(predictions),
                "total_predictions": sum(len(p.get("detections", [])) for p in predictions),
                "total_ground_truth": sum(len(gt.get("detections", [])) for gt in ground_truth),
                "evaluation_date": str(Path().cwd())
            },
            "metrics": metrics,
            "target_achievement": {
                "target_accuracy": EVALUATION_METRICS["target_accuracy"],
                "achieved_accuracy": metrics["detection_accuracy"],
                "target_met": metrics["detection_accuracy"] >= EVALUATION_METRICS["target_accuracy"]
            }
        }
        
        # Save report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON report
            report_path = output_path / "evaluation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate plots
            self.plot_precision_recall_curve(
                predictions, ground_truth,
                str(output_path / "precision_recall_curve.png")
            )
            
            self.plot_confusion_matrix(
                predictions, ground_truth,
                str(output_path / "confusion_matrix.png")
            )
            
            logger.info(f"Evaluation report saved to: {output_path}")
        
        return report

def main():
    """Main evaluation function for testing"""
    evaluator = LogoDetectionEvaluator()
    
    # This would typically load actual prediction and ground truth data
    # For demonstration, we'll create sample data
    sample_predictions = [
        {"detections": [{"bbox": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}, "confidence": 0.9}]},
        {"detections": []},
        {"detections": [{"bbox": {"x1": 20, "y1": 20, "x2": 60, "y2": 60}, "confidence": 0.8}]}
    ]
    
    sample_ground_truth = [
        {"detections": [{"bbox": {"x1": 12, "y1": 12, "x2": 52, "y2": 52}}]},
        {"detections": []},
        {"detections": [{"bbox": {"x1": 18, "y1": 18, "x2": 58, "y2": 58}}]}
    ]
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(
        sample_predictions, sample_ground_truth, "output/evaluation"
    )
    
    print("Evaluation Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
