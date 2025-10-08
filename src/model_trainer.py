"""
Model training module for Japanese sport sponsor logo detection
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import wandb
import yaml
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDetectionTrainer:
    """
    Trainer class for Japanese sport sponsor logo detection model
    """
    
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self.model = None
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "mAP": []
        }
        
        # Initialize wandb for experiment tracking
        self._init_wandb()
        
    def _init_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        try:
            wandb.init(
                project="japanese-sport-logo-detection",
                config={
                    "model": MODEL_CONFIG["model_name"],
                    "input_size": MODEL_CONFIG["input_size"],
                    "epochs": TRAINING_CONFIG["epochs"],
                    "batch_size": TRAINING_CONFIG["batch_size"],
                    "learning_rate": TRAINING_CONFIG["learning_rate"],
                    "target_accuracy": EVALUATION_METRICS["target_accuracy"]
                }
            )
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
    
    def load_model(self) -> None:
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_name)
            logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train(self, dataset_yaml: str) -> Dict:
        """Train the model"""
        if self.model is None:
            self.load_model()
        
        logger.info("Starting training...")
        
        try:
            # Train the model
            results = self.model.train(
                data=dataset_yaml,
                epochs=TRAINING_CONFIG["epochs"],
                batch=TRAINING_CONFIG["batch_size"],
                imgsz=MODEL_CONFIG["input_size"],
                device=TRAINING_CONFIG["device"],
                patience=TRAINING_CONFIG["patience"],
                save_period=TRAINING_CONFIG["save_period"],
                project=str(MODELS_DIR),
                name="logo_detection",
                exist_ok=True,
                verbose=True
            )
            
            # Extract training metrics
            self._extract_training_metrics(results)
            
            # Save training history
            self._save_training_history()
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _extract_training_metrics(self, results) -> None:
        """Extract metrics from training results"""
        try:
            # This would need to be adapted based on the actual YOLO results format
            # For now, we'll create placeholder metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                
                # Extract loss values
                if 'train/box_loss' in metrics:
                    self.training_history["train_loss"].append(metrics['train/box_loss'])
                if 'val/box_loss' in metrics:
                    self.training_history["val_loss"].append(metrics['val/box_loss'])
                
                # Extract precision, recall, F1
                if 'metrics/precision' in metrics:
                    self.training_history["precision"].append(metrics['metrics/precision'])
                if 'metrics/recall' in metrics:
                    self.training_history["recall"].append(metrics['metrics/recall'])
                if 'metrics/f1' in metrics:
                    self.training_history["f1"].append(metrics['metrics/f1'])
                if 'metrics/mAP50' in metrics:
                    self.training_history["mAP"].append(metrics['metrics/mAP50'])
            
        except Exception as e:
            logger.warning(f"Could not extract training metrics: {e}")
    
    def _save_training_history(self) -> None:
        """Save training history to file"""
        history_path = MODELS_DIR / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def evaluate(self, dataset_yaml: str) -> Dict:
        """Evaluate the trained model"""
        if self.model is None:
            logger.error("No model loaded for evaluation")
            return {}
        
        logger.info("Evaluating model...")
        
        try:
            # Run validation
            results = self.model.val(data=dataset_yaml)
            
            # Extract evaluation metrics
            eval_metrics = self._extract_evaluation_metrics(results)
            
            # Log metrics to wandb
            if wandb.run:
                wandb.log(eval_metrics)
            
            # Check if target accuracy is achieved
            if eval_metrics.get('mAP50', 0) >= EVALUATION_METRICS["target_accuracy"]:
                logger.info("🎉 Target accuracy achieved!")
            else:
                logger.warning(f"Target accuracy not achieved. Current mAP50: {eval_metrics.get('mAP50', 0):.3f}")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}
    
    def _extract_evaluation_metrics(self, results) -> Dict:
        """Extract evaluation metrics from results"""
        metrics = {}
        
        try:
            # Extract key metrics
            if hasattr(results, 'box'):
                metrics['mAP50'] = results.box.map50
                metrics['mAP50-95'] = results.box.map
                metrics['precision'] = results.box.mp
                metrics['recall'] = results.box.mr
            
            # Calculate F1 score
            if 'precision' in metrics and 'recall' in metrics:
                p, r = metrics['precision'], metrics['recall']
                if p + r > 0:
                    metrics['f1'] = 2 * (p * r) / (p + r)
                else:
                    metrics['f1'] = 0.0
            
            logger.info(f"Evaluation metrics: {metrics}")
            
        except Exception as e:
            logger.warning(f"Could not extract evaluation metrics: {e}")
        
        return metrics
    
    def predict(self, image_path: str, save_results: bool = True) -> Dict:
        """Make predictions on a single image"""
        if self.model is None:
            logger.error("No model loaded for prediction")
            return {}
        
        try:
            # Run prediction
            results = self.model(image_path)
            
            # Extract prediction data
            prediction_data = {
                "image_path": image_path,
                "detections": [],
                "confidence_scores": [],
                "bounding_boxes": []
            }
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        prediction_data["detections"].append({
                            "class": int(cls),
                            "confidence": float(conf),
                            "bbox": box.tolist()
                        })
                        prediction_data["confidence_scores"].append(float(conf))
                        prediction_data["bounding_boxes"].append(box.tolist())
            
            # Save results if requested
            if save_results:
                self._save_prediction_results(image_path, prediction_data)
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {}
    
    def _save_prediction_results(self, image_path: str, prediction_data: Dict) -> None:
        """Save prediction results"""
        try:
            # Create output directory
            output_dir = OUTPUT_DIR / "predictions"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save prediction data as JSON
            image_name = Path(image_path).stem
            json_path = output_dir / f"{image_name}_predictions.json"
            
            with open(json_path, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            
            logger.info(f"Prediction results saved to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving prediction results: {e}")
    
    def plot_training_curves(self) -> None:
        """Plot training curves"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16)
            
            # Loss curves
            if self.training_history["train_loss"] and self.training_history["val_loss"]:
                axes[0, 0].plot(self.training_history["train_loss"], label='Train Loss')
                axes[0, 0].plot(self.training_history["val_loss"], label='Val Loss')
                axes[0, 0].set_title('Loss Curves')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Precision curve
            if self.training_history["precision"]:
                axes[0, 1].plot(self.training_history["precision"], label='Precision')
                axes[0, 1].set_title('Precision')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Precision')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Recall curve
            if self.training_history["recall"]:
                axes[1, 0].plot(self.training_history["recall"], label='Recall')
                axes[1, 0].set_title('Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Recall')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # F1 curve
            if self.training_history["f1"]:
                axes[1, 1].plot(self.training_history["f1"], label='F1 Score')
                axes[1, 1].set_title('F1 Score')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('F1 Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = MODELS_DIR / "training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting training curves: {e}")
    
    def optimize_for_accuracy(self) -> None:
        """Optimize model for higher accuracy"""
        logger.info("Optimizing model for accuracy...")
        
        try:
            # Load the best model
            best_model_path = MODELS_DIR / "logo_detection" / "weights" / "best.pt"
            if best_model_path.exists():
                self.model = YOLO(str(best_model_path))
                logger.info("Loaded best model for optimization")
            
            # Apply test-time augmentation
            self._apply_tta_optimization()
            
            # Fine-tune confidence threshold
            self._optimize_confidence_threshold()
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
    
    def _apply_tta_optimization(self) -> None:
        """Apply test-time augmentation for better accuracy"""
        # This would involve running inference multiple times with different augmentations
        # and combining the results
        logger.info("Applying test-time augmentation optimization...")
        # Implementation would go here
    
    def _optimize_confidence_threshold(self) -> None:
        """Optimize confidence threshold for better precision/recall balance"""
        logger.info("Optimizing confidence threshold...")
        # Implementation would go here
    
    def save_model(self, model_path: str = None) -> str:
        """Save the trained model"""
        if self.model is None:
            logger.error("No model to save")
            return ""
        
        if model_path is None:
            model_path = str(MODELS_DIR / "logo_detection_final.pt")
        
        try:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""

def main():
    """Main training function"""
    trainer = LogoDetectionTrainer()
    
    # Load model
    trainer.load_model()
    
    # Check if dataset exists
    dataset_yaml = PROCESSED_DATA_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error(f"Dataset not found at {dataset_yaml}")
        logger.info("Please run data_preprocessor.py first")
        return
    
    # Train model
    logger.info("Starting training process...")
    results = trainer.train(str(dataset_yaml))
    
    # Evaluate model
    logger.info("Evaluating model...")
    eval_metrics = trainer.evaluate(str(dataset_yaml))
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Optimize for accuracy
    trainer.optimize_for_accuracy()
    
    # Save final model
    final_model_path = trainer.save_model()
    
    logger.info("Training process completed!")
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Print final metrics
    if eval_metrics:
        logger.info("Final Evaluation Metrics:")
        for metric, value in eval_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
