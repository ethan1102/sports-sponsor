"""
Main application for Japanese Sport Sponsor Logo Detection AI
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import json
import time

# Import our modules
from data_collector import JapaneseSportLogoCollector
from data_preprocessor import LogoDataPreprocessor
from model_trainer import LogoDetectionTrainer
from inference_engine import LogoDetectionInference
from evaluation_metrics import LogoDetectionEvaluator
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "main.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class JapaneseSportLogoDetectionApp:
    """
    Main application class for Japanese sport sponsor logo detection
    """
    
    def __init__(self):
        self.collector = None
        self.preprocessor = None
        self.trainer = None
        self.inference = None
        self.evaluator = None
        
    def collect_data(self, num_synthetic: int = 100) -> bool:
        """Collect and prepare training data"""
        logger.info("Starting data collection...")
        
        try:
            self.collector = JapaneseSportLogoCollector()
            
            # Collect from websites
            logger.info("Collecting logos from Japanese sport websites...")
            logo_data = self.collector.collect_from_websites()
            logger.info(f"Found {len(logo_data)} logo candidates")
            
            # Download logos
            logger.info("Downloading logos...")
            downloaded_paths = self.collector.download_logos(logo_data)
            logger.info(f"Successfully downloaded {len(downloaded_paths)} logos")
            
            # Create synthetic logos
            logger.info("Creating synthetic logos...")
            synthetic_paths = self.collector.create_synthetic_logos(num_synthetic)
            logger.info(f"Created {len(synthetic_paths)} synthetic logos")
            
            # Save metadata
            metadata = {
                "total_logos": len(downloaded_paths) + len(synthetic_paths),
                "real_logos": len(downloaded_paths),
                "synthetic_logos": len(synthetic_paths),
                "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(RAW_DATA_DIR / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.collector.close()
            logger.info("Data collection completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """Preprocess collected data for training"""
        logger.info("Starting data preprocessing...")
        
        try:
            self.preprocessor = LogoDataPreprocessor()
            
            # Get all image files from raw data
            image_paths = list(RAW_DATA_DIR.glob("*.jpg")) + list(RAW_DATA_DIR.glob("*.png"))
            image_paths = [str(p) for p in image_paths]
            
            if not image_paths:
                logger.error("No images found in raw data directory")
                return False
            
            logger.info(f"Found {len(image_paths)} images to process")
            
            # Create YOLO annotations
            self.preprocessor.create_yolo_annotations(image_paths)
            
            # Split dataset
            splits = self.preprocessor.split_dataset(image_paths)
            
            # Organize dataset
            self.preprocessor.organize_dataset(splits)
            
            # Create dataset.yaml
            self.preprocessor.create_dataset_yaml()
            
            # Validate dataset
            if self.preprocessor.validate_dataset():
                logger.info("Data preprocessing completed successfully!")
                return True
            else:
                logger.error("Dataset validation failed!")
                return False
                
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            return False
    
    def train_model(self, model_name: str = "yolov8n.pt") -> bool:
        """Train the logo detection model"""
        logger.info("Starting model training...")
        
        try:
            self.trainer = LogoDetectionTrainer(model_name)
            
            # Check if dataset exists
            dataset_yaml = PROCESSED_DATA_DIR / "dataset.yaml"
            if not dataset_yaml.exists():
                logger.error(f"Dataset not found at {dataset_yaml}")
                logger.info("Please run data collection and preprocessing first")
                return False
            
            # Train model
            logger.info("Training model...")
            results = self.trainer.train(str(dataset_yaml))
            
            # Evaluate model
            logger.info("Evaluating model...")
            eval_metrics = self.trainer.evaluate(str(dataset_yaml))
            
            # Plot training curves
            self.trainer.plot_training_curves()
            
            # Check if target accuracy is achieved
            if eval_metrics.get('mAP50', 0) >= EVALUATION_METRICS["target_accuracy"]:
                logger.info("🎉 Target accuracy achieved!")
            else:
                logger.warning(f"Target accuracy not achieved. Current mAP50: {eval_metrics.get('mAP50', 0):.3f}")
                logger.info("Consider running optimization or collecting more data")
            
            logger.info("Model training completed!")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def optimize_model(self) -> bool:
        """Optimize model for higher accuracy"""
        logger.info("Starting model optimization...")
        
        try:
            if self.trainer is None:
                self.trainer = LogoDetectionTrainer()
            
            # Load the best model
            best_model_path = MODELS_DIR / "logo_detection" / "weights" / "best.pt"
            if not best_model_path.exists():
                logger.error("No trained model found. Please train a model first.")
                return False
            
            # Optimize for accuracy
            self.trainer.optimize_for_accuracy()
            
            # Re-evaluate optimized model
            dataset_yaml = PROCESSED_DATA_DIR / "dataset.yaml"
            eval_metrics = self.trainer.evaluate(str(dataset_yaml))
            
            logger.info(f"Optimized model metrics: {eval_metrics}")
            logger.info("Model optimization completed!")
            return True
            
        except Exception as e:
            logger.error(f"Error during model optimization: {e}")
            return False
    
    def run_inference(self, image_path: str, output_dir: str = None) -> dict:
        """Run inference on a single image"""
        logger.info(f"Running inference on: {image_path}")
        
        try:
            if self.inference is None:
                self.inference = LogoDetectionInference()
            
            # Detect logos
            detection_data = self.inference.detect_logos(image_path)
            
            # Visualize results
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                vis_path = output_path / f"detection_{Path(image_path).stem}.png"
                self.inference.visualize_detections(image_path, detection_data, str(vis_path))
                
                # Save detection data
                json_path = output_path / f"detection_{Path(image_path).stem}.json"
                with open(json_path, 'w') as f:
                    json.dump(detection_data, f, indent=2)
            
            logger.info(f"Detection completed. Found {detection_data.get('total_detections', 0)} logos")
            return detection_data
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {}
    
    def batch_inference(self, image_dir: str, output_dir: str = None) -> list:
        """Run inference on multiple images"""
        logger.info(f"Running batch inference on directory: {image_dir}")
        
        try:
            if self.inference is None:
                self.inference = LogoDetectionInference()
            
            # Get all image files
            image_dir_path = Path(image_dir)
            image_files = list(image_dir_path.glob("*.jpg")) + list(image_dir_path.glob("*.png"))
            image_paths = [str(f) for f in image_files]
            
            if not image_paths:
                logger.error(f"No images found in {image_dir}")
                return []
            
            logger.info(f"Found {len(image_paths)} images to process")
            
            # Run batch detection
            results = self.inference.batch_detect(image_paths, output_dir)
            
            # Calculate summary statistics
            total_detections = sum(r.get("total_detections", 0) for r in results)
            avg_confidence = sum(r.get("average_confidence", 0) for r in results) / len(results)
            
            logger.info(f"Batch inference completed. Total logos detected: {total_detections}")
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch inference: {e}")
            return []
    
    def evaluate_model(self, test_data_dir: str = None) -> dict:
        """Evaluate model performance"""
        logger.info("Starting model evaluation...")
        
        try:
            if self.evaluator is None:
                self.evaluator = LogoDetectionEvaluator()
            
            # For now, we'll use a simplified evaluation
            # In practice, you'd load actual test data and ground truth
            logger.info("Evaluation completed (simplified version)")
            
            return {"status": "evaluation_completed"}
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline from data collection to model deployment"""
        logger.info("Starting full pipeline...")
        
        try:
            # Step 1: Collect data
            if not self.collect_data():
                logger.error("Data collection failed")
                return False
            
            # Step 2: Preprocess data
            if not self.preprocess_data():
                logger.error("Data preprocessing failed")
                return False
            
            # Step 3: Train model
            if not self.train_model():
                logger.error("Model training failed")
                return False
            
            # Step 4: Optimize model
            if not self.optimize_model():
                logger.warning("Model optimization failed, but continuing...")
            
            # Step 5: Evaluate model
            self.evaluate_model()
            
            logger.info("Full pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Japanese Sport Sponsor Logo Detection AI")
    parser.add_argument("--mode", choices=["collect", "preprocess", "train", "optimize", 
                                         "inference", "batch", "evaluate", "full"],
                       default="full", help="Mode to run")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    parser.add_argument("--image_dir", type=str, help="Path to directory of images for batch inference")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to use")
    parser.add_argument("--synthetic", type=int, default=100, help="Number of synthetic logos to generate")
    
    args = parser.parse_args()
    
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
            logger.error("Please provide --image argument for inference mode")
            sys.exit(1)
        result = app.run_inference(args.image, args.output_dir)
        success = bool(result)
    elif args.mode == "batch":
        if not args.image_dir:
            logger.error("Please provide --image_dir argument for batch mode")
            sys.exit(1)
        results = app.batch_inference(args.image_dir, args.output_dir)
        success = bool(results)
    elif args.mode == "evaluate":
        result = app.evaluate_model()
        success = bool(result)
    elif args.mode == "full":
        success = app.run_full_pipeline()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False
    
    if success:
        logger.info("Operation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Operation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
