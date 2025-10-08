"""
Inference engine for Japanese sport sponsor logo detection
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import logging
import json
import time
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import base64
from io import BytesIO
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDetectionInference:
    """
    Inference engine for real-time Japanese sport sponsor logo detection
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.load_model()
        
    def load_model(self) -> None:
        """Load the trained YOLO model"""
        try:
            if self.model_path is None:
                # Try to find the best model
                best_model_path = MODELS_DIR / "logo_detection" / "weights" / "best.pt"
                if best_model_path.exists():
                    self.model_path = str(best_model_path)
                else:
                    # Fallback to a default model
                    self.model_path = "yolov8n.pt"
            
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded model from: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def detect_logos(self, image: Union[str, np.ndarray], 
                    confidence_threshold: float = None,
                    iou_threshold: float = None) -> Dict:
        """
        Detect logos in an image
        
        Args:
            image: Path to image file or numpy array
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        # Use default thresholds if not provided
        if confidence_threshold is None:
            confidence_threshold = MODEL_CONFIG["confidence_threshold"]
        if iou_threshold is None:
            iou_threshold = MODEL_CONFIG["iou_threshold"]
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=MODEL_CONFIG["max_detections"]
            )
            
            # Process results
            detection_data = self._process_detection_results(results, image)
            
            return detection_data
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return {}
    
    def _process_detection_results(self, results, image_input) -> Dict:
        """Process YOLO detection results"""
        detection_data = {
            "image_path": str(image_input) if isinstance(image_input, (str, Path)) else "numpy_array",
            "detections": [],
            "total_detections": 0,
            "average_confidence": 0.0,
            "processing_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Extract bounding boxes, confidences, and classes
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Process each detection
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        detection = {
                            "id": i,
                            "class": int(cls),
                            "class_name": "logo",
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3]),
                                "width": float(box[2] - box[0]),
                                "height": float(box[3] - box[1])
                            },
                            "center": {
                                "x": float((box[0] + box[2]) / 2),
                                "y": float((box[1] + box[3]) / 2)
                            }
                        }
                        detection_data["detections"].append(detection)
                    
                    # Calculate statistics
                    detection_data["total_detections"] = len(detection_data["detections"])
                    if detection_data["total_detections"] > 0:
                        confidences_list = [d["confidence"] for d in detection_data["detections"]]
                        detection_data["average_confidence"] = np.mean(confidences_list)
            
            detection_data["processing_time"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Error processing detection results: {e}")
        
        return detection_data
    
    def visualize_detections(self, image: Union[str, np.ndarray], 
                           detection_data: Dict, 
                           save_path: str = None) -> np.ndarray:
        """
        Visualize detection results on the image
        
        Args:
            image: Input image (path or numpy array)
            detection_data: Detection results from detect_logos()
            save_path: Optional path to save the visualization
            
        Returns:
            Image with detection visualizations
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
                if img is None:
                    logger.error(f"Could not load image: {image}")
                    return None
            else:
                img = image.copy()
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_rgb)
            
            # Draw bounding boxes
            for detection in detection_data.get("detections", []):
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                
                # Create rectangle
                rect = patches.Rectangle(
                    (bbox["x1"], bbox["y1"]),
                    bbox["width"],
                    bbox["height"],
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add confidence label
                ax.text(
                    bbox["x1"], bbox["y1"] - 5,
                    f"Logo: {confidence:.2f}",
                    fontsize=10,
                    color='red',
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                )
            
            # Set title
            total_detections = detection_data.get("total_detections", 0)
            avg_conf = detection_data.get("average_confidence", 0.0)
            ax.set_title(f"Logo Detection Results - {total_detections} logos found (avg conf: {avg_conf:.2f})")
            ax.axis('off')
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to: {save_path}")
            
            # Convert back to numpy array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return img_array
            
        except Exception as e:
            logger.error(f"Error visualizing detections: {e}")
            return None
    
    def batch_detect(self, image_paths: List[str], 
                    output_dir: str = None) -> List[Dict]:
        """
        Detect logos in multiple images
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results and visualizations
            
        Returns:
            List of detection results for each image
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Detect logos
                detection_data = self.detect_logos(image_path)
                all_results.append(detection_data)
                
                # Save visualization if output directory provided
                if output_dir and detection_data.get("total_detections", 0) > 0:
                    vis_path = output_path / f"detection_{i:04d}_{Path(image_path).stem}.png"
                    self.visualize_detections(image_path, detection_data, str(vis_path))
                
                # Save detection data as JSON
                if output_dir:
                    json_path = output_path / f"detection_{i:04d}_{Path(image_path).stem}.json"
                    with open(json_path, 'w') as f:
                        json.dump(detection_data, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                all_results.append({"error": str(e), "image_path": image_path})
        
        logger.info(f"Batch processing completed. Processed {len(all_results)} images.")
        return all_results
    
    def detect_video(self, video_path: str, output_path: str = None,
                    frame_skip: int = 1) -> None:
        """
        Detect logos in video frames
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            frame_skip: Process every Nth frame
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_frames = 0
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Video properties: {width}x{height} @ {fps}fps")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_count % frame_skip == 0:
                    # Detect logos
                    detection_data = self.detect_logos(frame)
                    
                    # Draw detections on frame
                    if detection_data.get("total_detections", 0) > 0:
                        for detection in detection_data["detections"]:
                            bbox = detection["bbox"]
                            confidence = detection["confidence"]
                            
                            # Draw bounding box
                            cv2.rectangle(
                                frame,
                                (int(bbox["x1"]), int(bbox["y1"])),
                                (int(bbox["x2"]), int(bbox["y2"])),
                                (0, 0, 255), 2
                            )
                            
                            # Draw confidence
                            cv2.putText(
                                frame,
                                f"Logo: {confidence:.2f}",
                                (int(bbox["x1"]), int(bbox["y1"] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2
                            )
                    
                    processed_frames += 1
                
                # Write frame to output video
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames...")
            
            # Cleanup
            cap.release()
            if output_path:
                out.release()
            
            logger.info(f"Video processing completed. Processed {processed_frames} frames.")
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        try:
            return {
                "model_path": self.model_path,
                "model_type": type(self.model).__name__,
                "device": str(next(self.model.model.parameters()).device),
                "input_size": MODEL_CONFIG["input_size"],
                "confidence_threshold": MODEL_CONFIG["confidence_threshold"],
                "iou_threshold": MODEL_CONFIG["iou_threshold"]
            }
        except Exception as e:
            return {"error": f"Could not get model info: {e}"}

def main():
    """Main inference function for testing"""
    # Initialize inference engine
    inference = LogoDetectionInference()
    
    # Test with sample images
    test_images_dir = Path("test_images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        if test_images:
            logger.info(f"Found {len(test_images)} test images")
            
            # Process images
            results = inference.batch_detect(
                [str(img) for img in test_images],
                output_dir="output/detections"
            )
            
            # Print summary
            total_detections = sum(r.get("total_detections", 0) for r in results)
            logger.info(f"Total logos detected: {total_detections}")
            
        else:
            logger.info("No test images found in test_images directory")
    else:
        logger.info("No test_images directory found")

if __name__ == "__main__":
    main()
