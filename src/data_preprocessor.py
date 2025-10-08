"""
Data preprocessing module for Japanese sport sponsor logo detection
"""
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
import shutil
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDataPreprocessor:
    """
    Preprocesses logo data for YOLO training
    """
    
    def __init__(self):
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline"""
        return A.Compose([
            A.HorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip"]),
            A.VerticalFlip(p=AUGMENTATION_CONFIG["vertical_flip"]),
            A.Rotate(limit=AUGMENTATION_CONFIG["rotation"], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=AUGMENTATION_CONFIG["brightness_contrast"],
                contrast_limit=AUGMENTATION_CONFIG["brightness_contrast"],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(255 * AUGMENTATION_CONFIG["hue_saturation"]),
                sat_shift_limit=int(255 * AUGMENTATION_CONFIG["hue_saturation"]),
                val_shift_limit=int(255 * AUGMENTATION_CONFIG["hue_saturation"]),
                p=0.3
            ),
            A.GaussianBlur(blur_limit=3, p=AUGMENTATION_CONFIG["blur"]),
            A.GaussNoise(var_limit=(10.0, 50.0), p=AUGMENTATION_CONFIG["noise"]),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=AUGMENTATION_CONFIG["cutout"]
            ),
            A.Resize(MODEL_CONFIG["input_size"], MODEL_CONFIG["input_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def create_yolo_annotations(self, image_paths: List[str]) -> None:
        """Create YOLO format annotations for all images"""
        logger.info("Creating YOLO annotations...")
        
        for image_path in image_paths:
            try:
                # Load image to get dimensions
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                    
                height, width = image.shape[:2]
                
                # Create bounding box for the entire image (logo detection)
                # In a real scenario, you would have precise bounding box coordinates
                # For now, we'll assume the entire image is a logo
                bbox = self._create_logo_bbox(image)
                
                if bbox is None:
                    continue
                
                # Convert to YOLO format (normalized coordinates)
                x_center, y_center, bbox_width, bbox_height = bbox
                x_center_norm = x_center / width
                y_center_norm = y_center / height
                width_norm = bbox_width / width
                height_norm = bbox_height / height
                
                # Create annotation file
                annotation_path = Path(image_path).with_suffix('.txt')
                with open(annotation_path, 'w') as f:
                    f.write(f"0 {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                
                logger.debug(f"Created annotation: {annotation_path}")
                
            except Exception as e:
                logger.error(f"Error creating annotation for {image_path}: {e}")
    
    def _create_logo_bbox(self, image: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Create bounding box for logo in image"""
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to find logo regions
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # If no contours found, use entire image
                h, w = image.shape[:2]
                return w/2, h/2, w, h
            
            # Find the largest contour (likely the logo)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Ensure minimum size
            min_w, min_h = LOGO_CHARACTERISTICS["min_size"]
            if w < min_w or h < min_h:
                h, w = image.shape[:2]
                return w/2, h/2, w, h
            
            # Return center coordinates and dimensions
            return x + w/2, y + h/2, w, h
            
        except Exception as e:
            logger.error(f"Error creating logo bbox: {e}")
            return None
    
    def split_dataset(self, image_paths: List[str], train_ratio: float = 0.7, 
                     val_ratio: float = 0.2, test_ratio: float = 0.1) -> Dict[str, List[str]]:
        """Split dataset into train/val/test sets"""
        logger.info("Splitting dataset...")
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        # First split: train + val vs test
        train_val_paths, test_paths = train_test_split(
            image_paths, test_size=test_ratio, random_state=42
        )
        
        # Second split: train vs val
        train_paths, val_paths = train_test_split(
            train_val_paths, test_size=val_ratio/(train_ratio + val_ratio), random_state=42
        )
        
        splits = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths
        }
        
        logger.info(f"Dataset split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        return splits
    
    def organize_dataset(self, splits: Dict[str, List[str]]) -> None:
        """Organize dataset into train/val/test directories"""
        logger.info("Organizing dataset...")
        
        # Create directories
        for split in ["train", "val", "test"]:
            (PROCESSED_DATA_DIR / split).mkdir(parents=True, exist_ok=True)
            (PROCESSED_DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
            (PROCESSED_DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy files to appropriate directories
        for split, paths in splits.items():
            for image_path in paths:
                try:
                    # Copy image
                    image_name = Path(image_path).name
                    dest_image = PROCESSED_DATA_DIR / split / "images" / image_name
                    shutil.copy2(image_path, dest_image)
                    
                    # Copy annotation
                    annotation_path = Path(image_path).with_suffix('.txt')
                    if annotation_path.exists():
                        dest_annotation = PROCESSED_DATA_DIR / split / "labels" / annotation_path.name
                        shutil.copy2(annotation_path, dest_annotation)
                    
                except Exception as e:
                    logger.error(f"Error organizing {image_path}: {e}")
    
    def create_dataset_yaml(self) -> None:
        """Create dataset.yaml file for YOLO training"""
        yaml_content = f"""
# Japanese Sport Sponsor Logo Detection Dataset
path: {PROCESSED_DATA_DIR.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {MODEL_CONFIG['num_classes']}
names: ['logo']
"""
        
        yaml_path = PROCESSED_DATA_DIR / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created dataset.yaml at {yaml_path}")
    
    def preprocess_images(self, image_paths: List[str]) -> List[str]:
        """Preprocess images for training"""
        logger.info("Preprocessing images...")
        
        processed_paths = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply augmentation
                augmented = self.augmentation_pipeline(image=image)
                processed_image = augmented["image"]
                
                # Save processed image
                processed_path = Path(image_path).parent / f"processed_{Path(image_path).name}"
                processed_image_np = processed_image.permute(1, 2, 0).numpy()
                processed_image_np = (processed_image_np * 255).astype(np.uint8)
                
                cv2.imwrite(str(processed_path), cv2.cvtColor(processed_image_np, cv2.COLOR_RGB2BGR))
                processed_paths.append(str(processed_path))
                
            except Exception as e:
                logger.error(f"Error preprocessing {image_path}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed_paths)} images")
        return processed_paths
    
    def validate_dataset(self) -> bool:
        """Validate the processed dataset"""
        logger.info("Validating dataset...")
        
        for split in ["train", "val", "test"]:
            images_dir = PROCESSED_DATA_DIR / split / "images"
            labels_dir = PROCESSED_DATA_DIR / split / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                logger.error(f"Missing directories for {split} split")
                return False
            
            # Check if number of images matches labels
            image_files = list(images_dir.glob("*"))
            label_files = list(labels_dir.glob("*.txt"))
            
            if len(image_files) != len(label_files):
                logger.error(f"Mismatch between images and labels in {split} split")
                return False
            
            # Validate annotation format
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                logger.error(f"Invalid annotation format in {label_file}")
                                return False
                            
                            # Check if values are in valid range [0, 1]
                            for part in parts[1:]:
                                if not 0 <= float(part) <= 1:
                                    logger.error(f"Invalid coordinate value in {label_file}")
                                    return False
                except Exception as e:
                    logger.error(f"Error validating {label_file}: {e}")
                    return False
        
        logger.info("Dataset validation passed!")
        return True

def main():
    """Main preprocessing function"""
    preprocessor = LogoDataPreprocessor()
    
    # Get all image files from raw data
    image_paths = list(RAW_DATA_DIR.glob("*.jpg")) + list(RAW_DATA_DIR.glob("*.png"))
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        logger.error("No images found in raw data directory")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Create YOLO annotations
    preprocessor.create_yolo_annotations(image_paths)
    
    # Split dataset
    splits = preprocessor.split_dataset(image_paths)
    
    # Organize dataset
    preprocessor.organize_dataset(splits)
    
    # Create dataset.yaml
    preprocessor.create_dataset_yaml()
    
    # Validate dataset
    if preprocessor.validate_dataset():
        logger.info("Dataset preprocessing completed successfully!")
    else:
        logger.error("Dataset validation failed!")

if __name__ == "__main__":
    main()
