"""
Configuration file for Japanese Sport Sponsor Logo Detection AI
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from src/
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Model configuration
MODEL_CONFIG = {
    "model_name": "yolov8n",  # Start with nano, can upgrade to larger models
    "input_size": 640,
    "num_classes": 1,  # Single class for logo detection
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100
}

# Training configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.01,
    "weight_decay": 0.0005,
    "patience": 20,  # Early stopping patience
    "save_period": 10,  # Save checkpoint every N epochs
    "device": "cuda" if os.system("nvidia-smi") == 0 else "cpu"
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.2,
    "rotation": 15,  # degrees
    "brightness_contrast": 0.2,
    "hue_saturation": 0.1,
    "blur": 0.1,
    "noise": 0.1,
    "cutout": 0.1
}

# Japanese sport categories to focus on
SPORT_CATEGORIES = [
    "baseball", "soccer", "basketball", "volleyball", 
    "tennis", "golf", "sumo", "martial_arts", "swimming",
    "track_field", "gymnastics", "figure_skating"
]

# Common Japanese sponsor logo characteristics
LOGO_CHARACTERISTICS = {
    "min_size": (20, 20),  # Minimum logo size in pixels
    "max_size": (200, 200),  # Maximum logo size in pixels
    "aspect_ratios": [1.0, 1.5, 2.0, 0.5, 0.67],  # Common aspect ratios
    "color_spaces": ["RGB", "BGR", "GRAY"],
    "formats": ["PNG", "JPEG", "JPG"]
}

# Evaluation metrics
EVALUATION_METRICS = {
    "target_accuracy": 0.90,  # 90% detection rate target
    "precision_threshold": 0.85,
    "recall_threshold": 0.90,
    "f1_threshold": 0.87,
    "mAP_threshold": 0.85
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "training.log"
}
