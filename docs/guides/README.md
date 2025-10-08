# Japanese Sport Sponsor Logo Detection AI

A comprehensive AI system for detecting Japanese sport sponsor logos with over 90% accuracy using YOLO-based computer vision.

## 🎯 Features

- **High Accuracy**: Designed to achieve 90%+ detection rate
- **Real-time Detection**: Fast inference for live video streams
- **Comprehensive Data Pipeline**: Automated data collection and preprocessing
- **Advanced Optimization**: Multiple strategies for accuracy improvement
- **Easy to Use**: Simple command-line interface

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd japanese-sport-logo-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the full pipeline:
```bash
python main.py --mode full
```

### Basic Usage

#### Data Collection
```bash
python main.py --mode collect --synthetic 200
```

#### Data Preprocessing
```bash
python main.py --mode preprocess
```

#### Model Training
```bash
python main.py --mode train --model yolov8n.pt
```

#### Single Image Inference
```bash
python main.py --mode inference --image path/to/image.jpg --output_dir results/
```

#### Batch Processing
```bash
python main.py --mode batch --image_dir path/to/images/ --output_dir results/
```

## 📁 Project Structure

```
japanese-sport-logo-detection/
├── main.py                     # Main application entry point
├── config.py                   # Configuration settings
├── data_collector.py           # Data collection from websites
├── data_preprocessor.py        # Data preprocessing pipeline
├── model_trainer.py            # YOLO model training
├── inference_engine.py         # Real-time inference
├── evaluation_metrics.py       # Performance evaluation
├── optimization_strategies.py  # Advanced optimization
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Data directories
│   ├── raw/                   # Raw collected data
│   ├── processed/             # Processed training data
│   └── annotations/           # YOLO format annotations
├── models/                     # Trained models
├── logs/                       # Training logs
└── output/                     # Inference results
```

## 🔧 Configuration

The system is highly configurable through `config.py`:

### Model Configuration
- **Model Type**: YOLOv8 (nano, small, medium, large)
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.5
- **IoU Threshold**: 0.45

### Training Configuration
- **Epochs**: 100
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Device**: Auto-detect (CUDA/CPU)

### Data Augmentation
- Horizontal/Vertical Flip
- Rotation (±15°)
- Brightness/Contrast Adjustment
- Gaussian Blur and Noise
- Coarse Dropout

## 📊 Performance Optimization

### Strategies for 90%+ Accuracy

1. **Data Quality**
   - Collect diverse Japanese sport logos
   - Generate synthetic training data
   - Apply advanced data augmentation

2. **Model Architecture**
   - Use YOLOv8 with optimal configuration
   - Implement ensemble methods
   - Apply test-time augmentation

3. **Training Optimization**
   - Hyperparameter tuning
   - Advanced loss functions (Focal Loss, IoU Loss)
   - Learning rate scheduling

4. **Post-processing**
   - Non-maximum suppression optimization
   - Confidence threshold tuning
   - Multi-scale detection

## 🎮 Usage Examples

### Training a New Model

```python
from main import JapaneseSportLogoDetectionApp

app = JapaneseSportLogoDetectionApp()

# Collect data
app.collect_data(num_synthetic=500)

# Preprocess data
app.preprocess_data()

# Train model
app.train_model("yolov8n.pt")

# Optimize for accuracy
app.optimize_model()
```

### Running Inference

```python
from inference_engine import LogoDetectionInference

# Initialize inference engine
inference = LogoDetectionInference("path/to/model.pt")

# Detect logos in image
results = inference.detect_logos("image.jpg")

# Visualize results
inference.visualize_detections("image.jpg", results, "output.png")
```

### Batch Processing

```python
# Process multiple images
results = inference.batch_detect(
    ["image1.jpg", "image2.jpg", "image3.jpg"],
    output_dir="results/"
)
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision
- **Detection Accuracy**: Overall detection performance

## 🔍 Advanced Features

### Test-Time Augmentation (TTA)
- Multiple augmented versions of input images
- Ensemble predictions for better accuracy
- Configurable augmentation strategies

### Active Learning
- Uncertainty-based sample selection
- Continuous model improvement
- Automated retraining pipeline

### Speed Optimization
- Model quantization (INT8)
- TensorRT optimization
- ONNX export for deployment

## 🛠️ Customization

### Adding New Sport Categories
Edit `config.py` to add new sport categories:

```python
SPORT_CATEGORIES = [
    "baseball", "soccer", "basketball", "volleyball", 
    "tennis", "golf", "sumo", "martial_arts", "swimming",
    "track_field", "gymnastics", "figure_skating",
    "your_new_sport"  # Add here
]
```

### Custom Data Sources
Modify `data_collector.py` to add new data sources:

```python
websites = [
    "https://your-sport-website.com/",
    # Add more websites
]
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Ultralytics YOLO
- CUDA (recommended for training)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.py`
   - Use smaller model (yolov8n instead of yolov8l)

2. **Low Detection Accuracy**
   - Collect more training data
   - Run optimization pipeline
   - Adjust confidence thresholds

3. **Slow Inference**
   - Enable model quantization
   - Use TensorRT optimization
   - Reduce input image size

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- OpenCV community
- PyTorch team
- Japanese sport organizations for data sources
