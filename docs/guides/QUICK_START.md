# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Installation
```bash
# Install dependencies
python scripts/setup/install.py

# Or manually
pip install -r scripts/setup/requirements.txt
```

### 2. Collect Data
```bash
# Collect Japanese sport logos
python src/main.py --mode collect --synthetic 100
```

### 3. Annotate Data
```bash
# Open annotation tool
python tools/annotation/run_annotation_tool.py

# Or batch annotation
python tools/annotation/run_batch_annotation.py
```

### 4. Train Model
```bash
# Full training pipeline
python scripts/training/train_model.py

# Or step by step
python src/main.py --mode preprocess
python src/main.py --mode train
python src/main.py --mode optimize
```

### 5. Run Inference
```bash
# Single image
python scripts/inference/run_inference.py --image path/to/image.jpg

# Batch processing
python scripts/inference/run_inference.py --image_dir path/to/images/
```

## 📁 Project Structure

```
japanese-sport-logo-detection/
├── src/                          # Source code
│   ├── main.py                   # Main application
│   ├── config.py                 # Configuration
│   ├── data_collector.py         # Data collection
│   ├── data_preprocessor.py      # Data preprocessing
│   ├── model_trainer.py          # Model training
│   ├── inference_engine.py       # Inference engine
│   ├── evaluation_metrics.py     # Evaluation
│   ├── optimization_strategies.py # Optimization
│   ├── annotation_tool.py        # Annotation tool
│   ├── batch_annotation_tool.py  # Batch annotation
│   └── annotation_validator.py   # Validation
├── data/                         # Data directories
│   ├── raw/                      # Raw images
│   ├── processed/                # Processed data
│   ├── annotations/              # Annotation files
│   └── test_images/              # Test images
├── models/                       # Model files
│   ├── checkpoints/              # Training checkpoints
│   └── exports/                  # Exported models
├── tools/                        # Annotation tools
│   ├── annotation/               # Annotation tools
│   ├── validation/               # Validation tools
│   └── evaluation/               # Evaluation tools
├── scripts/                      # Utility scripts
│   ├── setup/                    # Setup scripts
│   ├── training/                 # Training scripts
│   ├── inference/                # Inference scripts
│   └── evaluation/               # Evaluation scripts
├── docs/                         # Documentation
│   ├── guides/                   # User guides
│   └── api/                      # API documentation
├── tests/                        # Test files
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── logs/                         # Log files
└── output/                       # Output files
    ├── detections/               # Detection results
    └── evaluation/               # Evaluation results
```

## 🎯 Key Features

- **90%+ Detection Accuracy**: Optimized for high performance
- **Easy Annotation**: Interactive GUI tools
- **Batch Processing**: Handle large datasets efficiently
- **Quality Control**: Built-in validation and metrics
- **Real-time Inference**: Fast detection on images and videos

## 🔧 Configuration

Edit `src/config.py` to customize:
- Model parameters
- Training settings
- Data augmentation
- Evaluation metrics

## 📊 Workflow

1. **Data Collection** → Gather Japanese sport images
2. **Annotation** → Label logos with bounding boxes
3. **Training** → Train YOLO model
4. **Optimization** → Fine-tune for 90%+ accuracy
5. **Inference** → Detect logos in new images

## 🆘 Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce batch size in config
- **Low Accuracy**: Collect more training data
- **Slow Inference**: Enable model quantization

### Getting Help
- Check logs in `logs/` directory
- Run validation: `python tools/validation/validate_annotations.py`
- See full documentation in `docs/`

## 🎉 Success!

Once you see "Target accuracy achieved!" in the logs, your model is ready for production use!
