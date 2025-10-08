# Project Structure

## 📁 Japanese Sport Logo Detection AI

This document explains the organized folder structure of the Japanese Sport Logo Detection AI project.

```
japanese-sport-logo-detection/
├── 📁 src/                          # Source Code
│   ├── main.py                      # Main application entry point
│   ├── config.py                    # Configuration settings
│   ├── data_collector.py            # Data collection from websites
│   ├── data_preprocessor.py         # Data preprocessing pipeline
│   ├── model_trainer.py             # YOLO model training
│   ├── inference_engine.py          # Real-time inference
│   ├── evaluation_metrics.py        # Performance evaluation
│   ├── optimization_strategies.py   # Advanced optimization
│   ├── annotation_tool.py           # Interactive annotation tool
│   ├── batch_annotation_tool.py     # Batch annotation tool
│   └── annotation_validator.py      # Annotation validation
│
├── 📁 data/                         # Data Directories
│   ├── raw/                         # Raw collected images
│   ├── processed/                   # Processed training data
│   │   ├── train/                   # Training images
│   │   ├── val/                     # Validation images
│   │   └── test/                    # Test images
│   ├── annotations/                 # Annotation files (JSON)
│   └── test_images/                 # Test images for inference
│
├── 📁 models/                       # Model Files
│   ├── checkpoints/                 # Training checkpoints
│   │   ├── best.pt                  # Best model weights
│   │   ├── last.pt                  # Latest model weights
│   │   └── epoch_*.pt               # Epoch-specific weights
│   └── exports/                     # Exported models
│       ├── onnx/                    # ONNX format models
│       ├── tensorrt/                # TensorRT optimized models
│       └── quantized/               # Quantized models
│
├── 📁 tools/                        # Annotation & Validation Tools
│   ├── annotation/                  # Annotation tools
│   │   ├── run_annotation_tool.py   # Launch annotation tool
│   │   └── run_batch_annotation.py  # Launch batch annotation
│   ├── validation/                  # Validation tools
│   │   └── validate_annotations.py  # Annotation quality validation
│   └── evaluation/                  # Evaluation tools
│       └── evaluate_model.py        # Model performance evaluation
│
├── 📁 scripts/                      # Utility Scripts
│   ├── setup/                       # Setup & Installation
│   │   ├── install.py               # Installation script
│   │   └── requirements.txt         # Python dependencies
│   ├── training/                    # Training scripts
│   │   └── train_model.py           # Training pipeline
│   ├── inference/                   # Inference scripts
│   │   └── run_inference.py         # Inference pipeline
│   └── evaluation/                  # Evaluation scripts
│       └── evaluate_model.py        # Evaluation pipeline
│
├── 📁 docs/                         # Documentation
│   ├── guides/                      # User Guides
│   │   ├── QUICK_START.md           # Quick start guide
│   │   ├── ANNOTATION_GUIDE.md      # Annotation guide
│   │   ├── TRAINING_GUIDE.md        # Training guide
│   │   └── README.md                # Main documentation
│   └── api/                         # API Documentation
│       └── (Future API docs)
│
├── 📁 tests/                        # Test Files
│   ├── unit/                        # Unit tests
│   │   ├── test_data_collector.py   # Data collection tests
│   │   ├── test_preprocessor.py     # Preprocessing tests
│   │   └── test_inference.py        # Inference tests
│   └── integration/                 # Integration tests
│       ├── test_full_pipeline.py    # End-to-end tests
│       └── test_annotation_tools.py # Annotation tool tests
│
├── 📁 logs/                         # Log Files
│   ├── training.log                 # Training logs
│   ├── inference.log                # Inference logs
│   └── main.log                     # Main application logs
│
├── 📁 output/                       # Output Files
│   ├── detections/                  # Detection results
│   │   ├── images/                  # Images with detections
│   │   ├── json/                    # Detection data (JSON)
│   │   └── videos/                  # Video detection results
│   └── evaluation/                  # Evaluation results
│       ├── reports/                 # Evaluation reports
│       ├── plots/                   # Performance plots
│       └── metrics/                 # Detailed metrics
│
└── 📄 PROJECT_STRUCTURE.md          # This file
```

## 🎯 Directory Purposes

### Source Code (`src/`)
Contains all the core Python modules:
- **Main application**: Entry point and orchestration
- **Data pipeline**: Collection, preprocessing, and validation
- **Model components**: Training, inference, and optimization
- **Tools**: Annotation and evaluation utilities

### Data (`data/`)
Organized data storage:
- **Raw**: Original collected images
- **Processed**: Preprocessed training data in YOLO format
- **Annotations**: Human-annotated bounding boxes
- **Test images**: Images for testing and validation

### Models (`models/`)
Model storage and management:
- **Checkpoints**: Training snapshots and best models
- **Exports**: Optimized models for deployment

### Tools (`tools/`)
User-friendly interfaces:
- **Annotation**: GUI tools for data labeling
- **Validation**: Quality control and validation
- **Evaluation**: Performance assessment tools

### Scripts (`scripts/`)
Ready-to-run scripts:
- **Setup**: Installation and configuration
- **Training**: Model training pipelines
- **Inference**: Detection and prediction
- **Evaluation**: Performance measurement

### Documentation (`docs/`)
Comprehensive guides:
- **Quick Start**: Get running in 5 minutes
- **Annotation Guide**: How to label data
- **Training Guide**: Model training process
- **API Docs**: Technical documentation

### Tests (`tests/`)
Quality assurance:
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end testing

### Logs (`logs/`)
System monitoring:
- **Training logs**: Model training progress
- **Inference logs**: Detection operations
- **Application logs**: General system logs

### Output (`output/`)
Results and reports:
- **Detections**: Logo detection results
- **Evaluation**: Performance metrics and reports

## 🚀 Getting Started

1. **Installation**: `python scripts/setup/install.py`
2. **Quick Start**: Follow `docs/guides/QUICK_START.md`
3. **Annotation**: Use `tools/annotation/run_annotation_tool.py`
4. **Training**: Run `scripts/training/train_model.py`
5. **Inference**: Use `scripts/inference/run_inference.py`

## 📋 File Naming Conventions

- **Python files**: `snake_case.py`
- **Configuration**: `config.py`
- **Main scripts**: `main.py` or descriptive names
- **Documentation**: `UPPER_CASE.md`
- **Data files**: `descriptive_name.ext`
- **Model files**: `model_name.pt` or `checkpoint_epoch.pt`

## 🔧 Customization

- **Configuration**: Edit `src/config.py`
- **Data sources**: Modify `src/data_collector.py`
- **Model architecture**: Update `src/model_trainer.py`
- **Annotation tools**: Customize `src/annotation_tool.py`

This structure makes the project easy to navigate, understand, and maintain while keeping related files organized together.
