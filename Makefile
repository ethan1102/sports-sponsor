# Japanese Sport Logo Detection AI - Makefile
# Easy commands for common tasks

.PHONY: help install setup collect annotate train infer evaluate clean test

# Default target
help:
	@echo "🏆 Japanese Sport Logo Detection AI"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup project directories"
	@echo "  make collect     - Collect training data"
	@echo "  make annotate    - Launch annotation tool"
	@echo "  make train       - Train the model"
	@echo "  make infer       - Run inference on test images"
	@echo "  make evaluate    - Evaluate model performance"
	@echo "  make clean       - Clean temporary files"
	@echo "  make test        - Run tests"
	@echo ""
	@echo "Quick start:"
	@echo "  make install && make setup && make collect && make train"

# Installation
install:
	@echo "📦 Installing dependencies..."
	python scripts/setup/install.py

# Setup
setup:
	@echo "🔧 Setting up project..."
	python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/raw', 'data/processed', 'data/annotations', 'data/test_images', 'models/checkpoints', 'models/exports', 'logs', 'output', 'output/detections', 'output/evaluation']]"
	@echo "✅ Project setup complete!"

# Data collection
collect:
	@echo "📊 Collecting training data..."
	python run.py --mode collect --synthetic 200

# Annotation
annotate:
	@echo "🎨 Launching annotation tool..."
	python tools/annotation/run_annotation_tool.py

# Training
train:
	@echo "🤖 Training model..."
	python run.py --mode train

# Full training pipeline
train-full:
	@echo "🚀 Running full training pipeline..."
	python scripts/training/train_model.py

# Inference
infer:
	@echo "🔍 Running inference..."
	python run.py --mode inference --image data/test_images

# Batch inference
infer-batch:
	@echo "📁 Running batch inference..."
	python run.py --mode batch --image_dir data/test_images

# Evaluation
evaluate:
	@echo "📊 Evaluating model..."
	python run.py --mode evaluate

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf .pytest_cache/
	rm -rf logs/*.log
	@echo "✅ Cleanup complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

# Quick start (full pipeline)
quickstart: install setup collect train
	@echo "🎉 Quick start complete!"
	@echo "Run 'make infer' to test your model"

# Development setup
dev-setup: install setup
	@echo "🔧 Development setup complete!"
	@echo "Run 'make collect' to start data collection"

# Production deployment
deploy:
	@echo "🚀 Preparing for deployment..."
	python run.py --mode optimize
	@echo "✅ Model optimized for production!"

# Help for specific commands
help-collect:
	@echo "Data Collection Help:"
	@echo "  make collect          - Collect 200 synthetic logos"
	@echo "  python run.py --mode collect --synthetic 500  - Collect 500 logos"

help-annotate:
	@echo "Annotation Help:"
	@echo "  make annotate         - Launch single image annotation tool"
	@echo "  python tools/annotation/run_batch_annotation.py  - Batch annotation"

help-train:
	@echo "Training Help:"
	@echo "  make train            - Basic training"
	@echo "  make train-full       - Full training pipeline with optimization"
	@echo "  python run.py --mode train --model yolov8s.pt  - Use different model"

help-infer:
	@echo "Inference Help:"
	@echo "  make infer            - Test on data/test_images/"
	@echo "  make infer-batch      - Batch processing"
	@echo "  python run.py --mode inference --image path/to/image.jpg  - Single image"
