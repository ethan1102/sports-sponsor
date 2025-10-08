# Training Guide

## 🤖 How to Train Your Logo Detection Model

### Prerequisites

1. **Install Dependencies**
   ```bash
   python scripts/setup/install.py
   ```

2. **Prepare Data**
   - Collect Japanese sport images
   - Annotate logos using annotation tools
   - Validate annotation quality

### Training Pipeline

#### Option 1: Automated Training
```bash
# Run complete training pipeline
python scripts/training/train_model.py
```

#### Option 2: Step-by-Step Training
```bash
# 1. Collect data
python src/main.py --mode collect --synthetic 200

# 2. Preprocess data
python src/main.py --mode preprocess

# 3. Train model
python src/main.py --mode train --model yolov8n.pt

# 4. Optimize model
python src/main.py --mode optimize

# 5. Evaluate model
python src/main.py --mode evaluate
```

### Training Configuration

Edit `src/config.py` to customize training:

```python
TRAINING_CONFIG = {
    "epochs": 100,           # Number of training epochs
    "batch_size": 16,        # Batch size
    "learning_rate": 0.01,   # Learning rate
    "weight_decay": 0.0005,  # Weight decay
    "patience": 20,          # Early stopping patience
    "device": "cuda"         # Training device
}
```

### Model Selection

Choose the right model for your needs:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | Small | Fast | Good | Quick testing |
| YOLOv8s | Medium | Medium | Better | Balanced |
| YOLOv8m | Large | Slow | Best | Production |

### Data Requirements

#### Minimum Requirements
- **Images**: 100+ annotated images
- **Logos**: 200+ logo annotations
- **Variety**: Different sports, lighting, angles

#### Recommended
- **Images**: 500+ annotated images
- **Logos**: 1000+ logo annotations
- **Diversity**: Multiple sports, sponsors, conditions

### Training Process

#### 1. Data Collection
```bash
python src/main.py --mode collect --synthetic 200
```
- Collects from Japanese sport websites
- Generates synthetic training data
- Saves to `data/raw/`

#### 2. Data Preprocessing
```bash
python src/main.py --mode preprocess
```
- Creates YOLO format annotations
- Splits data into train/val/test
- Applies data augmentation
- Saves to `data/processed/`

#### 3. Model Training
```bash
python src/main.py --mode train
```
- Trains YOLO model
- Saves checkpoints to `models/checkpoints/`
- Logs training progress
- Implements early stopping

#### 4. Model Optimization
```bash
python src/main.py --mode optimize
```
- Applies advanced optimization strategies
- Test-time augmentation
- Hyperparameter tuning
- Ensemble methods

### Monitoring Training

#### Training Logs
- Check `logs/training.log` for detailed logs
- Monitor loss curves and metrics
- Watch for overfitting signs

#### Key Metrics to Watch
- **Loss**: Should decrease steadily
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **mAP**: Mean Average Precision

#### Target Metrics
- **Detection Accuracy**: 90%+
- **Precision**: 85%+
- **Recall**: 90%+
- **F1-Score**: 87%+

### Troubleshooting

#### Low Accuracy
1. **Collect More Data**: Add more annotated images
2. **Improve Annotations**: Check annotation quality
3. **Adjust Parameters**: Tune learning rate, batch size
4. **Data Augmentation**: Increase augmentation strength

#### Overfitting
1. **Reduce Model Size**: Use smaller model
2. **Increase Data**: Add more training data
3. **Regularization**: Increase weight decay
4. **Early Stopping**: Reduce patience value

#### Training Too Slow
1. **Reduce Batch Size**: Use smaller batches
2. **Use GPU**: Ensure CUDA is available
3. **Reduce Image Size**: Use smaller input size
4. **Fewer Epochs**: Reduce training time

### Advanced Training

#### Custom Data Augmentation
Edit `src/config.py`:
```python
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "rotation": 15,
    "brightness_contrast": 0.2,
    # Add more augmentations
}
```

#### Transfer Learning
```bash
# Use pre-trained model
python src/main.py --mode train --model yolov8n.pt
```

#### Multi-GPU Training
```python
# In config.py
TRAINING_CONFIG = {
    "device": "cuda:0,1,2,3"  # Multiple GPUs
}
```

### Model Evaluation

#### Run Evaluation
```bash
python scripts/evaluation/evaluate_model.py --test_data data/test_images
```

#### Key Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives

### Model Deployment

#### Export Model
```bash
# Export to ONNX
python src/main.py --mode export --format onnx

# Export to TensorRT
python src/main.py --mode export --format tensorrt
```

#### Optimize for Inference
```bash
# Quantize model
python src/main.py --mode optimize --quantize

# Optimize for speed
python src/main.py --mode optimize --speed
```

### Best Practices

1. **Start Small**: Begin with small dataset and model
2. **Iterate**: Improve data quality based on results
3. **Validate**: Use validation set to monitor progress
4. **Document**: Keep track of experiments and results
5. **Test**: Evaluate on diverse test data

### Common Issues

#### CUDA Out of Memory
- Reduce batch size
- Use smaller model
- Reduce image size
- Use gradient accumulation

#### Poor Detection Results
- Check annotation quality
- Increase training data
- Adjust confidence threshold
- Try different model size

#### Training Stalls
- Check learning rate
- Verify data loading
- Monitor loss curves
- Check for data issues

### Success Checklist

- [ ] Collected sufficient training data (100+ images)
- [ ] Annotated logos with high quality
- [ ] Validated annotation consistency
- [ ] Configured training parameters
- [ ] Monitored training progress
- [ ] Achieved target accuracy (90%+)
- [ ] Tested on diverse images
- [ ] Optimized for deployment
