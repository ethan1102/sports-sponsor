"""Test Step 2: Model Loading"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model_trainer import LogoDetectionTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_load():
    """Test loading YOLOv8 model"""
    print("=" * 60)
    print("Step 2: Model Loading Test")
    print("=" * 60)
    
    try:
        # Initialize trainer with default model
        model_name = "yolov8n.pt"
        print(f"\n1. Initializing LogoDetectionTrainer with model: {model_name}")
        trainer = LogoDetectionTrainer(model_name=model_name)
        
        # Load model
        print(f"\n2. Loading model: {model_name}")
        trainer.load_model()
        
        # Check if model is loaded
        if trainer.model is not None:
            print(f"✅ Model loaded successfully!")
            print(f"   Model type: {type(trainer.model).__name__}")
            
            # Try to get model info
            try:
                model_info = trainer.model.info()
                print(f"   Model info available: Yes")
            except:
                print(f"   Model info: Basic YOLO model")
            
            print(f"\n✅ Step 2 (Model Loading) completed successfully!")
            return True
        else:
            print(f"❌ Model is None after loading")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)
