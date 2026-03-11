"""Create sample synthetic images for preprocessing"""
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Create data/raw directory
base = Path(r'C:\NextStairs\sports-sponsor-logo-detection\data\raw')
base.mkdir(parents=True, exist_ok=True)

# Generate 20 synthetic images with simple logos
for i in range(20):
    # Create random sized image (64x64 to 128x128)
    size = np.random.randint(64, 129)
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    
    # Add a simple geometric shape as "logo"
    center = (size // 2, size // 2)
    radius = size // 4
    cv2.circle(img, center, radius, (255, 255, 255), -1)
    cv2.rectangle(img, 
                 (center[0] - radius//2, center[1] - radius//2),
                 (center[0] + radius//2, center[1] + radius//2),
                 (0, 0, 0), 2)
    
    # Save image
    filename = f'synthetic_{i:03d}.jpg'
    filepath = base / filename
    cv2.imwrite(str(filepath), img)
    print(f"Created: {filename}")

print(f"\nCreated {20} synthetic images in {base}")
