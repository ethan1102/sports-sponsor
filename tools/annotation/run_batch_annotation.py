#!/usr/bin/env python3
"""
Batch Annotation Tool Launcher
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from batch_annotation_tool import BatchAnnotationTool

def main():
    """Launch the batch annotation tool"""
    print("📦 Starting Batch Annotation Tool")
    print("=" * 35)
    
    app = BatchAnnotationTool()
    app.run()

if __name__ == "__main__":
    main()
