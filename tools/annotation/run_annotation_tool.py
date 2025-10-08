#!/usr/bin/env python3
"""
Annotation Tool Launcher
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from annotation_tool import LogoAnnotationTool

def main():
    """Launch the annotation tool"""
    print("🎨 Starting Logo Annotation Tool")
    print("=" * 35)
    
    app = LogoAnnotationTool()
    app.run()

if __name__ == "__main__":
    main()
