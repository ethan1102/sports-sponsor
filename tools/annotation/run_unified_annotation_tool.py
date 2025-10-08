#!/usr/bin/env python3
"""
Unified Annotation Tool Launcher
Professional annotation tool with single and batch processing capabilities
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from unified_annotation_tool import UnifiedAnnotationTool

def main():
    """Launch the unified annotation tool"""
    print("🎨 Starting Unified Professional Annotation Tool")
    print("=" * 50)
    print("Features:")
    print("  • Single image annotation")
    print("  • Batch processing")
    print("  • Auto-save functionality")
    print("  • Undo/Redo support")
    print("  • Real-time quality metrics")
    print("  • Enhanced UI with modern styling")
    print("  • Multiple export formats")
    print("=" * 50)
    
    app = UnifiedAnnotationTool()
    app.run()

if __name__ == "__main__":
    main()
