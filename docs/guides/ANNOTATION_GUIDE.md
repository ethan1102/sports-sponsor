# Professional Annotation Guide

## 🎨 How to Annotate Japanese Sport Logos with the Enhanced Tool

### Getting Started

#### Option 1: Unified Professional Tool (Recommended)
```bash
python tools/annotation/run_unified_annotation_tool.py
```

#### Option 2: Single Annotation Tool
```bash
python tools/annotation/run_annotation_tool.py
```

#### Option 3: Batch Annotation Tool
```bash
python tools/annotation/run_batch_annotation.py
```

### Quick Start Guide

1. **Launch the Tool**
   - Use the unified tool for best experience
   - Modern UI with professional styling
   - Supports both single and batch processing

2. **Load Images**
   - Click "📂 Load Images" button
   - Select directory containing your images
   - Supports: JPG, JPEG, PNG, BMP, TIFF, WebP

3. **Choose Mode**
   - **Single Mode**: Annotate one image at a time
   - **Batch Mode**: Process multiple images efficiently

4. **Start Annotating**
   - Click "🎯 Start Annotation"
   - Click and drag to draw bounding box around logo
   - Release mouse to complete annotation
   - Auto-save every 30 seconds (configurable)

5. **Multi-Selection (New!)**
   - Enable "Multi-Select Mode" checkbox
   - Click on existing annotations to select them
   - Use Ctrl+Click in listbox for multiple selection
   - Assign same label to selected annotations with "🏷️ Assign Same Label"
   - Delete multiple annotations at once

### Annotation Best Practices

#### ✅ Good Annotations
- **Tight Bounding Boxes**: Draw boxes as close to logo edges as possible
- **Complete Coverage**: Include the entire logo, not just part of it
- **Consistent Style**: Use same annotation style across all images
- **Multiple Logos**: Annotate all visible logos in each image

#### ❌ Avoid These
- **Loose Boxes**: Don't include too much background
- **Partial Logos**: Don't annotate cut-off or partially visible logos
- **Inconsistent Sizing**: Keep similar logos at similar sizes
- **Missing Logos**: Don't skip small or partially obscured logos

### Enhanced Features

#### 🎯 Professional UI
- **Modern Design**: Clean, intuitive interface with professional styling
- **Tabbed Interface**: Organized controls in separate tabs
- **Real-time Feedback**: Live status updates and activity indicators
- **Progress Tracking**: Visual progress bars and statistics

#### 🔄 Advanced Workflow
- **Auto-save**: Automatic saving every 30 seconds (configurable)
- **Undo/Redo**: Full history support with Ctrl+Z/Ctrl+Y
- **Smart Navigation**: Arrow keys, mouse wheel zoom, pan support
- **Batch Processing**: Process multiple images efficiently

#### 📊 Quality Assurance
- **Real-time Validation**: Instant feedback on annotation quality
- **Statistics Dashboard**: Live metrics and session statistics
- **Quality Scoring**: Automatic quality assessment (0-100 score)
- **Annotation Details**: Comprehensive annotation information

#### 🎨 Enhanced Display
- **Customizable Colors**: Choose annotation colors
- **Zoom & Pan**: Mouse wheel zoom, middle-click pan
- **Fit to Canvas**: Smart image fitting
- **Enhanced Labels**: Better annotation visualization

#### 🔘 Multi-Selection Features
- **Multi-Select Mode**: Select multiple annotations at once
- **Click to Select**: Click on annotations to toggle selection
- **Assign Same Label**: Give multiple annotations the same label/class
- **Bulk Operations**: Delete or modify multiple annotations at once
- **Visual Feedback**: Selected annotations are highlighted in red

### Keyboard Shortcuts

#### Navigation
| Key | Action |
|-----|--------|
| `N` / `→` | Next image |
| `P` / `←` | Previous image |
| `Tab` | Switch between tabs |
| `Esc` | Focus management |

#### File Operations
| Key | Action |
|-----|--------|
| `Ctrl+O` | Load images |
| `Ctrl+S` | Save annotations |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+Shift+Z` | Redo (alternative) |

#### Annotation Operations
| Key | Action |
|-----|--------|
| `C` | Start new annotation |
| `Delete` | Delete selected annotation(s) |
| `A` | Clear all annotations |
| `M` | Toggle multi-select mode |
| `Ctrl+L` | Assign same label to selected annotations |
| `R` | Refresh statistics |

#### Zoom & Display
| Key | Action |
|-----|--------|
| `+` / `=` | Zoom in |
| `-` | Zoom out |
| `0` | Fit to canvas |
| `Mouse Wheel` | Zoom in/out |
| `Middle Click + Drag` | Pan image |

### Batch Annotation

#### Unified Tool Batch Processing
The unified annotation tool includes powerful batch processing capabilities:

1. **Create Batches**
   - Set batch size (default: 20 images)
   - Automatically organize images into batches
   - Track progress across all batches

2. **Batch Controls**
   - Start/Pause/Stop batch processing
   - Automatic progression through unannotated images
   - Real-time batch statistics

3. **Batch Features**
   - Process multiple images efficiently
   - Track progress across large datasets
   - Auto-validation of annotations
   - Export to YOLO format

#### Legacy Batch Tool
For dedicated batch processing:
```bash
python tools/annotation/run_batch_annotation.py
```

### Quality Control

#### Validation
```bash
python tools/validation/validate_annotations.py --annotation_dir data/annotations
```

#### Common Issues to Fix
- **Out of bounds coordinates**: Coordinates outside [0,1] range
- **Invalid aspect ratios**: Extremely wide or tall boxes
- **Too small logos**: Logos smaller than minimum size
- **Missing annotations**: Images with no annotations

### Enhanced Annotation Format

Annotations are saved in enhanced JSON format with metadata:
```json
{
  "image_path": "path/to/image.jpg",
  "image_size": [height, width],
  "annotations": [
    {
      "id": "ann_1703123456789",
      "bbox": [x1, y1, x2, y2],  // Normalized coordinates [0,1]
      "class_name": "logo",
      "confidence": 1.0,
      "created_at": "2024-01-01T12:00:00.000Z",
      "modified_at": "2024-01-01T12:00:00.000Z",
      "selected": false
    }
  ],
  "annotation_count": 1,
  "saved_at": "2024-01-01T12:00:00.000Z",
  "tool_version": "2.0",
  "config": {
    "auto_save_enabled": true,
    "default_class": "logo",
    "annotation_color": "blue"
  }
}
```

#### New Features in Enhanced Format:
- **Unique IDs**: Each annotation has a unique identifier
- **Timestamps**: Creation and modification timestamps
- **Tool Version**: Version tracking for compatibility
- **Configuration**: Saved tool settings
- **Metadata**: Additional annotation information

### YOLO Export

Convert annotations to YOLO format:
```bash
python tools/annotation/run_annotation_tool.py
# Click "Export to YOLO" button
```

YOLO format:
```
0 center_x center_y width height
```

### Tips for Japanese Sport Logos

1. **Text Logos**: Include all text, even if partially obscured
2. **Symbol Logos**: Include the entire symbol/icon
3. **Small Logos**: Don't skip small sponsor logos
4. **Multiple Logos**: Annotate all logos in the image
5. **Consistency**: Use same annotation style throughout

### Advanced Features

#### Auto-save Configuration
- **Enable/Disable**: Toggle in Settings tab
- **Interval**: Default 30 seconds (configurable)
- **Silent Operation**: No interruption to workflow
- **Backup**: Automatic backup of annotations

#### Quality Metrics
- **Size Analysis**: Average annotation sizes
- **Aspect Ratio**: Min/max aspect ratios
- **Consistency**: Size variance analysis
- **Quality Score**: Overall quality assessment (0-100)

#### Statistics Dashboard
- **Session Stats**: Images processed, annotations created
- **Time Tracking**: Session duration, last activity
- **Progress**: Current image, completion percentage
- **Performance**: Annotation speed, quality trends

### Troubleshooting

#### Tool Won't Start
- Check Python dependencies: `pip install -r scripts/setup/requirements.txt`
- Verify image formats: JPG, JPEG, PNG, BMP, TIFF, WebP supported
- Ensure sufficient memory for large images

#### Annotations Not Saving
- Check write permissions in output directory
- Ensure sufficient disk space
- Verify JSON format is valid
- Check auto-save settings in Settings tab

#### Performance Issues
- Reduce batch size for large datasets
- Close other applications to free memory
- Use smaller images if possible
- Check auto-save interval (too frequent may slow down)

#### Poor Detection Results
- Review annotation quality using Statistics tab
- Check for consistent annotation style
- Ensure sufficient training data (100+ images minimum)
- Use Quality Score to identify issues

### Multi-Selection for Same Logos

When you have multiple instances of the same logo in one image:

1. **Enable Multi-Select Mode**: Check the "Multi-Select Mode" checkbox
2. **Select All Instances**: 
   - Click on each logo bounding box to select it
   - Or use Ctrl+Click in the annotation list
   - Selected annotations will turn red
3. **Assign Same Label**: 
   - Click "🏷️ Assign Same Label" button
   - Enter the label name (e.g., "Nike", "Adidas", "logo")
   - All selected annotations will get the same label
4. **Continue Annotating**: 
   - Clear selection and continue with other logos
   - Repeat for different logo types

**Example Workflow:**
- Image has 3 Nike logos and 2 Adidas logos
- Select all 3 Nike logos → Assign label "Nike"
- Select all 2 Adidas logos → Assign label "Adidas"
- Each logo keeps its individual bounding box but shares the same label

### Best Practices Summary

1. **Start Small**: Begin with 50-100 images
2. **Quality First**: Focus on annotation quality over quantity
3. **Consistent Style**: Use same approach throughout
4. **Regular Validation**: Check quality frequently using Statistics tab
5. **Iterative Improvement**: Refine based on detection results
6. **Use Auto-save**: Enable automatic saving to prevent data loss
7. **Monitor Quality**: Watch quality scores and adjust annotation style
8. **Batch Processing**: Use batch mode for large datasets
9. **Keyboard Shortcuts**: Learn shortcuts for faster workflow
10. **Regular Backups**: Export to YOLO format regularly
11. **Multi-Selection**: Use multi-select for same logos to save time

### Migration from Legacy Tools

If you have existing annotations from the old tool:
- The new tool automatically detects and converts legacy formats
- All existing annotations will be preserved
- New features (IDs, timestamps) will be added automatically
- No manual conversion required

### Support

For issues or questions:
- Check the Statistics tab for quality metrics
- Review the Settings tab for configuration options
- Use the keyboard shortcuts for faster workflow
- Monitor the status bar for real-time feedback
