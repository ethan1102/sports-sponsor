"""
Unified Professional Annotation Tool for Japanese Sport Sponsor Logo Detection
Combines single image annotation with batch processing capabilities
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import shutil
import time
import threading
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnnotationConfig:
    """Configuration for annotation tool"""
    auto_save_interval: int = 30  # seconds
    default_class: str = "logo"
    min_bbox_size: int = 10
    max_bbox_size: int = 500
    default_confidence: float = 1.0
    supported_formats: list = None
    batch_size: int = 20
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

@dataclass
class Annotation:
    """Data class for annotation"""
    bbox: Tuple[float, float, float, float]  # normalized coordinates
    class_name: str
    confidence: float
    created_at: str
    modified_at: str
    selected: bool = False
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"ann_{int(time.time() * 1000)}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.modified_at:
            self.modified_at = self.created_at

class UnifiedAnnotationTool:
    """
    Unified professional annotation tool with single and batch processing capabilities
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎨 Unified Professional Logo Annotation Tool")
        self.root.geometry("1800x1000")
        self.root.minsize(1400, 800)
        
        # Configuration
        self.config = AnnotationConfig()
        
        # Annotation data
        self.current_image_path = None
        self.current_image = None
        self.display_image = None
        self.annotations: List[Annotation] = []
        self.current_annotation = None
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        
        # Multi-selection support
        self.multi_select_mode = False
        self.selected_annotations = set()
        self.temp_annotations = []  # For grouping multiple boxes
        
        # Image display variables
        self.scale_factor = 1.0
        self.canvas_width = 1000
        self.canvas_height = 700
        self.image_x = 0
        self.image_y = 0
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        
        # File management
        self.image_files = []
        self.current_index = 0
        self.annotation_dir = None
        self.output_dir = None
        
        # Batch processing
        self.batch_mode = False
        self.current_batch = []
        self.batch_index = 0
        self.batch_progress = 0
        
        # History for undo/redo
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
        # Auto-save
        self.auto_save_timer = None
        self.last_save_time = None
        
        # Statistics
        self.stats = {
            'total_annotations': 0,
            'images_processed': 0,
            'session_start': datetime.now(),
            'last_activity': None,
            'batch_annotations': 0
        }
        
        # UI setup
        self.setup_styles()
        self.setup_ui()
        self.setup_bindings()
        self.start_auto_save()
    
    def setup_styles(self):
        """Setup modern styling for the UI"""
        style = ttk.Style()
        
        # Configure modern theme
        style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Warning.TLabel', font=('Arial', 10), foreground='#f39c12')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        style.configure('Secondary.TButton', font=('Arial', 9))
        style.configure('Batch.TButton', font=('Arial', 10, 'bold'), foreground='#e67e22')
        
        # Configure notebook style
        style.configure('TNotebook', tabposition='n')
        style.configure('TNotebook.Tab', padding=[20, 10])
        
    def setup_ui(self):
        """Setup the unified user interface"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top toolbar
        self.create_toolbar(main_container)
        
        # Main content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Left panel - Image display
        left_panel = ttk.LabelFrame(content_frame, text="🖼️ Image Display", style='Header.TLabel')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Image canvas with scrollbars
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height, 
                               bg='#f8f9fa', cursor='crosshair', relief=tk.SUNKEN, bd=2)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Image info panel
        self.create_image_info_panel(left_panel)
        
        # Right panel - Controls
        right_panel = ttk.Frame(content_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Create notebook for organized tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Single annotation tab
        self.create_single_annotation_tab()
        
        # Batch processing tab
        self.create_batch_processing_tab()
        
        # Statistics tab
        self.create_statistics_tab()
        
        # Settings tab
        self.create_settings_tab()
        
        # Status bar
        self.create_status_bar()
    
    def create_toolbar(self, parent):
        """Create the enhanced top toolbar"""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # File operations
        file_frame = ttk.LabelFrame(toolbar, text="📁 File Operations")
        file_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_frame, text="📂 Load Images", command=self.load_images, 
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(file_frame, text="💾 Save", command=self.save_annotations, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(file_frame, text="📤 Export", command=self.export_to_yolo, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        
        # Navigation
        nav_frame = ttk.LabelFrame(toolbar, text="🧭 Navigation")
        nav_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(nav_frame, text="⬅️", command=self.previous_image, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        ttk.Button(nav_frame, text="➡️", command=self.next_image, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        
        self.image_label = ttk.Label(nav_frame, text="No image loaded", style='Info.TLabel')
        self.image_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Mode toggle
        mode_frame = ttk.LabelFrame(toolbar, text="🔄 Mode")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.mode_var = tk.StringVar(value="Single")
        ttk.Radiobutton(mode_frame, text="Single", variable=self.mode_var, 
                       value="Single", command=self.toggle_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_frame, text="Batch", variable=self.mode_var, 
                       value="Batch", command=self.toggle_mode).pack(side=tk.LEFT, padx=2)
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(toolbar, text="🔍 Zoom")
        zoom_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(zoom_frame, text="🔍+", command=self.zoom_in, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        ttk.Button(zoom_frame, text="🔍-", command=self.zoom_out, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        ttk.Button(zoom_frame, text="📐 Fit", command=self.fit_to_canvas, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        
        # History controls
        history_frame = ttk.LabelFrame(toolbar, text="🔄 History")
        history_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(history_frame, text="↶ Undo", command=self.undo, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        ttk.Button(history_frame, text="↷ Redo", command=self.redo, 
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=2, pady=5)
        
        # Progress indicator
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(toolbar, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.X, expand=True)
    
    def create_single_annotation_tab(self):
        """Create the single annotation controls tab"""
        annotation_frame = ttk.Frame(self.notebook)
        self.notebook.add(annotation_frame, text="🎯 Single Annotation")
        
        # Annotation controls
        controls_frame = ttk.LabelFrame(annotation_frame, text="🎮 Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="🎯 Start Annotation", 
                  command=self.start_annotation, style='Primary.TButton').pack(fill=tk.X, padx=5, pady=2)
        
        # Multi-selection controls
        multi_frame = ttk.Frame(controls_frame)
        multi_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.multi_select_var = tk.BooleanVar()
        ttk.Checkbutton(multi_frame, text="Multi-Select Mode", 
                       variable=self.multi_select_var, command=self.toggle_multi_select).pack(side=tk.LEFT)
        
        ttk.Button(multi_frame, text="🏷️ Assign Same Label", 
                  command=self.assign_same_label, style='Secondary.TButton').pack(side=tk.RIGHT, padx=(5, 0))
        
        ttk.Button(controls_frame, text="🗑️ Delete Selected", 
                  command=self.delete_selected, style='Secondary.TButton').pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(controls_frame, text="🧹 Clear All", 
                  command=self.clear_annotations, style='Secondary.TButton').pack(fill=tk.X, padx=5, pady=2)
        
        # Annotation list
        list_frame = ttk.LabelFrame(annotation_frame, text="📋 Annotation List")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.annotation_listbox = tk.Listbox(list_container, font=('Consolas', 9), 
                                           selectmode=tk.MULTIPLE, bg='#f8f9fa')
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        self.annotation_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Annotation details
        details_frame = ttk.LabelFrame(annotation_frame, text="📊 Details")
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.details_text = tk.Text(details_frame, height=4, font=('Consolas', 9), 
                                   bg='#f8f9fa', wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, padx=5, pady=5)
    
    def create_batch_processing_tab(self):
        """Create the batch processing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="📦 Batch Processing")
        
        # Batch setup
        setup_frame = ttk.LabelFrame(batch_frame, text="⚙️ Batch Setup")
        setup_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Batch size
        size_frame = ttk.Frame(setup_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(size_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value=str(self.config.batch_size))
        ttk.Entry(size_frame, textvariable=self.batch_size_var, width=5).pack(side=tk.RIGHT)
        
        # Batch controls
        batch_controls = ttk.LabelFrame(batch_frame, text="🎮 Batch Controls")
        batch_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(batch_controls, text="📦 Create Batches", 
                  command=self.create_batches, style='Batch.TButton').pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(batch_controls, text="▶️ Start Batch", 
                  command=self.start_batch_processing, style='Primary.TButton').pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(batch_controls, text="⏸️ Pause Batch", 
                  command=self.pause_batch_processing, style='Secondary.TButton').pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(batch_controls, text="🛑 Stop Batch", 
                  command=self.stop_batch_processing, style='Secondary.TButton').pack(fill=tk.X, padx=5, pady=2)
        
        # Batch info
        info_frame = ttk.LabelFrame(batch_frame, text="📊 Batch Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.batch_info_text = tk.Text(info_frame, height=10, font=('Consolas', 9), 
                                      bg='#f8f9fa', wrap=tk.WORD)
        self.batch_info_text.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Batch progress
        progress_frame = ttk.LabelFrame(batch_frame, text="📈 Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.batch_progress_var = tk.DoubleVar()
        self.batch_progress_bar = ttk.Progressbar(progress_frame, variable=self.batch_progress_var, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.batch_progress_label = ttk.Label(progress_frame, text="Ready for batch processing")
        self.batch_progress_label.pack(pady=2)
    
    def create_statistics_tab(self):
        """Create the statistics tab"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📈 Statistics")
        
        # Session stats
        session_frame = ttk.LabelFrame(stats_frame, text="⏱️ Session Statistics")
        session_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.session_text = tk.Text(session_frame, height=8, font=('Consolas', 9), 
                                   bg='#f8f9fa', wrap=tk.WORD)
        self.session_text.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Quality metrics
        quality_frame = ttk.LabelFrame(stats_frame, text="✅ Quality Metrics")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.quality_text = tk.Text(quality_frame, height=6, font=('Consolas', 9), 
                                   bg='#f8f9fa', wrap=tk.WORD)
        self.quality_text.pack(fill=tk.BOTH, padx=5, pady=5)
    
    def create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # General settings
        general_frame = ttk.LabelFrame(settings_frame, text="🔧 General")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Auto-save setting
        auto_save_frame = ttk.Frame(general_frame)
        auto_save_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.auto_save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_save_frame, text="Auto-save every 30s", 
                       variable=self.auto_save_var, command=self.toggle_auto_save).pack(side=tk.LEFT)
        
        # Default class setting
        class_frame = ttk.Frame(general_frame)
        class_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(class_frame, text="Default Class:").pack(side=tk.LEFT)
        self.default_class_var = tk.StringVar(value="logo")
        class_entry = ttk.Entry(class_frame, textvariable=self.default_class_var, width=10)
        class_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Display settings
        display_frame = ttk.LabelFrame(settings_frame, text="🎨 Display")
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Annotation color
        color_frame = ttk.Frame(display_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(color_frame, text="Annotation Color:").pack(side=tk.LEFT)
        self.annotation_color = tk.StringVar(value="blue")
        ttk.Button(color_frame, text="🎨 Choose", command=self.choose_color).pack(side=tk.LEFT, padx=(5, 0))
    
    def create_image_info_panel(self, parent):
        """Create image information panel"""
        info_frame = ttk.LabelFrame(parent, text="ℹ️ Image Information")
        info_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        self.info_text = tk.Text(info_frame, height=6, width=50, wrap=tk.WORD, 
                                font=('Consolas', 9), bg='#f8f9fa')
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load images to begin annotation")
        
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     relief=tk.SUNKEN, style='Info.TLabel')
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=2)
        
        # Activity indicator
        self.activity_var = tk.StringVar()
        self.activity_label = ttk.Label(status_frame, textvariable=self.activity_var, 
                                       relief=tk.SUNKEN, style='Success.TLabel')
        self.activity_label.pack(side=tk.RIGHT, padx=2, pady=2)
    
    def toggle_mode(self):
        """Toggle between single and batch mode"""
        mode = self.mode_var.get()
        if mode == "Batch":
            self.batch_mode = True
            self.notebook.select(1)  # Switch to batch tab
            self.update_status("📦 Batch mode activated")
        else:
            self.batch_mode = False
            self.notebook.select(0)  # Switch to single tab
            self.update_status("🎯 Single mode activated")
    
    def create_batches(self):
        """Create batches from loaded images"""
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load images first")
            return
        
        try:
            batch_size = int(self.batch_size_var.get())
            self.config.batch_size = batch_size
            
            # Create batches
            self.batches = []
            for i in range(0, len(self.image_files), batch_size):
                batch = self.image_files[i:i + batch_size]
                self.batches.append(batch)
            
            self.current_batch = self.batches[0] if self.batches else []
            self.batch_index = 0
            self.batch_progress = 0
            
            self.update_batch_info()
            self.update_status(f"📦 Created {len(self.batches)} batches")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid batch size")
    
    def update_batch_info(self):
        """Update batch information display"""
        if not hasattr(self, 'batches') or not self.batches:
            self.batch_info_text.delete(1.0, tk.END)
            self.batch_info_text.insert(1.0, "No batches created")
            return
        
        info = f"Batch Processing Information:\n"
        info += f"Total Images: {len(self.image_files)}\n"
        info += f"Batch Size: {self.config.batch_size}\n"
        info += f"Total Batches: {len(self.batches)}\n"
        info += f"Current Batch: {self.batch_index + 1}/{len(self.batches)}\n\n"
        
        info += f"Current Batch Images:\n"
        for i, image_path in enumerate(self.current_batch):
            filename = Path(image_path).name
            status = "✅" if self.is_annotated(image_path) else "⭕"
            info += f"  {i+1}. {status} {filename}\n"
        
        info += f"\nBatch Statistics:\n"
        annotated_count = sum(1 for img in self.current_batch if self.is_annotated(img))
        info += f"Annotated: {annotated_count}/{len(self.current_batch)}\n"
        info += f"Progress: {annotated_count/len(self.current_batch)*100:.1f}%\n"
        
        self.batch_info_text.delete(1.0, tk.END)
        self.batch_info_text.insert(1.0, info)
    
    def is_annotated(self, image_path: str) -> bool:
        """Check if image has annotations"""
        if not self.output_dir:
            return False
        
        annotation_file = self.output_dir / f"{Path(image_path).stem}.json"
        return annotation_file.exists()
    
    def start_batch_processing(self):
        """Start batch processing"""
        if not hasattr(self, 'batches') or not self.batches:
            messagebox.showwarning("No Batches", "Please create batches first")
            return
        
        self.batch_processing_active = True
        self.process_next_batch_item()
    
    def process_next_batch_item(self):
        """Process next item in current batch"""
        if not self.batch_processing_active or not self.current_batch:
            return
        
        # Find next unannotated image
        for i, image_path in enumerate(self.current_batch):
            if not self.is_annotated(image_path):
                # Load this image
                self.current_index = self.image_files.index(image_path)
                self.load_current_image()
                self.batch_progress = (i / len(self.current_batch)) * 100
                self.batch_progress_var.set(self.batch_progress)
                self.batch_progress_label.config(text=f"Processing: {Path(image_path).name}")
                return
        
        # Current batch complete, move to next batch
        self.next_batch()
    
    def next_batch(self):
        """Move to next batch"""
        if self.batch_index < len(self.batches) - 1:
            self.batch_index += 1
            self.current_batch = self.batches[self.batch_index]
            self.batch_progress = 0
            self.batch_progress_var.set(0)
            self.batch_progress_label.config(text=f"Starting batch {self.batch_index + 1}")
            self.update_batch_info()
            
            # Continue processing
            if self.batch_processing_active:
                self.process_next_batch_item()
        else:
            # All batches complete
            self.batch_processing_active = False
            self.batch_progress_label.config(text="All batches completed!")
            messagebox.showinfo("Batch Complete", "All batches have been processed!")
    
    def pause_batch_processing(self):
        """Pause batch processing"""
        self.batch_processing_active = False
        self.update_status("⏸️ Batch processing paused")
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        self.batch_processing_active = False
        self.batch_progress_label.config(text="Batch processing stopped")
        self.update_status("🛑 Batch processing stopped")
    
    # Include all the methods from the single annotation tool
    # (setup_bindings, load_images, load_current_image, etc.)
    # For brevity, I'll include the key methods that need to be adapted
    
    def setup_bindings(self):
        """Setup keyboard and mouse bindings"""
        # Mouse bindings
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # Keyboard bindings
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()
        
        # Listbox selection
        self.annotation_listbox.bind("<<ListboxSelect>>", self.on_annotation_select)
        
        # Canvas bindings for panning
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)
        
        # Mouse wheel for zoom
        self.canvas.bind("<MouseWheel>", self.zoom_with_mouse)
        self.canvas.bind("<Button-4>", self.zoom_with_mouse)
        self.canvas.bind("<Button-5>", self.zoom_with_mouse)
    
    def load_images(self):
        """Load images from directory with enhanced features"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if not directory:
            return
        
        self.annotation_dir = directory
        self.output_dir = Path(directory) / "annotations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Find image files with more formats
        self.image_files = []
        
        for ext in self.config.supported_formats:
            self.image_files.extend(Path(directory).glob(f"*{ext}"))
            self.image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        self.image_files = [str(f) for f in self.image_files]
        
        if not self.image_files:
            messagebox.showwarning("No Images", "No image files found in the selected directory")
            return
        
        # Sort files for consistent ordering
        self.image_files.sort()
        
        self.current_index = 0
        self.load_current_image()
        self.update_status(f"📁 Loaded {len(self.image_files)} images")
        self.update_progress()
        self.update_statistics()
        self.activity_var.set(f"📁 Loaded {len(self.image_files)} images")
        self.root.after(2000, lambda: self.activity_var.set(""))
    
    def update_progress(self):
        """Update progress bar"""
        if self.image_files:
            progress = (self.current_index / len(self.image_files)) * 100
            self.progress_var.set(progress)
    
    def load_current_image(self):
        """Load the current image with enhanced features"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        self.current_image_path = self.image_files[self.current_index]
        
        # Load image
        self.current_image = cv2.imread(self.current_image_path)
        if self.current_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
            return
        
        # Convert BGR to RGB
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Load existing annotations
        self.load_image_annotations()
        
        # Save to history before making changes
        self.save_to_history()
        
        # Display image
        self.display_image_on_canvas()
        self.update_image_info()
        self.update_progress()
        self.update_statistics()
        
        # Update stats
        self.stats['images_processed'] = max(self.stats['images_processed'], self.current_index + 1)
        
        # Update batch info if in batch mode
        if self.batch_mode and hasattr(self, 'batches'):
            self.update_batch_info()
    
    # Include all essential methods from single annotation tool
    def start_auto_save(self):
        """Start auto-save timer"""
        if self.auto_save_var.get():
            self.auto_save_timer = threading.Timer(self.config.auto_save_interval, self.auto_save)
            self.auto_save_timer.start()
    
    def auto_save(self):
        """Auto-save annotations"""
        if self.current_image_path and self.annotations:
            try:
                self.save_annotations(silent=True)
                self.last_save_time = datetime.now()
                self.activity_var.set("💾 Auto-saved")
                self.root.after(2000, lambda: self.activity_var.set(""))
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
        
        # Schedule next auto-save
        if self.auto_save_var.get():
            self.start_auto_save()
    
    def toggle_auto_save(self):
        """Toggle auto-save feature"""
        if self.auto_save_var.get():
            self.start_auto_save()
            self.update_status("Auto-save enabled")
        else:
            if self.auto_save_timer:
                self.auto_save_timer.cancel()
            self.update_status("Auto-save disabled")
    
    def save_to_history(self):
        """Save current state to history for undo/redo"""
        if len(self.history) >= self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        
        # Create deep copy of current state
        state = {
            'annotations': [Annotation(**ann.__dict__) for ann in self.annotations],
            'current_index': self.current_index,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(state)
        self.history_index = len(self.history) - 1
        self.stats['last_activity'] = datetime.now()
    
    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.restore_state(state)
            self.update_status("Undo completed")
            self.activity_var.set("↶ Undone")
            self.root.after(1000, lambda: self.activity_var.set(""))
    
    def redo(self):
        """Redo last undone action"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.restore_state(state)
            self.update_status("Redo completed")
            self.activity_var.set("↷ Redone")
            self.root.after(1000, lambda: self.activity_var.set(""))
    
    def restore_state(self, state):
        """Restore application state from history"""
        self.annotations = state['annotations']
        self.current_index = state['current_index']
        self.update_annotation_list()
        self.display_image_on_canvas()
    
    def choose_color(self):
        """Choose annotation color"""
        color = colorchooser.askcolor(title="Choose Annotation Color")[1]
        if color:
            self.annotation_color.set(color)
            self.display_image_on_canvas()
    
    def start_pan(self, event):
        """Start panning the image"""
        self.dragging = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def pan_image(self, event):
        """Pan the image"""
        if self.dragging and self.display_image:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.pan_x += dx
            self.pan_y += dy
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            self.display_image_on_canvas()
    
    def stop_pan(self, event):
        """Stop panning the image"""
        self.dragging = False
    
    def zoom_with_mouse(self, event):
        """Zoom with mouse wheel"""
        if event.delta > 0 or event.num == 4:
            self.zoom_in()
        elif event.delta < 0 or event.num == 5:
            self.zoom_out()
    
    def zoom_in(self):
        """Zoom in on image"""
        self.scale_factor *= 1.2
        self.display_image_on_canvas()
    
    def zoom_out(self):
        """Zoom out on image"""
        self.scale_factor *= 0.8
        self.display_image_on_canvas()
    
    def fit_to_canvas(self):
        """Fit image to canvas"""
        if self.current_image is not None:
            img_height, img_width = self.current_image.shape[:2]
            scale_x = self.canvas_width / img_width
            scale_y = self.canvas_height / img_height
            self.scale_factor = min(scale_x, scale_y) * 0.9
            self.display_image_on_canvas()
    
    def previous_image(self):
        """Load previous image"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Load next image"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def display_image_on_canvas(self):
        """Display the current image on canvas"""
        if self.current_image is None:
            return
        
        # Calculate scale factor to fit image in canvas
        img_height, img_width = self.current_image.shape[:2]
        scale_x = self.canvas_width / img_width
        scale_y = self.canvas_height / img_height
        self.scale_factor = min(scale_x, scale_y) * 0.9
        
        # Resize image
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        resized_image = cv2.resize(self.current_image, (new_width, new_height))
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(resized_image)
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and draw image
        self.canvas.delete("all")
        
        # Center image on canvas
        self.image_x = (self.canvas_width - new_width) // 2
        self.image_y = (self.canvas_height - new_height) // 2
        
        self.canvas.create_image(self.image_x, self.image_y, anchor=tk.NW, image=self.display_image)
        
        # Draw existing annotations
        self.draw_annotations()
        
        # Update image label
        filename = Path(self.current_image_path).name
        self.image_label.config(text=f"{self.current_index + 1}/{len(self.image_files)}: {filename}")
    
    def draw_annotations(self):
        """Draw all annotations on the canvas"""
        for i, annotation in enumerate(self.annotations):
            self.draw_single_annotation(annotation, i)
    
    def draw_single_annotation(self, annotation, index):
        """Draw a single annotation on the canvas with enhanced styling"""
        x1, y1, x2, y2 = annotation.bbox
        
        # Convert to canvas coordinates
        canvas_x1 = self.image_x + x1 * self.scale_factor
        canvas_y1 = self.image_y + y1 * self.scale_factor
        canvas_x2 = self.image_x + x2 * self.scale_factor
        canvas_y2 = self.image_y + y2 * self.scale_factor
        
        # Determine color and style
        if annotation.selected:
            color = 'red'
            width = 3
            dash = None
        else:
            color = self.annotation_color.get()
            width = 2
            dash = (5, 5)
        
        # Draw rectangle with enhanced styling
        rect_id = self.canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline=color, width=width, dash=dash, tags=f"annotation_{index}"
        )
        
        # Draw label with background
        label_text = f"{annotation.class_name} {index + 1}"
        
        # Create label background
        label_bg = self.canvas.create_rectangle(
            canvas_x1 - 2, canvas_y1 - 15, 
            canvas_x1 + len(label_text) * 6 + 4, canvas_y1 - 2,
            fill=color, outline=color, tags=f"label_bg_{index}"
        )
        
        # Create label text
        label_id = self.canvas.create_text(
            canvas_x1 + 2, canvas_y1 - 8,
            text=label_text, fill='white', font=('Arial', 9, 'bold'),
            tags=f"label_{index}", anchor='w'
        )
    
    def start_annotation(self):
        """Start drawing a new annotation"""
        self.drawing = True
        self.current_annotation = None
        self.update_status("🎯 Click and drag to draw bounding box")
        self.activity_var.set("🎯 Drawing mode active")
        self.root.after(3000, lambda: self.activity_var.set(""))
    
    def on_mouse_click(self, event):
        """Handle mouse click events"""
        # Check if clicking on an existing annotation for multi-selection
        if self.multi_select_mode and not self.drawing:
            clicked_annotation = self.get_clicked_annotation(event.x, event.y)
            if clicked_annotation is not None:
                self.toggle_annotation_selection(clicked_annotation)
                return
        
        if not self.drawing:
            return
        
        # Convert canvas coordinates to image coordinates
        self.start_x = (event.x - self.image_x) / self.scale_factor
        self.start_y = (event.y - self.image_y) / self.scale_factor
        
        # Ensure coordinates are within image bounds
        if self.current_image is not None:
            img_height, img_width = self.current_image.shape[:2]
            self.start_x = max(0, min(self.start_x, img_width))
            self.start_y = max(0, min(self.start_y, img_height))
    
    def on_mouse_drag(self, event):
        """Handle mouse drag events"""
        if not self.drawing:
            return
        
        # Convert canvas coordinates to image coordinates
        current_x = (event.x - self.image_x) / self.scale_factor
        current_y = (event.y - self.image_y) / self.scale_factor
        
        # Ensure coordinates are within image bounds
        if self.current_image is not None:
            img_height, img_width = self.current_image.shape[:2]
            current_x = max(0, min(current_x, img_width))
            current_y = max(0, min(current_y, img_height))
        
        # Remove previous temporary rectangle
        self.canvas.delete("temp_rect")
        
        # Draw temporary rectangle
        canvas_x1 = self.image_x + self.start_x * self.scale_factor
        canvas_y1 = self.image_y + self.start_y * self.scale_factor
        canvas_x2 = self.image_x + current_x * self.scale_factor
        canvas_y2 = self.image_y + current_y * self.scale_factor
        
        self.canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline='green', width=2, tags="temp_rect"
        )
    
    def on_mouse_release(self, event):
        """Handle mouse release events"""
        if not self.drawing:
            return
        
        # Convert canvas coordinates to image coordinates
        end_x = (event.x - self.image_x) / self.scale_factor
        end_y = (event.y - self.image_y) / self.scale_factor
        
        # Ensure coordinates are within image bounds
        if self.current_image is not None:
            img_height, img_width = self.current_image.shape[:2]
            end_x = max(0, min(end_x, img_width))
            end_y = max(0, min(end_y, img_height))
        
        # Remove temporary rectangle
        self.canvas.delete("temp_rect")
        
        # Create annotation if rectangle is large enough
        width = abs(end_x - self.start_x)
        height = abs(end_y - self.start_y)
        
        if width > self.config.min_bbox_size and height > self.config.min_bbox_size:
            # Normalize coordinates
            img_height, img_width = self.current_image.shape[:2]
            x1 = min(self.start_x, end_x) / img_width
            y1 = min(self.start_y, end_y) / img_height
            x2 = max(self.start_x, end_x) / img_width
            y2 = max(self.start_y, end_y) / img_height
            
            # Create annotation using the new Annotation dataclass
            annotation = Annotation(
                bbox=(x1, y1, x2, y2),
                class_name=self.default_class_var.get() or self.config.default_class,
                confidence=self.config.default_confidence,
                created_at=datetime.now().isoformat(),
                modified_at=datetime.now().isoformat()
            )
            
            self.annotations.append(annotation)
            self.stats['total_annotations'] += 1
            self.update_annotation_list()
            self.display_image_on_canvas()
            self.update_statistics()
            self.activity_var.set(f"✅ Added annotation {len(self.annotations)}")
            self.root.after(2000, lambda: self.activity_var.set(""))
        
        self.drawing = False
        self.update_status("Ready")
    
    def delete_selected(self):
        """Delete selected annotations"""
        selected_indices = list(self.annotation_listbox.curselection())
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select annotation(s) to delete")
            return
        
        # Confirm deletion
        count = len(selected_indices)
        if count > 1:
            if not messagebox.askyesno("Confirm Deletion", f"Delete {count} selected annotations?"):
                return
        else:
            if not messagebox.askyesno("Confirm Deletion", "Delete selected annotation?"):
                return
        
        # Delete in reverse order to maintain indices
        for index in sorted(selected_indices, reverse=True):
            if 0 <= index < len(self.annotations):
                del self.annotations[index]
        
        # Clear selection
        self.clear_selection()
        self.update_annotation_list()
        self.display_image_on_canvas()
        self.update_statistics()
        
        self.update_status(f"🗑️ Deleted {count} annotation(s)")
        self.activity_var.set(f"🗑️ Deleted {count} annotation(s)")
        self.root.after(2000, lambda: self.activity_var.set(""))
    
    def clear_annotations(self):
        """Clear all annotations for current image"""
        if messagebox.askyesno("Confirm", "Clear all annotations for this image?"):
            self.annotations = []
            self.update_annotation_list()
            self.display_image_on_canvas()
            self.update_status("All annotations cleared")
    
    def update_annotation_list(self):
        """Update the annotation listbox with enhanced information"""
        self.annotation_listbox.delete(0, tk.END)
        
        for i, annotation in enumerate(self.annotations):
            x1, y1, x2, y2 = annotation.bbox
            width = (x2 - x1) * 100
            height = (y2 - y1) * 100
            
            # Format creation time
            created_time = ""
            if annotation.created_at:
                try:
                    dt = datetime.fromisoformat(annotation.created_at.replace('Z', '+00:00'))
                    created_time = dt.strftime('%H:%M:%S')
                except:
                    created_time = "Unknown"
            
            list_text = f"{annotation.class_name} {i+1}: ({x1:.3f}, {y1:.3f}) - ({x2:.3f}, {y2:.3f}) [{width:.1f}%×{height:.1f}%] @{created_time}"
            self.annotation_listbox.insert(tk.END, list_text)
        
        # Update details if annotation is selected
        self.update_annotation_details()
    
    def update_annotation_details(self):
        """Update annotation details display"""
        selected_indices = self.annotation_listbox.curselection()
        if not selected_indices:
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, "No annotation selected")
            return
        
        index = selected_indices[0]
        if 0 <= index < len(self.annotations):
            annotation = self.annotations[index]
            x1, y1, x2, y2 = annotation.bbox
            
            details = f"Annotation {index + 1} Details:\n"
            details += f"ID: {annotation.id}\n"
            details += f"Class: {annotation.class_name}\n"
            details += f"Confidence: {annotation.confidence:.2f}\n"
            details += f"Coordinates: ({x1:.4f}, {y1:.4f}) - ({x2:.4f}, {y2:.4f})\n"
            details += f"Size: {((x2-x1)*(y2-y1)*100):.2f}% of image\n"
            details += f"Created: {annotation.created_at}\n"
            details += f"Modified: {annotation.modified_at}"
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(1.0, details)
    
    def on_annotation_select(self, event):
        """Handle annotation listbox selection"""
        selected_indices = list(self.annotation_listbox.curselection())
        
        # Update selected annotations set
        self.selected_annotations.clear()
        for annotation in self.annotations:
            annotation.selected = False
        
        # Select annotations from listbox
        for index in selected_indices:
            if 0 <= index < len(self.annotations):
                self.annotations[index].selected = True
                self.selected_annotations.add(index)
        
        self.display_image_on_canvas()
        self.update_annotation_details()
        
        # Update status
        count = len(selected_indices)
        if count > 0:
            self.update_status(f"🔘 {count} annotation(s) selected")
        else:
            self.update_status("No annotations selected")
    
    def load_image_annotations(self):
        """Load annotations for current image with enhanced format support"""
        if not self.current_image_path:
            return
        
        annotation_file = self.output_dir / f"{Path(self.current_image_path).stem}.json"
        
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                # Handle both old and new annotation formats
                annotations_data = data.get('annotations', [])
                self.annotations = []
                
                for ann_data in annotations_data:
                    # Check if it's the new format (with id, created_at, etc.)
                    if isinstance(ann_data, dict) and 'id' in ann_data:
                        annotation = Annotation(
                            bbox=tuple(ann_data['bbox']),
                            class_name=ann_data.get('class_name', 'logo'),
                            confidence=ann_data.get('confidence', 1.0),
                            created_at=ann_data.get('created_at', ''),
                            modified_at=ann_data.get('modified_at', ''),
                            selected=ann_data.get('selected', False),
                            id=ann_data.get('id', '')
                        )
                    else:
                        # Legacy format support
                        annotation = Annotation(
                            bbox=tuple(ann_data['bbox']),
                            class_name=ann_data.get('class', 'logo'),
                            confidence=ann_data.get('confidence', 1.0),
                            created_at=datetime.now().isoformat(),
                            modified_at=datetime.now().isoformat(),
                            selected=ann_data.get('selected', False)
                        )
                    
                    self.annotations.append(annotation)
                
                # Restore settings if available
                config_data = data.get('config', {})
                if config_data:
                    self.default_class_var.set(config_data.get('default_class', 'logo'))
                    self.annotation_color.set(config_data.get('annotation_color', 'blue'))
                    
            except Exception as e:
                logger.error(f"Error loading annotations: {e}")
                self.annotations = []
        else:
            self.annotations = []
        
        self.update_annotation_list()
    
    def save_annotations(self, silent=False):
        """Save annotations for current image with enhanced features"""
        if not self.current_image_path or not self.output_dir:
            if not silent:
                messagebox.showwarning("No Image", "Please load an image first")
            return
        
        annotation_file = self.output_dir / f"{Path(self.current_image_path).stem}.json"
        
        try:
            # Convert annotations to serializable format
            annotations_data = []
            for ann in self.annotations:
                ann_dict = {
                    'id': ann.id,
                    'bbox': list(ann.bbox),
                    'class_name': ann.class_name,
                    'confidence': ann.confidence,
                    'created_at': ann.created_at,
                    'modified_at': ann.modified_at,
                    'selected': ann.selected
                }
                annotations_data.append(ann_dict)
            
            data = {
                'image_path': self.current_image_path,
                'image_size': self.current_image.shape[:2] if self.current_image is not None else None,
                'annotations': annotations_data,
                'annotation_count': len(self.annotations),
                'saved_at': datetime.now().isoformat(),
                'tool_version': '2.0',
                'config': {
                    'auto_save_enabled': self.auto_save_var.get(),
                    'default_class': self.default_class_var.get(),
                    'annotation_color': self.annotation_color.get()
                }
            }
            
            with open(annotation_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save_time = datetime.now()
            
            if not silent:
                self.update_status(f"💾 Annotations saved to {annotation_file.name}")
                self.activity_var.set("💾 Saved")
                self.root.after(2000, lambda: self.activity_var.set(""))
            else:
                logger.info(f"Auto-saved annotations to {annotation_file}")
            
        except Exception as e:
            if not silent:
                messagebox.showerror("Error", f"Failed to save annotations: {e}")
            else:
                logger.error(f"Auto-save failed: {e}")
    
    def export_to_yolo(self):
        """Export annotations to YOLO format"""
        if not self.output_dir:
            messagebox.showwarning("No Output Directory", "Please load images first")
            return
        
        yolo_dir = self.output_dir.parent / "yolo_annotations"
        yolo_dir.mkdir(exist_ok=True)
        
        # Process all images
        exported_count = 0
        
        for image_path in self.image_files:
            annotation_file = self.output_dir / f"{Path(image_path).stem}.json"
            
            if annotation_file.exists():
                try:
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        annotations = data.get('annotations', [])
                    
                    # Convert to YOLO format
                    yolo_annotations = []
                    for annotation in annotations:
                        bbox = annotation['bbox']
                        x1, y1, x2, y2 = bbox
                        
                        # Convert to YOLO format (center_x, center_y, width, height)
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                    
                    # Save YOLO annotation file
                    yolo_file = yolo_dir / f"{Path(image_path).stem}.txt"
                    with open(yolo_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    exported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error exporting {image_path}: {e}")
        
        messagebox.showinfo("Export Complete", f"Exported {exported_count} annotations to YOLO format")
        self.update_status(f"Exported {exported_count} annotations to YOLO format")
    
    def update_image_info(self):
        """Update image information display"""
        if not self.current_image_path:
            self.info_text.delete(1.0, tk.END)
            return
        
        info = f"Image: {Path(self.current_image_path).name}\n"
        info += f"Size: {self.current_image.shape[1]}x{self.current_image.shape[0]}\n"
        info += f"Annotations: {len(self.annotations)}\n"
        info += f"Scale: {self.scale_factor:.2f}\n\n"
        
        info += "Annotations:\n"
        for i, annotation in enumerate(self.annotations):
            bbox = annotation.bbox
            x1, y1, x2, y2 = bbox
            width = (x2 - x1) * 100
            height = (y2 - y1) * 100
            info += f"  {i+1}. ({x1:.3f}, {y1:.3f}) - ({x2:.3f}, {y2:.3f})\n"
            info += f"     Size: {width:.1f}% x {height:.1f}%\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
    
    def update_statistics(self):
        """Update statistics display"""
        if hasattr(self, 'session_text'):
            session_info = f"Session Started: {self.stats['session_start'].strftime('%H:%M:%S')}\n"
            session_info += f"Images Processed: {self.stats['images_processed']}\n"
            session_info += f"Total Annotations: {self.stats['total_annotations']}\n"
            session_info += f"Current Image: {self.current_index + 1}/{len(self.image_files) if self.image_files else 0}\n"
            session_info += f"Annotations in Current: {len(self.annotations)}\n"
            
            if self.stats['last_activity']:
                session_info += f"Last Activity: {self.stats['last_activity'].strftime('%H:%M:%S')}\n"
            
            if self.last_save_time:
                session_info += f"Last Save: {self.last_save_time.strftime('%H:%M:%S')}\n"
            
            self.session_text.delete(1.0, tk.END)
            self.session_text.insert(1.0, session_info)
        
        if hasattr(self, 'quality_text'):
            quality_info = self.calculate_quality_metrics()
            self.quality_text.delete(1.0, tk.END)
            self.quality_text.insert(1.0, quality_info)
    
    def calculate_quality_metrics(self):
        """Calculate annotation quality metrics"""
        if not self.annotations:
            return "No annotations to analyze"
        
        # Calculate various quality metrics
        bbox_sizes = []
        aspect_ratios = []
        
        for ann in self.annotations:
            x1, y1, x2, y2 = ann.bbox
            width = x2 - x1
            height = y2 - y1
            bbox_sizes.append(width * height)
            aspect_ratios.append(width / height if height > 0 else 1)
        
        # Calculate statistics
        avg_size = sum(bbox_sizes) / len(bbox_sizes) if bbox_sizes else 0
        size_variance = sum((s - avg_size) ** 2 for s in bbox_sizes) / len(bbox_sizes) if bbox_sizes else 0
        
        quality_info = f"Annotation Quality Metrics:\n"
        quality_info += f"Average Size: {avg_size:.3f}\n"
        quality_info += f"Size Variance: {size_variance:.6f}\n"
        quality_info += f"Min Aspect Ratio: {min(aspect_ratios):.2f}\n"
        quality_info += f"Max Aspect Ratio: {max(aspect_ratios):.2f}\n"
        
        # Quality score (0-100)
        quality_score = 100
        if size_variance > 0.01:
            quality_score -= 20
        if any(ar < 0.1 or ar > 10 for ar in aspect_ratios):
            quality_score -= 15
        if avg_size < 0.001:
            quality_score -= 10
        
        quality_info += f"Quality Score: {quality_score}/100\n"
        
        return quality_info
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def on_key_press(self, event):
        """Handle enhanced keyboard shortcuts"""
        # Navigation shortcuts
        if event.keysym == 'Right' or event.keysym == 'n':
            self.next_image()
        elif event.keysym == 'Left' or event.keysym == 'p':
            self.previous_image()
        
        # File operations
        elif event.keysym == 's' and event.state & 0x4:  # Ctrl+S
            self.save_annotations()
        elif event.keysym == 'o' and event.state & 0x4:  # Ctrl+O
            self.load_images()
        
        # Annotation operations
        elif event.keysym == 'c':
            self.start_annotation()
        elif event.keysym == 'Delete':
            self.delete_selected()
        elif event.keysym == 'a':
            self.clear_annotations()
        elif event.keysym == 'm':
            self.toggle_multi_select()
        elif event.keysym == 'l' and event.state & 0x4:  # Ctrl+L
            self.assign_same_label()
        
        # History operations
        elif event.keysym == 'z' and event.state & 0x4:  # Ctrl+Z
            if event.state & 0x1:  # Ctrl+Shift+Z
                self.redo()
            else:
                self.undo()
        elif event.keysym == 'y' and event.state & 0x4:  # Ctrl+Y
            self.redo()
        
        # Zoom operations
        elif event.keysym == 'plus' or event.keysym == 'equal':
            self.zoom_in()
        elif event.keysym == 'minus':
            self.zoom_out()
        elif event.keysym == '0':
            self.fit_to_canvas()
        
        # Tab navigation
        elif event.keysym == 'Tab':
            current_tab = self.notebook.index(self.notebook.select())
            next_tab = (current_tab + 1) % self.notebook.index("end")
            self.notebook.select(next_tab)
        
        # Statistics update
        elif event.keysym == 'r':
            self.update_statistics()
        
        # Focus management
        elif event.keysym == 'Escape':
            self.clear_selection()
    
    def toggle_multi_select(self):
        """Toggle multi-selection mode"""
        self.multi_select_mode = self.multi_select_var.get()
        if self.multi_select_mode:
            self.update_status("🔘 Multi-select mode enabled - Click annotations to select multiple")
            self.activity_var.set("🔘 Multi-select mode")
            self.root.after(3000, lambda: self.activity_var.set(""))
        else:
            self.clear_selection()
            self.update_status("Single selection mode")
    
    def clear_selection(self):
        """Clear all selected annotations"""
        self.selected_annotations.clear()
        for annotation in self.annotations:
            annotation.selected = False
        self.display_image_on_canvas()
        self.update_annotation_details()
    
    def assign_same_label(self):
        """Assign the same label to all selected annotations"""
        selected_indices = list(self.annotation_listbox.curselection())
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select annotations to assign label")
            return
        
        # Ask for the label to assign
        label = simpledialog.askstring("Assign Label", 
                                     f"Enter label for {len(selected_indices)} selected annotation(s):",
                                     initialvalue=self.default_class_var.get() or self.config.default_class)
        
        if not label:
            return
        
        # Update all selected annotations with the same label
        updated_count = 0
        for index in selected_indices:
            if 0 <= index < len(self.annotations):
                self.annotations[index].class_name = label
                self.annotations[index].modified_at = datetime.now().isoformat()
                updated_count += 1
        
        # Update display
        self.update_annotation_list()
        self.display_image_on_canvas()
        self.update_statistics()
        
        self.update_status(f"🏷️ Assigned label '{label}' to {updated_count} annotation(s)")
        self.activity_var.set(f"🏷️ Label assigned: {label}")
        self.root.after(2000, lambda: self.activity_var.set(""))
    
    def get_clicked_annotation(self, x, y):
        """Get the annotation that was clicked on"""
        # Convert canvas coordinates to image coordinates
        img_x = (x - self.image_x) / self.scale_factor
        img_y = (y - self.image_y) / self.scale_factor
        
        # Check each annotation in reverse order (top-most first)
        for i in range(len(self.annotations) - 1, -1, -1):
            annotation = self.annotations[i]
            x1, y1, x2, y2 = annotation.bbox
            
            # Convert to image coordinates
            img_x1 = x1 * self.current_image.shape[1] if self.current_image is not None else 0
            img_y1 = y1 * self.current_image.shape[0] if self.current_image is not None else 0
            img_x2 = x2 * self.current_image.shape[1] if self.current_image is not None else 0
            img_y2 = y2 * self.current_image.shape[0] if self.current_image is not None else 0
            
            if img_x1 <= img_x <= img_x2 and img_y1 <= img_y <= img_y2:
                return i
        
        return None
    
    def toggle_annotation_selection(self, annotation_index):
        """Toggle selection of an annotation"""
        if annotation_index in self.selected_annotations:
            self.selected_annotations.remove(annotation_index)
            self.annotations[annotation_index].selected = False
        else:
            self.selected_annotations.add(annotation_index)
            self.annotations[annotation_index].selected = True
        
        # Update listbox selection
        self.update_listbox_selection()
        self.display_image_on_canvas()
        self.update_annotation_details()
        
        # Update status
        count = len(self.selected_annotations)
        if count > 0:
            self.update_status(f"🔘 {count} annotation(s) selected")
        else:
            self.update_status("No annotations selected")
    
    def update_listbox_selection(self):
        """Update listbox selection to match selected annotations"""
        self.annotation_listbox.selection_clear(0, tk.END)
        for index in self.selected_annotations:
            if 0 <= index < len(self.annotations):
                self.annotation_listbox.selection_set(index)
    
    def run(self):
        """Run the unified annotation tool"""
        self.root.mainloop()

def main():
    """Main function to run the unified annotation tool"""
    app = UnifiedAnnotationTool()
    app.run()

if __name__ == "__main__":
    main()
