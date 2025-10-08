"""
Batch Annotation Tool for Japanese Sport Sponsor Logo Detection
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchAnnotationTool:
    """
    Batch annotation tool for processing multiple images efficiently
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Batch Logo Annotation Tool")
        self.root.geometry("1200x800")
        
        # Data
        self.image_files = []
        self.annotation_data = {}
        self.current_batch = []
        self.batch_size = 10
        self.current_batch_index = 0
        
        # Directories
        self.input_dir = None
        self.output_dir = None
        self.yolo_output_dir = None
        
        # UI setup
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame - Controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Directory selection
        dir_frame = ttk.LabelFrame(top_frame, text="Directory Setup")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(dir_frame, text="Select Input Directory", 
                  command=self.select_input_dir).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(dir_frame, text="Select Output Directory", 
                  command=self.select_output_dir).pack(side=tk.LEFT, padx=(0, 5))
        
        self.input_label = ttk.Label(dir_frame, text="No input directory selected")
        self.input_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Batch controls
        batch_frame = ttk.LabelFrame(top_frame, text="Batch Processing")
        batch_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value="10")
        batch_size_entry = ttk.Entry(batch_frame, textvariable=self.batch_size_var, width=5)
        batch_size_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(batch_frame, text="Load Images", 
                  command=self.load_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(batch_frame, text="Process Batch", 
                  command=self.process_batch).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(batch_frame, text="Export All", 
                  command=self.export_all).pack(side=tk.LEFT, padx=(0, 5))
        
        # Progress frame
        progress_frame = ttk.LabelFrame(top_frame, text="Progress")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(pady=(0, 5))
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame - Image list
        left_frame = ttk.LabelFrame(content_frame, text="Images")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Image list with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        scrollbar1 = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar1.set)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right frame - Batch info
        right_frame = ttk.LabelFrame(content_frame, text="Batch Information")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Batch info
        self.batch_info_text = tk.Text(right_frame, height=15, width=40)
        self.batch_info_text.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Batch controls
        batch_controls = ttk.Frame(right_frame)
        batch_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(batch_controls, text="Previous Batch", 
                  command=self.previous_batch).pack(fill=tk.X, pady=2)
        ttk.Button(batch_controls, text="Next Batch", 
                  command=self.next_batch).pack(fill=tk.X, pady=2)
        ttk.Button(batch_controls, text="Auto-Annotate", 
                  command=self.auto_annotate).pack(fill=tk.X, pady=2)
        ttk.Button(batch_controls, text="Validate Batch", 
                  command=self.validate_batch).pack(fill=tk.X, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_input_dir(self):
        """Select input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir = Path(directory)
            self.input_label.config(text=f"Input: {self.input_dir.name}")
            self.update_status(f"Input directory: {directory}")
    
    def select_output_dir(self):
        """Select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = Path(directory)
            self.yolo_output_dir = self.output_dir / "yolo_annotations"
            self.yolo_output_dir.mkdir(exist_ok=True)
            self.update_status(f"Output directory: {directory}")
    
    def load_images(self):
        """Load all images from input directory"""
        if not self.input_dir:
            messagebox.showwarning("No Directory", "Please select an input directory first")
            return
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.image_files = []
        
        for ext in image_extensions:
            self.image_files.extend(self.input_dir.glob(f"*{ext}"))
            self.image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        self.image_files = [str(f) for f in self.image_files]
        
        if not self.image_files:
            messagebox.showwarning("No Images", "No image files found in the input directory")
            return
        
        # Update UI
        self.image_listbox.delete(0, tk.END)
        for i, image_path in enumerate(self.image_files):
            filename = Path(image_path).name
            self.image_listbox.insert(tk.END, f"{i+1:03d}: {filename}")
        
        # Create batches
        self.create_batches()
        self.update_batch_info()
        self.update_status(f"Loaded {len(self.image_files)} images")
    
    def create_batches(self):
        """Create batches from image files"""
        self.batch_size = int(self.batch_size_var.get())
        self.current_batch_index = 0
        
        # Split images into batches
        self.batches = []
        for i in range(0, len(self.image_files), self.batch_size):
            batch = self.image_files[i:i + self.batch_size]
            self.batches.append(batch)
        
        self.current_batch = self.batches[0] if self.batches else []
    
    def update_batch_info(self):
        """Update batch information display"""
        if not self.batches:
            self.batch_info_text.delete(1.0, tk.END)
            return
        
        info = f"Total Images: {len(self.image_files)}\n"
        info += f"Batch Size: {self.batch_size}\n"
        info += f"Total Batches: {len(self.batches)}\n"
        info += f"Current Batch: {self.current_batch_index + 1}/{len(self.batches)}\n\n"
        
        info += f"Current Batch Images:\n"
        for i, image_path in enumerate(self.current_batch):
            filename = Path(image_path).name
            status = "✓" if self.is_annotated(image_path) else "○"
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
    
    def previous_batch(self):
        """Go to previous batch"""
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self.current_batch = self.batches[self.current_batch_index]
            self.update_batch_info()
            self.update_status(f"Batch {self.current_batch_index + 1}/{len(self.batches)}")
    
    def next_batch(self):
        """Go to next batch"""
        if self.current_batch_index < len(self.batches) - 1:
            self.current_batch_index += 1
            self.current_batch = self.batches[self.current_batch_index]
            self.update_batch_info()
            self.update_status(f"Batch {self.current_batch_index + 1}/{len(self.batches)}")
    
    def process_batch(self):
        """Process current batch"""
        if not self.current_batch:
            messagebox.showwarning("No Batch", "No batch to process")
            return
        
        # Open annotation tool for each image in batch
        for image_path in self.current_batch:
            if not self.is_annotated(image_path):
                self.open_annotation_tool(image_path)
                break
        
        self.update_batch_info()
    
    def open_annotation_tool(self, image_path: str):
        """Open annotation tool for specific image"""
        # This would open the main annotation tool
        # For now, just show a message
        messagebox.showinfo("Annotation Tool", f"Opening annotation tool for: {Path(image_path).name}")
    
    def auto_annotate(self):
        """Auto-annotate using AI model"""
        if not self.current_batch:
            messagebox.showwarning("No Batch", "No batch to process")
            return
        
        # This would use the trained model to auto-annotate
        messagebox.showinfo("Auto-Annotation", "Auto-annotation feature coming soon!")
    
    def validate_batch(self):
        """Validate annotations in current batch"""
        if not self.current_batch:
            messagebox.showwarning("No Batch", "No batch to validate")
            return
        
        validation_results = []
        
        for image_path in self.current_batch:
            if self.is_annotated(image_path):
                result = self.validate_annotations(image_path)
                validation_results.append(result)
        
        # Show validation results
        if validation_results:
            valid_count = sum(1 for r in validation_results if r['valid'])
            total_count = len(validation_results)
            
            messagebox.showinfo("Validation Results", 
                              f"Validated {valid_count}/{total_count} annotations\n"
                              f"Success rate: {valid_count/total_count*100:.1f}%")
        else:
            messagebox.showinfo("Validation Results", "No annotations found in current batch")
    
    def validate_annotations(self, image_path: str) -> Dict:
        """Validate annotations for a specific image"""
        annotation_file = self.output_dir / f"{Path(image_path).stem}.json"
        
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                annotations = data.get('annotations', [])
            
            # Basic validation
            valid = True
            issues = []
            
            for i, annotation in enumerate(annotations):
                bbox = annotation.get('bbox', [])
                if len(bbox) != 4:
                    valid = False
                    issues.append(f"Annotation {i+1}: Invalid bbox format")
                    continue
                
                x1, y1, x2, y2 = bbox
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    valid = False
                    issues.append(f"Annotation {i+1}: Bbox coordinates out of range")
            
            return {
                'image_path': image_path,
                'valid': valid,
                'issues': issues,
                'annotation_count': len(annotations)
            }
            
        except Exception as e:
            return {
                'image_path': image_path,
                'valid': False,
                'issues': [f"Error reading file: {e}"],
                'annotation_count': 0
            }
    
    def export_all(self):
        """Export all annotations to YOLO format"""
        if not self.output_dir or not self.yolo_output_dir:
            messagebox.showwarning("No Output Directory", "Please select output directory first")
            return
        
        # Process all images
        exported_count = 0
        total_images = len(self.image_files)
        
        self.progress_var.set(0)
        self.progress_label.config(text="Exporting annotations...")
        
        for i, image_path in enumerate(self.image_files):
            try:
                annotation_file = self.output_dir / f"{Path(image_path).stem}.json"
                
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        annotations = data.get('annotations', [])
                    
                    # Convert to YOLO format
                    yolo_annotations = []
                    for annotation in annotations:
                        bbox = annotation['bbox']
                        x1, y1, x2, y2 = bbox
                        
                        # Convert to YOLO format
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        yolo_annotations.append(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                    
                    # Save YOLO annotation file
                    yolo_file = self.yolo_output_dir / f"{Path(image_path).stem}.txt"
                    with open(yolo_file, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    exported_count += 1
                
                # Update progress
                progress = (i + 1) / total_images * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Exporting... {i+1}/{total_images}")
                self.root.update_idletasks()
                
            except Exception as e:
                logger.error(f"Error exporting {image_path}: {e}")
        
        self.progress_var.set(100)
        self.progress_label.config(text="Export complete")
        
        messagebox.showinfo("Export Complete", 
                          f"Exported {exported_count} annotations to YOLO format\n"
                          f"Output directory: {self.yolo_output_dir}")
        
        self.update_status(f"Exported {exported_count} annotations")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def run(self):
        """Run the batch annotation tool"""
        self.root.mainloop()

def main():
    """Main function to run the batch annotation tool"""
    app = BatchAnnotationTool()
    app.run()

if __name__ == "__main__":
    main()
