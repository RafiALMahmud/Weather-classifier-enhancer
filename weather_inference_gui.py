"""
Weather Classifier GUI - Single Image Inference with Upload Button
Shows top 2 predictions, confidence, and inference time
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import time
import onnxruntime as ort
from pathlib import Path

# Class names
CLASS_NAMES = ["rainy", "foggy", "sand", "snow"]

# Color mapping for each weather class (BGR format for display)
CLASS_COLORS = {
    "rainy": (100, 100, 255),    # Blue
    "foggy": (200, 200, 200),    # Gray
    "sand": (0, 200, 255),       # Orange/Yellow
    "snow": (255, 255, 255)      # White
}


class WeatherClassifierGUI:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Weather Classifier - Single Image Inference")
        self.root.geometry("900x750")
        self.root.configure(bg='#2b2b2b')
        
        # Load ONNX model
        self.load_model(model_path)
        
        # Create GUI elements
        self.create_widgets()
        
        # Current image
        self.current_image = None
        self.current_image_path = None
    
    def load_model(self, model_path):
        """Load ONNX model"""
        try:
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.provider = self.session.get_providers()[0]
            
            print(f"Model loaded successfully!")
            print(f"Provider: {self.provider}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create GUI elements"""
        
        # Title
        title_label = tk.Label(
            self.root, 
            text="Weather Classification System",
            font=("Arial", 24, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        title_label.pack(pady=20)
        
        # Model info
        info_label = tk.Label(
            self.root,
            text=f"Model: ONNX | Device: {self.provider}",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#aaaaaa'
        )
        info_label.pack()
        
        # Upload button
        upload_btn = tk.Button(
            self.root,
            text="📁 Upload Image",
            font=("Arial", 14, "bold"),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            command=self.upload_image,
            padx=30,
            pady=15,
            cursor="hand2"
        )
        upload_btn.pack(pady=20)
        
        # Image display frame
        self.image_frame = tk.Frame(self.root, bg='#1e1e1e', bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="No image loaded\n\nClick 'Upload Image' to start",
            font=("Arial", 14),
            bg='#1e1e1e',
            fg='#888888'
        )
        self.image_label.pack(expand=True)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.results_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Top prediction label
        self.prediction_label = tk.Label(
            self.results_frame,
            text="1st: -",
            font=("Arial", 18, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        self.prediction_label.pack()
        
        # Second best prediction label
        self.second_prediction_label = tk.Label(
            self.results_frame,
            text="2nd: -",
            font=("Arial", 14, "bold"),
            bg='#2b2b2b',
            fg='#aaaaaa'
        )
        self.second_prediction_label.pack(pady=5)
        
        # Inference time label
        self.time_label = tk.Label(
            self.results_frame,
            text="Inference Time: -",
            font=("Arial", 12),
            bg='#2b2b2b',
            fg='#888888'
        )
        self.time_label.pack(pady=5)
        
        # All predictions frame
        self.all_predictions_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.all_predictions_frame.pack(pady=10, fill=tk.X, padx=40)
        
        self.all_predictions_label = tk.Label(
            self.all_predictions_frame,
            text="",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#cccccc',
            justify=tk.LEFT
        )
        self.all_predictions_label.pack()
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.process_image(file_path)
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize (ImageNet stats)
        img_normalized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_tensor = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)
        
        return img_tensor, img
    
    def process_image(self, image_path):
        """Process and classify the image"""
        try:
            # Preprocess
            input_tensor, original_img = self.preprocess_image(image_path)
            
            # Inference with timing
            start_time = time.time()
            outputs = self.session.run(None, {self.input_name: input_tensor})[0]
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Compute softmax for confidence
            exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))  # Numerical stability
            softmax_probs = exp_outputs / np.sum(exp_outputs)
            
            # Get top 2 predictions
            top2_indices = np.argsort(softmax_probs)[-2:][::-1]  # Get indices of top 2, descending
            
            pred_class_1 = CLASS_NAMES[top2_indices[0]]
            confidence_1 = softmax_probs[top2_indices[0]] * 100
            
            pred_class_2 = CLASS_NAMES[top2_indices[1]]
            confidence_2 = softmax_probs[top2_indices[1]] * 100
            
            # Display image
            self.display_image(original_img, pred_class_1, confidence_1, pred_class_2, confidence_2)
            
            # Update results
            self.update_results(pred_class_1, confidence_1, pred_class_2, confidence_2, 
                              inference_time, softmax_probs)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
    
    def display_image(self, img, prediction1, confidence1, prediction2, confidence2):
        """Display the image with top 2 predictions overlay"""
        # Resize for display (max 600x400)
        h, w = img.shape[:2]
        max_width, max_height = 600, 400
        
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        img_display = cv2.resize(img, (new_w, new_h))
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        
        # Get colors for classes
        color1 = CLASS_COLORS.get(prediction1, (255, 255, 255))
        color2 = CLASS_COLORS.get(prediction2, (200, 200, 200))
        
        # Draw background rectangle for text (taller to fit both predictions)
        cv2.rectangle(img_rgb, (10, 10), (280, 110), (0, 0, 0), -1)
        cv2.rectangle(img_rgb, (10, 10), (280, 110), color1, 2)
        
        # Draw first prediction
        text1 = f"1st: {prediction1.upper()}"
        conf_text1 = f"{confidence1:.1f}%"
        cv2.putText(img_rgb, text1, (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, color1, 2)
        cv2.putText(img_rgb, conf_text1, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw second prediction
        text2 = f"2nd: {prediction2.upper()}"
        conf_text2 = f"{confidence2:.1f}%"
        cv2.putText(img_rgb, text2, (20, 85), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, color2, 1)
        cv2.putText(img_rgb, conf_text2, (20, 102), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def update_results(self, prediction1, confidence1, prediction2, confidence2, 
                       inference_time, all_probs):
        """Update result labels"""
        # First prediction
        color_hex_1 = self.bgr_to_hex(CLASS_COLORS.get(prediction1, (255, 255, 255)))
        self.prediction_label.configure(
            text=f"1st: {prediction1.upper()} ({confidence1:.2f}%)",
            fg=color_hex_1
        )
        
        # Second prediction
        color_hex_2 = self.bgr_to_hex(CLASS_COLORS.get(prediction2, (200, 200, 200)))
        self.second_prediction_label.configure(
            text=f"2nd: {prediction2.upper()} ({confidence2:.2f}%)",
            fg=color_hex_2
        )
        
        # Inference time
        time_color = '#00ff00' if inference_time < 1000 else '#ffaa00'
        self.time_label.configure(
            text=f"Inference Time: {inference_time:.2f} ms ({1000/inference_time:.1f} FPS)",
            fg=time_color
        )
        
        # All predictions
        all_pred_text = "All Class Probabilities:\n"
        sorted_indices = np.argsort(all_probs)[::-1]  # Sort descending
        
        for idx in sorted_indices:
            class_name = CLASS_NAMES[idx]
            prob = all_probs[idx] * 100
            bar_length = int(prob / 5)  # Scale bar
            bar = "█" * bar_length + "░" * (20 - bar_length)
            all_pred_text += f"{class_name:8s}: {bar} {prob:5.2f}%\n"
        
        self.all_predictions_label.configure(text=all_pred_text)
    
    def bgr_to_hex(self, bgr_color):
        """Convert BGR tuple to hex color"""
        b, g, r = bgr_color
        return f'#{r:02x}{g:02x}{b:02x}'


def main():
    # Default model path
    default_model = "./outputs/model_quantized.onnx"
    
    # Check if model exists
    if not Path(default_model).exists():
        # Try non-quantized version
        default_model = "./outputs/model.onnx"
        if not Path(default_model).exists():
            print("Error: Model not found!")
            print("Please train the model first using:")
            print("  python weather_classifier_complete.py --mode train")
            return
    
    print(f"Loading model: {default_model}")
    
    # Create GUI
    root = tk.Tk()
    app = WeatherClassifierGUI(root, default_model)
    root.mainloop()


if __name__ == '__main__':
    main()