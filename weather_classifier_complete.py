"""
Deployable 4-Class Weather Classifier
Handles DAWN (rainy, sand, snow via XML) + FoggyCityscape (foggy via JSON)
Realtime inference target: <1 second per frame on CPU/GPU

Based on: "Confidence-Adaptive, Retinex-Enhanced Self-Training for 
Robust Object Detection in Adverse Weather" thesis paper
"""

import os
import json
import xml.etree.ElementTree as ET
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import onnx
import onnxruntime as ort

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Class mapping
CLASS_NAMES = ["rainy", "foggy", "sand", "snow"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class WeatherDataset(Dataset):
    """Dataset handler for DAWN XML and FoggyCityscape JSON"""
    
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def parse_dawn_xml(xml_path: str) -> str:
    """Parse DAWN PASCAL VOC XML to extract weather class"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Try multiple possible tag structures
        # Structure 1: <object><name>
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None and name.text:
                return name.text.lower()
        
        # Structure 2: Direct <name> tag
        name = root.find('name')
        if name is not None and name.text:
            return name.text.lower()
        
        # Structure 3: <annotation><object><name>
        for obj in root.findall('.//object'):
            name = obj.find('name')
            if name is not None and name.text:
                return name.text.lower()
        
        # Structure 4: Check filename tag as fallback
        filename = root.find('filename')
        if filename is not None and filename.text:
            fname = filename.text.lower()
            if 'rain' in fname:
                return 'rain'
            elif 'sand' in fname:
                return 'sand'
            elif 'snow' in fname:
                return 'snow'
        
    except Exception as e:
        # Silently skip errors - we'll debug at higher level
        pass
    
    return None


def parse_foggycityscape_json(json_path: str) -> bool:
    """Parse FoggyCityscape JSON - returns True if valid foggy image"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check various possible JSON structures for Cityscapes format
        # Could be 'objects', 'imgHeight', or just any valid JSON structure
        if isinstance(data, dict):
            # If it has standard Cityscapes fields, it's valid
            if 'imgHeight' in data or 'imgWidth' in data:
                return True
            # If it has objects array (even if empty for some images)
            if 'objects' in data:
                return True
            # If it's any non-empty dict, assume it's labeled
            if len(data) > 0:
                return True
    except Exception as e:
        # Don't print every error, just skip silently
        pass
    
    return False


def collect_dawn_samples(dawn_root: str, classes: List[str]) -> List[Tuple[str, int]]:
    """Collect samples from DAWN dataset (rainy, sand, snow, foggy)
    
    Note: DAWN XML files contain object detection labels (car, person, etc.),
    NOT weather labels. The weather class is determined by the folder structure.
    We use the folder name as the weather label and collect all images directly.
    """
    samples = []
    
    # Map lowercase class names to actual DAWN folder names
    dawn_folder_map = {
        'rainy': 'Rain',
        'sand': 'Sand',
        'snow': 'Snow',
        'foggy': 'Fog'  # Added fog support
    }
    
    for cls in classes:
        # Get actual folder name (capitalized)
        dawn_folder = dawn_folder_map.get(cls)
        if not dawn_folder:
            continue
        
        cls_dir = os.path.join(dawn_root, dawn_folder)
        
        # Check if folder exists
        if not os.path.exists(cls_dir):
            print(f"Warning: Folder not found: {cls_dir}")
            continue
        
        # Images are in the nested subfolder: e.g., Rain/Rain/, Sand/Sand/, Snow/Snow/, Fog/Fog/
        img_base_dir = os.path.join(cls_dir, dawn_folder)
        
        if not os.path.exists(img_base_dir):
            print(f"Warning: Image directory not found: {img_base_dir}")
            # Try parent directory as fallback
            img_base_dir = cls_dir
        
        if not os.path.exists(img_base_dir):
            print(f"Error: Cannot find image directory for {dawn_folder}")
            continue
        
        # Collect all image files directly from the folder
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(img_base_dir).glob(ext)))
        
        print(f"Found {len(image_files)} images in {dawn_folder}")
        
        # Add all images with the folder-based weather label
        for img_file in image_files:
            samples.append((str(img_file), CLASS_TO_IDX[cls]))
    
    return samples


def collect_foggycityscape_samples(foggy_root: str) -> List[Tuple[str, int]]:
    """Collect samples from FoggyCityscape dataset (foggy class)"""
    samples = []
    
    images_dir = os.path.join(foggy_root, 'images')
    labels_dir = os.path.join(foggy_root, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Warning: FoggyCityscape directories not found")
        print(f"  Images dir: {images_dir} (exists: {os.path.exists(images_dir)})")
        print(f"  Labels dir: {labels_dir} (exists: {os.path.exists(labels_dir)})")
        return samples
    
    # Find all JSON label files
    json_files = list(Path(labels_dir).glob('*.json'))
    print(f"Found {len(json_files)} JSON files in FoggyCityscape")
    
    valid_json_count = 0
    found_image_count = 0
    
    for json_file in json_files:
        if parse_foggycityscape_json(str(json_file)):
            valid_json_count += 1
            
            # Extract base identifier from JSON filename
            # aachen_000000_000019_gtFine_polygons.json -> aachen_000000_000019
            json_stem = json_file.stem
            
            # Remove _gtFine_polygons or similar suffixes
            if '_gtFine' in json_stem:
                base_id = json_stem.split('_gtFine')[0]
            else:
                base_id = json_stem
            
            # Look for matching images with pattern: base_id_leftImg8bit_foggy_beta_*.png
            # Use glob to find all foggy variants
            pattern = f"{base_id}_leftImg8bit_foggy_beta_*.png"
            matching_images = list(Path(images_dir).glob(pattern))
            
            # If found, add all foggy variants (or just pick one, e.g., beta_0.01)
            if matching_images:
                # Use the middle beta value (0.01) if available, otherwise first one
                selected_img = None
                for img in matching_images:
                    if 'beta_0.01' in img.name:
                        selected_img = img
                        break
                
                if not selected_img:
                    selected_img = matching_images[0]
                
                samples.append((str(selected_img), CLASS_TO_IDX['foggy']))
                found_image_count += 1
    
    print(f"  Valid JSON files: {valid_json_count}")
    print(f"  Matched images: {found_image_count}")
    
    if valid_json_count > 0 and found_image_count == 0:
        print(f"  WARNING: Found valid JSONs but no matching images!")
        print(f"  Sample JSON stem: {json_files[0].stem}")
        # List first few image files to help debug
        image_files = list(Path(images_dir).iterdir())[:5]
        print(f"  Sample image files: {[f.name for f in image_files]}")
    
    return samples


def load_datasets(dawn_root: str, foggy_root: str, 
                  train_split: float = 0.8) -> Tuple[List, List]:
    """Load and split datasets into train/test"""
    
    print("\n=== Loading Datasets ===")
    
    # Collect DAWN samples (rainy, sand, snow, fog)
    dawn_samples = collect_dawn_samples(dawn_root, ['rainy', 'sand', 'snow', 'foggy'])
    print(f"DAWN samples collected: {len(dawn_samples)}")
    
    # Collect FoggyCityscape samples (foggy) - additional foggy data
    foggy_samples = collect_foggycityscape_samples(foggy_root)
    print(f"FoggyCityscape samples collected: {len(foggy_samples)}")
    
    # Merge all samples
    all_samples = dawn_samples + foggy_samples
    print(f"Total samples: {len(all_samples)}")
    
    # Print class distribution
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    for _, label in all_samples:
        class_counts[CLASS_NAMES[label]] += 1
    
    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # IMPORTANT: Shuffle with stratification to maintain class balance
    # Group samples by class
    samples_by_class = {i: [] for i in range(len(CLASS_NAMES))}
    for sample in all_samples:
        samples_by_class[sample[1]].append(sample)
    
    # Split each class separately to ensure stratification
    train_samples = []
    test_samples = []
    
    for class_idx, class_samples in samples_by_class.items():
        random.shuffle(class_samples)
        split_idx = int(len(class_samples) * train_split)
        train_samples.extend(class_samples[:split_idx])
        test_samples.extend(class_samples[split_idx:])
    
    # Final shuffle of train/test sets
    random.shuffle(train_samples)
    random.shuffle(test_samples)
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Print class distribution in train/test
    train_class_counts = {cls: 0 for cls in CLASS_NAMES}
    test_class_counts = {cls: 0 for cls in CLASS_NAMES}
    
    for _, label in train_samples:
        train_class_counts[CLASS_NAMES[label]] += 1
    for _, label in test_samples:
        test_class_counts[CLASS_NAMES[label]] += 1
    
    print("\nTrain class distribution:")
    for cls, count in train_class_counts.items():
        print(f"  {cls}: {count}")
    
    print("\nTest class distribution:")
    for cls, count in test_class_counts.items():
        print(f"  {cls}: {count}")
    
    return train_samples, test_samples


def get_transforms(mode='train'):
    """Get image transforms for train/test with stronger augmentation"""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Resize larger first
            transforms.RandomCrop((224, 224)),  # Then random crop
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),  # Increased rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger color jitter
            transforms.RandomGrayscale(p=0.1),  # Occasionally grayscale
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # Random erasing for regularization
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class WeatherClassifier(nn.Module):
    """MobileNetV3-Small based weather classifier with dropout"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        self.backbone = timm.create_model('mobilenetv3_small_100', 
                                          pretrained=pretrained, 
                                          num_classes=0)  # Remove classifier
        
        # Get feature dimension
        num_features = self.backbone.num_features
        
        # Custom classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def train_model(model, train_loader, test_loader, device, 
                epochs=30, lr=1e-3, patience=5, save_dir='./outputs'):
    """Train the weather classifier with stronger regularization and class weighting"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate class weights to handle imbalance
    # Count samples per class from train_loader
    class_counts = [0] * len(CLASS_NAMES)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1
    
    print(f"\nTraining set class counts: {dict(zip(CLASS_NAMES, class_counts))}")
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts)
    class_weights = [total_samples / (len(class_counts) * count) if count > 0 else 1.0 
                     for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Class weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in class_weights]))}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    best_acc = 0.0
    patience_counter = 0
    
    print("\n=== Training Started ===")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/len(train_loader):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        # Per-class validation accuracy
        class_correct = [0] * len(CLASS_NAMES)
        class_total = [0] * len(CLASS_NAMES)
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Track per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # Print per-class validation accuracy
        per_class_acc = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                         for i in range(len(CLASS_NAMES))]
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Train Acc={100.*train_correct/train_total:.2f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        print(f"  Per-class Val Acc: " + 
              ", ".join([f"{CLASS_NAMES[i]}={per_class_acc[i]:.1f}%" 
                        for i in range(len(CLASS_NAMES))]))
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping and checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"  → Best model saved (Val Acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\nTraining completed. Best Val Acc: {best_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate model and print classification metrics with confusion matrix"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\n=== Evaluation ===")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    
    # Per-class metrics
    precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\nPer-Class Metrics:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:10s} - P: {precision_pc[i]*100:.2f}%, "
              f"R: {recall_pc[i]*100:.2f}%, F1: {f1_pc[i]*100:.2f}%, "
              f"Support: {support[i]}")
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nConfusion Matrix:")
    print("Predicted →")
    print(f"{'Actual ↓':12s}" + "".join([f"{cls:10s}" for cls in CLASS_NAMES]))
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:12s}" + "".join([f"{cm[i][j]:10d}" for j in range(len(CLASS_NAMES))]))
    
    # Show misclassification patterns
    print(f"\nMisclassification Analysis:")
    for i, true_cls in enumerate(CLASS_NAMES):
        for j, pred_cls in enumerate(CLASS_NAMES):
            if i != j and cm[i][j] > 0:
                error_rate = 100.0 * cm[i][j] / support[i] if support[i] > 0 else 0
                print(f"  {true_cls} → {pred_cls}: {cm[i][j]} samples ({error_rate:.1f}%)")


def export_models(model, device, save_dir='./outputs'):
    """Export model to TorchScript and ONNX formats"""
    
    print("\n=== Exporting Models ===")
    
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Export TorchScript
    torchscript_path = os.path.join(save_dir, 'model.pt')
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, torchscript_path)
    print(f"TorchScript model saved: {torchscript_path}")
    
    # Export ONNX
    onnx_path = os.path.join(save_dir, 'model.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved: {onnx_path}")
    
    # Quantize ONNX model
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = os.path.join(save_dir, 'model_quantized.onnx')
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized ONNX model saved: {quantized_path}")
    except Exception as e:
        print(f"Warning: ONNX quantization failed: {e}")


def benchmark_onnx(onnx_path: str, num_runs=100):
    """Benchmark ONNX model inference speed"""
    
    print(f"\n=== Benchmarking ONNX Model ===")
    print(f"Model: {onnx_path}")
    
    # Create ONNX Runtime session
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Execution provider: {session.get_providers()[0]}")
    
    # Prepare dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    import time
    latencies = []
    
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        start = time.time()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.time() - start) * 1000)  # Convert to ms
    
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1000.0 / mean_latency
    
    print(f"\nInference Performance:")
    print(f"  Mean latency: {mean_latency:.2f} ms")
    print(f"  Std latency:  {std_latency:.2f} ms")
    print(f"  FPS:          {fps:.2f}")
    print(f"  Target met:   {'✓ YES' if mean_latency < 1000 else '✗ NO'} (<1000ms)")


def realtime_inference(model_path: str, source=0):
    """Run realtime inference on webcam or video"""
    
    print("\n=== Realtime Inference ===")
    print(f"Model: {model_path}")
    print("Press 'q' to quit\n")
    
    # Load ONNX model
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    import time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).numpy()
        
        # Inference
        outputs = session.run(None, {input_name: input_tensor})[0]
        latency = (time.time() - start_time) * 1000
        
        # Get prediction
        pred_idx = np.argmax(outputs[0])
        confidence = np.exp(outputs[0][pred_idx]) / np.sum(np.exp(outputs[0]))
        pred_class = CLASS_NAMES[pred_idx]
        
        # Display results
        text = f"{pred_class} ({confidence*100:.1f}%)"
        latency_text = f"Latency: {latency:.1f}ms"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        cv2.putText(frame, latency_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 0), 2)
        
        cv2.imshow('Weather Classifier', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Weather Classifier Training and Inference')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'],
                       help='Mode: train or infer')
    parser.add_argument('--dawn_root', type=str, 
                       default='E:\\Daddyyaga\\DAWN\\DAWN\\',
                       help='DAWN dataset root path')
    parser.add_argument('--foggy_root', type=str,
                       default='E:\\Daddyyaga\\labeled_dataset\\labeled_dataset\\',
                       help='FoggyCityscape dataset root path')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--model_path', type=str, default='./outputs/model_quantized.onnx',
                       help='Path to model for inference')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source for inference (0 for webcam, or video path)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Load datasets
        train_samples, test_samples = load_datasets(args.dawn_root, args.foggy_root)
        
        if len(train_samples) == 0 or len(test_samples) == 0:
            print("Error: No samples found. Check dataset paths.")
            return
        
        # Create datasets and dataloaders
        train_dataset = WeatherDataset(train_samples, transform=get_transforms('train'))
        test_dataset = WeatherDataset(test_samples, transform=get_transforms('test'))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True)
        
        # Create model
        model = WeatherClassifier(num_classes=len(CLASS_NAMES), pretrained=True)
        model = model.to(device)
        
        # Train model
        model = train_model(model, train_loader, test_loader, device, 
                          epochs=args.epochs, lr=args.lr)
        
        # Evaluate model
        evaluate_model(model, test_loader, device)
        
        # Export models
        export_models(model, device)
        
        # Benchmark ONNX models
        benchmark_onnx('./outputs/model.onnx')
        if os.path.exists('./outputs/model_quantized.onnx'):
            benchmark_onnx('./outputs/model_quantized.onnx')
    
    elif args.mode == 'infer':
        # Check if model exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}")
            print("Please train the model first using --mode train")
            return
        
        # Convert source to int if it's a digit (webcam)
        source = int(args.source) if args.source.isdigit() else args.source
        
        # Run realtime inference
        realtime_inference(args.model_path, source)


if __name__ == '__main__':
    main()