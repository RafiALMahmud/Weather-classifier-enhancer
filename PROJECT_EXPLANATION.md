# Project Explanation

## Goal
This project classifies weather conditions from road-scene images into four categories:
- rainy
- foggy
- sand
- snow

## Why this project matters
Adverse weather severely affects computer vision systems. A lightweight weather classifier can act as a front-end decision module for downstream systems such as:
- object detection
- image enhancement
- autonomous driving perception
- traffic monitoring

## How the training pipeline works

### Dataset collection
The training script combines two sources:
- **DAWN** for rainy, sand, snow, and fog-related samples
- **FoggyCityscape** for additional foggy examples

### Preprocessing and augmentation
Images are resized and normalized using ImageNet statistics. During training, augmentation increases robustness through:
- random crop
- horizontal flip
- rotation
- color jitter
- grayscale conversion
- sharpness adjustment
- random erasing

### Model architecture
The classifier uses **MobileNetV3-Small** as a pretrained feature extractor and adds a custom classification head with dropout and two linear layers.

This choice makes the model practical for realtime use while remaining accurate enough for weather recognition.

### Training strategy
The model uses:
- cross entropy loss with label smoothing
- class weighting to reduce imbalance effects
- AdamW optimizer
- ReduceLROnPlateau learning rate scheduling
- early stopping
- gradient clipping

### Evaluation
The script prints:
- overall accuracy
- weighted precision
- weighted recall
- weighted F1-score
- per-class metrics
- confusion matrix
- misclassification analysis

### Deployment support
After training, the project exports:
- TorchScript model
- ONNX model
- quantized ONNX model

The ONNX export is then used by:
- realtime webcam/video inference
- the GUI application

## GUI explanation
The GUI lets a user upload one image and instantly receive:
- top 1 prediction
- top 2 prediction
- confidence percentages
- inference speed
- full probability breakdown across all 4 classes

## Practical use case
A detector or enhancer system can first run this classifier. Based on the predicted weather class, the system can then switch to weather-specific processing such as:
- dehazing for fog
- deraining for rainy scenes
- contrast restoration for sandstorm scenes
- snow-scene adaptation
