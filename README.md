# Weather Classifier Enhancer

A 4-class adverse weather classification project built for training and fast inference on **rainy, foggy, sand, and snow** conditions.

This repository contains:
- a full training + evaluation + export pipeline
- realtime ONNX inference support
- a desktop GUI for single-image prediction
- a GitHub Actions workflow for basic CI validation
- SSH setup notes for pushing this repo from your own machine

## Project overview

The main training script combines:
- **DAWN** samples for rainy, sand, snow, and foggy classes
- **FoggyCityscape** samples for extra foggy supervision

The model uses **MobileNetV3-Small** as a lightweight backbone so the exported ONNX model can run with low latency on CPU or GPU.

## Included files

```text
Weather-classifier-enhancer/
├── .github/
│   └── workflows/
│       └── python-ci.yml
├── weather_classifier_complete.py
├── weather_inference_gui.py
├── requirements.txt
├── .gitignore
├── PROJECT_EXPLANATION.md
├── SSH_PUSH_GUIDE.md
└── README.md
```

## Main scripts

### 1) `weather_classifier_complete.py`
This is the end-to-end core pipeline.

It supports:
- dataset loading from DAWN and FoggyCityscape
- train/test split with stratification
- augmentation and normalization
- MobileNetV3-Small classifier training
- weighted loss for class imbalance
- evaluation with accuracy, precision, recall, F1, and confusion matrix
- export to TorchScript and ONNX
- ONNX quantization
- realtime webcam/video inference
- inference speed benchmarking

### 2) `weather_inference_gui.py`
This is a Tkinter desktop GUI for single-image weather classification.

It supports:
- image upload
- ONNX inference
- top-2 predicted classes
- confidence scores
- inference timing
- visual overlay on the uploaded image

## Dataset structure expected

### DAWN
The script expects a structure similar to:

```text
DAWN/
├── Rain/
│   └── Rain/
├── Sand/
│   └── Sand/
├── Snow/
│   └── Snow/
└── Fog/
    └── Fog/
```

### FoggyCityscape
The script expects:

```text
labeled_dataset/
├── images/
└── labels/
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

### Windows
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

Main dependencies:
- Python 3.10 or newer
- PyTorch
- torchvision
- timm
- opencv-python
- numpy
- scikit-learn
- onnx
- onnxruntime
- pillow
- tqdm

If you want GPU inference with ONNX Runtime, install the GPU build of ONNX Runtime appropriate for your CUDA setup.

## Training

Example:

```bash
python weather_classifier_complete.py \
  --mode train \
  --dawn_root "E:\\Daddyyaga\\DAWN\\DAWN\\" \
  --foggy_root "E:\\Daddyyaga\\labeled_dataset\\labeled_dataset\\" \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3
```

### Training outputs
The script saves exported artifacts under `./outputs`:
- `best_model.pth`
- `model.pt`
- `model.onnx`
- `model_quantized.onnx`

## Realtime inference

Use webcam:

```bash
python weather_classifier_complete.py --mode infer --model_path ./outputs/model_quantized.onnx --source 0
```

Use a video file:

```bash
python weather_classifier_complete.py --mode infer --model_path ./outputs/model_quantized.onnx --source path/to/video.mp4
```

## GUI inference

After training, run:

```bash
python weather_inference_gui.py
```

The GUI automatically looks for:
- `./outputs/model_quantized.onnx`
- then `./outputs/model.onnx`

## GitHub Actions workflow

This repo includes a workflow at:

```text
.github/workflows/python-ci.yml
```

It performs:
- repository checkout
- Python setup
- dependency installation
- syntax compilation check

This is a safe starter workflow for a research/prototype project because it validates the codebase without forcing full dataset-dependent training during CI.

## Notes

- The default dataset paths inside the script are Windows-specific. Change them to match your machine.
- Full training requires the datasets locally available.
- The GUI requires a trained ONNX model inside the `outputs` folder.

## Suggested next improvements

- add a sample `config.yaml`
- add model checkpoint metadata logging
- save confusion matrix image automatically
- add dataset path validation before training starts
- package the GUI as a standalone executable

## Uploaded source files used

The repository content here was built from the two uploaded scripts:
- training/inference pipeline fileciteturn0file0
- Tkinter GUI inference app fileciteturn0file1
