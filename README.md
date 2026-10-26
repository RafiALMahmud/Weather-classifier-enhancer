# Weather Classifier Enhancer

> Four-class road-scene weather classification, from model training to ONNX inference.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)

**[GitHub repository](https://github.com/RafiALMahmud/Weather-classifier-enhancer)** · [Project explanation](PROJECT_EXPLANATION.md) · [Installation](#installation) · [Training](#training) · [Realtime inference](#realtime-inference) · [Desktop GUI](#gui-inference) · [CI notes](#github-actions-workflow)

| Input | Classification | Outputs and interfaces |
|---|---|---|
| Road-scene images | Rainy, foggy, sand and snow | TorchScript, ONNX and quantized ONNX exports |
| Webcam or video | Frame-level inference | Realtime inference pipeline |
| An uploaded image | Class predictions and confidence scores | Tkinter desktop GUI |

**Project format:** a local training and desktop-inference workflow. Datasets and a trained model are required for the relevant commands below; no hosted web demo is listed here.

---

A 4-class adverse weather classification project built for training and fast inference on **rainy, foggy, sand, and snow** conditions.

This repository contains:
- a full training + evaluation + export pipeline
- realtime ONNX inference support
- a desktop GUI for single-image prediction
- a GitHub Actions workflow and a separate syntax-check example
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
│       └── python-publish.yml
├── weather_classifier_complete.py
├── weather_inference_gui.py
├── python-ci.yml                  # root-level syntax-check example
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

The active workflow is [`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml). It sets up Python, installs dependencies and invokes the training script with `|| true`. Script errors therefore do not fail that step; a successful workflow run is not proof of successful model training.

A separate [`python-ci.yml`](python-ci.yml) example is stored at the repository root. It defines:

- repository checkout
- Python setup
- dependency installation
- syntax compilation checks for both Python scripts

GitHub discovers workflows under `.github/workflows/`, so the root-level syntax-check example is not an active workflow in its current location. Full training still requires the datasets and configuration described above.

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
- [Training/inference pipeline](weather_classifier_complete.py)
- [Tkinter GUI inference app](weather_inference_gui.py)
