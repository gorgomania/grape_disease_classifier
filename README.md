# Grape Disease Classifier

A neural network based on **EfficientNet-B0** for identifying diseases in grape leaves from photos. Supports 4 classes:

| Class           | Description  |
|-----------------|--------------|
| **Healthy**     | Healthy      |
| **Black Rot**   | Black Rot    |
| **ESCA**        | ESCA         |
| **Leaf Blight** | Leaf Blight  |

## Project Structure

```
├── train_grape.py            # Model training
├── convert.py                # Conversion .pth → .onnx
├── test_model_pth.py         # PyTorch model evaluation
├── test_model_onnx.py        # ONNX model evaluation
├── create_vineyard_archives.py  # Vineyard dataset generation
├── best_grape_model.pth      # Trained model (PyTorch)
└── grape_model.onnx          # Deployment model (ONNX)
```

## Dataset

[Grape Leaf Disease Augmented Dataset](https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset) — an augmented grape leaf disease dataset (Hugging Face).

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Training

Prepare the dataset in the `data/` folder:
```
data/
├── train/
│   ├── Healthy/
│   ├── Black Rot/
│   ├── ESCA/
│   └── Leaf Blight/
└── test/
    ├── Healthy/
    ├── Black Rot/
    ├── ESCA/
    └── Leaf Blight/
```

Run training:
```bash
python train_grape.py
```

The model trains for 15 epochs using mixed precision (optimized for GTX 1060 3GB). Best weights are saved to `best_grape_model.pth`.

### 2. Convert to ONNX

```bash
python convert.py
```

Creates `grape_model.onnx` for deployment without a PyTorch dependency.

### 3. Evaluation

Evaluate the PyTorch model:
```bash
python test_model_pth.py
```

Evaluate the ONNX model:
```bash
python test_model_onnx.py
```

Both scripts print a classification report and confusion matrix.

### 4. Vineyard Dataset Generation

The `create_vineyard_archives.py` script creates a synthetic vineyard dataset from real images — it simulates placement by rows and bushes with a specified disease distribution (default: 85% healthy, 5% each disease). The result is packed into 1 GB ZIP archives.

```bash
python create_vineyard_archives.py
# Enter vineyard structure, e.g.: [10, 12, 15, 10]
```

## Metrics

Test set: **3,610 images**.

### PyTorch (.pth)

| Class           | Precision | Recall  | F1-score | Images |
|-----------------|-----------|---------|----------|--------|
| **Black Rot**   | 99.89%    | 99.68%  | 99.79%   | 944    |
| **ESCA**        | 99.79%    | 99.90%  | 99.84%   | 960    |
| **Healthy**     | 100.00%   | 99.88%  | 99.94%   | 846    |
| **Leaf Blight** | 99.77%    | 100.00% | 99.88%   | 860    |
| **Total**       | 99.86%    | 99.86%  | 99.86%   | 3,610  |

#### Confusion Matrix (PTH)

|                 | Black Rot | ESCA | Healthy | Leaf Blight |
|-----------------|-----------|------|---------|-------------|
| **Black Rot**   | 941       | 2    | 0       | 1           |
| **ESCA**        | 1         | 959  | 0       | 0           |
| **Healthy**     | 0         | 0    | 845     | 1           |
| **Leaf Blight** | 0         | 0    | 0       | 860         |

### ONNX (.onnx)

| Class           | Precision | Recall  | F1-score | Images |
|-----------------|-----------|---------|----------|--------|
| **Black Rot**   | 99.68%    | 99.58%  | 99.63%   | 944    |
| **ESCA**        | 99.69%    | 99.79%  | 99.74%   | 960    |
| **Healthy**     | 99.65%    | 99.76%  | 99.70%   | 846    |
| **Leaf Blight** | 99.77%    | 99.65%  | 99.71%   | 860    |
| **Total**       | 99.70%    | 99.70%  | 99.70%   | 3,610  |

#### Confusion Matrix (ONNX)

|                 | Black Rot | ESCA | Healthy | Leaf Blight |
|-----------------|-----------|------|---------|-------------|
| **Black Rot**   | 940       | 3    | 0       | 1           |
| **ESCA**        | 2         | 958  | 0       | 0           |
| **Healthy**     | 1         | 0    | 844     | 1           |
| **Leaf Blight** | 0         | 0    | 3       | 857         |

## Model

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned on grape data)
- **Input size:** 224×224 RGB
- **Optimizer:** Adam (lr=0.0005)
- **Epochs:** 15
- **Batch size:** 16 (for 3GB VRAM)
