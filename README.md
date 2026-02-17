# SparseML Sparsity Experiments

Small experiments around training and pruning PyTorch models using [SparseML](https://github.com/neuralmagic/sparseml).

This repo contains scripts and notebooks for:
- Training baseline models (MNIST / CIFAR-100)
- Applying unstructured magnitude pruning via SparseML recipes
- Tracking accuracy/sparsity over epochs
- Exporting a pruned MNIST model to ONNX

## Project layout

- `MNIST.py` – trains a small fully-connected MNIST model, then sparsifies at multiple targets (60/70/75/80%).
- `cifar-100.py` – trains a small fully-connected CIFAR-100 model and saves plots into `cifar-100_vizualizations/`.
- `mobilenet_v2.py` – trains MobileNetV2 on CIFAR-100 (baseline training only).
- `saved_model.py` – trains + sparsifies MNIST (80%) and exports `saved_model/sparse_model.onnx` plus `random_prediction.png`.
- `recipe*.yaml` – example SparseML pruning recipes.
- `*.ipynb` – exploratory notebooks (`deepSparse.ipynb`, `prediction.ipynb`, etc.).

## Setup

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### Install PyTorch (important)

`requirements.txt` pins CUDA 11.8 wheels (`torch==2.0.0+cu118`, `torchvision==0.15.1+cu118`, `torchaudio==2.0.1+cu118`). Install those from the PyTorch CUDA index first, then install the rest:

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

If you want **CPU-only** or a different CUDA version, update those 3 pinned lines in `requirements.txt` accordingly.

If you hit install errors with Python 3.12, try Python 3.10/3.11 (SparseML/PyTorch wheel availability varies by version).

## Run

All training scripts download datasets automatically into `./data/` (which is gitignored).

```bash
python MNIST.py
python cifar-100.py
python mobilenet_v2.py
python saved_model.py
```

Notes:
- `MNIST.py`, `cifar-100.py`, and `saved_model.py` write/overwrite `temp_recipe.yaml`.
- `cifar-100.py` saves plots under `cifar-100_vizualizations/`.
- `saved_model.py` writes `saved_model/sparse_model.onnx` and `random_prediction.png`.

## Git / large files

GitHub rejects files over 100MB. Keep datasets and large training artifacts out of git:
- `data/` is ignored (downloaded at runtime)
- `sparse_model/training/*.pth` is ignored
- `__pycache__/` and `*.pyc` are ignored

If you need to version large artifacts, use Git LFS instead of committing them directly.
