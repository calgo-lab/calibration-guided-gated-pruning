# joint-calib-sparsity
Joint optimization of calibration and sparsity in deep neural networks, integrating calibration-aware training objectives with structured pruning mechanisms to preserve predictive reliability under model compression.


## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately depending on your hardware setup.

### PyTorch (CUDA 12.4)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Jupyter Notebook setup
```python
!pip install -r requirements.txt
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** If installing inside a notebook, restart the kernel after installation.

## Data

This repository supports experiments on:
- CIFAR-10
- CIFAR-100
- ImageNet


## Running Experiments

The experiments are organized as notebook-based pipelines, with one notebook per dataset and method.

### Main method

#### WRN-28-10 on CIFAR-10
Run:
- `notebooks/wrn28_10_cifar10.ipynb`

#### WRN-28-10 on CIFAR-100
Run:
- `notebooks/wrn28_10_cifar100.ipynb`

#### ResNet-50 on ImageNet
Run:
- `notebooks/resnet50_imagenet.ipynb`

### Gate Decorator (GD) baseline

#### WRN-28-10 on CIFAR-10
Run:
- `notebooks/wrn28_10_cifar10_gd.ipynb`

#### WRN-28-10 on CIFAR-100
Run:
- `notebooks/wrn28_10_cifar100_gd.ipynb`

#### ResNet-50 on ImageNet
Run:
- `notebooks/resnet50_imagenet_gd.ipynb`

### Notes
- Please update dataset paths before running the notebooks.
- For ImageNet experiments, ensure the dataset is prepared using the utilities provided under `data/`.
- The notebooks contain the end-to-end pipeline for model setup, pruning, fine-tuning, and evaluation.
