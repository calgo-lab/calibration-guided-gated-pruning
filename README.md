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

The experiments are organized as notebook-based pipelines. Each notebook contains the end-to-end workflow for model setup, training, pruning, fine-tuning, and evaluation.

### Main Method

#### WRN-28-10 on CIFAR-10
Run:
- `notebooks/wrn28_10_cifar10.ipynb`

#### WRN-28-10 on CIFAR-100
Run:
- `notebooks/wrn28_10_cifar100.ipynb`

#### ResNet-50 on ImageNet
Run:
- `notebooks/resnet50_imagenet.ipynb`

### Gate Decorator (GD) Baseline

#### WRN-28-10 on CIFAR-10
Run:
- `notebooks/wrn28_10_cifar10_gd.ipynb`

#### WRN-28-10 on CIFAR-100
Run:
- `notebooks/wrn28_10_cifar100_gd.ipynb`

#### ResNet-50 on ImageNet
Run:
- `notebooks/resnet50_imagenet_gd.ipynb`

### Hyperparameter Sweep for Calibration Weight

To study the effect of the calibration regularization strength, we additionally provide HPO notebooks for sweeping the calibration weight $\beta$ in the joint objective.

#### WRN-28-10 on CIFAR-10
Run:
- `notebooks/hpo_beta_wrn28_10_cifar10.ipynb`

#### WRN-28-10 on CIFAR-100
Run:
- `notebooks/hpo_beta_wrn28_10_cifar100.ipynb`

#### ResNet-50 on ImageNet
Run:
- `notebooks/hpo_beta_resnet50_imagenet.ipynb`

### Notes

- Please update dataset paths before running the notebooks.
- For ImageNet experiments, ensure the dataset is prepared using the utilities provided under `data/`.
- The HPO notebooks are used to select the calibration weight $\beta$ before running the final main-method experiments.
- The GD baseline corresponds to gated structured pruning without the calibration-aware objective.
