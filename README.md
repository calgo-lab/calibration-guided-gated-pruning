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
