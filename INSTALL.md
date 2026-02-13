# Installation Guide

## Quick Start with UV (Recommended)

UV is a fast Python package installer and resolver. It's the recommended way to install this framework.

### Installing UV

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installing the Framework

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Install PyTorch (CPU version)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and dependencies
uv pip install torch-geometric torch-geometric-temporal

# Install the framework
uv pip install -e .
```

## Alternative: Installation with pip

If you prefer using pip:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate     # On Windows

# Install PyTorch first (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric (required for GNN models)
pip install torch-geometric torch-geometric-temporal

# Install the framework
pip install -e .
```

## GPU Support

For CUDA support (NVIDIA GPUs):

```bash
# Install PyTorch with CUDA 11.8 (adjust version as needed)
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
uv pip install torch-geometric torch-geometric-temporal
uv pip install -e .
```

## Minimal Installation (Without GNN Libraries)

If you only want the data processing utilities:

```bash
uv pip install -e .
```

This installs only the core dependencies. To use GNN models, you'll need to install the optional dependencies:

```bash
uv pip install -e ".[full]"
```

## Development Installation

For development with testing and formatting tools:

```bash
uv pip install -e ".[dev,full]"
```

## Verifying Installation

Test your installation:

```bash
python -c "from wildfire_gnn.data import WildfireDataset; print('Dataset module loaded successfully')"
python -c "from wildfire_gnn.models import LSTMConvModel; print('Models loaded successfully')"
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'torch_geometric'

**Solution**: Install PyTorch Geometric:
```bash
uv pip install torch-geometric torch-geometric-temporal
```

### Issue: CUDA not available

**Solution**: Either:
1. Install CPU-only version: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Or install CUDA version matching your GPU drivers

### Issue: Long installation times

**Solution**: Use UV which is much faster than pip, or pre-install PyTorch:
```bash
uv pip install torch  # Install PyTorch first
uv pip install torch-geometric torch-geometric-temporal  # Then GNN libraries
uv pip install -e .  # Finally install the framework
```

## System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- For GPU: CUDA-compatible NVIDIA GPU with 4GB+ VRAM
