# Wildfire Prediction using Graph Neural Networks

A comprehensive framework for wildfire prediction using state-of-the-art Graph Neural Networks (GNNs) with time-series analysis capabilities. This project implements multiple GNN architectures optimized for spatiotemporal wildfire forecasting.

## Overview

This framework provides a unified implementation of various time-series GNN models for wildfire prediction, including:
- **LSTMConv**: LSTM-based Graph Convolutional Networks
- **GConvLSTM**: Graph Convolutional LSTM
- **TGCN**: Temporal Graph Convolutional Network
- **A3TGCN**: Attention Temporal Graph Convolutional Network

## State-of-the-Art Wildfire Datasets

### 1. **FIRMS (Fire Information for Resource Management System)**
- **Source**: NASA's Active Fire Data
- **Coverage**: Global, real-time
- **Resolution**: MODIS (1km), VIIRS (375m)
- **Data**: Fire detections, brightness temperature, fire radiative power
- **URL**: https://firms.modaps.eosdis.nasa.gov/

### 2. **EFFIS (European Forest Fire Information System)**
- **Source**: European Commission's Copernicus
- **Coverage**: European countries
- **Data**: Burnt areas, fire danger forecasts, historical fire records
- **URL**: https://effis.jrc.ec.europa.eu/

### 3. **MTBS (Monitoring Trends in Burn Severity)**
- **Source**: USGS and USDA Forest Service
- **Coverage**: United States (1984-present)
- **Data**: Burn severity maps, perimeter data, pre/post-fire imagery
- **URL**: https://www.mtbs.gov/

### 4. **CalFire (California Department of Forestry and Fire Protection)**
- **Source**: State of California
- **Coverage**: California
- **Data**: Historical fire perimeters, incident information
- **URL**: https://www.fire.ca.gov/

### 5. **GWIS (Global Wildfire Information System)**
- **Source**: Copernicus Emergency Management Service
- **Coverage**: Global
- **Data**: Fire danger forecast, emissions, burnt areas
- **URL**: https://gwis.jrc.ec.europa.eu/

### 6. **FPA-FOD (Fire Program Analysis - Fire Occurrence Database)**
- **Source**: USFS
- **Coverage**: United States (1992-2018)
- **Records**: 2.3+ million wildfires
- **URL**: https://www.fs.usda.gov/rds/archive/

### 7. **Environmental Variables**
- **Weather Data**: ERA5 (ECMWF), NOAA
- **Vegetation**: NDVI from MODIS/Landsat
- **Topography**: SRTM DEM
- **Land Cover**: ESA CCI Land Cover

## Features

- **Unified Framework**: Same codebase for multiple GNN architectures
- **Modular Design**: Easy to add new models and datasets
- **Time-Series Focus**: Specialized for spatiotemporal prediction
- **PyTorch Geometric**: Built on industry-standard libraries
- **UV Compatible**: Modern Python package management with uv
- **Configurable**: YAML-based configuration system
- **Reproducible**: Seed setting and deterministic training

## Installation

**Important**: This framework requires PyTorch and PyTorch Geometric. See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Quick Install with UV (Recommended)

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric and dependencies
uv pip install torch-geometric torch-geometric-temporal

# Install the framework
uv pip install -e .
```

### Quick Install with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric torch-geometric-temporal

# Install the framework
pip install -e .
```

For GPU support, troubleshooting, and other installation options, see [INSTALL.md](INSTALL.md).

## Quick Start

### 1. Prepare Data

```python
from wildfire_gnn.data import WildfireDataset

# Load and prepare dataset
dataset = WildfireDataset(
    data_dir='data/raw',
    graph_type='spatial',
    time_steps=10,
    prediction_horizon=3
)
```

### 2. Train a Model

```bash
# Train LSTMConv model
python examples/train.py --config configs/lstmconv.yaml

# Train TGCN model
python examples/train.py --config configs/tgcn.yaml
```

### 3. Make Predictions

```python
from wildfire_gnn.models import LSTMConvModel
from wildfire_gnn.utils import load_checkpoint

# Load trained model
model = load_checkpoint('models/checkpoints/lstmconv_best.pth')

# Make predictions
predictions = model.predict(test_data)
```

## Project Structure

```
wildfire-gnn-prediction/
├── src/wildfire_gnn/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstmconv.py      # LSTMConv GNN
│   │   ├── gconvlstm.py     # Graph Convolutional LSTM
│   │   ├── tgcn.py          # Temporal GCN
│   │   ├── a3tgcn.py        # Attention Temporal GCN
│   │   └── base.py          # Base model class
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset loaders
│   │   ├── processors.py    # Data preprocessing
│   │   └── graph_builder.py # Graph construction
│   └── utils/
│       ├── __init__.py
│       ├── training.py      # Training utilities
│       ├── metrics.py       # Evaluation metrics
│       └── visualization.py # Plotting tools
├── configs/
│   ├── lstmconv.yaml
│   ├── tgcn.yaml
│   └── a3tgcn.yaml
├── examples/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── pyproject.toml
└── README.md
```

## Model Architectures

### LSTMConv
Combines LSTM cells with graph convolution for capturing both temporal dynamics and spatial dependencies.

### GConvLSTM
Graph Convolutional LSTM that applies convolution operations within LSTM cells.

### TGCN (Temporal Graph Convolutional Network)
Efficient architecture combining GRU with graph convolution.

### A3TGCN (Attention Temporal GCN)
Enhanced TGCN with attention mechanisms for better feature learning.

## Configuration

Example configuration file (`configs/lstmconv.yaml`):

```yaml
model:
  name: lstmconv
  hidden_dim: 64
  num_layers: 2
  dropout: 0.2

data:
  time_steps: 10
  prediction_horizon: 3
  batch_size: 32
  train_ratio: 0.7
  val_ratio: 0.15

training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-5
  patience: 20
```

## Performance Metrics

The framework evaluates models using:
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **F1-Score**: For binary fire/no-fire classification

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{wildfire_gnn_2024,
  title={Wildfire Prediction using Graph Neural Networks},
  author={Your Name},
  year={2024},
  url={https://github.com/joelchaconcastillo/GCN-willdprediction}
}
```

## References

1. Yu, B., Yin, H., & Zhu, Z. (2017). Spatio-temporal graph convolutional networks.
2. Seo, Y., Defferrard, M., Vandergheynst, P., & Bresson, X. (2018). Structured sequence modeling with graph convolutional recurrent networks.
3. Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). Adaptive graph convolutional recurrent network for traffic forecasting.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue on GitHub.