# Wildfire GNN Framework - Getting Started Guide

## Introduction

This framework provides a unified platform for wildfire prediction using various Graph Neural Network (GNN) architectures with time-series capabilities. It's designed to make it easy to experiment with different models while maintaining a consistent data format and training pipeline.

## Core Concepts

### 1. Spatial Graphs
Wildfire spread is inherently spatial. This framework represents geographical locations as nodes in a graph, with edges representing spatial relationships (e.g., proximity, wind direction).

### 2. Time-Series Prediction
Wildfire prediction requires analyzing temporal patterns. The framework uses sequences of historical observations to predict future fire risk or spread.

### 3. Unified Framework
All models share the same:
- Data format and loaders
- Training pipeline
- Evaluation metrics
- Configuration system

## Quick Start

### Step 1: Install the Framework

```bash
# See INSTALL.md for detailed instructions
uv venv
source .venv/bin/activate
uv pip install torch torch-geometric torch-geometric-temporal
uv pip install -e .
```

### Step 2: Test Your Installation

```bash
python tests/test_installation.py
```

### Step 3: Run the Simple Example

```bash
python examples/simple_example.py
```

This trains and compares three models on synthetic data.

## Data Format

### Input Data Structure

The framework expects data in one of two formats:

#### Option 1: Numpy Arrays (for custom data)

```python
from wildfire_gnn.data import WildfireDataset
import numpy as np

# Your data
coordinates = np.array([[lat1, lon1], [lat2, lon2], ...])  # [num_nodes, 2]
features = np.array([...])  # [num_samples, num_nodes, num_features]
targets = np.array([...])   # [num_samples, num_nodes, target_dim]

# Create dataset
dataset = WildfireDataset(
    coordinates=coordinates,
    features=features,
    targets=targets,
    graph_type='knn',  # or 'radius', 'distance'
    k=5,
    time_steps=10,
    prediction_horizon=1,
)
```

#### Option 2: Directory Structure

```
data/
├── coordinates.npy  # [num_nodes, 2]
├── features.npy     # [num_samples, num_nodes, num_features]
└── targets.npy      # [num_samples, num_nodes, target_dim]
```

```python
dataset = WildfireDataset(data_dir='data/', ...)
```

### Feature Engineering

Typical features for wildfire prediction include:
- **Meteorological**: Temperature, humidity, wind speed/direction, precipitation
- **Vegetation**: NDVI (Normalized Difference Vegetation Index), fuel moisture
- **Topographic**: Elevation, slope, aspect
- **Temporal**: Day of year, hour of day, season
- **Historical**: Previous fire occurrences, burn severity

### Target Variables

Common targets:
- **Fire probability**: Binary (fire/no fire) or continuous (0-1)
- **Fire intensity**: Fire radiative power, burn severity
- **Spread rate**: Velocity of fire spread
- **Burned area**: Hectares burned

## Available Models

### 1. LSTMConv
- **Best for**: Long-term temporal dependencies
- **Architecture**: LSTM cells + Chebyshev graph convolution
- **Use case**: Multi-day fire risk prediction

```yaml
# configs/lstmconv.yaml
model:
  name: lstmconv
  hidden_channels: 64
  num_layers: 2
  K: 3  # Chebyshev polynomial order
```

### 2. TGCN (Temporal GCN)
- **Best for**: Efficient processing, shorter sequences
- **Architecture**: GRU cells + graph convolution
- **Use case**: Real-time fire spread prediction

```yaml
# configs/tgcn.yaml
model:
  name: tgcn
  hidden_channels: 64
  num_layers: 2
```

### 3. A3TGCN (Attention Temporal GCN)
- **Best for**: Learning complex spatiotemporal patterns
- **Architecture**: TGCN + attention mechanism
- **Use case**: Complex multi-factor fire prediction

```yaml
# configs/a3tgcn.yaml
model:
  name: a3tgcn
  hidden_channels: 64
  num_layers: 2
  periods: 12  # Attention window
```

### 4. GConvLSTM
- **Best for**: Balancing spatial and temporal features
- **Architecture**: Graph convolution within LSTM cells
- **Use case**: Combined short/long-term prediction

```yaml
# configs/gconvlstm.yaml
model:
  name: gconvlstm
  hidden_channels: 64
  num_layers: 2
  K: 3
```

## Training a Model

### Using the Training Script

```bash
python examples/train.py --config configs/lstmconv.yaml
```

This will:
1. Load the configuration
2. Create a synthetic dataset (or load your data if specified)
3. Split into train/validation/test sets
4. Train the model with early stopping
5. Save the best model checkpoint
6. Generate training history plots

### Custom Training Code

```python
from wildfire_gnn.models import LSTMConvModel
from wildfire_gnn.data import WildfireDataset
from wildfire_gnn.utils import Trainer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Create dataset
dataset = WildfireDataset(...)
train_ds, val_ds, test_ds = dataset.split()

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Create model
model = LSTMConvModel(
    num_nodes=100,
    in_channels=5,
    hidden_channels=64,
    out_channels=1,
    num_layers=2,
)

# Create trainer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

trainer = Trainer(model, optimizer, criterion, device='cpu')

# Train
edge_index, edge_weight = dataset.get_graph()
history = trainer.fit(
    train_loader, val_loader,
    edge_index, edge_weight,
    epochs=100
)
```

## Evaluation

### Using the Evaluation Script

```bash
python examples/evaluate.py \
    --config configs/lstmconv.yaml \
    --checkpoint models/checkpoints/best_model.pth
```

### Custom Evaluation

```python
from wildfire_gnn.utils import evaluate_model

predictions, targets, metrics = evaluate_model(
    model=model,
    data_loader=test_loader,
    edge_index=edge_index,
    edge_weight=edge_weight,
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

## Making Predictions

```bash
python examples/predict.py \
    --config configs/lstmconv.yaml \
    --checkpoint models/checkpoints/best_model.pth \
    --output predictions.npy
```

## Working with Real Data

### Example: FIRMS Fire Data

```python
import pandas as pd
import numpy as np
from wildfire_gnn.data import WildfireDataset

# Load FIRMS data (example structure)
df = pd.read_csv('firms_data.csv')

# Extract coordinates (unique locations)
coords = df[['latitude', 'longitude']].drop_duplicates().values

# Create time-series features
# Group by location and time, pivot to get feature matrix
# ... your data processing code ...

# Create dataset
dataset = WildfireDataset(
    coordinates=coords,
    features=features,
    targets=targets,
    graph_type='knn',
    k=8,
    time_steps=14,  # 2 weeks of history
    prediction_horizon=7,  # Predict 1 week ahead
)
```

## Tips and Best Practices

### 1. Graph Construction
- **KNN (k-nearest neighbors)**: Best for regularly distributed sensors
- **Radius**: Best for irregularly distributed points
- **Distance**: Use with caution (fully connected can be slow)

### 2. Hyperparameter Tuning
- Start with `hidden_channels=64`, increase if underfitting
- Use 2-3 layers; more layers need more data
- Dropout 0.2-0.3 helps prevent overfitting

### 3. Data Normalization
- Always normalize features (built into WildfireDataset)
- Consider log-transform for skewed distributions
- Standardize temporal features (time of day, day of year)

### 4. Training
- Use early stopping to prevent overfitting
- Monitor validation loss, not just training loss
- Start with small learning rate (1e-3 or 1e-4)

### 5. Model Selection
- Try all models; performance depends on data
- TGCN is fastest for experimentation
- A3TGCN often performs best but is slower

## Troubleshooting

### Poor Performance

1. **Check data quality**: Are features informative?
2. **Increase model capacity**: More hidden channels, more layers
3. **Tune hyperparameters**: Learning rate, batch size
4. **Add more data**: GNNs benefit from more training samples
5. **Feature engineering**: Add domain-specific features

### Memory Issues

1. **Reduce batch size**
2. **Use fewer hidden channels**
3. **Reduce number of nodes** (subsample if dataset is too large)
4. **Use CPU instead of GPU** for small models

### Slow Training

1. **Use GPU**: Set `device='cuda'` if available
2. **Increase batch size**: Better GPU utilization
3. **Use TGCN**: Faster than LSTM-based models
4. **Reduce sequence length**: Shorter `time_steps`

## Next Steps

1. **Prepare your data**: Format it according to the framework
2. **Experiment with models**: Try all four architectures
3. **Tune hyperparameters**: Use validation set to optimize
4. **Evaluate thoroughly**: Test on held-out data
5. **Deploy**: Save best model and use for predictions

## Additional Resources

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- PyTorch Geometric Temporal: https://pytorch-geometric-temporal.readthedocs.io/
- FIRMS Data: https://firms.modaps.eosdis.nasa.gov/
- Wildfire Datasets: See README.md for comprehensive list

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues, questions, or feature requests, please open an issue on GitHub.
