"""
Wildfire GNN Prediction Framework
A comprehensive framework for wildfire prediction using Graph Neural Networks.
"""

__version__ = "0.1.0"

from wildfire_gnn.models import (
    LSTMConvModel,
    GConvLSTMModel,
    TGCNModel,
    A3TGCNModel,
)
from wildfire_gnn.data import WildfireDataset, GraphBuilder
from wildfire_gnn.utils import Trainer, evaluate_model

__all__ = [
    "LSTMConvModel",
    "GConvLSTMModel",
    "TGCNModel",
    "A3TGCNModel",
    "WildfireDataset",
    "GraphBuilder",
    "Trainer",
    "evaluate_model",
]
