"""
Wildfire GNN Prediction Framework
A comprehensive framework for wildfire prediction using Graph Neural Networks.
"""

__version__ = "0.1.0"

# Core data utilities are always available
from wildfire_gnn.data import WildfireDataset, GraphBuilder
from wildfire_gnn.utils import Trainer, evaluate_model

# Models require torch-geometric, so import them lazily
def _get_models():
    """Lazy import of models that require torch-geometric."""
    try:
        from wildfire_gnn.models import (
            LSTMConvModel,
            GConvLSTMModel,
            TGCNModel,
            A3TGCNModel,
        )
        return {
            'LSTMConvModel': LSTMConvModel,
            'GConvLSTMModel': GConvLSTMModel,
            'TGCNModel': TGCNModel,
            'A3TGCNModel': A3TGCNModel,
        }
    except ImportError as e:
        raise ImportError(
            "GNN models require torch-geometric. "
            "Install with: pip install torch-geometric torch-geometric-temporal"
        ) from e


# Export only data and utils by default
__all__ = [
    "WildfireDataset",
    "GraphBuilder",
    "Trainer",
    "evaluate_model",
]


def __getattr__(name):
    """Lazy loading of model classes."""
    models = _get_models()
    if name in models:
        return models[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
