"""
Base model class for all GNN models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseGNNModel(nn.Module, ABC):
    """Base class for all GNN models in the framework."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize base GNN model.

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features
            out_channels: Number of output features per node
            num_layers: Number of layers in the model
            dropout: Dropout probability
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the model.

        Args:
            x: Node features [batch_size, num_nodes, in_channels] or
               [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Output tensor [batch_size, num_nodes, out_channels] or
            [num_nodes, out_channels]
        """
        pass

    def reset_parameters(self):
        """Reset model parameters."""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'num_nodes': self.num_nodes,
            'in_channels': self.in_channels,
            'hidden_channels': self.hidden_channels,
            'out_channels': self.out_channels,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
            'model_class': self.__class__.__name__,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        config.update(kwargs)
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
