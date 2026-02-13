"""
Graph Convolutional LSTM (GConvLSTM) for wildfire prediction.
"""

import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvLSTM as GConvLSTMLayer
from .base import BaseGNNModel


class GConvLSTMModel(BaseGNNModel):
    """
    Graph Convolutional LSTM.
    
    Applies convolution operations within LSTM cells for spatiotemporal
    wildfire prediction.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        K: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize GConvLSTM model.

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features
            out_channels: Number of output features per node
            num_layers: Number of GConvLSTM layers
            K: Filter size for Chebyshev polynomials
            dropout: Dropout probability
        """
        super().__init__(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.K = K

        # GConvLSTM layers
        self.gconvlstm_layers = nn.ModuleList()
        
        # First layer
        self.gconvlstm_layers.append(GConvLSTMLayer(in_channels, hidden_channels, K=K))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.gconvlstm_layers.append(
                GConvLSTMLayer(hidden_channels, hidden_channels, K=K)
            )

        # Output layer
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, h=None, c=None):
        """
        Forward pass.

        Args:
            x: Node features [batch_size, time_steps, num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            h: Hidden state list for each layer
            c: Cell state list for each layer

        Returns:
            Output tensor [batch_size, num_nodes, out_channels]
            Tuple of (hidden_state_list, cell_state_list)
        """
        batch_size, time_steps, num_nodes, _ = x.size()

        # Initialize states if not provided
        if h is None:
            h = [None] * self.num_layers
        if c is None:
            c = [None] * self.num_layers

        # Process through time steps
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # [batch_size, num_nodes, in_channels]
            
            # Process through GConvLSTM layers
            for layer_idx, gconvlstm_layer in enumerate(self.gconvlstm_layers):
                # Process each sample in batch
                h_list = []
                c_list = []
                for b in range(batch_size):
                    x_b = x_t[b]  # [num_nodes, channels]
                    h_b = h[layer_idx][b] if h[layer_idx] is not None else None
                    c_b = c[layer_idx][b] if c[layer_idx] is not None else None
                    h_new, c_new = gconvlstm_layer(x_b, edge_index, edge_weight, h_b, c_b)
                    h_list.append(h_new)
                    c_list.append(c_new)
                
                # Stack batch
                x_t = torch.stack(h_list, dim=0)  # [batch_size, num_nodes, hidden]
                h[layer_idx] = h_list
                c[layer_idx] = c_list
                
                # Apply dropout
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

        # Final output
        out = self.fc(x_t)  # [batch_size, num_nodes, out_channels]
        
        return out, (h, c)

    def predict(self, x, edge_index, edge_weight=None):
        """
        Make predictions without returning hidden states.

        Args:
            x: Node features [batch_size, time_steps, num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Predictions [batch_size, num_nodes, out_channels]
        """
        out, _ = self.forward(x, edge_index, edge_weight)
        return out

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config['K'] = self.K
        return config
