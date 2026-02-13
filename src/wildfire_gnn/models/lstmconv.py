"""
LSTM-based Graph Convolutional Network for wildfire prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from .base import BaseGNNModel


class LSTMConvModel(BaseGNNModel):
    """
    LSTM-based Graph Convolutional Network.
    
    Combines LSTM cells with Chebyshev graph convolution for capturing
    both temporal dynamics and spatial dependencies in wildfire spread.
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
        bias: bool = True,
    ):
        """
        Initialize LSTMConv model.

        Args:
            num_nodes: Number of nodes in the graph
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features
            out_channels: Number of output features per node
            num_layers: Number of LSTM layers
            K: Order of Chebyshev polynomials
            dropout: Dropout probability
            bias: Whether to use bias in convolution
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

        # Graph convolution layers
        self.conv_input = ChebConv(in_channels, hidden_channels, K=K, bias=bias)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layers
        self.conv_output = ChebConv(hidden_channels, hidden_channels, K=K, bias=bias)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, h=None, c=None):
        """
        Forward pass.

        Args:
            x: Node features [batch_size, time_steps, num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            h: Hidden state for LSTM
            c: Cell state for LSTM

        Returns:
            Output tensor [batch_size, num_nodes, out_channels]
            Hidden state tuple (h, c)
        """
        batch_size, time_steps, num_nodes, _ = x.size()

        # Process each time step through graph convolution
        conv_outputs = []
        for t in range(time_steps):
            # Apply graph convolution to each node feature at time t
            # x_t shape: [batch_size, num_nodes, in_channels]
            x_t = x[:, t, :, :]
            
            # Reshape for graph conv: [batch_size * num_nodes, in_channels]
            x_t = x_t.reshape(batch_size * num_nodes, -1)
            
            # Expand edge_index for batch processing
            edge_index_batch = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
            edge_index_batch = edge_index_batch.reshape(2, -1)
            
            # Add batch offset to edge indices
            batch_offset = torch.arange(batch_size, device=x.device)
            batch_offset = batch_offset.view(-1, 1, 1) * num_nodes
            edge_index_batch = edge_index_batch.view(2, batch_size, -1)
            edge_index_batch = edge_index_batch + batch_offset
            edge_index_batch = edge_index_batch.view(2, -1)
            
            # Apply graph convolution
            x_conv = self.conv_input(x_t, edge_index_batch, edge_weight)
            x_conv = torch.relu(x_conv)
            x_conv = self.dropout(x_conv)
            
            # Reshape back: [batch_size, num_nodes, hidden_channels]
            x_conv = x_conv.reshape(batch_size, num_nodes, -1)
            conv_outputs.append(x_conv)

        # Stack time steps: [batch_size, time_steps, num_nodes, hidden_channels]
        conv_outputs = torch.stack(conv_outputs, dim=1)

        # Process through LSTM for each node
        # Reshape to [batch_size * num_nodes, time_steps, hidden_channels]
        lstm_input = conv_outputs.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, time_steps, -1
        )

        # LSTM forward
        if h is not None and c is not None:
            lstm_out, (h_new, c_new) = self.lstm(lstm_input, (h, c))
        else:
            lstm_out, (h_new, c_new) = self.lstm(lstm_input)

        # Take last time step output: [batch_size * num_nodes, hidden_channels]
        lstm_out = lstm_out[:, -1, :]

        # Apply final graph convolution
        edge_index_batch = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)
        edge_index_batch = edge_index_batch.reshape(2, -1)
        batch_offset = torch.arange(batch_size, device=x.device)
        batch_offset = batch_offset.view(-1, 1, 1) * num_nodes
        edge_index_batch = edge_index_batch.view(2, batch_size, -1)
        edge_index_batch = edge_index_batch + batch_offset
        edge_index_batch = edge_index_batch.view(2, -1)

        out = self.conv_output(lstm_out, edge_index_batch, edge_weight)
        out = torch.relu(out)
        out = self.dropout(out)

        # Final linear layer
        out = self.fc(out)

        # Reshape to [batch_size, num_nodes, out_channels]
        out = out.reshape(batch_size, num_nodes, -1)

        return out, (h_new, c_new)

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
