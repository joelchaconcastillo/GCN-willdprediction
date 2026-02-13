"""
Graph builder for wildfire prediction.
"""

import numpy as np
import torch
from scipy.spatial import distance_matrix
from typing import Tuple, Optional


class GraphBuilder:
    """Build spatial graphs from geographical coordinates."""

    def __init__(self, method: str = 'knn', k: int = 5, threshold: float = None):
        """
        Initialize graph builder.

        Args:
            method: Graph construction method ('knn', 'radius', 'distance')
            k: Number of nearest neighbors for 'knn' method
            threshold: Distance threshold for 'radius' and 'distance' methods
        """
        self.method = method
        self.k = k
        self.threshold = threshold

    def build_graph(
        self, coordinates: np.ndarray
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build graph from coordinates.

        Args:
            coordinates: Node coordinates [num_nodes, 2] (lat, lon)

        Returns:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
        """
        num_nodes = coordinates.shape[0]
        
        # Compute pairwise distances
        dist_matrix = distance_matrix(coordinates, coordinates)

        if self.method == 'knn':
            edge_index, edge_weight = self._build_knn_graph(dist_matrix)
        elif self.method == 'radius':
            edge_index, edge_weight = self._build_radius_graph(dist_matrix)
        elif self.method == 'distance':
            edge_index, edge_weight = self._build_distance_graph(dist_matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return edge_index, edge_weight

    def _build_knn_graph(
        self, dist_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build k-nearest neighbors graph."""
        num_nodes = dist_matrix.shape[0]
        edge_list = []
        edge_weights = []

        for i in range(num_nodes):
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i]
            nearest_indices = np.argsort(distances)[1 : self.k + 1]
            
            for j in nearest_indices:
                edge_list.append([i, j])
                # Use inverse distance as weight
                weight = 1.0 / (distances[j] + 1e-6)
                edge_weights.append(weight)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        return edge_index, edge_weight

    def _build_radius_graph(
        self, dist_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build radius graph (connect nodes within threshold distance)."""
        if self.threshold is None:
            raise ValueError("Threshold must be specified for radius graph")

        edge_list = []
        edge_weights = []
        num_nodes = dist_matrix.shape[0]

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if dist_matrix[i, j] <= self.threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Add both directions
                    weight = 1.0 / (dist_matrix[i, j] + 1e-6)
                    edge_weights.append(weight)
                    edge_weights.append(weight)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        return edge_index, edge_weight

    def _build_distance_graph(
        self, dist_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build fully connected graph with distance-based weights."""
        num_nodes = dist_matrix.shape[0]
        edge_list = []
        edge_weights = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
                    # Gaussian kernel for weights
                    if self.threshold is not None:
                        weight = np.exp(-(dist_matrix[i, j] ** 2) / (2 * self.threshold ** 2))
                    else:
                        weight = 1.0 / (dist_matrix[i, j] + 1e-6)
                    edge_weights.append(weight)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        return edge_index, edge_weight

    @staticmethod
    def build_temporal_edges(
        num_nodes: int, time_steps: int
    ) -> torch.Tensor:
        """
        Build temporal edges connecting same nodes across time.

        Args:
            num_nodes: Number of spatial nodes
            time_steps: Number of time steps

        Returns:
            edge_index: Temporal edge indices [2, num_edges]
        """
        edge_list = []
        
        for t in range(time_steps - 1):
            for i in range(num_nodes):
                # Connect node i at time t to node i at time t+1
                source = t * num_nodes + i
                target = (t + 1) * num_nodes + i
                edge_list.append([source, target])
                edge_list.append([target, source])  # Bidirectional

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        return edge_index
