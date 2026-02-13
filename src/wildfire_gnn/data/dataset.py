"""
Dataset class for wildfire prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict
import os
from .graph_builder import GraphBuilder


class WildfireDataset(Dataset):
    """
    Dataset for wildfire spatiotemporal prediction.
    
    This dataset handles time-series data structured as graphs,
    where nodes represent spatial locations and edges represent
    spatial relationships.
    """

    def __init__(
        self,
        data_dir: str = None,
        coordinates: np.ndarray = None,
        features: np.ndarray = None,
        targets: np.ndarray = None,
        graph_type: str = 'knn',
        k: int = 5,
        time_steps: int = 10,
        prediction_horizon: int = 1,
        normalize: bool = True,
    ):
        """
        Initialize wildfire dataset.

        Args:
            data_dir: Directory containing data files (optional)
            coordinates: Node coordinates [num_nodes, 2]
            features: Node features [num_samples, num_nodes, num_features]
            targets: Target values [num_samples, num_nodes, target_dim]
            graph_type: Type of graph construction ('knn', 'radius', 'distance')
            k: Number of neighbors for knn graph
            time_steps: Number of historical time steps to use
            prediction_horizon: Number of future steps to predict
            normalize: Whether to normalize features
        """
        self.data_dir = data_dir
        self.graph_type = graph_type
        self.k = k
        self.time_steps = time_steps
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize

        if data_dir is not None:
            self._load_from_directory(data_dir)
        elif coordinates is not None and features is not None:
            self.coordinates = coordinates
            self.features = features
            self.targets = targets if targets is not None else features
        else:
            # Generate synthetic data for demonstration
            self._generate_synthetic_data()

        # Build graph structure
        self.graph_builder = GraphBuilder(method=graph_type, k=k)
        self.edge_index, self.edge_weight = self.graph_builder.build_graph(
            self.coordinates
        )

        # Normalize features if requested
        if self.normalize:
            self._normalize_features()

        # Create sequences
        self._create_sequences()

    def _load_from_directory(self, data_dir: str):
        """Load data from directory."""
        # This is a placeholder - implement based on your data format
        # Expected files: coordinates.npy, features.npy, targets.npy
        coords_path = os.path.join(data_dir, 'coordinates.npy')
        features_path = os.path.join(data_dir, 'features.npy')
        targets_path = os.path.join(data_dir, 'targets.npy')

        if os.path.exists(coords_path):
            self.coordinates = np.load(coords_path)
        else:
            raise FileNotFoundError(f"Coordinates file not found: {coords_path}")

        if os.path.exists(features_path):
            self.features = np.load(features_path)
        else:
            raise FileNotFoundError(f"Features file not found: {features_path}")

        if os.path.exists(targets_path):
            self.targets = np.load(targets_path)
        else:
            self.targets = self.features

    def _generate_synthetic_data(self):
        """Generate synthetic wildfire data for demonstration."""
        np.random.seed(42)
        
        # Generate grid of locations
        num_nodes = 100
        grid_size = int(np.sqrt(num_nodes))
        x = np.linspace(0, 10, grid_size)
        y = np.linspace(0, 10, grid_size)
        xx, yy = np.meshgrid(x, y)
        self.coordinates = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Generate synthetic time series
        num_samples = 1000
        num_features = 5  # e.g., temperature, humidity, wind speed, NDVI, elevation
        
        # Create features with temporal and spatial patterns
        self.features = np.random.randn(num_samples, num_nodes, num_features)
        
        # Add some spatial correlation
        for t in range(num_samples):
            for i in range(num_nodes):
                # Add neighbor influence
                neighbors = np.random.choice(num_nodes, size=3, replace=False)
                self.features[t, i] += 0.3 * self.features[t, neighbors].mean(axis=0)
        
        # Create targets (e.g., fire probability or intensity)
        # Simple model: target depends on features + spatial neighbors
        self.targets = np.zeros((num_samples, num_nodes, 1))
        for t in range(num_samples):
            # Fire probability increases with temperature, decreases with humidity
            fire_prob = (
                0.3 * self.features[t, :, 0]  # temperature
                - 0.2 * self.features[t, :, 1]  # humidity
                + 0.1 * self.features[t, :, 2]  # wind speed
            )
            self.targets[t, :, 0] = 1.0 / (1.0 + np.exp(-fire_prob))  # sigmoid

    def _normalize_features(self):
        """Normalize features to zero mean and unit variance."""
        # Compute statistics across all samples and nodes
        self.feature_mean = self.features.mean(axis=(0, 1), keepdims=True)
        self.feature_std = self.features.std(axis=(0, 1), keepdims=True) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std

        if self.targets is not None and self.targets.shape[-1] > 1:
            self.target_mean = self.targets.mean(axis=(0, 1), keepdims=True)
            self.target_std = self.targets.std(axis=(0, 1), keepdims=True) + 1e-8
            self.targets = (self.targets - self.target_mean) / self.target_std

    def _create_sequences(self):
        """Create input-output sequences."""
        num_samples = len(self.features)
        self.sequences = []

        for i in range(num_samples - self.time_steps - self.prediction_horizon + 1):
            # Input: [time_steps, num_nodes, num_features]
            x = self.features[i : i + self.time_steps]
            # Target: [prediction_horizon, num_nodes, target_dim]
            y = self.targets[
                i + self.time_steps : i + self.time_steps + self.prediction_horizon
            ]
            self.sequences.append((x, y))

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence.

        Returns:
            x: Input sequence [time_steps, num_nodes, num_features]
            y: Target sequence [prediction_horizon, num_nodes, target_dim]
        """
        x, y = self.sequences[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def get_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get graph structure.

        Returns:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges]
        """
        return self.edge_index, self.edge_weight

    def split(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple['WildfireDataset', 'WildfireDataset', 'WildfireDataset']:
        """
        Split dataset into train, validation, and test sets.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation

        Returns:
            train_dataset, val_dataset, test_dataset
        """
        n = len(self)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_sequences = self.sequences[:train_size]
        val_sequences = self.sequences[train_size : train_size + val_size]
        test_sequences = self.sequences[train_size + val_size :]

        # Create new dataset instances
        train_dataset = self._create_split_dataset(train_sequences)
        val_dataset = self._create_split_dataset(val_sequences)
        test_dataset = self._create_split_dataset(test_sequences)

        return train_dataset, val_dataset, test_dataset

    def _create_split_dataset(self, sequences):
        """Create a new dataset instance with given sequences."""
        dataset = WildfireDataset.__new__(WildfireDataset)
        dataset.coordinates = self.coordinates
        dataset.edge_index = self.edge_index
        dataset.edge_weight = self.edge_weight
        dataset.graph_builder = self.graph_builder
        dataset.sequences = sequences
        dataset.time_steps = self.time_steps
        dataset.prediction_horizon = self.prediction_horizon
        dataset.normalize = self.normalize
        if hasattr(self, 'feature_mean'):
            dataset.feature_mean = self.feature_mean
            dataset.feature_std = self.feature_std
        if hasattr(self, 'target_mean'):
            dataset.target_mean = self.target_mean
            dataset.target_std = self.target_std
        return dataset
