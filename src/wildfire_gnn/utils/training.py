"""
Training utilities for GNN models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm
import os


class Trainer:
    """Trainer for GNN models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        patience: int = 20,
        checkpoint_dir: str = 'models/checkpoints',
    ):
        """
        Initialize trainer.

        Args:
            model: GNN model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use ('cpu' or 'cuda')
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            edge_index: Graph edge indices
            edge_weight: Graph edge weights

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        for batch_x, batch_y in tqdm(train_loader, desc='Training', leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(batch_x, edge_index, edge_weight)
            else:
                predictions, _ = self.model(batch_x, edge_index, edge_weight)

            # For multi-step prediction, use only the last prediction
            if batch_y.dim() == 4:  # [batch, time, nodes, features]
                batch_y = batch_y[:, -1, :, :]  # Take last time step

            # Compute loss
            loss = self.criterion(predictions, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(
        self,
        val_loader: DataLoader,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            edge_index: Graph edge indices
            edge_weight: Graph edge weights

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc='Validation', leave=False):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(batch_x, edge_index, edge_weight)
                else:
                    predictions, _ = self.model(batch_x, edge_index, edge_weight)

                # For multi-step prediction, use only the last prediction
                if batch_y.dim() == 4:
                    batch_y = batch_y[:, -1, :, :]

                # Compute loss
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            edge_index: Graph edge indices
            edge_weight: Graph edge weights
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Training history
        """
        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, edge_index, edge_weight)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, edge_index, edge_weight)
            self.history['val_loss'].append(val_loss)

            epoch_time = time.time() - start_time

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pth')
                if verbose:
                    print(f"  → New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping after {epoch+1} epochs")
                    break

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
