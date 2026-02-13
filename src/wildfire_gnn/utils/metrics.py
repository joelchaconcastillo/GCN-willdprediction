"""
Evaluation metrics for wildfire prediction.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple


def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, denormalize: bool = False
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        denormalize: Whether values need to be denormalized

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten for metrics computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Compute metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }

    return metrics


def compute_classification_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics for binary fire/no-fire prediction.

    Args:
        y_true: Ground truth values (binary or probability)
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification

    Returns:
        Dictionary of classification metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Binarize predictions
    y_pred_binary = (y_pred_flat > threshold).astype(int)
    y_true_binary = (y_true_flat > threshold).astype(int)

    # Compute confusion matrix elements
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

    # Compute metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }

    return metrics


def evaluate_model(model, data_loader, edge_index, edge_weight=None, device='cpu'):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: Data loader
        edge_index: Graph edge indices
        edge_weight: Graph edge weights
        device: Device to use

    Returns:
        predictions, targets, metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []

    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            if hasattr(model, 'predict'):
                predictions = model.predict(batch_x, edge_index, edge_weight)
            else:
                predictions, _ = model(batch_x, edge_index, edge_weight)

            # Handle multi-step predictions
            if batch_y.dim() == 4:
                batch_y = batch_y[:, -1, :, :]

            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(targets, predictions)

    return predictions, targets, metrics


class EarlyStopping:
    """Early stopping utility."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop
