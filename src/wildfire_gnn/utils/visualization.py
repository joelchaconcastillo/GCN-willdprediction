"""
Visualization utilities for wildfire prediction.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, Tuple
import torch


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss' keys
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_predictions(
    coordinates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_idx: Optional[int] = None,
    save_path: Optional[str] = None,
):
    """
    Plot spatial predictions.

    Args:
        coordinates: Node coordinates [num_nodes, 2]
        y_true: Ground truth values [num_nodes, features]
        y_pred: Predicted values [num_nodes, features]
        node_idx: Optional node index to highlight
        save_path: Path to save the figure
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Take first feature if multiple
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        y_true = y_true[:, 0]
    else:
        y_true = y_true.squeeze()
    
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = y_pred.squeeze()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot ground truth
    scatter1 = axes[0].scatter(
        coordinates[:, 0], coordinates[:, 1], 
        c=y_true, cmap='YlOrRd', s=100, alpha=0.7
    )
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot predictions
    scatter2 = axes[1].scatter(
        coordinates[:, 0], coordinates[:, 1], 
        c=y_pred, cmap='YlOrRd', s=100, alpha=0.7
    )
    axes[1].set_title('Predictions', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Plot error
    error = np.abs(y_true - y_pred)
    scatter3 = axes[2].scatter(
        coordinates[:, 0], coordinates[:, 1], 
        c=error, cmap='viridis', s=100, alpha=0.7
    )
    axes[2].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(scatter3, ax=axes[2])
    
    if node_idx is not None:
        for ax in axes:
            ax.scatter(
                coordinates[node_idx, 0], coordinates[node_idx, 1],
                c='blue', marker='*', s=500, edgecolors='black', linewidths=2
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_time_series(
    time_steps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Plot time series for a specific node.

    Args:
        time_steps: Time step indices
        y_true: Ground truth values [time_steps, num_nodes, features]
        y_pred: Predicted values [time_steps, num_nodes, features]
        node_idx: Node index to plot
        save_path: Path to save the figure
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Extract node data
    y_true_node = y_true[:, node_idx]
    y_pred_node = y_pred[:, node_idx]
    
    # Take first feature if multiple
    if y_true_node.ndim > 1 and y_true_node.shape[-1] > 1:
        y_true_node = y_true_node[:, 0]
    else:
        y_true_node = y_true_node.squeeze()
    
    if y_pred_node.ndim > 1 and y_pred_node.shape[-1] > 1:
        y_pred_node = y_pred_node[:, 0]
    else:
        y_pred_node = y_pred_node.squeeze()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_steps, y_true_node, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(time_steps, y_pred_node, 'r--', label='Prediction', linewidth=2)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Time Series Prediction (Node {node_idx})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(
    metrics_dict: dict,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of metrics across different models.

    Args:
        metrics_dict: Dictionary mapping model names to metric dictionaries
        save_path: Path to save the figure
    """
    models = list(metrics_dict.keys())
    metric_names = list(metrics_dict[models[0]].keys())
    
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5*len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]
    
    for idx, metric_name in enumerate(metric_names):
        values = [metrics_dict[model][metric_name] for model in models]
        axes[idx].bar(models, values, color='steelblue', alpha=0.7)
        axes[idx].set_title(metric_name.upper(), fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
