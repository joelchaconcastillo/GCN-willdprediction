"""
Simple example demonstrating the wildfire GNN framework.

This script shows how to:
1. Create a synthetic dataset
2. Train multiple models
3. Compare their performance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wildfire_gnn.models import LSTMConvModel, TGCNModel, A3TGCNModel
from wildfire_gnn.data import WildfireDataset
from wildfire_gnn.utils import Trainer, evaluate_model, plot_metrics_comparison


def train_and_evaluate_model(model_class, model_name, dataset, config):
    """Train and evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print('='*60)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_ratio=0.7, val_ratio=0.15
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Get graph structure
    edge_index, edge_weight = dataset.get_graph()
    
    # Create model
    model = model_class(**config['model_params'])
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create optimizer and trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config['device'],
        patience=10,
        checkpoint_dir=f"models/checkpoints/{model_name}",
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        epochs=config['epochs'],
        verbose=True,
    )
    
    # Evaluate on test set
    print(f"\nEvaluating {model_name} on test set...")
    predictions, targets, metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=config['device'],
    )
    
    print(f"\n{model_name} Test Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")
    
    return metrics


def main():
    print("Wildfire GNN Framework - Simple Example")
    print("="*60)
    
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 20,  # Small number for quick demo
        'device': 'cpu',
        'model_params': {
            'num_nodes': 100,
            'in_channels': 5,
            'hidden_channels': 32,
            'out_channels': 1,
            'num_layers': 2,
            'dropout': 0.2,
        }
    }
    
    # Create synthetic dataset
    print("\nCreating synthetic wildfire dataset...")
    dataset = WildfireDataset(
        graph_type='knn',
        k=5,
        time_steps=10,
        prediction_horizon=1,
        normalize=True,
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of nodes: {dataset.coordinates.shape[0]}")
    
    # Train and evaluate different models
    results = {}
    
    # LSTMConv model
    lstmconv_params = config['model_params'].copy()
    lstmconv_params['K'] = 3
    lstmconv_config = config.copy()
    lstmconv_config['model_params'] = lstmconv_params
    results['LSTMConv'] = train_and_evaluate_model(
        LSTMConvModel, 'LSTMConv', dataset, lstmconv_config
    )
    
    # TGCN model
    results['TGCN'] = train_and_evaluate_model(
        TGCNModel, 'TGCN', dataset, config
    )
    
    # A3TGCN model
    a3tgcn_params = config['model_params'].copy()
    a3tgcn_params['periods'] = 12
    a3tgcn_config = config.copy()
    a3tgcn_config['model_params'] = a3tgcn_params
    results['A3TGCN'] = train_and_evaluate_model(
        A3TGCNModel, 'A3TGCN', dataset, a3tgcn_config
    )
    
    # Compare results
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(f"{'Model':<15} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<15} "
            f"{metrics['mse']:<10.4f} "
            f"{metrics['rmse']:<10.4f} "
            f"{metrics['mae']:<10.4f} "
            f"{metrics['r2']:<10.4f}"
        )
    
    # Plot comparison
    os.makedirs('results', exist_ok=True)
    metrics_to_plot = {
        model: {k: v for k, v in m.items() if k in ['mse', 'rmse', 'mae', 'r2']}
        for model, m in results.items()
    }
    plot_metrics_comparison(metrics_to_plot, save_path='results/model_comparison.png')
    print("\nModel comparison plot saved to results/model_comparison.png")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == '__main__':
    main()
