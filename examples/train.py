"""
Training script for wildfire GNN models.

Example usage:
    python examples/train.py --config configs/lstmconv.yaml
    python examples/train.py --config configs/tgcn.yaml --epochs 50
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wildfire_gnn.models import LSTMConvModel, GConvLSTMModel, TGCNModel, A3TGCNModel
from wildfire_gnn.data import WildfireDataset
from wildfire_gnn.utils import Trainer, plot_training_history


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> nn.Module:
    """Create model based on configuration."""
    model_name = config['model']['name'].lower()
    model_config = config['model'].copy()
    del model_config['name']

    if model_name == 'lstmconv':
        model = LSTMConvModel(**model_config)
    elif model_name == 'gconvlstm':
        model = GConvLSTMModel(**model_config)
    elif model_name == 'tgcn':
        model = TGCNModel(**model_config)
    elif model_name == 'a3tgcn':
        model = A3TGCNModel(**model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train wildfire GNN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (override config)')
    parser.add_argument('--device', type=str, default=None, help='Device (override config)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.device is not None:
        config['training']['device'] = args.device

    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])

    # Set device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Training {config['model']['name']} model...")

    # Create dataset
    print("\nCreating dataset...")
    dataset = WildfireDataset(
        data_dir=config['data']['data_dir'],
        graph_type=config['data']['graph_type'],
        k=config['data']['k'],
        time_steps=config['data']['time_steps'],
        prediction_horizon=config['data']['prediction_horizon'],
        normalize=config['data']['normalize'],
    )

    # Split dataset
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
    )

    # Get graph structure
    edge_index, edge_weight = dataset.get_graph()

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    criterion = nn.MSELoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=config['training']['patience'],
        checkpoint_dir=config['experiment']['checkpoint_dir'],
    )

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        epochs=config['training']['epochs'],
        verbose=True,
    )

    # Plot training history
    os.makedirs(config['experiment']['results_dir'], exist_ok=True)
    plot_path = os.path.join(
        config['experiment']['results_dir'],
        f"{config['model']['name']}_training_history.png"
    )
    plot_training_history(history, save_path=plot_path)
    print(f"\nTraining history saved to {plot_path}")

    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
