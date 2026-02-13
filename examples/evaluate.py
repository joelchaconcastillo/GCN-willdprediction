"""
Evaluation script for wildfire GNN models.

Example usage:
    python examples/evaluate.py --config configs/lstmconv.yaml --checkpoint models/checkpoints/best_model.pth
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wildfire_gnn.models import LSTMConvModel, GConvLSTMModel, TGCNModel, A3TGCNModel
from wildfire_gnn.data import WildfireDataset
from wildfire_gnn.utils import evaluate_model, plot_predictions


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> torch.nn.Module:
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
    parser = argparse.ArgumentParser(description='Evaluate wildfire GNN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Evaluating {config['model']['name']} model...")

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

    print(f"Test size: {len(test_dataset)}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
    )

    # Get graph structure
    edge_index, edge_weight = dataset.get_graph()

    # Create and load model
    print("\nLoading model...")
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    print("\nEvaluating on test set...")
    predictions, targets, metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=device,
    )

    # Print metrics
    print("\nTest Metrics:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Visualize predictions
    print("\nGenerating visualizations...")
    os.makedirs(config['experiment']['results_dir'], exist_ok=True)
    
    # Plot first sample
    plot_path = os.path.join(
        config['experiment']['results_dir'],
        f"{config['model']['name']}_predictions.png"
    )
    plot_predictions(
        coordinates=dataset.coordinates,
        y_true=targets[0].numpy(),
        y_pred=predictions[0].numpy(),
        save_path=plot_path,
    )
    print(f"Predictions visualization saved to {plot_path}")

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
