"""
Prediction script for wildfire GNN models.

Example usage:
    python examples/predict.py --config configs/lstmconv.yaml --checkpoint models/checkpoints/best_model.pth
"""

import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wildfire_gnn.models import LSTMConvModel, GConvLSTMModel, TGCNModel, A3TGCNModel
from wildfire_gnn.data import WildfireDataset


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
    parser = argparse.ArgumentParser(description='Make predictions with wildfire GNN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.npy', help='Output file for predictions')
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
    print(f"Making predictions with {config['model']['name']} model...")

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

    # Split dataset (use test set for predictions)
    _, _, test_dataset = dataset.split(
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
    )

    print(f"Making predictions on {len(test_dataset)} samples...")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
    )

    # Get graph structure
    edge_index, edge_weight = dataset.get_graph()
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Create and load model
    print("\nLoading model...")
    model = create_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Make predictions
    print("\nMaking predictions...")
    all_predictions = []
    all_inputs = []

    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            
            if hasattr(model, 'predict'):
                predictions = model.predict(batch_x, edge_index, edge_weight)
            else:
                predictions, _ = model(batch_x, edge_index, edge_weight)
            
            all_predictions.append(predictions.cpu().numpy())
            all_inputs.append(batch_x.cpu().numpy())

    # Concatenate results
    predictions = np.concatenate(all_predictions, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)

    # Save predictions
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(args.output, predictions)
    print(f"\nPredictions saved to {args.output}")
    print(f"Prediction shape: {predictions.shape}")
    
    # Also save inputs for reference
    input_output = args.output.replace('.npy', '_inputs.npy')
    np.save(input_output, inputs)
    print(f"Input data saved to {input_output}")

    print("\nPrediction completed!")


if __name__ == '__main__':
    main()
