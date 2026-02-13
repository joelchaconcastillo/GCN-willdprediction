"""
Basic test to verify the framework installation.

This script tests core functionality without requiring full PyTorch Geometric installation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that basic modules can be imported."""
    print("Testing imports...")
    try:
        from wildfire_gnn.data import WildfireDataset, GraphBuilder
        print("✓ Data modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data modules: {e}")
        return False

    try:
        from wildfire_gnn.utils import (
            compute_metrics,
            plot_training_history,
        )
        print("✓ Utility modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utility modules: {e}")
        return False

    # Optional: Test model imports (requires torch-geometric)
    try:
        from wildfire_gnn.models import LSTMConvModel
        print("✓ Model modules imported successfully (torch-geometric is installed)")
    except ModuleNotFoundError as e:
        if "torch_geometric" in str(e):
            print("⚠ Model modules require torch-geometric (install with: pip install torch-geometric torch-geometric-temporal)")
        else:
            print(f"✗ Failed to import model modules: {e}")
            return False

    return True


def test_dataset():
    """Test dataset creation."""
    print("\nTesting dataset creation...")
    try:
        from wildfire_gnn.data import WildfireDataset

        # Create synthetic dataset
        dataset = WildfireDataset(
            time_steps=5,
            prediction_horizon=1,
        )

        print(f"✓ Dataset created with {len(dataset)} samples")
        print(f"  - Number of nodes: {dataset.coordinates.shape[0]}")
        print(f"  - Number of features: {dataset.features.shape[-1]}")

        # Test data access
        x, y = dataset[0]
        print(f"✓ Data access works - input shape: {x.shape}, target shape: {y.shape}")

        # Test graph building
        edge_index, edge_weight = dataset.get_graph()
        print(f"✓ Graph structure created - edges: {edge_index.shape[1]}")

        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_builder():
    """Test graph builder."""
    print("\nTesting graph builder...")
    try:
        import numpy as np
        from wildfire_gnn.data import GraphBuilder

        # Create sample coordinates
        coords = np.random.rand(20, 2) * 10

        # Test KNN graph
        builder = GraphBuilder(method='knn', k=3)
        edge_index, edge_weight = builder.build_graph(coords)
        print(f"✓ KNN graph built - edges: {edge_index.shape[1]}")

        # Test radius graph
        builder = GraphBuilder(method='radius', threshold=2.0)
        edge_index, edge_weight = builder.build_graph(coords)
        print(f"✓ Radius graph built - edges: {edge_index.shape[1]}")

        return True
    except Exception as e:
        print(f"✗ Graph builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics...")
    try:
        import torch
        from wildfire_gnn.utils import compute_metrics

        # Create dummy data
        y_true = torch.randn(100, 10, 1)
        y_pred = y_true + torch.randn(100, 10, 1) * 0.1

        metrics = compute_metrics(y_true, y_pred)
        print(f"✓ Metrics computed:")
        for name, value in metrics.items():
            print(f"    {name}: {value:.4f}")

        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Wildfire GNN Framework - Installation Test")
    print("="*60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_dataset()
    all_passed &= test_graph_builder()
    all_passed &= test_metrics()

    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("  1. Install torch-geometric: pip install torch-geometric torch-geometric-temporal")
        print("  2. Try the simple example: python examples/simple_example.py")
        print("  3. Train a model: python examples/train.py --config configs/lstmconv.yaml")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
