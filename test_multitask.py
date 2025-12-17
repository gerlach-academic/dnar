"""
Test script to verify that the @multitask decorator works correctly,
especially that backpropagation affects the right algorithm-specific parameters.
"""

import torch
from configs.base_config import Config
from models import Dnar
from utils import multitask


def test_multitask_backprop():
    """
    Test that gradients flow to the correct algorithm-specific parameters.
    """
    print("=" * 60)
    print("Testing Multitask Backpropagation")
    print("=" * 60)
    
    # Create a minimal config
    config = Config(
        algorithm="bfs",  # Default algorithm for spec
        h=32,
        num_node_states=1,
        num_edge_states=2,
        output_type="pointer",
        output_idx=0,
        stepwise_training=False,
        num_iterations=100,
    )
    
    # Create model with multitask enabled
    model = Dnar(config, multitask_num_algorithms=3)
    
    # Print multitask info
    info = model.get_multitask_info()
    print(f"\nMultitask Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get references to algorithm-specific embeddings BEFORE any forward pass
    # We need to access them through the registry
    registry = model._multitask_registry
    
    # Find the first embedding variant
    first_emb_key = list(registry.embedding_variants.keys())[0]
    emb_variants = registry.embedding_variants[first_emb_key]
    
    # Store initial weights for comparison
    alg0_emb_initial = emb_variants[0].weight.clone().detach()
    alg1_emb_initial = emb_variants[1].weight.clone().detach()
    alg2_emb_initial = emb_variants[2].weight.clone().detach()
    
    print(f"\nInitial embedding weights (first 3 values):")
    print(f"  Algorithm 0: {alg0_emb_initial[0, :3].tolist()}")
    print(f"  Algorithm 1: {alg1_emb_initial[0, :3].tolist()}")
    print(f"  Algorithm 2: {alg2_emb_initial[0, :3].tolist()}")
    
    # Create a simple mock batch
    batch = create_mock_batch(num_nodes=5, num_edges=10, num_steps=3, config=config)
    
    # Optimizer for all model parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # === Test 1: Forward pass with algorithm "bfs" (index 0) ===
    print("\n--- Test 1: Forward with 'bfs' ---")
    optimizer.zero_grad()
    # Pass training_step > 0 to enable loss computation and gradient flow
    output, loss = model(batch, training_step=1, multitask_algorithm="bfs")
    
    # Use the model's loss for backprop (it has proper gradient connections)
    loss.backward()
    
    # Check gradients
    print(f"  Gradients on alg 0 emb: {emb_variants[0].weight.grad is not None and emb_variants[0].weight.grad.abs().sum() > 0}")
    print(f"  Gradients on alg 1 emb: {emb_variants[1].weight.grad is not None and emb_variants[1].weight.grad.abs().sum() > 0}")
    print(f"  Gradients on alg 2 emb: {emb_variants[2].weight.grad is not None and emb_variants[2].weight.grad.abs().sum() > 0}")
    
    # Step optimizer
    optimizer.step()
    
    # Check which weights changed
    alg0_changed = not torch.allclose(emb_variants[0].weight, alg0_emb_initial)
    alg1_changed = not torch.allclose(emb_variants[1].weight, alg1_emb_initial)
    alg2_changed = not torch.allclose(emb_variants[2].weight, alg2_emb_initial)
    
    print(f"  Weights changed for alg 0: {alg0_changed} (expected: True)")
    print(f"  Weights changed for alg 1: {alg1_changed} (expected: False)")
    print(f"  Weights changed for alg 2: {alg2_changed} (expected: False)")
    
    # Update references for next test
    alg0_emb_after_test1 = emb_variants[0].weight.clone().detach()
    alg1_emb_after_test1 = emb_variants[1].weight.clone().detach()
    
    # === Test 2: Forward pass with algorithm "dijkstra" (index 1) ===
    print("\n--- Test 2: Forward with 'dijkstra' ---")
    optimizer.zero_grad()
    # Pass training_step > 0 to enable loss computation and gradient flow
    output, loss = model(batch, training_step=2, multitask_algorithm="dijkstra")
    
    # Use the model's loss for backprop
    loss.backward()
    
    # Check gradients
    print(f"  Gradients on alg 0 emb: {emb_variants[0].weight.grad is not None and emb_variants[0].weight.grad.abs().sum() > 0}")
    print(f"  Gradients on alg 1 emb: {emb_variants[1].weight.grad is not None and emb_variants[1].weight.grad.abs().sum() > 0}")
    print(f"  Gradients on alg 2 emb: {emb_variants[2].weight.grad is not None and emb_variants[2].weight.grad.abs().sum() > 0}")
    
    optimizer.step()
    
    # Check which weights changed from test1 state
    alg0_changed = not torch.allclose(emb_variants[0].weight, alg0_emb_after_test1)
    alg1_changed = not torch.allclose(emb_variants[1].weight, alg1_emb_after_test1)
    alg2_changed = not torch.allclose(emb_variants[2].weight, alg2_emb_initial)
    
    print(f"  Weights changed for alg 0: {alg0_changed} (expected: False)")
    print(f"  Weights changed for alg 1: {alg1_changed} (expected: True)")
    print(f"  Weights changed for alg 2: {alg2_changed} (expected: False)")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("If 'expected' matches actual, backpropagation is working correctly.")
    print("=" * 60)


def create_mock_batch(num_nodes, num_edges, num_steps, config):
    """Create a minimal mock batch for testing."""
    from torch_geometric.data import Data
    
    # Create a simple graph structure
    # Self-loops for each node + some regular edges
    edge_index = []
    for i in range(num_nodes):
        edge_index.append([i, i])  # Self-loop
    for i in range(num_nodes - 1):
        edge_index.append([i, i + 1])  # Forward edge
        edge_index.append([i + 1, i])  # Backward edge
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    actual_num_edges = edge_index.shape[1]
    
    # Create features
    node_fts = torch.zeros(num_nodes, num_steps, config.num_node_states)
    edge_fts = torch.zeros(actual_num_edges, num_steps, config.num_edge_states)
    scalars = torch.rand(actual_num_edges, num_steps, 1)
    
    # Ground truth for output
    y = torch.zeros(actual_num_edges)
    y[0] = 1.0  # Just mark first edge as target
    
    batch = Data(
        edge_index=edge_index,
        node_fts=node_fts,
        edge_fts=edge_fts,
        scalars=scalars,
        y=y,
        batch=torch.zeros(num_nodes, dtype=torch.long),  # Single graph
    )
    
    return batch


if __name__ == "__main__":
    test_multitask_backprop()
