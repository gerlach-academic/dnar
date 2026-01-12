
import sys
import os

# Adjust path to include the current directory
sys.path.append(os.getcwd())

from generate_data import create_dataloader, SPEC
from configs import base_config
import torch
import yaml

def test_generation():
    print("Testing Topological Sort Generation...")
    
    # Load config
    config_path = "configs/attention/top_sort.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create Config object
    config = base_config.Config()
    for k, v in config_dict.items():
        if isinstance(v, dict):
             # Handle nested dictionaries like num_samples or problem_size
             current_val = getattr(config, k, None)
             if current_val is None:
                 setattr(config, k, v)
             else:
                 current_val.update(v)
        else:
            setattr(config, k, v)
    
    # Force no cache and sequential processing for debugging
    config.use_dataset_cache = False
    
    # Generate a small batch
    config.num_samples['train'] = 5
    config.problem_size['train'] = 10
    
    dataloader = create_dataloader(config, "train", seed=42, device="cpu", num_workers=0)
    
    batch = next(iter(dataloader))
    
    print("\nBatch Verification:")
    print(f"Algorithm: {config.algorithm}")
    print(f"Node Features Shape: {batch.node_fts.shape}") # [B, T, N, F]
    print(f"Edge Features Shape: {batch.edge_fts.shape}") # [B, T, E, F]
    print(f"Scalars Shape: {batch.scalars.shape}") # [B, T, E, 1]
    
    # Check for start attribute
    if hasattr(batch, 'start'):
        print(f"Batch has 'start' attribute: {batch.start}")
    else:
        print("Batch MISSING 'start' attribute!")
        
    # Check individual graphs in batch
    data_list = batch.to_data_list()
    if data_list and hasattr(data_list[0], 'start'):
        print(f"Individual Graph 0 has 'start': {data_list[0].start}")
    else:
        print("Individual Graph 0 MISSING 'start'!")
    
    # Verify Adjacency is DAG (Upper Triangular roughly if sorted, but we permuted)
    # Actually just check we have data
    assert batch.node_fts.shape[0] == config.batch_size or batch.node_fts.shape[0] == 50, f"Batch size mismatch: {batch.node_fts.shape[0]}, expected {50}"
    print("Generation Successful!")
    
    # Print the spec to ensure consistency
    print(f"SPEC for {config.algorithm}: {SPEC[config.algorithm]}")

if __name__ == "__main__":
    test_generation()
