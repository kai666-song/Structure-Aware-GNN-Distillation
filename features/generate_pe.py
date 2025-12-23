"""
Random Walk Positional Encoding (RWPE) Generator
================================================

This script generates positional encodings based on random walk probabilities.
The key insight: k-step random walk diagonal elements p_ii^(k) capture the 
probability of returning to the same node, which encodes local structural info.

Reference: 
- "Graph Neural Networks with Learnable Structural and Positional Representations"
  (Dwivedi et al., ICLR 2022)

Usage:
    python features/generate_pe.py --dataset actor --k 16
    python features/generate_pe.py --all --k 16
"""

import os
import sys
import argparse
import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_data_new


def compute_rwpe(adj, k=16, add_self_loop=True):
    """
    Compute Random Walk Positional Encoding.
    
    For each node i, compute the diagonal elements of P^1, P^2, ..., P^k
    where P = D^{-1}A is the random walk transition matrix.
    
    p_ii^(k) = probability of returning to node i after k steps
    
    Args:
        adj: scipy sparse adjacency matrix (N x N)
        k: number of random walk steps (default: 16)
        add_self_loop: whether to add self-loops before computing (default: True)
    
    Returns:
        pe: torch.Tensor of shape (N, k) containing positional encodings
    """
    num_nodes = adj.shape[0]
    
    # Convert to CSR format for efficient operations
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()
    
    # Optionally add self-loops
    if add_self_loop:
        adj = adj + sp.eye(num_nodes, format='csr')
    
    # Compute degree matrix D
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1  # Avoid division by zero
    
    # Compute D^{-1}
    d_inv = 1.0 / degree
    d_inv_diag = sp.diags(d_inv, format='csr')
    
    # Random walk matrix P = D^{-1} A
    P = d_inv_diag @ adj
    
    # Initialize PE matrix
    pe = np.zeros((num_nodes, k), dtype=np.float32)
    
    # Compute P^1, P^2, ..., P^k and extract diagonals
    print(f"Computing {k}-step random walk PE for {num_nodes} nodes...")
    
    # P_power starts as identity, then we multiply by P each step
    P_power = sp.eye(num_nodes, format='csr')
    
    for step in tqdm(range(k), desc="Computing RWPE"):
        P_power = P_power @ P
        # Extract diagonal: p_ii^(step+1)
        pe[:, step] = P_power.diagonal()
    
    return torch.from_numpy(pe)


def generate_rwpe(dataset_name, k=16, save_dir='./data', split_idx=0):
    """
    Generate and save RWPE for a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'actor', 'squirrel', 'cora')
        k: Number of random walk steps
        save_dir: Directory to save the PE file
        split_idx: Split index (for loading data, PE is same across splits)
    
    Returns:
        pe: torch.Tensor of shape (N, k)
    """
    print(f"\n{'='*60}")
    print(f"Generating RWPE for {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    adj, features, labels, *_ = load_data_new(dataset_name, split_idx=split_idx)
    
    num_nodes = adj.shape[0]
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {adj.nnz}")
    print(f"PE dimension: {k}")
    
    # Compute RWPE
    pe = compute_rwpe(adj, k=k)
    
    # Normalize PE (important for stable training)
    # Use standardization: (x - mean) / std
    pe_mean = pe.mean(dim=0, keepdim=True)
    pe_std = pe.std(dim=0, keepdim=True)
    pe_std[pe_std < 1e-6] = 1.0  # Avoid division by zero
    pe_normalized = (pe - pe_mean) / pe_std
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'pe_rw_{dataset_name}.pt')
    torch.save({
        'pe': pe_normalized,
        'pe_raw': pe,
        'k': k,
        'num_nodes': num_nodes,
        'dataset': dataset_name
    }, save_path)
    
    print(f"\nSaved to: {save_path}")
    print(f"PE shape: {pe_normalized.shape}")
    print(f"PE stats (normalized): mean={pe_normalized.mean():.4f}, std={pe_normalized.std():.4f}")
    print(f"PE stats (raw): min={pe.min():.4f}, max={pe.max():.4f}")
    
    return pe_normalized


def load_pe(dataset_name, data_dir='./data'):
    """
    Load pre-computed positional encoding.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing PE files
    
    Returns:
        pe: torch.Tensor of shape (N, k)
    """
    pe_path = os.path.join(data_dir, f'pe_rw_{dataset_name}.pt')
    
    if not os.path.exists(pe_path):
        raise FileNotFoundError(
            f"PE file not found: {pe_path}\n"
            f"Run: python features/generate_pe.py --dataset {dataset_name}"
        )
    
    data = torch.load(pe_path)
    return data['pe']


def main():
    parser = argparse.ArgumentParser(description='Generate Random Walk Positional Encoding')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'squirrel', 'chameleon', 'cora', 'citeseer', 'pubmed'],
                       help='Dataset name')
    parser.add_argument('--all', action='store_true',
                       help='Generate PE for all heterophilic datasets')
    parser.add_argument('--k', type=int, default=16,
                       help='Number of random walk steps (PE dimension)')
    parser.add_argument('--save_dir', type=str, default='./data',
                       help='Directory to save PE files')
    args = parser.parse_args()
    
    if args.all:
        datasets = ['actor', 'squirrel', 'chameleon']
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        pe = generate_rwpe(dataset, k=args.k, save_dir=args.save_dir)
        print(f"\n✓ {dataset}: PE shape = {pe.shape}")


if __name__ == '__main__':
    main()
