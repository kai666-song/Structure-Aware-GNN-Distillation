"""
Teacher-based Homophily Profiling Generator
===========================================

This script computes per-node homophily weights based on Teacher predictions.
These weights indicate how "consistent" each node's neighborhood is according
to the Teacher model, which can be used for adaptive loss weighting.

Key Insight:
- High homophily nodes: neighbors agree with the node → smooth loss is good
- Low homophily nodes: neighbors disagree → sharp/independent loss is better

IMPORTANT: We use Teacher's predictions, NOT ground truth labels!
This is because ground truth is unavailable at test time.

Usage:
    python features/generate_homophily.py --dataset actor --teacher_logits checkpoints/glognn_actor_logits.pt
    python features/generate_homophily.py --dataset actor --run_teacher
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


def compute_homophily_from_predictions(adj, predictions, soft=False):
    """
    Compute per-node homophily based on predictions.
    
    For each node i:
        h_i = (1 / |N(i)|) * sum_{j in N(i)} I(pred_i == pred_j)
    
    Args:
        adj: scipy sparse adjacency matrix (N x N)
        predictions: torch.Tensor of shape (N,) with predicted class labels
                    OR (N, C) with soft predictions (logits/probabilities)
        soft: If True, use soft predictions (cosine similarity)
              If False, use hard predictions (class match)
    
    Returns:
        homophily: torch.Tensor of shape (N, 1) with homophily scores in [0, 1]
    """
    num_nodes = adj.shape[0]
    
    # Convert to COO for edge iteration
    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    
    if soft and predictions.dim() == 2:
        # Soft homophily: cosine similarity between prediction vectors
        # Normalize predictions
        pred_norm = predictions / (predictions.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute similarity for each edge
        src_pred = pred_norm[row]  # (E, C)
        dst_pred = pred_norm[col]  # (E, C)
        edge_sim = (src_pred * dst_pred).sum(dim=1)  # (E,)
        
        # Aggregate: average similarity per node
        homophily = torch.zeros(num_nodes, dtype=torch.float32)
        degree = torch.zeros(num_nodes, dtype=torch.float32)
        
        for i, (r, c, sim) in enumerate(zip(row, col, edge_sim)):
            homophily[r] += sim.item()
            degree[r] += 1
        
        # Normalize by degree
        degree[degree == 0] = 1
        homophily = homophily / degree
        
    else:
        # Hard homophily: class match ratio
        if predictions.dim() == 2:
            predictions = predictions.argmax(dim=1)
        
        predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        
        # Count matching neighbors for each node
        match_count = np.zeros(num_nodes, dtype=np.float32)
        degree = np.zeros(num_nodes, dtype=np.float32)
        
        for r, c in zip(row, col):
            degree[r] += 1
            if predictions[r] == predictions[c]:
                match_count[r] += 1
        
        # Normalize by degree
        degree[degree == 0] = 1
        homophily = match_count / degree
        homophily = torch.from_numpy(homophily)
    
    return homophily.unsqueeze(1)  # (N, 1)


def generate_homophily_weights(dataset_name, teacher_logits=None, 
                               teacher_logits_path=None, 
                               save_dir='./data', split_idx=0,
                               soft=True):
    """
    Generate and save homophily weights for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        teacher_logits: Pre-computed teacher logits (N, C) tensor
        teacher_logits_path: Path to saved teacher logits
        save_dir: Directory to save the weights
        split_idx: Split index for data loading
        soft: Use soft (cosine similarity) or hard (class match) homophily
    
    Returns:
        homophily: torch.Tensor of shape (N, 1)
    """
    print(f"\n{'='*60}")
    print(f"Generating Homophily Weights for {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    adj, features, labels, *_ = load_data_new(dataset_name, split_idx=split_idx)
    num_nodes = adj.shape[0]
    
    # Get teacher predictions
    if teacher_logits is not None:
        logits = teacher_logits
    elif teacher_logits_path is not None:
        print(f"Loading teacher logits from: {teacher_logits_path}")
        data = torch.load(teacher_logits_path)
        if isinstance(data, dict):
            logits = data.get('logits', data.get('soft_labels', None))
            if logits is None:
                logits = list(data.values())[0]
        else:
            logits = data
    else:
        # Try to load from default path
        default_path = os.path.join(save_dir, f'teacher_logits_{dataset_name}.pt')
        if os.path.exists(default_path):
            print(f"Loading teacher logits from: {default_path}")
            data = torch.load(default_path)
            logits = data.get('logits', data) if isinstance(data, dict) else data
        else:
            raise ValueError(
                f"No teacher logits provided!\n"
                f"Either pass teacher_logits tensor, or run:\n"
                f"  python baselines/save_teacher_logits.py --dataset {dataset_name}"
            )
    
    print(f"Teacher logits shape: {logits.shape}")
    print(f"Using {'soft' if soft else 'hard'} homophily computation")
    
    # Compute homophily weights
    homophily = compute_homophily_from_predictions(adj, logits, soft=soft)
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'homophily_weights_{dataset_name}.pt')
    torch.save({
        'homophily': homophily,
        'num_nodes': num_nodes,
        'dataset': dataset_name,
        'soft': soft
    }, save_path)
    
    print(f"\nSaved to: {save_path}")
    print(f"Homophily shape: {homophily.shape}")
    print(f"Homophily stats: mean={homophily.mean():.4f}, std={homophily.std():.4f}")
    print(f"Homophily range: [{homophily.min():.4f}, {homophily.max():.4f}]")
    
    # Distribution analysis
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("\nHomophily Distribution:")
    for i in range(len(bins)-1):
        mask = (homophily >= bins[i]) & (homophily < bins[i+1])
        count = mask.sum().item()
        pct = 100 * count / num_nodes
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:5d} nodes ({pct:5.1f}%)")
    
    return homophily


def load_homophily_weights(dataset_name, data_dir='./data'):
    """
    Load pre-computed homophily weights.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing weight files
    
    Returns:
        homophily: torch.Tensor of shape (N, 1)
    """
    weight_path = os.path.join(data_dir, f'homophily_weights_{dataset_name}.pt')
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Homophily weights not found: {weight_path}\n"
            f"Run: python features/generate_homophily.py --dataset {dataset_name}"
        )
    
    data = torch.load(weight_path)
    return data['homophily']


def main():
    parser = argparse.ArgumentParser(description='Generate Teacher-based Homophily Weights')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'squirrel', 'chameleon', 'cora', 'citeseer', 'pubmed'],
                       help='Dataset name')
    parser.add_argument('--teacher_logits', type=str, default=None,
                       help='Path to teacher logits file')
    parser.add_argument('--save_dir', type=str, default='./data',
                       help='Directory to save weight files')
    parser.add_argument('--soft', action='store_true',
                       help='Use soft (cosine similarity) homophily - NOT recommended')
    parser.add_argument('--hard', action='store_true', default=True,
                       help='Use hard (class match) homophily - RECOMMENDED')
    parser.add_argument('--run_teacher', action='store_true',
                       help='Run GloGNN++ teacher to generate logits first')
    args = parser.parse_args()
    
    # Default to hard homophily (class match) - more reliable for heterophilic graphs
    soft = args.soft and not args.hard
    
    if args.run_teacher:
        print("Running GloGNN++ teacher to generate logits...")
        # Import and run teacher
        from baselines.save_teacher_logits import save_teacher_logits
        save_teacher_logits(args.dataset, save_dir=args.save_dir)
    
    homophily = generate_homophily_weights(
        args.dataset,
        teacher_logits_path=args.teacher_logits,
        save_dir=args.save_dir,
        soft=soft
    )
    
    print(f"\n✓ {args.dataset}: Homophily weights shape = {homophily.shape}")


if __name__ == '__main__':
    main()
