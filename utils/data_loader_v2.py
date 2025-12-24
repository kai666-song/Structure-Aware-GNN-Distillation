"""
Unified Data Loader v2 - Using GloGNN's Official Splits
========================================================

This module provides standardized data loading that uses GloGNN's official
60/20/20 splits to ensure fair comparison with published baselines.

CRITICAL: This replaces the previous data loading that used PyG's Geom-GCN
splits (48/32/20), which would make comparisons unfair.

Usage:
    from utils.data_loader_v2 import load_data_with_glognn_splits
    
    adj, features, labels, splits = load_data_with_glognn_splits(
        dataset='actor', 
        split_idx=0
    )
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_sparse(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1)).flatten()
    # Handle zero rows
    rowsum[rowsum == 0] = 1
    r_inv = 1.0 / rowsum
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx, dtype=torch.float32):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float64 if dtype == torch.float64 else np.float32))
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)


def load_heterophilic_data_raw(dataset: str, data_dir: str):
    """
    Load raw heterophilic dataset (Actor/Film, Chameleon, Squirrel).
    
    This follows GloGNN's exact data loading procedure.
    
    Args:
        dataset: Dataset name ('actor'/'film', 'chameleon', 'squirrel')
        data_dir: Path to GloGNN's new_data directory
    
    Returns:
        adj: scipy sparse adjacency matrix (unnormalized)
        features: scipy sparse feature matrix (unnormalized)
        labels: numpy array of labels
    """
    # Map dataset name
    glognn_name = 'film' if dataset == 'actor' else dataset
    
    # File paths
    graph_file = os.path.join(data_dir, glognn_name, 'out1_graph_edges.txt')
    features_file = os.path.join(data_dir, glognn_name, 'out1_node_feature_label.txt')
    
    if not os.path.exists(graph_file):
        raise FileNotFoundError(
            f"Graph file not found: {graph_file}\n"
            f"Please ensure GloGNN repository is cloned at ../GloGNN"
        )
    
    # Load graph edges
    graph_dict = defaultdict(list)
    with open(graph_file) as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip().split('\t')
            assert len(parts) == 2
            src, dst = int(parts[0]), int(parts[1])
            graph_dict[src].append(dst)
            graph_dict[dst].append(src)  # Undirected
    
    # Sort for consistency
    graph_dict_ordered = defaultdict(list)
    for key in sorted(graph_dict):
        graph_dict_ordered[key] = sorted(graph_dict[key])
    
    # Build adjacency matrix
    num_nodes = max(graph_dict_ordered.keys()) + 1
    rows, cols = [], []
    for src, dsts in graph_dict_ordered.items():
        for dst in dsts:
            rows.append(src)
            cols.append(dst)
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
    
    # Load features and labels
    graph_node_features_dict = {}
    graph_labels_dict = {}
    
    with open(features_file) as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip().split('\t')
            assert len(parts) == 3
            node_id = int(parts[0])
            
            if glognn_name == 'film':
                # Film/Actor: sparse binary features (932-dim)
                feature_blank = np.zeros(932, dtype=np.float32)
                feature_indices = np.array(parts[1].split(','), dtype=np.int32)
                feature_blank[feature_indices] = 1
                graph_node_features_dict[node_id] = feature_blank
            else:
                # Chameleon/Squirrel: dense features
                graph_node_features_dict[node_id] = np.array(
                    parts[1].split(','), dtype=np.float32
                )
            
            graph_labels_dict[node_id] = int(parts[2])
    
    # Convert to arrays
    features_list = [graph_node_features_dict[i] for i in range(num_nodes)]
    features = sp.csr_matrix(np.vstack(features_list))
    
    labels = np.array([graph_labels_dict[i] for i in range(num_nodes)])
    
    return adj, features, labels


def load_glognn_split(dataset: str, split_idx: int, splits_dir: str, num_nodes: int):
    """
    Load GloGNN's official split masks.
    
    Args:
        dataset: Dataset name
        split_idx: Split index (0-9)
        splits_dir: Path to GloGNN's splits directory
        num_nodes: Number of nodes in the graph
    
    Returns:
        train_mask, val_mask, test_mask: boolean numpy arrays
    """
    glognn_name = 'film' if dataset == 'actor' else dataset
    split_file = os.path.join(splits_dir, f"{glognn_name}_split_0.6_0.2_{split_idx}.npz")
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Please ensure GloGNN repository is cloned at ../GloGNN"
        )
    
    with np.load(split_file) as data:
        train_mask = data['train_mask'].astype(bool)
        val_mask = data['val_mask'].astype(bool)
        test_mask = data['test_mask'].astype(bool)
    
    # Verify mask length matches num_nodes
    if len(train_mask) != num_nodes:
        raise ValueError(
            f"Split mask length ({len(train_mask)}) does not match "
            f"number of nodes ({num_nodes}). Data mismatch!"
        )
    
    return train_mask, val_mask, test_mask


def load_data_with_glognn_splits(
    dataset: str,
    split_idx: int = 0,
    normalize_features: bool = True,
    normalize_adj: bool = True,
    add_self_loops: bool = True,
    glognn_data_dir: str = None,
    glognn_splits_dir: str = None,
    dtype: torch.dtype = torch.float64,  # GloGNN uses float64
):
    """
    Load dataset with GloGNN's official 60/20/20 splits.
    
    This is the ONLY data loading function that should be used for
    fair comparison with GloGNN and other baselines.
    
    Args:
        dataset: Dataset name ('actor', 'chameleon', 'squirrel')
        split_idx: Split index (0-9)
        normalize_features: Whether to row-normalize features
        normalize_adj: Whether to normalize adjacency matrix
        add_self_loops: Whether to add self-loops before normalization
        glognn_data_dir: Path to GloGNN's new_data directory
        glognn_splits_dir: Path to GloGNN's splits directory
        dtype: Data type for tensors (default: float64 for GloGNN compatibility)
    
    Returns:
        dict with keys:
            - adj: torch sparse tensor (normalized if requested)
            - adj_raw: scipy sparse matrix (unnormalized, for spectral decomposition)
            - features: torch tensor
            - labels: torch LongTensor
            - train_mask, val_mask, test_mask: torch BoolTensor
            - idx_train, idx_val, idx_test: torch LongTensor
            - num_nodes, num_features, num_classes: int
    """
    # Default paths - try multiple possible locations
    if glognn_data_dir is None:
        possible_data_dirs = [
            os.path.join('..', 'GloGNN', 'small-scale', 'new_data'),
            os.path.join('..', '..', 'GloGNN', 'small-scale', 'new_data'),
            os.path.join('.', 'GloGNN', 'small-scale', 'new_data'),
        ]
        for d in possible_data_dirs:
            if os.path.exists(d):
                glognn_data_dir = d
                break
        if glognn_data_dir is None:
            raise FileNotFoundError(
                f"GloGNN data directory not found. Tried: {possible_data_dirs}\n"
                f"Please clone GloGNN: git clone https://github.com/RecklessRonan/GloGNN.git"
            )
    
    if glognn_splits_dir is None:
        possible_splits_dirs = [
            os.path.join('..', 'GloGNN', 'small-scale', 'splits'),
            os.path.join('..', '..', 'GloGNN', 'small-scale', 'splits'),
            os.path.join('.', 'GloGNN', 'small-scale', 'splits'),
        ]
        for d in possible_splits_dirs:
            if os.path.exists(d):
                glognn_splits_dir = d
                break
        if glognn_splits_dir is None:
            raise FileNotFoundError(
                f"GloGNN splits directory not found. Tried: {possible_splits_dirs}"
            )
    
    # Load raw data
    adj_raw, features_raw, labels = load_heterophilic_data_raw(dataset, glognn_data_dir)
    num_nodes = adj_raw.shape[0]
    
    # Load splits (pass num_nodes for validation)
    train_mask, val_mask, test_mask = load_glognn_split(
        dataset, split_idx, glognn_splits_dir, num_nodes
    )
    
    # Process adjacency matrix
    adj = adj_raw.copy()
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0])
    if normalize_adj:
        adj = normalize_sparse(adj)
    
    # Process features
    features = features_raw.copy()
    if normalize_features:
        features = normalize_sparse(features)
    
    # Convert to tensors with specified dtype
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj, dtype=dtype)
    
    if dtype == torch.float64:
        features_tensor = torch.DoubleTensor(np.array(features.todense()))
    else:
        features_tensor = torch.FloatTensor(np.array(features.todense()))
    
    labels_tensor = torch.LongTensor(labels)
    
    train_mask_tensor = torch.BoolTensor(train_mask)
    val_mask_tensor = torch.BoolTensor(val_mask)
    test_mask_tensor = torch.BoolTensor(test_mask)
    
    idx_train = torch.where(train_mask_tensor)[0]
    idx_val = torch.where(val_mask_tensor)[0]
    idx_test = torch.where(test_mask_tensor)[0]
    
    num_features = features_tensor.shape[1]
    num_classes = int(labels.max()) + 1
    
    # Print split statistics
    print(f"Dataset: {dataset}, Split: {split_idx}")
    print(f"  Nodes: {num_nodes}, Features: {num_features}, Classes: {num_classes}")
    print(f"  Train: {train_mask.sum()} ({100*train_mask.sum()/num_nodes:.1f}%)")
    print(f"  Val:   {val_mask.sum()} ({100*val_mask.sum()/num_nodes:.1f}%)")
    print(f"  Test:  {test_mask.sum()} ({100*test_mask.sum()/num_nodes:.1f}%)")
    
    return {
        'adj': adj_tensor,
        'adj_raw': adj_raw,  # For spectral decomposition
        'features': features_tensor,
        'labels': labels_tensor,
        'train_mask': train_mask_tensor,
        'val_mask': val_mask_tensor,
        'test_mask': test_mask_tensor,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': num_classes,
    }
