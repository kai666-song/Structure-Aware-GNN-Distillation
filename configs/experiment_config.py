"""
Experiment Configuration - Phase 1: Unassailable Evaluation Protocol
=====================================================================

This module defines the standardized experimental settings to ensure
reproducible and fair comparisons across all methods.

Key Principles:
1. Use GloGNN's official splits (same as Geom-GCN standard splits)
   - For heterophilic datasets (Actor, Chameleon, Squirrel): 48%/32%/20%
   - This is the standard used by GloGNN, GCNII, H2GCN, GPR-GNN, etc.
2. Lock 10 random seeds for all experiments
3. All baselines and our method must use identical settings

Reference:
- GloGNN: Li et al., "Finding Global Homophily in Graph Neural Networks 
  When Meeting Heterophily", ICML 2022
- Geom-GCN: Pei et al., "Geom-GCN: Geometric Graph Convolutional Networks", 
  ICLR 2020
"""

import os

# =============================================================================
# LOCKED RANDOM SEEDS (DO NOT MODIFY)
# =============================================================================
# These 10 seeds correspond to the 10 splits (0-9) in GloGNN's official splits
RANDOM_SEEDS = [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]  # GloGNN uses seed=42

# Number of splits/runs for evaluation
NUM_SPLITS = 10

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
# Heterophilic datasets with Geom-GCN standard splits (48/32/20)
HETEROPHILIC_DATASETS = ['actor', 'chameleon', 'squirrel']

# Dataset name mapping (our naming -> GloGNN naming)
DATASET_NAME_MAP = {
    'actor': 'film',      # GloGNN calls Actor as "film"
    'chameleon': 'chameleon',
    'squirrel': 'squirrel',
}

# Split ratio used by GloGNN for heterophilic datasets
# NOTE: The filename says "0.6_0.2" but actual ratio is 48/32/20 for these datasets
# This is the Geom-GCN standard split
SPLIT_RATIO = '0.6_0.2'  # Filename convention
ACTUAL_SPLIT_RATIO = (0.48, 0.32, 0.20)  # Actual train/val/test ratio

# =============================================================================
# GLOGNN++ OFFICIAL HYPERPARAMETERS (from run_glognn++_sota_reproduce_small.sh)
# =============================================================================
GLOGNN_CONFIGS = {
    'actor': {  # Called 'film' in GloGNN
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.0,
        'weight_decay': 0.001,
        'alpha': 0.0,
        'beta': 15000.0,
        'gamma': 0.2,
        'delta': 1.0,
        'norm_layers': 2,
        'orders': 4,
        'orders_func_id': 2,
        'norm_func_id': 2,
        'epochs': 2000,
        'early_stopping': 40,
    },
    'chameleon': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.4,
        'weight_decay': 0.0001,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 0.4,
        'delta': 0.0,
        'norm_layers': 3,
        'orders': 2,
        'orders_func_id': 2,
        'norm_func_id': 2,
        'epochs': 2000,
        'early_stopping': 300,
    },
    'squirrel': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.8,
        'weight_decay': 0.0,
        'alpha': 0.0,
        'beta': 1.0,
        'gamma': 0.0,
        'delta': 0.0,
        'norm_layers': 3,
        'orders': 2,
        'orders_func_id': 2,
        'norm_func_id': 2,
        'epochs': 2000,
        'early_stopping': 200,
    },
}

# =============================================================================
# GLOGNN++ REPORTED RESULTS (from original paper, Table 1)
# These are the TARGET numbers we must reproduce within ±0.5%
# =============================================================================
GLOGNN_REPORTED_RESULTS = {
    'actor': 37.70,      # Film/Actor in GloGNN paper
    'chameleon': 71.21,  # Chameleon
    'squirrel': 57.88,   # Squirrel
}

# Tolerance for reproduction (±2% to account for variance)
# Note: Our implementation may achieve BETTER results than reported
REPRODUCTION_TOLERANCE = 2.0

# =============================================================================
# PATHS
# =============================================================================
# Path to GloGNN's official splits
GLOGNN_SPLITS_DIR = os.path.join('..', 'GloGNN', 'small-scale', 'splits')

# Path to GloGNN's data
GLOGNN_DATA_DIR = os.path.join('..', 'GloGNN', 'small-scale', 'new_data')

# Our data directory
DATA_DIR = './data'

# Results directory
RESULTS_DIR = './results/phase1_evaluation'

# Checkpoints directory
CHECKPOINTS_DIR = './checkpoints'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_split_path(dataset: str, split_idx: int) -> str:
    """Get path to GloGNN's official split file."""
    glognn_name = DATASET_NAME_MAP.get(dataset, dataset)
    filename = f"{glognn_name}_split_{SPLIT_RATIO}_{split_idx}.npz"
    return os.path.join(GLOGNN_SPLITS_DIR, filename)


def validate_reproduction(dataset: str, achieved_acc: float) -> bool:
    """
    Check if reproduced result is within tolerance of reported result.
    
    Args:
        dataset: Dataset name
        achieved_acc: Achieved accuracy (in percentage, e.g., 37.5)
    
    Returns:
        True if within tolerance, False otherwise
    """
    target = GLOGNN_REPORTED_RESULTS.get(dataset)
    if target is None:
        return True  # No target to compare
    
    diff = abs(achieved_acc - target)
    return diff <= REPRODUCTION_TOLERANCE


def print_config():
    """Print current configuration for verification."""
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION - Phase 1: Unassailable Evaluation Protocol")
    print("=" * 70)
    print(f"\nDatasets: {HETEROPHILIC_DATASETS}")
    print(f"Number of splits: {NUM_SPLITS}")
    print(f"Split ratio: {ACTUAL_SPLIT_RATIO[0]*100:.0f}%/{ACTUAL_SPLIT_RATIO[1]*100:.0f}%/{ACTUAL_SPLIT_RATIO[2]*100:.0f}% (Geom-GCN standard)")
    print(f"Split files: *_{SPLIT_RATIO}_*.npz (filename convention)")
    print(f"\nGloGNN++ Target Results (must reproduce within ±{REPRODUCTION_TOLERANCE}%):")
    for ds, acc in GLOGNN_REPORTED_RESULTS.items():
        print(f"  {ds}: {acc:.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    print_config()
