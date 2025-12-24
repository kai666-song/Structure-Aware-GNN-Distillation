"""
Task 3: GLNN Baseline Implementation
=====================================

GLNN (Graph-Less Neural Networks) is the TRUE competitor we must beat.
It uses simple soft-label distillation from GNN teacher to MLP student.

Reference: Zhang et al., "Graph-less Neural Networks: Teaching Old MLPs 
New Tricks via Distillation", ICLR 2022

This is the baseline that proves whether our spectral decomposition
actually provides value beyond vanilla knowledge distillation.

Usage:
    python baselines/glnn_baseline.py --dataset actor
    python baselines/glnn_baseline.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader_v2 import load_data_with_glognn_splits
from configs.experiment_config import (
    NUM_SPLITS,
    RESULTS_DIR,
    GLOGNN_REPORTED_RESULTS,
)


class MLP(nn.Module):
    """
    Standard MLP Student Model (GLNN style).
    
    Architecture follows GLNN paper:
    - 2-layer MLP with ReLU activation
    - Dropout for regularization
    - No graph structure used
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.layers.append(nn.Linear(nfeat, nhid))
        self.bns.append(nn.BatchNorm1d(nhid))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        
        # Output layer
        self.layers.append(nn.Linear(nhid, nclass))
        
    def forward(self, x):
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.bns)):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.layers[-1](x)
        return x


class SoftTargetLoss(nn.Module):
    """
    Standard Soft Target Loss (Hinton et al., 2015).
    
    L = KL(softmax(s/T), softmax(t/T)) * T^2
    
    This is the EXACT loss used by GLNN.
    """
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature
    
    def forward(self, logits_student, logits_teacher):
        p_s = F.log_softmax(logits_student / self.T, dim=1)
        p_t = F.softmax(logits_teacher / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss


def accuracy(output, labels):
    """Compute accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def load_teacher_logits(dataset: str, split_idx: int):
    """Load pre-saved teacher logits."""
    path = os.path.join(
        'checkpoints', f'glognn_teacher_{dataset}',
        f'split_{split_idx}', 'teacher_logits.pt'
    )
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Teacher logits not found: {path}\n"
            f"Run: python baselines/verify_glognn_teacher.py --dataset {dataset}"
        )
    
    data = torch.load(path)
    return data['logits']


# GLNN Hyperparameters (from GLNN paper)
GLNN_CONFIGS = {
    'actor': {
        'hidden': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'temperature': 4.0,
        'lambda_kd': 1.0,  # Weight for KD loss
        'epochs': 500,
        'patience': 100,
    },
    'chameleon': {
        'hidden': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'temperature': 4.0,
        'lambda_kd': 1.0,
        'epochs': 500,
        'patience': 100,
    },
    'squirrel': {
        'hidden': 256,
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'temperature': 4.0,
        'lambda_kd': 1.0,
        'epochs': 500,
        'patience': 100,
    },
}


def train_glnn(dataset: str, split_idx: int, config: dict, device: str = 'cpu'):
    """
    Train GLNN (MLP with soft-label distillation) on a single split.
    
    Args:
        dataset: Dataset name
        split_idx: Split index (0-9)
        config: Hyperparameter configuration
        device: 'cpu' or 'cuda'
    
    Returns:
        test_acc: Test accuracy
        val_acc: Best validation accuracy
    """
    # Load data (use float32 for MLP, not float64)
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    # Load teacher logits and convert to float32
    teacher_logits = load_teacher_logits(dataset, split_idx).float().to(device)
    
    # Initialize model
    model = MLP(
        nfeat=data['num_features'],
        nhid=config['hidden'],
        nclass=data['num_classes'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss()
    kd_loss_fn = SoftTargetLoss(temperature=config['temperature'])
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        # GLNN Loss: CE + lambda * KD
        loss_ce = ce_loss_fn(logits[train_mask], labels[train_mask])
        loss_kd = kd_loss_fn(logits, teacher_logits)
        loss = loss_ce + config['lambda_kd'] * loss_kd
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(features)
            acc_val = accuracy(logits[val_mask], labels[val_mask])
            acc_test = accuracy(logits[test_mask], labels[test_mask])
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_test_acc = acc_test
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            break
    
    return best_test_acc.item(), best_val_acc.item()


def train_vanilla_mlp(dataset: str, split_idx: int, config: dict, device: str = 'cpu'):
    """
    Train Vanilla MLP (no distillation) as additional baseline.
    
    Args:
        dataset: Dataset name
        split_idx: Split index (0-9)
        config: Hyperparameter configuration
        device: 'cpu' or 'cuda'
    
    Returns:
        test_acc: Test accuracy
        val_acc: Best validation accuracy
    """
    # Load data (use float32 for MLP)
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    # Initialize model
    model = MLP(
        nfeat=data['num_features'],
        nhid=config['hidden'],
        nclass=data['num_classes'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    ce_loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        loss = ce_loss_fn(logits[train_mask], labels[train_mask])
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            acc_val = accuracy(logits[val_mask], labels[val_mask])
            acc_test = accuracy(logits[test_mask], labels[test_mask])
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_test_acc = acc_test
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    return best_test_acc.item(), best_val_acc.item()


def run_glnn_baseline(dataset: str, device: str = 'cpu'):
    """
    Run GLNN baseline on all 10 splits.
    
    Args:
        dataset: Dataset name
        device: 'cpu' or 'cuda'
    
    Returns:
        results: Dictionary with mean, std, and all accuracies
    """
    config = GLNN_CONFIGS[dataset]
    teacher_target = GLOGNN_REPORTED_RESULTS[dataset]
    
    print(f"\n{'='*70}")
    print(f"Running GLNN Baseline on {dataset.upper()}")
    print(f"Teacher (GloGNN++) Target: {teacher_target:.2f}%")
    print(f"{'='*70}")
    
    glnn_accs = []
    mlp_accs = []
    
    for split_idx in range(NUM_SPLITS):
        print(f"\nSplit {split_idx}...")
        
        # Train GLNN
        glnn_test, glnn_val = train_glnn(dataset, split_idx, config, device)
        glnn_accs.append(glnn_test * 100)
        
        # Train Vanilla MLP
        mlp_test, mlp_val = train_vanilla_mlp(dataset, split_idx, config, device)
        mlp_accs.append(mlp_test * 100)
        
        print(f"  GLNN: {glnn_test*100:.2f}%, Vanilla MLP: {mlp_test*100:.2f}%")
    
    glnn_mean = np.mean(glnn_accs)
    glnn_std = np.std(glnn_accs)
    mlp_mean = np.mean(mlp_accs)
    mlp_std = np.std(mlp_accs)
    
    print(f"\n{'='*70}")
    print(f"GLNN BASELINE RESULTS for {dataset.upper()}")
    print(f"{'='*70}")
    print(f"Teacher (GloGNN++): {teacher_target:.2f}%")
    print(f"GLNN:               {glnn_mean:.2f}% ± {glnn_std:.2f}%")
    print(f"Vanilla MLP:        {mlp_mean:.2f}% ± {mlp_std:.2f}%")
    print(f"GLNN vs Teacher:    {glnn_mean - teacher_target:+.2f}%")
    print(f"GLNN vs MLP:        {glnn_mean - mlp_mean:+.2f}%")
    
    results = {
        'dataset': dataset,
        'teacher_target': teacher_target,
        'glnn_mean': glnn_mean,
        'glnn_std': glnn_std,
        'glnn_accs': glnn_accs,
        'mlp_mean': mlp_mean,
        'mlp_std': mlp_std,
        'mlp_accs': mlp_accs,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save results
    results_dir = os.path.join(RESULTS_DIR, 'glnn_baseline')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f'{dataset}_glnn.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}/{dataset}_glnn.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run GLNN Baseline')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'chameleon', 'squirrel'])
    parser.add_argument('--all', action='store_true',
                       help='Run on all datasets')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    datasets = ['actor', 'chameleon', 'squirrel'] if args.all else [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        results = run_glnn_baseline(dataset, device=args.device)
        all_results.append(results)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("GLNN BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Teacher':<10} {'GLNN':<18} {'MLP':<18} {'GLNN-T':<10}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['dataset']:<12} "
              f"{r['teacher_target']:<10.2f} "
              f"{r['glnn_mean']:.2f}±{r['glnn_std']:.2f}{'':>4} "
              f"{r['mlp_mean']:.2f}±{r['mlp_std']:.2f}{'':>4} "
              f"{r['glnn_mean']-r['teacher_target']:+.2f}")
    
    print(f"\nNote: Your method must beat GLNN to claim improvement over vanilla KD.")


if __name__ == '__main__':
    main()
