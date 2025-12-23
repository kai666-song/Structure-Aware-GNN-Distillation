"""
Phase 5: Hyperparameter Sensitivity Analysis
=============================================

Analyzes sensitivity to key hyperparameters:
1. PE Dimension (k): [4, 8, 16, 32, 64]
2. Spectral Loss Weight (λ): [0.1, 0.5, 1.0, 2.0, 5.0]

Goal: Prove our method is ROBUST across a range of hyperparameters,
not just tuned to a single magic number.

Usage:
    python analysis/sensitivity.py --dataset actor --param pe_dim
    python analysis/sensitivity.py --dataset actor --param lambda_spectral
    python analysis/sensitivity.py --dataset actor --all
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
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EnhancedMLP
from utils.data_utils import load_data_new
from kd_losses.adaptive_kd import HybridAdaptiveLoss
from features.generate_pe import compute_rwpe


def load_teacher_logits(dataset, data_dir='./data'):
    path = os.path.join(data_dir, f'teacher_logits_{dataset}.pt')
    data = torch.load(path)
    return data.get('logits', data) if isinstance(data, dict) else data


def load_homophily_weights(dataset, data_dir='./data'):
    path = os.path.join(data_dir, f'homophily_weights_{dataset}.pt')
    data = torch.load(path)
    return data['homophily']


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).double().sum() / len(labels)


def run_single_config(dataset, pe_dim, lambda_spectral, split_idx=0, 
                      epochs=200, patience=30, device='cpu'):
    """Run a single configuration and return test accuracy."""
    
    # Load data
    result = load_data_new(dataset, split_idx=split_idx)
    adj, features, labels = result[0], result[1], result[2]
    train_mask, val_mask, test_mask = result[6], result[7], result[8]
    
    # Convert features
    if hasattr(features, 'todense'):
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(features)
    
    # Generate PE with specified dimension
    if pe_dim > 0:
        pe = compute_rwpe(adj, k=pe_dim)
        # Normalize PE
        pe_mean = pe.mean(dim=0, keepdim=True)
        pe_std = pe.std(dim=0, keepdim=True)
        pe_std[pe_std < 1e-6] = 1.0
        pe = (pe - pe_mean) / pe_std
        features = torch.cat([features, pe], dim=1)
    
    # Load teacher logits and homophily
    teacher_logits = load_teacher_logits(dataset)
    homophily = load_homophily_weights(dataset)
    
    # Convert masks
    if isinstance(train_mask, np.ndarray):
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    # Move to device
    features = features.to(device)
    labels = labels.to(device)
    teacher_logits = teacher_logits.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    num_features = features.shape[1]
    num_classes = int(labels.max().item()) + 1
    
    # Initialize model
    model = EnhancedMLP(
        nfeat=num_features,
        nhid=256,
        nclass=num_classes,
        dropout=0.5,
        num_layers=3
    ).to(device)
    
    # Initialize loss
    loss_fn = HybridAdaptiveLoss(
        adj=adj,
        homophily_weights=homophily,
        lambda_spectral=lambda_spectral,
        lambda_soft=0.5,
        temperature=4.0,
        device=device
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        loss, _ = loss_fn(logits, teacher_logits, labels, train_mask=train_mask)
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask]).item()
            test_acc = accuracy(logits[test_mask], labels[test_mask]).item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_test_acc


def sweep_pe_dimension(dataset, num_splits=5, device='cpu'):
    """Sweep PE dimension."""
    pe_dims = [0, 4, 8, 16, 32, 64]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"PE Dimension Sensitivity - {dataset.upper()}")
    print(f"{'='*60}")
    
    for pe_dim in pe_dims:
        print(f"\nPE Dim = {pe_dim}...")
        accs = []
        for split_idx in range(num_splits):
            acc = run_single_config(
                dataset, pe_dim=pe_dim, lambda_spectral=1.0,
                split_idx=split_idx, device=device
            )
            accs.append(acc)
            print(f"  Split {split_idx}: {acc*100:.2f}%")
        
        results[pe_dim] = {
            'mean': np.mean(accs) * 100,
            'std': np.std(accs) * 100,
            'all': [a * 100 for a in accs]
        }
        print(f"  Mean: {results[pe_dim]['mean']:.2f}% ± {results[pe_dim]['std']:.2f}%")
    
    return results


def sweep_lambda_spectral(dataset, num_splits=5, device='cpu'):
    """Sweep spectral loss weight."""
    lambdas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Lambda Spectral Sensitivity - {dataset.upper()}")
    print(f"{'='*60}")
    
    for lam in lambdas:
        print(f"\nλ_spectral = {lam}...")
        accs = []
        for split_idx in range(num_splits):
            acc = run_single_config(
                dataset, pe_dim=16, lambda_spectral=lam,
                split_idx=split_idx, device=device
            )
            accs.append(acc)
            print(f"  Split {split_idx}: {acc*100:.2f}%")
        
        results[lam] = {
            'mean': np.mean(accs) * 100,
            'std': np.std(accs) * 100,
            'all': [a * 100 for a in accs]
        }
        print(f"  Mean: {results[lam]['mean']:.2f}% ± {results[lam]['std']:.2f}%")
    
    return results


def plot_sensitivity(results, param_name, dataset, save_dir='./figures'):
    """Plot sensitivity curve."""
    os.makedirs(save_dir, exist_ok=True)
    
    x = list(results.keys())
    means = [results[k]['mean'] for k in x]
    stds = [results[k]['std'] for k in x]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.errorbar(range(len(x)), means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#e74c3c')
    
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Sensitivity to {param_name} - {dataset.upper()}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line for best result
    best_idx = np.argmax(means)
    ax.axhline(y=means[best_idx], color='green', linestyle='--', alpha=0.5,
               label=f'Best: {means[best_idx]:.2f}%')
    ax.legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'sensitivity_{param_name}_{dataset}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Sensitivity Analysis')
    parser.add_argument('--dataset', type=str, default='actor')
    parser.add_argument('--param', type=str, default='all',
                       choices=['pe_dim', 'lambda_spectral', 'all'])
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    
    all_results = {}
    
    if args.param in ['pe_dim', 'all']:
        results = sweep_pe_dimension(args.dataset, args.num_splits, device)
        all_results['pe_dim'] = results
        plot_sensitivity(results, 'PE Dimension', args.dataset)
    
    if args.param in ['lambda_spectral', 'all']:
        results = sweep_lambda_spectral(args.dataset, args.num_splits, device)
        all_results['lambda_spectral'] = results
        plot_sensitivity(results, 'Lambda Spectral', args.dataset)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for param, results in all_results.items():
        print(f"\n{param}:")
        for k, v in results.items():
            print(f"  {k}: {v['mean']:.2f}% ± {v['std']:.2f}%")
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    save_path = f'./results/sensitivity_{args.dataset}.json'
    with open(save_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nResults saved to: {save_path}")


if __name__ == '__main__':
    main()
