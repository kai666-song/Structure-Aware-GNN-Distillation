"""
Phase 4: Ablation Study & Analysis
==================================

This script runs comprehensive ablation experiments to answer:
Q1: Is the improvement from PE or from Spectral Loss?
Q2: Which nodes benefit most from our method?
Q3: How sensitive are the hyperparameters?

Ablation Variants:
- Variant A (No Structure): Plain MLP + KL Loss (baseline)
- Variant B (PE Only): Enhanced MLP + PE + KL Loss
- Variant C (Full Method): Enhanced MLP + PE + Spectral Loss

Usage:
    python run_ablation.py --dataset actor --num_runs 10
    python run_ablation.py --dataset actor --variant A
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
from tqdm import tqdm

from models import EnhancedMLP, ResMLP, MLPBatchNorm, MLP
from utils.data_utils import load_data_new
from kd_losses.adaptive_kd import HybridAdaptiveLoss
from kd_losses.st import SoftTarget


def load_teacher_logits(dataset, data_dir='./data'):
    """Load pre-saved teacher logits."""
    path = os.path.join(data_dir, f'teacher_logits_{dataset}.pt')
    if os.path.exists(path):
        data = torch.load(path)
        return data.get('logits', data) if isinstance(data, dict) else data
    raise FileNotFoundError(f"Teacher logits not found: {path}")


def load_homophily_weights(dataset, data_dir='./data'):
    """Load pre-computed homophily weights."""
    path = os.path.join(data_dir, f'homophily_weights_{dataset}.pt')
    if os.path.exists(path):
        data = torch.load(path)
        return data['homophily']
    raise FileNotFoundError(f"Homophily weights not found: {path}")


def load_positional_encoding(dataset, data_dir='./data'):
    """Load pre-computed positional encoding."""
    path = os.path.join(data_dir, f'pe_rw_{dataset}.pt')
    if os.path.exists(path):
        data = torch.load(path)
        return data['pe']
    return None


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


class SimpleLoss(nn.Module):
    """Simple CE + KL loss without spectral decomposition."""
    def __init__(self, temperature=4.0, lambda_kd=1.0):
        super().__init__()
        self.T = temperature
        self.lambda_kd = lambda_kd
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits_s, logits_t, labels, train_mask=None, **kwargs):
        # CE loss
        if train_mask is not None:
            loss_ce = self.ce(logits_s[train_mask], labels[train_mask])
        else:
            loss_ce = self.ce(logits_s, labels)
        
        # KL loss
        p_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T * self.T)
        
        loss = loss_ce + self.lambda_kd * loss_kd
        
        return loss, {'loss_ce': loss_ce.item(), 'loss_kd': loss_kd.item()}


def run_variant(args, variant, split_idx=0):
    """
    Run a single variant experiment.
    
    Variants:
    - 'A': Plain MLP, no PE, simple KL loss
    - 'B': Enhanced MLP with PE, simple KL loss  
    - 'C': Enhanced MLP with PE, Spectral loss (full method)
    """
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Determine configuration based on variant
    use_pe = variant in ['B', 'C']
    use_spectral = variant == 'C'
    use_enhanced = variant in ['B', 'C']
    
    # Load data
    adj, features, labels, y_train, y_val, y_test, \
        train_mask, val_mask, test_mask, *_ = \
        load_data_new(args.dataset, split_idx=split_idx, use_pe=False)
    
    # Convert features
    if hasattr(features, 'todense'):
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(features)
    
    # Add PE if needed
    if use_pe:
        pe = load_positional_encoding(args.dataset)
        if pe is not None:
            features = torch.cat([features, pe], dim=1)
    
    # Load teacher logits
    teacher_logits = load_teacher_logits(args.dataset)
    
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
    if use_enhanced:
        model = EnhancedMLP(
            nfeat=num_features,
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout,
            num_layers=args.num_layers
        )
    else:
        model = MLPBatchNorm(
            nfeat=num_features,
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout,
            num_layers=2
        )
    model = model.to(device)
    
    # Initialize loss
    if use_spectral:
        homophily = load_homophily_weights(args.dataset)
        loss_fn = HybridAdaptiveLoss(
            adj=adj,
            homophily_weights=homophily,
            lambda_spectral=args.lambda_spectral,
            lambda_soft=args.lambda_soft,
            temperature=args.temperature,
            alpha_low=args.alpha_low,
            alpha_high=args.alpha_high,
            device=device
        )
    else:
        loss_fn = SimpleLoss(temperature=args.temperature, lambda_kd=1.0)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        loss, _ = loss_fn(logits, teacher_logits, labels, train_mask=train_mask)
        
        loss.backward()
        optimizer.step()
        
        # Evaluate
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
        
        if patience_counter >= args.patience:
            break
    
    return best_test_acc, best_val_acc


def run_ablation_study(args):
    """Run complete ablation study."""
    
    variants = {
        'A': 'Plain MLP + KL (No Structure)',
        'B': 'Enhanced MLP + PE + KL (PE Only)',
        'C': 'Enhanced MLP + PE + Spectral (Full Method)'
    }
    
    results = {}
    
    for variant, desc in variants.items():
        print(f"\n{'='*70}")
        print(f"Running Variant {variant}: {desc}")
        print(f"{'='*70}")
        
        test_accs = []
        val_accs = []
        
        for split_idx in range(args.num_runs):
            print(f"  Split {split_idx}...", end=" ")
            test_acc, val_acc = run_variant(args, variant, split_idx)
            test_accs.append(test_acc)
            val_accs.append(val_acc)
            print(f"Test: {test_acc*100:.2f}%")
        
        mean_test = np.mean(test_accs) * 100
        std_test = np.std(test_accs) * 100
        
        results[variant] = {
            'description': desc,
            'mean_test': mean_test,
            'std_test': std_test,
            'all_test': [a * 100 for a in test_accs]
        }
        
        print(f"\nVariant {variant} Result: {mean_test:.2f}% ± {std_test:.2f}%")
    
    # Print summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Variant':<10} {'Description':<40} {'Accuracy':<15}")
    print("-"*70)
    for v, r in results.items():
        print(f"{v:<10} {r['description']:<40} {r['mean_test']:.2f}% ± {r['std_test']:.2f}%")
    print("="*70)
    
    # Calculate improvements
    if 'A' in results and 'B' in results:
        pe_gain = results['B']['mean_test'] - results['A']['mean_test']
        print(f"\nPE Contribution (B - A): +{pe_gain:.2f}%")
    if 'B' in results and 'C' in results:
        spectral_gain = results['C']['mean_test'] - results['B']['mean_test']
        print(f"Spectral Loss Contribution (C - B): +{spectral_gain:.2f}%")
    if 'A' in results and 'C' in results:
        total_gain = results['C']['mean_test'] - results['A']['mean_test']
        print(f"Total Improvement (C - A): +{total_gain:.2f}%")
    
    # Save results
    save_path = os.path.join(args.results_dir, f'ablation_{args.dataset}.json')
    with open(save_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'results': results,
            'config': vars(args),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='actor')
    parser.add_argument('--num_runs', type=int, default=10)
    
    # Model
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Loss
    parser.add_argument('--lambda_spectral', type=float, default=1.0)
    parser.add_argument('--lambda_soft', type=float, default=0.5)
    parser.add_argument('--alpha_low', type=float, default=1.0)
    parser.add_argument('--alpha_high', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=50)
    
    # System
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--results_dir', type=str, default='./results')
    
    # Single variant mode
    parser.add_argument('--variant', type=str, default=None,
                       choices=['A', 'B', 'C'],
                       help='Run single variant only')
    
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    if args.variant:
        # Run single variant
        test_accs = []
        for split_idx in range(args.num_runs):
            test_acc, _ = run_variant(args, args.variant, split_idx)
            test_accs.append(test_acc)
            print(f"Split {split_idx}: {test_acc*100:.2f}%")
        print(f"\nVariant {args.variant}: {np.mean(test_accs)*100:.2f}% ± {np.std(test_accs)*100:.2f}%")
    else:
        # Run full ablation
        run_ablation_study(args)


if __name__ == '__main__':
    main()
