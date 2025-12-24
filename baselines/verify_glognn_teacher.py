"""
Task 2: Teacher Model Verification (GloGNN++)
==============================================

This script verifies that our GloGNN++ implementation reproduces the
original paper's results within ±0.5% tolerance.

CRITICAL: If this verification fails, ALL subsequent distillation 
experiments are INVALID and must be halted.

Target Results (from GloGNN paper Table 1):
- Actor (Film): 37.70%
- Chameleon: 71.21%
- Squirrel: 57.88%

Usage:
    python baselines/verify_glognn_teacher.py --dataset actor
    python baselines/verify_glognn_teacher.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter
import math
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader_v2 import load_data_with_glognn_splits
from configs.experiment_config import (
    GLOGNN_CONFIGS, 
    GLOGNN_REPORTED_RESULTS,
    REPRODUCTION_TOLERANCE,
    NUM_SPLITS,
    RESULTS_DIR,
)


# =============================================================================
# GloGNN++ Model (Exact copy from GloGNN repository)
# =============================================================================

class MLP_NORM(nn.Module):
    """
    GloGNN++ Model - Exact implementation from the original paper.
    
    Reference: Li et al., "Finding Global Homophily in Graph Neural Networks 
    When Meeting Heterophily", ICML 2022
    """
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, 
                 gamma, delta, norm_func_id, norm_layers, orders, 
                 orders_func_id, device='cpu'):
        super(MLP_NORM, self).__init__()
        
        # Use float64 for GloGNN (as in original implementation)
        self.fc1 = nn.Linear(nfeat, nhid).double()
        self.fc2 = nn.Linear(nhid, nclass).double()
        self.fc3 = nn.Linear(nnodes, nhid).double()
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha, dtype=torch.float64)
        self.beta = torch.tensor(beta, dtype=torch.float64)
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.delta = torch.tensor(delta, dtype=torch.float64)
        self.norm_layers = norm_layers
        self.orders = orders
        self.class_eye = torch.eye(nclass, dtype=torch.float64)
        self.nodes_eye = torch.eye(nnodes, dtype=torch.float64)
        self.device = device

        self.orders_weight = Parameter(
            torch.ones(orders, 1, dtype=torch.float64) / orders, requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.zeros(nclass, orders, dtype=torch.float64), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.zeros(orders, orders, dtype=torch.float64), requires_grad=True
        )
        self.diag_weight = Parameter(
            torch.ones(nclass, 1, dtype=torch.float64) / nclass, requires_grad=True
        )
        
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if norm_func_id == 1:
            self.norm = self._norm_func1
        else:
            self.norm = self._norm_func2

        if orders_func_id == 1:
            self.order_func = self._order_func1
        elif orders_func_id == 2:
            self.order_func = self._order_func2
        else:
            self.order_func = self._order_func3

    def to(self, device):
        super().to(device)
        self.device = device
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.gamma = self.gamma.to(device)
        self.delta = self.delta.to(device)
        self.class_eye = self.class_eye.to(device)
        self.nodes_eye = self.nodes_eye.to(device)
        return self

    def forward(self, x, adj):
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)
        xA = self.fc3(adj)
        x = F.relu(self.delta * xX + (1 - self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        return x  # Return logits, not log_softmax

    def _norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def _norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def _order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def _order_func2(self, x, res, adj):
        tmp_orders = torch.spmm(adj, res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def _order_func3(self, x, res, adj):
        orders_para = torch.mm(
            torch.relu(torch.mm(x, self.orders_weight_matrix)),
            self.orders_weight_matrix2
        )
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders


def accuracy(output, labels):
    """Compute accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train_glognn(dataset: str, split_idx: int, config: dict, device: str = 'cpu'):
    """
    Train GloGNN++ on a single split.
    
    Args:
        dataset: Dataset name
        split_idx: Split index (0-9)
        config: Hyperparameter configuration
        device: 'cpu' or 'cuda'
    
    Returns:
        test_acc: Test accuracy
        val_acc: Best validation accuracy
        logits: Final model logits (for distillation)
    """
    # Load data
    data = load_data_with_glognn_splits(dataset, split_idx)
    
    # Convert to float64 (GloGNN uses float64)
    features = data['features'].to(torch.float64).to(device)
    adj = data['adj'].to(torch.float64).to(device)
    labels = data['labels'].to(device)
    idx_train = data['idx_train'].to(device)
    idx_val = data['idx_val'].to(device)
    idx_test = data['idx_test'].to(device)
    
    num_nodes = data['num_nodes']
    num_features = data['num_features']
    num_classes = data['num_classes']
    
    # Initialize model
    model = MLP_NORM(
        nnodes=num_nodes,
        nfeat=num_features,
        nhid=config['hidden'],
        nclass=num_classes,
        dropout=config['dropout'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        delta=config['delta'],
        norm_func_id=config['norm_func_id'],
        norm_layers=config['norm_layers'],
        orders=config['orders'],
        orders_func_id=config['orders_func_id'],
        device=device,
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    best_logits = None
    cost_val = []
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_test_acc = acc_test
            best_logits = output.detach().cpu()
        
        # Early stopping
        cost_val.append(loss_val.item())
        if epoch > config['early_stopping']:
            if cost_val[-1] > np.mean(cost_val[-(config['early_stopping']+1):-1]):
                break
    
    return best_test_acc.item(), best_val_acc.item(), best_logits


def verify_teacher(dataset: str, device: str = 'cpu', save_logits: bool = True):
    """
    Verify GloGNN++ teacher performance on all 10 splits.
    
    Args:
        dataset: Dataset name
        device: 'cpu' or 'cuda'
        save_logits: Whether to save teacher logits for distillation
    
    Returns:
        mean_acc: Mean test accuracy
        std_acc: Standard deviation
        passed: Whether verification passed (within tolerance)
    """
    config = GLOGNN_CONFIGS[dataset]
    target = GLOGNN_REPORTED_RESULTS[dataset]
    
    print(f"\n{'='*70}")
    print(f"Verifying GloGNN++ Teacher on {dataset.upper()}")
    print(f"Target: {target:.2f}% (tolerance: ±{REPRODUCTION_TOLERANCE}%)")
    print(f"{'='*70}")
    
    all_test_accs = []
    all_logits = []
    
    for split_idx in range(NUM_SPLITS):
        print(f"\nSplit {split_idx}...")
        test_acc, val_acc, logits = train_glognn(dataset, split_idx, config, device)
        all_test_accs.append(test_acc * 100)
        all_logits.append(logits)
        print(f"  Test: {test_acc*100:.2f}%, Val: {val_acc*100:.2f}%")
    
    mean_acc = np.mean(all_test_accs)
    std_acc = np.std(all_test_accs)
    
    # Check if within tolerance
    diff = abs(mean_acc - target)
    passed = diff <= REPRODUCTION_TOLERANCE
    
    print(f"\n{'='*70}")
    print(f"VERIFICATION RESULTS for {dataset.upper()}")
    print(f"{'='*70}")
    print(f"Achieved: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Target:   {target:.2f}%")
    print(f"Diff:     {diff:.2f}%")
    print(f"Status:   {'✓ PASSED' if passed else '✗ FAILED'}")
    
    if not passed:
        print(f"\n⚠️  WARNING: Reproduction failed!")
        print(f"    The achieved accuracy ({mean_acc:.2f}%) differs from the")
        print(f"    reported result ({target:.2f}%) by more than {REPRODUCTION_TOLERANCE}%.")
        print(f"    Please check hyperparameters and data loading.")
    
    # Save logits for distillation
    if save_logits:
        save_dir = os.path.join('checkpoints', f'glognn_teacher_{dataset}')
        os.makedirs(save_dir, exist_ok=True)
        
        for split_idx, logits in enumerate(all_logits):
            split_dir = os.path.join(save_dir, f'split_{split_idx}')
            os.makedirs(split_dir, exist_ok=True)
            torch.save({
                'logits': logits,
                'config': config,
                'test_acc': all_test_accs[split_idx],
            }, os.path.join(split_dir, 'teacher_logits.pt'))
        
        print(f"\nTeacher logits saved to: {save_dir}")
    
    # Save verification results
    results_dir = os.path.join(RESULTS_DIR, 'teacher_verification')
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'dataset': dataset,
        'target': target,
        'achieved_mean': float(mean_acc),
        'achieved_std': float(std_acc),
        'all_accs': [float(a) for a in all_test_accs],
        'diff': float(diff),
        'passed': bool(passed),
        'config': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in config.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(results_dir, f'{dataset}_verification.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return mean_acc, std_acc, passed


def main():
    parser = argparse.ArgumentParser(description='Verify GloGNN++ Teacher')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'chameleon', 'squirrel'])
    parser.add_argument('--all', action='store_true',
                       help='Verify all datasets')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save teacher logits')
    args = parser.parse_args()
    
    datasets = ['actor', 'chameleon', 'squirrel'] if args.all else [args.dataset]
    
    all_passed = True
    results_summary = []
    
    for dataset in datasets:
        mean_acc, std_acc, passed = verify_teacher(
            dataset, 
            device=args.device,
            save_logits=not args.no_save
        )
        results_summary.append({
            'dataset': dataset,
            'mean': mean_acc,
            'std': std_acc,
            'passed': passed,
        })
        if not passed:
            all_passed = False
    
    # Print summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    for r in results_summary:
        status = '✓' if r['passed'] else '✗'
        print(f"{status} {r['dataset']}: {r['mean']:.2f}% ± {r['std']:.2f}%")
    
    if all_passed:
        print(f"\n✓ All verifications PASSED. Teacher models are qualified.")
        print(f"  You may proceed with distillation experiments.")
    else:
        print(f"\n✗ Some verifications FAILED.")
        print(f"  DO NOT proceed with distillation until all teachers are verified.")
        sys.exit(1)


if __name__ == '__main__':
    main()
