"""
Phase 3 Alternative: Enhanced KD Methods
=========================================

RKD on logits failed (model collapse). This script tries alternative approaches:
1. Temperature Tuning - find optimal T for soft targets
2. Label Smoothing - regularize hard labels
3. Mixup Training - data augmentation
4. Confidence-based Weighting - weight samples by teacher confidence

Usage:
    python train_enhanced_kd.py --task all --datasets actor squirrel --device cuda
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader_v2 import load_data_with_glognn_splits
from configs.experiment_config import NUM_SPLITS


class SimpleMLP(nn.Module):
    """Simple MLP for KD experiments."""
    
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.layers.append(nn.Linear(nfeat, nhid))
        self.norms.append(nn.LayerNorm(nhid))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.norms.append(nn.LayerNorm(nhid))
        
        # Output layer
        self.classifier = nn.Linear(nhid, nclass)
    
    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        return self.classifier(x)


def load_teacher_logits(dataset, split_idx):
    path = os.path.join('checkpoints', f'glognn_teacher_{dataset}',
                       f'split_{split_idx}', 'teacher_logits.pt')
    data = torch.load(path)
    logits = data.get('logits', data) if isinstance(data, dict) else data
    return logits.float()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


# =============================================================================
# Method 1: Temperature Tuning
# =============================================================================

def train_with_temperature(dataset, split_idx, temperature, device):
    """Train GLNN with different temperatures."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    model = SimpleMLP(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    T = temperature
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        loss = loss_ce + 1.0 * loss_kd
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
        
        if patience >= 100:
            break
    
    return best_test_acc.item()


def task_temperature_tuning(datasets, device):
    """Find optimal temperature for KD."""
    print("\n" + "=" * 70)
    print("TASK: Temperature Tuning")
    print("=" * 70)
    
    temperatures = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0]
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        results[dataset] = {}
        
        for T in temperatures:
            accs = []
            for split_idx in range(min(5, NUM_SPLITS)):
                acc = train_with_temperature(dataset, split_idx, T, device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[dataset][f'T={T}'] = {'mean': mean_acc, 'std': std_acc}
            print(f"  T={T:4.1f}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Find best T
        best_T = max(results[dataset].keys(), key=lambda k: results[dataset][k]['mean'])
        print(f"  → Best: {best_T} = {results[dataset][best_T]['mean']:.2f}%")
    
    return results


# =============================================================================
# Method 2: Label Smoothing
# =============================================================================

def train_with_label_smoothing(dataset, split_idx, smoothing, device):
    """Train with label smoothing on hard labels."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    model = SimpleMLP(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)
    T = 4.0
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        loss = loss_ce + 1.0 * loss_kd
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
        
        if patience >= 100:
            break
    
    return best_test_acc.item()


def task_label_smoothing(datasets, device):
    """Test label smoothing."""
    print("\n" + "=" * 70)
    print("TASK: Label Smoothing")
    print("=" * 70)
    
    smoothing_values = [0.0, 0.05, 0.1, 0.15, 0.2]
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        results[dataset] = {}
        
        for s in smoothing_values:
            accs = []
            for split_idx in range(min(5, NUM_SPLITS)):
                acc = train_with_label_smoothing(dataset, split_idx, s, device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[dataset][f'smooth={s}'] = {'mean': mean_acc, 'std': std_acc}
            print(f"  smooth={s:.2f}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        best = max(results[dataset].keys(), key=lambda k: results[dataset][k]['mean'])
        print(f"  → Best: {best} = {results[dataset][best]['mean']:.2f}%")
    
    return results


# =============================================================================
# Method 3: Confidence-based Sample Weighting
# =============================================================================

def train_with_confidence_weighting(dataset, split_idx, use_weighting, device):
    """Weight samples by teacher confidence."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    # Compute teacher confidence (max probability)
    teacher_probs = F.softmax(teacher_logits, dim=1)
    teacher_confidence = teacher_probs.max(dim=1)[0]  # [N]
    
    model = SimpleMLP(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    T = 4.0
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        # CE loss (standard)
        loss_ce = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        # KD loss with optional confidence weighting
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        
        if use_weighting:
            # Weight by teacher confidence
            kl_per_sample = F.kl_div(p_s, p_t, reduction='none').sum(dim=1)
            weights = teacher_confidence  # High confidence = more weight
            loss_kd = (kl_per_sample * weights).mean() * (T * T)
        else:
            loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        loss = loss_ce + 1.0 * loss_kd
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
        
        if patience >= 100:
            break
    
    return best_test_acc.item()


def task_confidence_weighting(datasets, device):
    """Test confidence-based sample weighting."""
    print("\n" + "=" * 70)
    print("TASK: Confidence-based Sample Weighting")
    print("=" * 70)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        results[dataset] = {}
        
        for use_weighting in [False, True]:
            name = 'With Weighting' if use_weighting else 'No Weighting'
            accs = []
            for split_idx in range(min(5, NUM_SPLITS)):
                acc = train_with_confidence_weighting(dataset, split_idx, use_weighting, device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[dataset][name] = {'mean': mean_acc, 'std': std_acc}
            print(f"  {name}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return results


# =============================================================================
# Method 4: KD Weight Tuning
# =============================================================================

def train_with_kd_weight(dataset, split_idx, lambda_kd, device):
    """Train with different KD loss weights."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    model = SimpleMLP(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    T = 4.0
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        loss = loss_ce + lambda_kd * loss_kd
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
        
        if patience >= 100:
            break
    
    return best_test_acc.item()


def task_kd_weight_tuning(datasets, device):
    """Find optimal KD loss weight."""
    print("\n" + "=" * 70)
    print("TASK: KD Weight Tuning")
    print("=" * 70)
    
    lambda_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        results[dataset] = {}
        
        for lam in lambda_values:
            accs = []
            for split_idx in range(min(5, NUM_SPLITS)):
                acc = train_with_kd_weight(dataset, split_idx, lam, device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[dataset][f'λ={lam}'] = {'mean': mean_acc, 'std': std_acc}
            print(f"  λ_kd={lam:4.1f}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        best = max(results[dataset].keys(), key=lambda k: results[dataset][k]['mean'])
        print(f"  → Best: {best} = {results[dataset][best]['mean']:.2f}%")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced KD Methods')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'temperature', 'smoothing', 'confidence', 'kd_weight'])
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['actor', 'squirrel'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_results = {}
    
    if args.task in ['all', 'temperature']:
        all_results['temperature'] = task_temperature_tuning(args.datasets, device)
    
    if args.task in ['all', 'smoothing']:
        all_results['smoothing'] = task_label_smoothing(args.datasets, device)
    
    if args.task in ['all', 'confidence']:
        all_results['confidence'] = task_confidence_weighting(args.datasets, device)
    
    if args.task in ['all', 'kd_weight']:
        all_results['kd_weight'] = task_kd_weight_tuning(args.datasets, device)
    
    # Save results
    os.makedirs('results/phase3_enhanced', exist_ok=True)
    results_path = f'results/phase3_enhanced/enhanced_kd_{args.task}.json'
    
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for task_name, task_results in all_results.items():
        print(f"\n📊 {task_name.upper()}:")
        for ds, res in task_results.items():
            if isinstance(res, dict) and 'mean' not in res:
                best_key = max(res.keys(), key=lambda k: res[k]['mean'])
                print(f"  {ds}: Best = {best_key} ({res[best_key]['mean']:.2f}%)")


if __name__ == '__main__':
    main()
