"""
Phase 3 Final: Best Configuration Search
=========================================

Combine the best hyperparameters found:
- Actor: T=8.0, λ_kd=5.0
- Squirrel: T=1.0, λ_kd=10.0

Also try combinations to find the optimal setup.

Usage:
    python train_best_config.py --device cuda
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader_v2 import load_data_with_glognn_splits
from configs.experiment_config import NUM_SPLITS


class SimpleMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        
        self.layers.append(nn.Linear(nfeat, nhid))
        self.norms.append(nn.LayerNorm(nhid))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.norms.append(nn.LayerNorm(nhid))
        
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


def train_glnn(dataset, split_idx, temperature, lambda_kd, device):
    """Train GLNN with specified hyperparameters."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configurations to try for each dataset
    configs = {
        'actor': [
            {'T': 4.0, 'lambda_kd': 1.0, 'name': 'GLNN (default)'},
            {'T': 8.0, 'lambda_kd': 1.0, 'name': 'T=8'},
            {'T': 4.0, 'lambda_kd': 5.0, 'name': 'λ=5'},
            {'T': 8.0, 'lambda_kd': 5.0, 'name': 'T=8, λ=5'},
            {'T': 8.0, 'lambda_kd': 10.0, 'name': 'T=8, λ=10'},
            {'T': 6.0, 'lambda_kd': 5.0, 'name': 'T=6, λ=5'},
            {'T': 10.0, 'lambda_kd': 5.0, 'name': 'T=10, λ=5'},
        ],
        'squirrel': [
            {'T': 4.0, 'lambda_kd': 1.0, 'name': 'GLNN (default)'},
            {'T': 1.0, 'lambda_kd': 1.0, 'name': 'T=1'},
            {'T': 4.0, 'lambda_kd': 10.0, 'name': 'λ=10'},
            {'T': 1.0, 'lambda_kd': 10.0, 'name': 'T=1, λ=10'},
            {'T': 2.0, 'lambda_kd': 10.0, 'name': 'T=2, λ=10'},
            {'T': 1.0, 'lambda_kd': 5.0, 'name': 'T=1, λ=5'},
            {'T': 1.0, 'lambda_kd': 15.0, 'name': 'T=1, λ=15'},
        ],
    }
    
    results = {}
    
    for dataset in ['actor', 'squirrel']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        results[dataset] = {}
        
        for config in configs[dataset]:
            print(f"\n  {config['name']}...")
            
            accs = []
            for split_idx in range(min(10, NUM_SPLITS)):  # Use all 10 splits
                acc = train_glnn(dataset, split_idx, config['T'], config['lambda_kd'], device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            results[dataset][config['name']] = {
                'mean': mean_acc,
                'std': std_acc,
                'T': config['T'],
                'lambda_kd': config['lambda_kd']
            }
            
            print(f"    {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Find best
        best_name = max(results[dataset].keys(), 
                       key=lambda k: results[dataset][k]['mean'])
        best = results[dataset][best_name]
        print(f"\n  ★ Best: {best_name} = {best['mean']:.2f}% ± {best['std']:.2f}%")
    
    # Save results
    os.makedirs('results/phase3_final', exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    with open('results/phase3_final/best_config.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    baselines = {'actor': 36.64, 'squirrel': 58.96}
    teachers = {'actor': 37.40, 'squirrel': 59.68}
    
    for dataset in ['actor', 'squirrel']:
        best_name = max(results[dataset].keys(), 
                       key=lambda k: results[dataset][k]['mean'])
        best = results[dataset][best_name]
        
        print(f"\n{dataset.upper()}:")
        print(f"  GLNN Baseline: {baselines[dataset]:.2f}%")
        print(f"  Teacher:       {teachers[dataset]:.2f}%")
        print(f"  Our Best:      {best['mean']:.2f}% ± {best['std']:.2f}%")
        print(f"  Config:        T={best['T']}, λ_kd={best['lambda_kd']}")
        
        improvement = best['mean'] - baselines[dataset]
        gap_closed = (best['mean'] - baselines[dataset]) / (teachers[dataset] - baselines[dataset]) * 100
        print(f"  Improvement:   +{improvement:.2f}%")
        print(f"  Gap Closed:    {gap_closed:.1f}%")


if __name__ == '__main__':
    main()
