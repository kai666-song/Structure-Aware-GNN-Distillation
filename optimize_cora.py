"""
Cora Optimization Script - Achieve SOTA-level MLP performance

Goals:
- MLP Baseline > 65%
- Distilled MLP > 80%

Key changes:
1. Use MLPBatchNorm instead of basic MLP
2. Grid search over weight_decay
3. Ablation study (gamma=0 vs gamma=optimal)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models import GCN, MLP, MLPBatchNorm
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


def run_experiment(args, use_batchnorm=True):
    """Run single experiment with given config."""
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Load data
    adj, features, labels, *_, idx_train, idx_val, idx_test = load_data_new('cora')
    
    # Preprocess
    features = preprocess_features(features)
    supports = preprocess_adj(adj)
    
    i = torch.from_numpy(features[0]).long().to(device)
    v = torch.from_numpy(features[1]).to(device)
    features = torch.sparse_coo_tensor(i.t(), v, features[2]).to(device)
    
    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    adj = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(device)
    
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    
    nfeat = features.shape[1]
    nclass = labels.max().item() + 1
    
    # Initialize models
    teacher = GCN(nfeat, args.hidden, nclass, args.dropout).to(device)
    
    if use_batchnorm:
        student = MLPBatchNorm(nfeat, args.hidden, nclass, args.dropout, num_layers=args.num_layers).to(device)
    else:
        student = MLP(nfeat, args.hidden, nclass, args.dropout).to(device)
    
    # Loss functions
    criterion_task = nn.CrossEntropyLoss()
    criterion_kd = SoftTarget(T=args.temperature)
    criterion_struct = AdaptiveRKDLoss(max_samples=2048)
    
    # Train Teacher
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=5e-4)
    best_val, best_state = 0, None
    
    for epoch in range(args.epochs):
        teacher.train()
        teacher_optimizer.zero_grad()
        out = teacher(features, adj)
        loss = criterion_task(out[idx_train], labels[idx_train])
        loss.backward()
        teacher_optimizer.step()
        
        teacher.eval()
        with torch.no_grad():
            val_acc = accuracy(teacher(features, adj)[idx_val], labels[idx_val]).item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
    
    teacher.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    with torch.no_grad():
        teacher_acc = accuracy(teacher(features, adj)[idx_test], labels[idx_test]).item() * 100
    
    # Train Student with distillation
    student_optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val, best_state = 0, None
    
    for epoch in range(args.epochs):
        student.train()
        student_optimizer.zero_grad()
        
        student_out = student(features, adj)
        with torch.no_grad():
            teacher_out = teacher(features, adj)
        
        loss_task = criterion_task(student_out[idx_train], labels[idx_train])
        loss_kd = criterion_kd(student_out, teacher_out)
        loss_struct = criterion_struct(student_out, teacher_out, mask=idx_train)
        
        loss = args.alpha * loss_task + args.beta * loss_kd + args.gamma * loss_struct
        loss.backward()
        student_optimizer.step()
        
        student.eval()
        with torch.no_grad():
            val_acc = accuracy(student(features, adj)[idx_val], labels[idx_val]).item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    
    student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    student.eval()
    with torch.no_grad():
        student_acc = accuracy(student(features, adj)[idx_test], labels[idx_test]).item() * 100
    
    return teacher_acc, student_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    
    print("="*70)
    print("CORA OPTIMIZATION - Comparing MLP vs MLPBatchNorm")
    print("="*70)
    
    # Grid search configurations
    configs = [
        # (use_batchnorm, weight_decay, gamma, description)
        (False, 5e-4, 1.0, "Basic MLP, wd=5e-4, gamma=1.0"),
        (False, 1e-5, 1.0, "Basic MLP, wd=1e-5, gamma=1.0"),
        (False, 0, 1.0, "Basic MLP, wd=0, gamma=1.0"),
        (True, 5e-4, 1.0, "MLPBatchNorm, wd=5e-4, gamma=1.0"),
        (True, 1e-5, 1.0, "MLPBatchNorm, wd=1e-5, gamma=1.0"),
        (True, 0, 1.0, "MLPBatchNorm, wd=0, gamma=1.0"),
        (True, 1e-5, 0.0, "MLPBatchNorm, wd=1e-5, gamma=0 (Ablation)"),
        (True, 1e-5, 0.5, "MLPBatchNorm, wd=1e-5, gamma=0.5"),
        (True, 1e-5, 2.0, "MLPBatchNorm, wd=1e-5, gamma=2.0"),
    ]
    
    results = []
    
    for use_bn, wd, gamma, desc in configs:
        print(f"\n>>> {desc}")
        args.weight_decay = wd
        args.gamma = gamma
        
        teacher_accs, student_accs = [], []
        for seed in range(args.num_runs):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)
            
            t_acc, s_acc = run_experiment(args, use_batchnorm=use_bn)
            teacher_accs.append(t_acc)
            student_accs.append(s_acc)
            print(f"  Seed {seed}: Teacher={t_acc:.2f}%, Student={s_acc:.2f}%")
        
        result = {
            'config': desc,
            'use_batchnorm': use_bn,
            'weight_decay': wd,
            'gamma': gamma,
            'teacher_mean': np.mean(teacher_accs),
            'teacher_std': np.std(teacher_accs),
            'student_mean': np.mean(student_accs),
            'student_std': np.std(student_accs)
        }
        results.append(result)
        print(f"  Result: Teacher={result['teacher_mean']:.2f}±{result['teacher_std']:.2f}, "
              f"Student={result['student_mean']:.2f}±{result['student_std']:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Config':<45} {'Student Acc':>15}")
    print("-"*70)
    for r in sorted(results, key=lambda x: -x['student_mean']):
        print(f"{r['config']:<45} {r['student_mean']:.2f} ± {r['student_std']:.2f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/cora_optimization.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/cora_optimization.json")


if __name__ == '__main__':
    main()
