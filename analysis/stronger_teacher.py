"""
Stronger Teacher Experiment (Task 1)

Test distillation with GCNII as a stronger teacher on heterophilic graphs.

Goal: Verify if Student MLP can follow a stronger Teacher's performance.
- Scenario A: Student improves to 36%+ -> Framework effectively transfers SOTA knowledge
- Scenario B: Student stays at 33% -> MLP capacity limit, but still good trade-off

Usage:
    python analysis/stronger_teacher.py --data actor --num_runs 5
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from models import GAT, GCNII, MLPBatchNorm, convert_adj_to_edge_index
from utils import load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


def train_teacher(teacher, features, edge_index, labels, idx_train, idx_val, 
                  config, device, teacher_name='GCNII'):
    """Train teacher model."""
    optimizer = optim.Adam(teacher.parameters(), lr=config['lr'], 
                           weight_decay=config['wd_teacher'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        teacher.train()
        optimizer.zero_grad()
        
        output = teacher(features, edge_index)
        loss = criterion(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        
        # Validation
        teacher.eval()
        with torch.no_grad():
            output = teacher(features, edge_index)
            val_acc = (output[idx_val].argmax(1) == labels[idx_val]).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    # Load best
    teacher.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    return teacher


def train_student_with_distillation(student, teacher, features_sparse, adj_sparse, 
                                     edge_index, labels, idx_train, idx_val, 
                                     config, device):
    """Train student with knowledge distillation."""
    criterion_task = nn.CrossEntropyLoss()
    criterion_kd = SoftTarget(T=4.0)
    criterion_rkd = AdaptiveRKDLoss(max_samples=2048)
    
    optimizer = optim.Adam(student.parameters(), lr=config['lr'],
                           weight_decay=config['wd_student'])
    
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        student.train()
        optimizer.zero_grad()
        
        student_out = student(features_sparse, adj_sparse)
        with torch.no_grad():
            teacher_out = teacher(features_sparse, edge_index)
        
        # Combined loss
        loss = (criterion_task(student_out[idx_train], labels[idx_train]) +
                criterion_kd(student_out, teacher_out) +
                criterion_rkd(student_out, teacher_out, mask=idx_train))
        
        loss.backward()
        optimizer.step()
        
        # Validation
        student.eval()
        with torch.no_grad():
            output = student(features_sparse, adj_sparse)
            val_acc = (output[idx_val].argmax(1) == labels[idx_val]).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    # Load best
    student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    student.eval()
    
    return student


def run_stronger_teacher_experiment(dataset='actor', num_runs=5, device='cuda'):
    """Compare GAT vs GCNII as teachers."""
    print(f"\n{'='*70}")
    print(f"STRONGER TEACHER EXPERIMENT: {dataset.upper()}")
    print(f"{'='*70}")
    
    # Load data
    adj, features, labels, *_, idx_train, idx_val, idx_test = load_data_new(dataset)
    
    features_processed = preprocess_features(features)
    supports = preprocess_adj(adj)
    
    i = torch.from_numpy(features_processed[0]).long().to(device)
    v = torch.from_numpy(features_processed[1]).to(device)
    features_sparse = torch.sparse_coo_tensor(i.t(), v, features_processed[2]).to(device)
    
    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    adj_sparse = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(device)
    
    edge_index = convert_adj_to_edge_index(adj_sparse).to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    
    nfeat = features_sparse.shape[1]
    nclass = labels.max().item() + 1
    
    print(f"\nDataset: {dataset}")
    print(f"  Nodes: {features_sparse.shape[0]}, Features: {nfeat}, Classes: {nclass}")
    
    # Config
    config = {
        'hidden': 64,
        'epochs': 500,
        'patience': 150,
        'lr': 0.01,
        'wd_teacher': 5e-4,
        'wd_student': 1e-5,
    }
    
    # Results storage
    results = {
        'gat': {'teacher_accs': [], 'student_accs': []},
        'gcnii': {'teacher_accs': [], 'student_accs': []},
    }
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        
        # ========== GAT Teacher ==========
        print("  Training GAT Teacher...")
        gat_teacher = GAT(nfeat, config['hidden'], nclass, dropout=0.6, heads=4).to(device)
        gat_teacher = train_teacher(gat_teacher, features_sparse, edge_index, labels,
                                    idx_train, idx_val, config, device, 'GAT')
        
        with torch.no_grad():
            gat_out = gat_teacher(features_sparse, edge_index)
            gat_acc = (gat_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
        results['gat']['teacher_accs'].append(gat_acc)
        print(f"    GAT Teacher: {gat_acc:.2f}%")
        
        # Student with GAT teacher
        print("  Training Student (GAT teacher)...")
        student_gat = MLPBatchNorm(nfeat, config['hidden'], nclass, dropout=0.5).to(device)
        student_gat = train_student_with_distillation(
            student_gat, gat_teacher, features_sparse, adj_sparse, edge_index,
            labels, idx_train, idx_val, config, device
        )
        
        with torch.no_grad():
            student_out = student_gat(features_sparse, adj_sparse)
            student_gat_acc = (student_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
        results['gat']['student_accs'].append(student_gat_acc)
        print(f"    Student (GAT): {student_gat_acc:.2f}%")
        
        # ========== GCNII Teacher ==========
        print("  Training GCNII Teacher...")
        gcnii_teacher = GCNII(nfeat, config['hidden'], nclass, dropout=0.5,
                              num_layers=8, alpha=0.1, theta=0.5).to(device)
        gcnii_teacher = train_teacher(gcnii_teacher, features_sparse, edge_index, labels,
                                      idx_train, idx_val, config, device, 'GCNII')
        
        with torch.no_grad():
            gcnii_out = gcnii_teacher(features_sparse, edge_index)
            gcnii_acc = (gcnii_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
        results['gcnii']['teacher_accs'].append(gcnii_acc)
        print(f"    GCNII Teacher: {gcnii_acc:.2f}%")
        
        # Student with GCNII teacher
        print("  Training Student (GCNII teacher)...")
        student_gcnii = MLPBatchNorm(nfeat, config['hidden'], nclass, dropout=0.5).to(device)
        student_gcnii = train_student_with_distillation(
            student_gcnii, gcnii_teacher, features_sparse, adj_sparse, edge_index,
            labels, idx_train, idx_val, config, device
        )
        
        with torch.no_grad():
            student_out = student_gcnii(features_sparse, adj_sparse)
            student_gcnii_acc = (student_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
        results['gcnii']['student_accs'].append(student_gcnii_acc)
        print(f"    Student (GCNII): {student_gcnii_acc:.2f}%")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*70}")
    print("STRONGER TEACHER RESULTS SUMMARY")
    print(f"{'='*70}")
    
    summary = {'dataset': dataset, 'num_runs': num_runs}
    
    for teacher_name in ['gat', 'gcnii']:
        t_mean = np.mean(results[teacher_name]['teacher_accs'])
        t_std = np.std(results[teacher_name]['teacher_accs'])
        s_mean = np.mean(results[teacher_name]['student_accs'])
        s_std = np.std(results[teacher_name]['student_accs'])
        gap = s_mean - t_mean
        
        summary[teacher_name] = {
            'teacher_mean': t_mean,
            'teacher_std': t_std,
            'student_mean': s_mean,
            'student_std': s_std,
            'gap': gap,
        }
        
        print(f"\n{teacher_name.upper()} Teacher:")
        print(f"  Teacher: {t_mean:.2f} ± {t_std:.2f}%")
        print(f"  Student: {s_mean:.2f} ± {s_std:.2f}%")
        print(f"  Gap: {gap:+.2f}%")
    
    # Analysis
    gat_student = summary['gat']['student_mean']
    gcnii_student = summary['gcnii']['student_mean']
    gcnii_teacher = summary['gcnii']['teacher_mean']
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    if gcnii_teacher > summary['gat']['teacher_mean']:
        print(f"✓ GCNII is a stronger teacher (+{gcnii_teacher - summary['gat']['teacher_mean']:.2f}%)")
    
    if gcnii_student > gat_student:
        print(f"✓ Student improves with stronger teacher (+{gcnii_student - gat_student:.2f}%)")
        print("  → Framework effectively transfers SOTA knowledge!")
    else:
        print(f"✗ Student doesn't improve with stronger teacher")
        print("  → MLP capacity may be the bottleneck")
    
    if gcnii_student > gcnii_teacher:
        print(f"✨ Student BEATS stronger teacher! (+{gcnii_student - gcnii_teacher:.2f}%)")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open(f'results/stronger_teacher_{dataset}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to results/stronger_teacher_{dataset}.json")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='actor')
    parser.add_argument('--num_runs', type=int, default=5)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    run_stronger_teacher_experiment(args.data, args.num_runs, device)
