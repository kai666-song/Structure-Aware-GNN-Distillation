"""
GLNN: Graph-Less Neural Network (Zhang et al., 2021)
=====================================================

GLNN is the most direct competitor to our method. It distills knowledge
from a GNN teacher to an MLP student using only soft labels (no structure loss).

Reference: "Graph-less Neural Networks: Teaching Old MLPs New Tricks Via Distillation"
Paper: https://arxiv.org/abs/2110.08727

Key differences from our method:
- GLNN: Only uses soft label distillation (KD loss)
- Ours: Adds RKD + TCD (structure-aware losses)

Expected performance on Actor: ~35-36% (we need to beat this!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLPBatchNorm, GAT, GCNII
from utils.data_utils import load_data_new
from kd_losses.st import SoftTarget


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class GLNN(nn.Module):
    """
    GLNN: Graph-Less Neural Network
    
    Architecture: Same as MLPBatchNorm (2-layer MLP with BatchNorm)
    Training: Distillation from GNN teacher using only soft labels
    
    This is the BASELINE we must beat.
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super(GLNN, self).__init__()
        self.mlp = MLPBatchNorm(nfeat, nhid, nclass, dropout, num_layers)
    
    def forward(self, x, adj=None):
        return self.mlp(x, adj)


def train_glnn(dataset='actor', num_runs=10, teacher_type='gcnii', 
               hidden=256, dropout=0.5, lr=0.01, epochs=500, 
               temperature=4.0, alpha=0.5, device='cuda'):
    """
    Train GLNN baseline with specified teacher.
    
    Args:
        dataset: Dataset name
        num_runs: Number of runs (uses different Geom-GCN splits)
        teacher_type: 'gat' or 'gcnii'
        hidden: Hidden dimension
        dropout: Dropout rate
        lr: Learning rate
        epochs: Training epochs
        temperature: KD temperature
        alpha: Weight for KD loss (1-alpha for task loss)
        device: Device to use
    
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print(f"GLNN Baseline (Teacher: {teacher_type.upper()})")
    print("=" * 70)
    print(f"Dataset: {dataset}, Runs: {num_runs}")
    print()
    
    teacher_accs = []
    student_accs = []
    
    for run in range(num_runs):
        set_seed(42 + run)
        split_idx = run % 10
        
        # Load data
        adj, features, labels, y_train, y_val, y_test, \
            train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = \
            load_data_new(dataset, split_idx=split_idx)
        
        # Convert features
        if hasattr(features, 'toarray'):
            features_np = features.toarray()
        elif hasattr(features, 'todense'):
            features_np = np.array(features.todense())
        else:
            features_np = features
        
        n_features = features_np.shape[1]
        n_classes = int(labels.max()) + 1
        
        # Prepare tensors
        import scipy.sparse as sp
        edge_index = torch.LongTensor(np.array(sp.coo_matrix(adj).nonzero())).to(device)
        features_tensor = torch.FloatTensor(features_np).to(device)
        labels_tensor = labels.to(device)
        
        # ============ Train Teacher ============
        if teacher_type == 'gcnii':
            teacher = GCNII(n_features, hidden, n_classes, num_layers=8, 
                           alpha=0.5, theta=1.0, dropout=dropout).to(device)
        else:
            teacher = GAT(n_features, 8, n_classes, dropout=0.6, heads=8).to(device)
        
        optimizer = optim.Adam(teacher.parameters(), lr=0.01, weight_decay=5e-4)
        
        best_val_acc = 0
        best_teacher_test_acc = 0
        
        for epoch in range(300):
            teacher.train()
            optimizer.zero_grad()
            
            if teacher_type == 'gcnii':
                output = teacher(features_tensor, edge_index)
            else:
                output = teacher(features_tensor, edge_index)
            
            loss = F.cross_entropy(output[idx_train], labels_tensor[idx_train])
            loss.backward()
            optimizer.step()
            
            # Validation
            teacher.eval()
            with torch.no_grad():
                output = teacher(features_tensor, edge_index)
                val_pred = output[idx_val].argmax(dim=1)
                val_acc = (val_pred == labels_tensor[idx_val]).float().mean().item()
                
                test_pred = output[idx_test].argmax(dim=1)
                test_acc = (test_pred == labels_tensor[idx_test]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_teacher_test_acc = test_acc
        
        teacher_accs.append(best_teacher_test_acc * 100)
        
        # ============ Train GLNN (Student) ============
        student = GLNN(n_features, hidden, n_classes, dropout).to(device)
        optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=5e-4)
        kd_loss_fn = SoftTarget(T=temperature)
        
        best_val_acc = 0
        best_student_test_acc = 0
        
        teacher.eval()
        
        for epoch in range(epochs):
            student.train()
            optimizer.zero_grad()
            
            student_out = student(features_tensor)
            
            with torch.no_grad():
                if teacher_type == 'gcnii':
                    teacher_out = teacher(features_tensor, edge_index)
                else:
                    teacher_out = teacher(features_tensor, edge_index)
            
            # GLNN loss: alpha * KD + (1-alpha) * Task
            loss_task = F.cross_entropy(student_out[idx_train], labels_tensor[idx_train])
            loss_kd = kd_loss_fn(student_out, teacher_out)
            
            loss = (1 - alpha) * loss_task + alpha * loss_kd
            
            loss.backward()
            optimizer.step()
            
            # Validation
            student.eval()
            with torch.no_grad():
                output = student(features_tensor)
                val_pred = output[idx_val].argmax(dim=1)
                val_acc = (val_pred == labels_tensor[idx_val]).float().mean().item()
                
                test_pred = output[idx_test].argmax(dim=1)
                test_acc = (test_pred == labels_tensor[idx_test]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_student_test_acc = test_acc
        
        student_accs.append(best_student_test_acc * 100)
        
        print(f"  Run {run+1}: Teacher={teacher_accs[-1]:.2f}%, GLNN={student_accs[-1]:.2f}%")
    
    # Summary
    teacher_mean = np.mean(teacher_accs)
    teacher_std = np.std(teacher_accs)
    student_mean = np.mean(student_accs)
    student_std = np.std(student_accs)
    
    print()
    print(f"Teacher ({teacher_type.upper()}): {teacher_mean:.2f} ± {teacher_std:.2f}%")
    print(f"GLNN (Student):                   {student_mean:.2f} ± {student_std:.2f}%")
    print(f"Gap: {student_mean - teacher_mean:+.2f}%")
    
    results = {
        'dataset': dataset,
        'teacher_type': teacher_type,
        'num_runs': num_runs,
        'teacher': {
            'mean': teacher_mean,
            'std': teacher_std,
            'all': teacher_accs
        },
        'glnn': {
            'mean': student_mean,
            'std': student_std,
            'all': student_accs
        },
        'gap': student_mean - teacher_mean
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = f'results/glnn_baseline_{dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GLNN Baseline')
    parser.add_argument('--dataset', type=str, default='actor')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--teacher', type=str, default='gcnii', choices=['gat', 'gcnii'])
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_glnn(
        dataset=args.dataset,
        num_runs=args.num_runs,
        teacher_type=args.teacher,
        hidden=args.hidden,
        device=device
    )
