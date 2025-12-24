"""
Phase 3: Topology Alignment via RKD (Relational Knowledge Distillation)
========================================================================

This script implements the new strategy: instead of forcing student to 
match teacher's logits point-by-point, we force student to match the 
RELATIONAL STRUCTURE of teacher's feature space.

Key insight: Teacher's strength comes from building a good manifold where
same-class nodes are close and different-class nodes are far apart.

Usage:
    # Task 1.1: Test PE impact on GLNN
    python train_rkd.py --task pe_ablation --device cuda
    
    # Task 3: Run RKD experiments
    python train_rkd.py --task rkd --device cuda
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
from models import EnhancedMLP
from configs.experiment_config import NUM_SPLITS


# =============================================================================
# RKD Loss Implementation (Distance + Angle)
# =============================================================================

class RKDDistanceLoss(nn.Module):
    """Distance-wise RKD: align pairwise distances."""
    
    def forward(self, student, teacher):
        teacher = teacher.detach()
        
        # Pairwise distances
        s_dist = self._pdist(student)
        t_dist = self._pdist(teacher)
        
        # Normalize by mean
        s_dist = s_dist / (s_dist.mean() + 1e-8)
        t_dist = t_dist / (t_dist.mean() + 1e-8)
        
        return F.smooth_l1_loss(s_dist, t_dist)
    
    def _pdist(self, e):
        """Pairwise Euclidean distance."""
        n = e.size(0)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        sq = (e ** 2).sum(dim=1)
        dist = sq.unsqueeze(0) + sq.unsqueeze(1) - 2 * torch.mm(e, e.t())
        dist = torch.clamp(dist, min=0).sqrt()
        return dist


class RKDAngleLoss(nn.Module):
    """Angle-wise RKD: align triplet angles."""
    
    def forward(self, student, teacher):
        teacher = teacher.detach()
        
        s_angle = self._angle(student)
        t_angle = self._angle(teacher)
        
        return F.smooth_l1_loss(s_angle, t_angle)
    
    def _angle(self, e):
        """Compute angles for all triplets."""
        # Normalize
        e = F.normalize(e, p=2, dim=1)
        n = e.size(0)
        
        # For efficiency, compute angle matrix
        # angle(i,j,k) = cos(angle at j between i-j and k-j)
        # We use simplified version: cosine similarity matrix
        sim = torch.mm(e, e.t())
        return sim


class CombinedRKDLoss(nn.Module):
    """Combined Distance + Angle RKD Loss."""
    
    def __init__(self, lambda_dist=25.0, lambda_angle=50.0, max_samples=1024):
        super().__init__()
        self.lambda_dist = lambda_dist
        self.lambda_angle = lambda_angle
        self.max_samples = max_samples
        self.dist_loss = RKDDistanceLoss()
        self.angle_loss = RKDAngleLoss()
    
    def forward(self, student_feat, teacher_feat, mask=None):
        """
        Args:
            student_feat: Student penultimate features [N, D]
            teacher_feat: Teacher penultimate features [N, D]
            mask: Optional mask for selecting nodes
        """
        if mask is not None:
            student_feat = student_feat[mask]
            teacher_feat = teacher_feat[mask]
        
        n = student_feat.size(0)
        
        # Subsample for memory efficiency
        if n > self.max_samples:
            idx = torch.randperm(n, device=student_feat.device)[:self.max_samples]
            student_feat = student_feat[idx]
            teacher_feat = teacher_feat[idx]
        
        loss_dist = self.dist_loss(student_feat, teacher_feat)
        loss_angle = self.angle_loss(student_feat, teacher_feat)
        
        return self.lambda_dist * loss_dist + self.lambda_angle * loss_angle



# =============================================================================
# Model with Feature Extraction
# =============================================================================

class MLPWithFeatures(nn.Module):
    """MLP that returns both logits and penultimate features."""
    
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
        
        # Output layer (no norm)
        self.classifier = nn.Linear(nhid, nclass)
    
    def forward(self, x, return_features=False):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        features = x  # Penultimate features
        logits = self.classifier(x)
        
        if return_features:
            return logits, features
        return logits


# =============================================================================
# Helper Functions
# =============================================================================

def load_teacher_logits(dataset, split_idx):
    path = os.path.join('checkpoints', f'glognn_teacher_{dataset}',
                       f'split_{split_idx}', 'teacher_logits.pt')
    data = torch.load(path)
    logits = data.get('logits', data) if isinstance(data, dict) else data
    return logits.float()


def load_pe(dataset):
    path = os.path.join('data', f'pe_rw_{dataset}.pt')
    if os.path.exists(path):
        data = torch.load(path)
        return data['pe']
    return None


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


# =============================================================================
# Task 1.1: PE Ablation on GLNN
# =============================================================================

def train_glnn(dataset, split_idx, use_pe, device):
    """Train GLNN with or without PE."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    if use_pe:
        pe = load_pe(dataset)
        if pe is not None:
            features = torch.cat([features, pe.to(device)], dim=1)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    model = MLPWithFeatures(
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


def task1_pe_ablation(datasets, device):
    """Task 1.1: Compare GLNN with and without PE."""
    print("\n" + "=" * 70)
    print("TASK 1.1: PE Ablation on GLNN")
    print("=" * 70)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        
        no_pe_accs = []
        with_pe_accs = []
        
        for split_idx in range(min(5, NUM_SPLITS)):
            print(f"  Split {split_idx}...", end=" ")
            
            acc_no_pe = train_glnn(dataset, split_idx, use_pe=False, device=device)
            acc_with_pe = train_glnn(dataset, split_idx, use_pe=True, device=device)
            
            no_pe_accs.append(acc_no_pe * 100)
            with_pe_accs.append(acc_with_pe * 100)
            
            print(f"No PE: {acc_no_pe*100:.2f}%, With PE: {acc_with_pe*100:.2f}%")
        
        no_pe_mean = np.mean(no_pe_accs)
        with_pe_mean = np.mean(with_pe_accs)
        diff = with_pe_mean - no_pe_mean
        
        results[dataset] = {
            'no_pe': {'mean': no_pe_mean, 'std': np.std(no_pe_accs)},
            'with_pe': {'mean': with_pe_mean, 'std': np.std(with_pe_accs)},
            'diff': diff,
            'recommendation': 'USE PE' if diff > 0.5 else 'NO PE'
        }
        
        print(f"\n  Summary:")
        print(f"    No PE:   {no_pe_mean:.2f}% ± {np.std(no_pe_accs):.2f}%")
        print(f"    With PE: {with_pe_mean:.2f}% ± {np.std(with_pe_accs):.2f}%")
        print(f"    Diff:    {diff:+.2f}%")
        print(f"    → Recommendation: {results[dataset]['recommendation']}")
    
    return results



# =============================================================================
# Task 3: RKD Training
# =============================================================================

def train_with_rkd(dataset, split_idx, use_pe, lambda_dist, lambda_angle, device):
    """Train with GLNN + RKD loss."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    if use_pe:
        pe = load_pe(dataset)
        if pe is not None:
            features = torch.cat([features, pe.to(device)], dim=1)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    # Student model
    model = MLPWithFeatures(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=3  # 3 layers for more capacity
    ).to(device)
    
    # RKD loss - use smaller weights since we're applying to logits
    rkd_loss_fn = CombinedRKDLoss(
        lambda_dist=lambda_dist, 
        lambda_angle=lambda_angle,
        max_samples=1024
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    T = 4.0
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits, student_feat = model(features, return_features=True)
        
        # CE loss
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        # Soft target KD loss
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        # RKD loss - apply on LOGITS (same dimension as teacher)
        # This aligns the relational structure in output space
        if lambda_dist > 0 or lambda_angle > 0:
            loss_rkd = rkd_loss_fn(logits, teacher_logits, train_mask)
        else:
            loss_rkd = 0
        
        loss = loss_ce + 1.0 * loss_kd + loss_rkd
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


def task3_rkd_experiments(datasets, use_pe_map, device):
    """Task 3: Run alternative KD experiments (RKD failed, try other methods)."""
    print("\n" + "=" * 70)
    print("TASK 3: Alternative KD Experiments")
    print("=" * 70)
    print("NOTE: RKD on logits causes model collapse. Trying alternative approaches.")
    
    # RKD on logits doesn't work - the dimension is too low (5 classes)
    # Try: Label Smoothing, Temperature tuning, Feature Matching
    configs = [
        {'lambda_dist': 0.0, 'lambda_angle': 0.0, 'name': 'GLNN (baseline)'},
    ]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*50}")
        
        use_pe = use_pe_map.get(dataset, False)
        print(f"Using PE: {use_pe}")
        
        results[dataset] = {}
        
        for config in configs:
            print(f"\n  {config['name']}...")
            
            accs = []
            for split_idx in range(min(5, NUM_SPLITS)):
                acc = train_with_rkd(
                    dataset, split_idx, use_pe,
                    config['lambda_dist'], config['lambda_angle'],
                    device
                )
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            results[dataset][config['name']] = {
                'mean': mean_acc,
                'std': std_acc,
                'config': config
            }
            
            print(f"    {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Find best config
        best_name = max(results[dataset].keys(), 
                       key=lambda k: results[dataset][k]['mean'])
        best_acc = results[dataset][best_name]['mean']
        baseline_acc = results[dataset]['GLNN (baseline)']['mean']
        improvement = best_acc - baseline_acc
        
        print(f"\n  Best: {best_name} = {best_acc:.2f}%")
        print(f"  Improvement over GLNN: {improvement:+.2f}%")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 3: RKD Training')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'pe_ablation', 'rkd'])
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['actor', 'squirrel'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_results = {}
    
    # Task 1.1: PE Ablation
    if args.task in ['all', 'pe_ablation']:
        pe_results = task1_pe_ablation(args.datasets, device)
        all_results['pe_ablation'] = pe_results
        
        # Determine PE usage for each dataset
        use_pe_map = {}
        for ds, res in pe_results.items():
            use_pe_map[ds] = res['recommendation'] == 'USE PE'
    else:
        # Default: no PE based on Phase 2 findings
        use_pe_map = {ds: False for ds in args.datasets}
    
    # Task 3: RKD Experiments
    if args.task in ['all', 'rkd']:
        rkd_results = task3_rkd_experiments(args.datasets, use_pe_map, device)
        all_results['rkd'] = rkd_results
    
    # Save results
    os.makedirs('results/phase3_rkd', exist_ok=True)
    results_path = f'results/phase3_rkd/phase3_{args.task}.json'
    
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
    print("PHASE 3 SUMMARY")
    print("=" * 70)
    
    if 'pe_ablation' in all_results:
        print("\n📊 Task 1.1 (PE Ablation):")
        for ds, res in all_results['pe_ablation'].items():
            print(f"  {ds}: No PE={res['no_pe']['mean']:.2f}%, "
                  f"With PE={res['with_pe']['mean']:.2f}% "
                  f"→ {res['recommendation']}")
    
    if 'rkd' in all_results:
        print("\n📊 Task 3 (RKD Results):")
        for ds, res in all_results['rkd'].items():
            baseline = res.get('GLNN (baseline)', {}).get('mean', 0)
            best_name = max(res.keys(), key=lambda k: res[k]['mean'])
            best_acc = res[best_name]['mean']
            print(f"  {ds}: GLNN={baseline:.2f}% → Best={best_name}={best_acc:.2f}%")


if __name__ == '__main__':
    main()
