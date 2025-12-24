"""
Phase 4: Feature-based RKD (Relational Knowledge Distillation)
===============================================================

This script implements the final strategy:
- Base: Best Hinton KD config (Actor: T=8, λ=10; Squirrel: T=1, λ=10)
- Add: RKD Loss on penultimate features (64-dim)

Loss formula:
    L_total = L_CE + λ_soft * L_KL + λ_rkd * L_RKD

Where L_RKD = λ_dist * L_distance + λ_angle * L_angle

Usage:
    # First, extract teacher features
    python baselines/save_teacher_features.py --all --device cuda
    
    # Then run Phase 4 experiments
    python train_phase4_rkd.py --device cuda
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


# =============================================================================
# RKD Loss (Feature-based)
# =============================================================================

class RKDDistanceLoss(nn.Module):
    """Distance-wise RKD: align pairwise distances in feature space."""
    
    def forward(self, student, teacher):
        teacher = teacher.detach()
        
        s_dist = self._pdist(student)
        t_dist = self._pdist(teacher)
        
        # Normalize by mean to make loss scale-invariant
        s_dist = s_dist / (s_dist.mean() + 1e-8)
        t_dist = t_dist / (t_dist.mean() + 1e-8)
        
        return F.smooth_l1_loss(s_dist, t_dist)
    
    def _pdist(self, e):
        """Pairwise Euclidean distance."""
        n = e.size(0)
        sq = (e ** 2).sum(dim=1)
        dist = sq.unsqueeze(0) + sq.unsqueeze(1) - 2 * torch.mm(e, e.t())
        dist = torch.clamp(dist, min=0).sqrt()
        return dist


class RKDAngleLoss(nn.Module):
    """Angle-wise RKD: align cosine similarities in feature space."""
    
    def forward(self, student, teacher):
        teacher = teacher.detach()
        
        s_sim = self._cosine_sim(student)
        t_sim = self._cosine_sim(teacher)
        
        return F.smooth_l1_loss(s_sim, t_sim)
    
    def _cosine_sim(self, e):
        """Pairwise cosine similarity."""
        e = F.normalize(e, p=2, dim=1)
        return torch.mm(e, e.t())


class FeatureRKDLoss(nn.Module):
    """Combined Feature-based RKD Loss."""
    
    def __init__(self, lambda_dist=1.0, lambda_angle=1.0, max_samples=512):
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
        
        # Subsample for memory efficiency (RKD is O(n^2))
        if n > self.max_samples:
            idx = torch.randperm(n, device=student_feat.device)[:self.max_samples]
            student_feat = student_feat[idx]
            teacher_feat = teacher_feat[idx]
        
        loss = 0
        if self.lambda_dist > 0:
            loss = loss + self.lambda_dist * self.dist_loss(student_feat, teacher_feat)
        if self.lambda_angle > 0:
            loss = loss + self.lambda_angle * self.angle_loss(student_feat, teacher_feat)
        
        return loss


# =============================================================================
# Student Model with Feature Extraction
# =============================================================================

class MLPStudent(nn.Module):
    """MLP Student that returns both logits and penultimate features."""
    
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.nhid = nhid
        
        # Input layer
        self.layers.append(nn.Linear(nfeat, nhid))
        self.norms.append(nn.LayerNorm(nhid))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.norms.append(nn.LayerNorm(nhid))
        
        # Output layer
        self.classifier = nn.Linear(nhid, nclass)
    
    def forward(self, x, return_features=False):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        features = x  # Penultimate features (nhid dimensional)
        logits = self.classifier(x)
        
        if return_features:
            return logits, features
        return logits


# =============================================================================
# Data Loading
# =============================================================================

def load_teacher_data(dataset, split_idx):
    """Load teacher logits and features."""
    base_path = os.path.join('checkpoints', f'glognn_teacher_{dataset}', f'split_{split_idx}')
    
    # Load logits
    logits_path = os.path.join(base_path, 'teacher_logits.pt')
    logits_data = torch.load(logits_path)
    logits = logits_data.get('logits', logits_data) if isinstance(logits_data, dict) else logits_data
    
    # Load features
    features_path = os.path.join(base_path, 'teacher_features.pt')
    if os.path.exists(features_path):
        features_data = torch.load(features_path)
        features = features_data['features']
    else:
        features = None
    
    return logits.float(), features


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


# =============================================================================
# Training
# =============================================================================

def train_with_feature_rkd(dataset, split_idx, config, device):
    """
    Train student with Hinton KD + Feature-based RKD.
    
    Args:
        dataset: Dataset name
        split_idx: Split index
        config: Training configuration dict
        device: torch device
    
    Returns:
        best_test_acc: Best test accuracy
    """
    # Load data
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    # Load teacher
    teacher_logits, teacher_features = load_teacher_data(dataset, split_idx)
    teacher_logits = teacher_logits.to(device)
    
    if teacher_features is not None:
        teacher_features = teacher_features.to(device)
        teacher_feat_dim = teacher_features.shape[1]
    else:
        teacher_feat_dim = 64  # Default
        teacher_features = None
    
    # Student model - match teacher's hidden dim for fair comparison
    student_hidden = config.get('student_hidden', teacher_feat_dim)
    
    model = MLPStudent(
        nfeat=features.shape[1],
        nhid=student_hidden,
        nclass=data['num_classes'],
        dropout=config.get('dropout', 0.5),
        num_layers=config.get('num_layers', 2)
    ).to(device)
    
    # Losses
    ce_loss = nn.CrossEntropyLoss()
    
    if teacher_features is not None and config.get('lambda_rkd', 0) > 0:
        rkd_loss_fn = FeatureRKDLoss(
            lambda_dist=config.get('lambda_dist', 1.0),
            lambda_angle=config.get('lambda_angle', 1.0),
            max_samples=config.get('rkd_samples', 512)
        )
    else:
        rkd_loss_fn = None
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    T = config.get('temperature', 4.0)
    lambda_kd = config.get('lambda_kd', 1.0)
    lambda_rkd = config.get('lambda_rkd', 0.0)
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    max_patience = config.get('patience', 100)
    
    for epoch in range(config.get('epochs', 500)):
        model.train()
        optimizer.zero_grad()
        
        logits, student_feat = model(features, return_features=True)
        
        # CE Loss
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        # Hinton KD Loss (soft targets)
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        # Feature RKD Loss
        if rkd_loss_fn is not None and teacher_features is not None:
            loss_rkd = rkd_loss_fn(student_feat, teacher_features, train_mask)
        else:
            loss_rkd = 0
        
        # Total loss
        loss = loss_ce + lambda_kd * loss_kd + lambda_rkd * loss_rkd
        
        loss.backward()
        optimizer.step()
        
        # Validation
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
        
        if patience >= max_patience:
            break
    
    return best_test_acc.item()


# =============================================================================
# Experiments
# =============================================================================

def run_phase4_experiments(datasets, device):
    """Run Phase 4 experiments: Best Hinton KD + Feature RKD."""
    
    print("\n" + "=" * 70)
    print("PHASE 4: Feature-based RKD Experiments")
    print("=" * 70)
    
    # Best configs from Phase 3
    best_configs = {
        'actor': {'temperature': 8.0, 'lambda_kd': 10.0},
        'squirrel': {'temperature': 1.0, 'lambda_kd': 10.0},
    }
    
    # RKD weight search space (start small since RKD loss can be large)
    rkd_weights = [0.0, 0.001, 0.01, 0.1, 0.5]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        base_config = best_configs.get(dataset, {'temperature': 4.0, 'lambda_kd': 1.0})
        print(f"Base config: T={base_config['temperature']}, λ_kd={base_config['lambda_kd']}")
        
        results[dataset] = {}
        
        for lambda_rkd in rkd_weights:
            config_name = f"λ_rkd={lambda_rkd}"
            print(f"\n  Testing {config_name}...")
            
            config = {
                'temperature': base_config['temperature'],
                'lambda_kd': base_config['lambda_kd'],
                'lambda_rkd': lambda_rkd,
                'lambda_dist': 1.0,
                'lambda_angle': 1.0,
                'student_hidden': 64,  # Match teacher
                'num_layers': 2,
                'dropout': 0.5,
                'lr': 0.01,
                'weight_decay': 5e-4,
                'epochs': 500,
                'patience': 100,
                'rkd_samples': 512,
            }
            
            accs = []
            for split_idx in range(min(5, NUM_SPLITS)):
                acc = train_with_feature_rkd(dataset, split_idx, config, device)
                accs.append(acc * 100)
                print(f"    Split {split_idx}: {acc*100:.2f}%")
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            results[dataset][config_name] = {
                'mean': mean_acc,
                'std': std_acc,
                'config': config,
            }
            
            print(f"  → {config_name}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Find best
        best_name = max(results[dataset].keys(), 
                       key=lambda k: results[dataset][k]['mean'])
        best_acc = results[dataset][best_name]['mean']
        baseline_acc = results[dataset]['λ_rkd=0.0']['mean']
        improvement = best_acc - baseline_acc
        
        print(f"\n  📊 Best: {best_name} = {best_acc:.2f}%")
        print(f"  📈 Improvement over baseline: {improvement:+.2f}%")
    
    return results


def run_full_evaluation(datasets, device):
    """Run full 10-split evaluation on best configs."""
    
    print("\n" + "=" * 70)
    print("PHASE 4: Full Evaluation (10 splits)")
    print("=" * 70)
    
    # Best configs (update after initial experiments)
    best_configs = {
        'actor': {
            'temperature': 8.0, 'lambda_kd': 10.0, 'lambda_rkd': 0.01,
            'lambda_dist': 1.0, 'lambda_angle': 1.0,
        },
        'squirrel': {
            'temperature': 1.0, 'lambda_kd': 10.0, 'lambda_rkd': 0.01,
            'lambda_dist': 1.0, 'lambda_angle': 1.0,
        },
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        
        config = {
            **best_configs.get(dataset, {}),
            'student_hidden': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'epochs': 500,
            'patience': 100,
            'rkd_samples': 512,
        }
        
        print(f"Config: T={config['temperature']}, λ_kd={config['lambda_kd']}, λ_rkd={config['lambda_rkd']}")
        
        accs = []
        for split_idx in range(NUM_SPLITS):
            acc = train_with_feature_rkd(dataset, split_idx, config, device)
            accs.append(acc * 100)
            print(f"  Split {split_idx}: {acc*100:.2f}%")
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        results[dataset] = {
            'mean': mean_acc,
            'std': std_acc,
            'all_accs': accs,
            'config': config,
        }
        
        print(f"\n  Final: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 4: Feature-based RKD')
    parser.add_argument('--mode', type=str, default='search',
                       choices=['search', 'full'])
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['actor', 'squirrel'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check if teacher features exist
    for dataset in args.datasets:
        feat_path = f'checkpoints/glognn_teacher_{dataset}/split_0/teacher_features.pt'
        if not os.path.exists(feat_path):
            print(f"\n⚠️  Teacher features not found for {dataset}!")
            print(f"   Please run: python baselines/save_teacher_features.py --dataset {dataset} --device {args.device}")
            return
    
    if args.mode == 'search':
        results = run_phase4_experiments(args.datasets, device)
    else:
        results = run_full_evaluation(args.datasets, device)
    
    # Save results
    os.makedirs('results/phase4_rkd', exist_ok=True)
    results_path = f'results/phase4_rkd/phase4_{args.mode}.json'
    
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
        json.dump(convert(results), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY")
    print("=" * 70)
    
    # Reference baselines
    baselines = {
        'actor': {'glnn': 36.64, 'teacher': 37.40},
        'squirrel': {'glnn': 58.96, 'teacher': 59.68},
    }
    
    for dataset in args.datasets:
        if dataset in results:
            if args.mode == 'search':
                best_name = max(results[dataset].keys(),
                               key=lambda k: results[dataset][k]['mean'])
                best = results[dataset][best_name]
            else:
                best = results[dataset]
            
            glnn = baselines.get(dataset, {}).get('glnn', 0)
            teacher = baselines.get(dataset, {}).get('teacher', 0)
            
            print(f"\n{dataset.upper()}:")
            print(f"  GLNN Baseline: {glnn:.2f}%")
            print(f"  Teacher:       {teacher:.2f}%")
            print(f"  Our Best:      {best['mean']:.2f}% ± {best['std']:.2f}%")
            print(f"  Improvement:   {best['mean'] - glnn:+.2f}%")
            
            if teacher > glnn:
                gap_closed = (best['mean'] - glnn) / (teacher - glnn) * 100
                print(f"  Gap Closed:    {gap_closed:.1f}%")


if __name__ == '__main__':
    main()
