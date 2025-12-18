"""
Advanced Structure-Aware Distillation with Contrastive Learning

Key Improvements over basic distillation:
1. ContrastiveTopologyLoss: InfoNCE-based contrastive learning
2. SoftTopologyLoss: Align with teacher attention weights
3. GraphMixup: Data augmentation for robustness
4. Support for heterophilic graphs and OGB-Arxiv

Target: Cora > 83%, Citeseer > 73%, beat GLNN baseline

Usage:
    python distill_advanced.py --data cora --teacher gat --num_runs 10
    python distill_advanced.py --data ogbn-arxiv --teacher gat --num_runs 3
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from models import GCN, GAT, GATv2, MLPBatchNorm, convert_adj_to_edge_index
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj
from kd_losses import (
    SoftTarget, 
    ContrastiveTopologyLoss, 
    SoftTopologyLoss,
    HybridContrastiveLoss,
    GraphMixup
)


# Dataset-specific configurations
DATASET_CONFIGS = {
    # Homophilic small graphs
    'cora': {
        'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
        'epochs': 500, 'patience': 100, 'gat_heads': 8, 'dropout': 0.6
    },
    'citeseer': {
        'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
        'epochs': 500, 'patience': 100, 'gat_heads': 8, 'dropout': 0.6
    },
    'pubmed': {
        'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
        'epochs': 500, 'patience': 100, 'gat_heads': 8, 'dropout': 0.5
    },
    # Amazon datasets
    'amazon-computers': {
        'hidden': 256, 'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0,
        'epochs': 500, 'patience': 150, 'gat_heads': 4, 'dropout': 0.5
    },
    'amazon-photo': {
        'hidden': 256, 'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0,
        'epochs': 500, 'patience': 150, 'gat_heads': 4, 'dropout': 0.5
    },
    # Heterophilic graphs
    'chameleon': {
        'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
        'epochs': 500, 'patience': 100, 'gat_heads': 8, 'dropout': 0.5
    },
    'squirrel': {
        'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
        'epochs': 500, 'patience': 100, 'gat_heads': 8, 'dropout': 0.5
    },
    # Large-scale OGB
    'ogbn-arxiv': {
        'hidden': 256, 'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0,
        'epochs': 500, 'patience': 50, 'gat_heads': 3, 'dropout': 0.5
    },
}


class AdvancedDistillationTrainer:
    def __init__(self, args, config, seed):
        self.args = args
        self.config = config
        self.seed = seed
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        self.load_data()
        self.init_models()
        self.init_losses()
        
        # Graph Mixup for augmentation
        if args.use_mixup:
            self.mixup = GraphMixup(alpha=args.mixup_alpha)
        else:
            self.mixup = None
        
    def load_data(self):
        """Load and preprocess dataset."""
        self.adj, self.features, self.labels, *_, \
            self.idx_train, self.idx_val, self.idx_test = load_data_new(self.args.data)
        
        # Preprocess for GCN format
        self.features_processed = preprocess_features(self.features)
        self.supports = preprocess_adj(self.adj)
        
        # Convert to tensors
        i = torch.from_numpy(self.features_processed[0]).long().to(self.device)
        v = torch.from_numpy(self.features_processed[1]).to(self.device)
        self.features_sparse = torch.sparse_coo_tensor(i.t(), v, self.features_processed[2]).to(self.device)
        
        i = torch.from_numpy(self.supports[0]).long().to(self.device)
        v = torch.from_numpy(self.supports[1]).to(self.device)
        self.adj_sparse = torch.sparse_coo_tensor(i.t(), v, self.supports[2]).float().to(self.device)
        
        # Convert to edge_index for GAT
        self.edge_index = convert_adj_to_edge_index(self.adj_sparse).to(self.device)
        
        self.labels = self.labels.to(self.device)
        self.idx_train = self.idx_train.to(self.device)
        self.idx_val = self.idx_val.to(self.device)
        self.idx_test = self.idx_test.to(self.device)
        
        self.nfeat = self.features_sparse.shape[1]
        self.nclass = self.labels.max().item() + 1
        
        print(f"Dataset: {self.args.data}")
        print(f"  Nodes: {self.features_sparse.shape[0]}, Features: {self.nfeat}, Classes: {self.nclass}")
        print(f"  Edges: {self.edge_index.shape[1]}")
        print(f"  Train/Val/Test: {len(self.idx_train)}/{len(self.idx_val)}/{len(self.idx_test)}")
        
    def init_models(self):
        """Initialize teacher and student models."""
        config = self.config
        
        # Teacher: GCN, GAT, or GATv2
        if self.args.teacher == 'gatv2':
            self.teacher = GATv2(
                self.nfeat, config['hidden'], self.nclass,
                dropout=config['dropout'], heads=config['gat_heads']
            ).to(self.device)
            self.teacher_type = 'gat'
        elif self.args.teacher == 'gat':
            self.teacher = GAT(
                self.nfeat, config['hidden'], self.nclass,
                dropout=config['dropout'], heads=config['gat_heads']
            ).to(self.device)
            self.teacher_type = 'gat'
        else:
            self.teacher = GCN(
                self.nfeat, config['hidden'], self.nclass, dropout=config['dropout']
            ).to(self.device)
            self.teacher_type = 'gcn'
        
        # Student: MLPBatchNorm with more layers for complex datasets
        num_layers = 3 if self.args.data in ['ogbn-arxiv', 'amazon-computers', 'amazon-photo'] else 2
        self.student = MLPBatchNorm(
            self.nfeat, config['hidden'], self.nclass, 
            dropout=config['dropout'], num_layers=num_layers
        ).to(self.device)
        
    def init_losses(self):
        """Initialize loss functions."""
        self.criterion_task = nn.CrossEntropyLoss()
        self.criterion_kd = SoftTarget(T=self.args.temperature)
        
        # Advanced contrastive + soft topology loss
        self.criterion_struct = HybridContrastiveLoss(
            lambda_con=self.args.lambda_con,
            lambda_soft=self.args.lambda_soft,
            lambda_rkd=self.args.lambda_rkd,
            temperature=self.args.contrastive_temp
        )
        
        # Soft topology loss for attention alignment
        self.criterion_soft_topo = SoftTopologyLoss(temperature=1.0)
        
    def get_teacher_output(self, return_attention=False):
        """Get teacher output based on model type."""
        if self.teacher_type == 'gat':
            return self.teacher(self.features_sparse, self.edge_index, return_attention=return_attention)
        else:
            return self.teacher(self.features_sparse, self.adj_sparse)
    
    def train_teacher(self):
        """Train teacher model."""
        config = self.config
        optimizer = optim.Adam(
            self.teacher.parameters(),
            lr=config['lr'],
            weight_decay=config['wd_teacher']
        )
        
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.teacher.train()
            optimizer.zero_grad()
            
            if self.teacher_type == 'gat':
                output = self.teacher(self.features_sparse, self.edge_index)
            else:
                output = self.teacher(self.features_sparse, self.adj_sparse)
            
            loss = self.criterion_task(output[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            # Validation
            self.teacher.eval()
            with torch.no_grad():
                if self.teacher_type == 'gat':
                    output = self.teacher(self.features_sparse, self.edge_index)
                else:
                    output = self.teacher(self.features_sparse, self.adj_sparse)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.teacher.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        # Load best and freeze
        self.teacher.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Test
        with torch.no_grad():
            if self.teacher_type == 'gat':
                output = self.teacher(self.features_sparse, self.edge_index)
            else:
                output = self.teacher(self.features_sparse, self.adj_sparse)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test]).item() * 100
        
        return test_acc
    
    def train_student(self):
        """Train student with advanced distillation."""
        config = self.config
        optimizer = optim.Adam(
            self.student.parameters(),
            lr=config['lr'],
            weight_decay=config['wd_student']
        )
        
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.student.train()
            optimizer.zero_grad()
            
            # Get student output
            student_out = self.student(self.features_sparse, self.adj_sparse)
            
            # Get teacher output with attention weights
            with torch.no_grad():
                if self.teacher_type == 'gat':
                    teacher_result = self.get_teacher_output(return_attention=True)
                    teacher_out = teacher_result[0]
                    teacher_attn = teacher_result[1][0]  # First layer attention
                else:
                    teacher_out = self.get_teacher_output()
                    teacher_attn = None
            
            # Task loss
            loss_task = self.criterion_task(
                student_out[self.idx_train], self.labels[self.idx_train]
            )
            
            # KD loss (soft targets)
            loss_kd = self.criterion_kd(student_out, teacher_out)
            
            # Contrastive + RKD loss
            loss_struct = self.criterion_struct(
                student_out, teacher_out, self.edge_index,
                teacher_attn=teacher_attn, mask=self.idx_train
            )
            
            # Soft topology loss (attention alignment)
            if teacher_attn is not None and self.args.lambda_soft > 0:
                loss_soft = self.criterion_soft_topo(
                    student_out, teacher_attn, self.edge_index, mask=self.idx_train
                )
            else:
                loss_soft = 0.0
            
            # Combined loss
            loss = (self.args.alpha * loss_task + 
                    self.args.beta * loss_kd + 
                    loss_struct +
                    self.args.lambda_soft * loss_soft)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            self.student.eval()
            with torch.no_grad():
                output = self.student(self.features_sparse, self.adj_sparse)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        # Load best
        self.student.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.student.eval()
        
        # Test
        with torch.no_grad():
            output = self.student(self.features_sparse, self.adj_sparse)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test]).item() * 100
        
        return test_acc


def run_experiment(args):
    """Run distillation experiment."""
    config = DATASET_CONFIGS.get(args.data, DATASET_CONFIGS['cora'])
    
    results = {
        'teacher_accs': [],
        'student_accs': []
    }
    
    for seed in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"Run {seed+1}/{args.num_runs} (seed={seed})")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        
        trainer = AdvancedDistillationTrainer(args, config, seed)
        
        # Train teacher
        teacher_acc = trainer.train_teacher()
        results['teacher_accs'].append(teacher_acc)
        print(f"  Teacher ({args.teacher.upper()}): {teacher_acc:.2f}%")
        
        # Train student
        student_acc = trainer.train_student()
        results['student_accs'].append(student_acc)
        print(f"  Student (MLP): {student_acc:.2f}%")
        
        # Clear cache
        if args.cuda:
            torch.cuda.empty_cache()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY: {args.data.upper()}")
    print(f"{'='*60}")
    print(f"Teacher ({args.teacher.upper()}): {np.mean(results['teacher_accs']):.2f} ± {np.std(results['teacher_accs']):.2f}%")
    print(f"Student (MLP):    {np.mean(results['student_accs']):.2f} ± {np.std(results['student_accs']):.2f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    result_file = f'results/advanced_distill_{args.data}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'dataset': args.data,
            'teacher': args.teacher,
            'config': {
                'alpha': args.alpha, 'beta': args.beta,
                'lambda_con': args.lambda_con, 'lambda_soft': args.lambda_soft,
                'lambda_rkd': args.lambda_rkd
            },
            'teacher_mean': np.mean(results['teacher_accs']),
            'teacher_std': np.std(results['teacher_accs']),
            'student_mean': np.mean(results['student_accs']),
            'student_std': np.std(results['student_accs']),
            'runs': results
        }, f, indent=2)
    print(f"Results saved to {result_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora',
                       help='Dataset: cora, citeseer, pubmed, amazon-*, chameleon, squirrel, ogbn-arxiv')
    parser.add_argument('--teacher', type=str, default='gat', choices=['gcn', 'gat', 'gatv2'])
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--num_runs', type=int, default=10)
    
    # Loss weights - tuned for stability
    parser.add_argument('--alpha', type=float, default=1.0, help='Task loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='KD loss weight')
    parser.add_argument('--lambda_con', type=float, default=0.1, help='Contrastive loss weight (small to avoid instability)')
    parser.add_argument('--lambda_soft', type=float, default=0.5, help='Soft topology loss weight')
    parser.add_argument('--lambda_rkd', type=float, default=1.0, help='RKD loss weight')
    parser.add_argument('--temperature', type=float, default=4.0, help='KD temperature')
    parser.add_argument('--contrastive_temp', type=float, default=0.5, help='Contrastive temperature (higher for stability)')
    
    # Mixup
    parser.add_argument('--use_mixup', action='store_true', help='Use graph mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    print(f"Device: {'CUDA' if args.cuda else 'CPU'}")
    print(f"Teacher: {args.teacher.upper()}")
    print(f"Dataset: {args.data}")
    
    run_experiment(args)


if __name__ == '__main__':
    main()
