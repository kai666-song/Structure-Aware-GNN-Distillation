"""
Structure-Aware Knowledge Distillation Training Script

Implements the "Trinity" distillation engine with three loss components:
1. L_task (Hard Loss): CrossEntropy with ground truth labels
2. L_kd (Soft Loss): KL divergence with teacher soft labels  
3. L_struct (Structure Loss): RKD loss for relational knowledge

Usage:
    python distill.py --data cora --alpha 1.0 --beta 1.0 --gamma 1.0
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from models import GCN, MLP
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


# Dataset-specific hyperparameters
# Student uses longer patience since KD training is more complex
DATASET_CONFIGS = {
    'cora': {
        'lr': 0.01, 'weight_decay': 5e-4, 'hidden': 64,
        'dropout': 0.5, 'epochs': 300, 'patience': 100,
    },
    'citeseer': {
        'lr': 0.01, 'weight_decay': 5e-4, 'hidden': 64,
        'dropout': 0.5, 'epochs': 300, 'patience': 100,
    },
    'pubmed': {
        'lr': 0.01, 'weight_decay': 5e-4, 'hidden': 64,
        'dropout': 0.5, 'epochs': 300, 'patience': 100,
    },
    'amazon-computers': {
        'lr': 0.01, 'weight_decay': 0, 'hidden': 256,
        'dropout': 0.5, 'epochs': 500, 'patience': 150,
    },
    'amazon-photo': {
        'lr': 0.01, 'weight_decay': 0, 'hidden': 256,
        'dropout': 0.5, 'epochs': 500, 'patience': 150,
    },
}


class DistillationTrainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        # Load data
        self.load_data()
        
        # Initialize models
        self.init_models()
        
        # Initialize loss functions
        self.init_losses()
        
    def load_data(self):
        """Load and preprocess dataset."""
        self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, \
            self.train_mask, self.val_mask, self.test_mask, \
            self.idx_train, self.idx_val, self.idx_test = load_data_new(self.args.data)
        
        print(f'Dataset: {self.args.data}')
        print(f'  Nodes: {self.features.shape[0]}, Features: {self.features.shape[1]}')
        print(f'  Classes: {self.labels.max().item() + 1}')
        print(f'  Train/Val/Test: {len(self.idx_train)}/{len(self.idx_val)}/{len(self.idx_test)}')
        
        # Preprocess
        self.features = preprocess_features(self.features)
        self.supports = preprocess_adj(self.adj)
        
        # Convert to tensors
        i = torch.from_numpy(self.features[0]).long().to(self.device)
        v = torch.from_numpy(self.features[1]).to(self.device)
        self.features = torch.sparse_coo_tensor(i.t(), v, self.features[2]).to(self.device)
        
        i = torch.from_numpy(self.supports[0]).long().to(self.device)
        v = torch.from_numpy(self.supports[1]).to(self.device)
        self.adj = torch.sparse_coo_tensor(i.t(), v, self.supports[2]).float().to(self.device)
        
        self.labels = self.labels.to(self.device)
        self.idx_train = self.idx_train.to(self.device)
        self.idx_val = self.idx_val.to(self.device)
        self.idx_test = self.idx_test.to(self.device)
        
        self.nfeat = self.features.shape[1]
        self.nclass = self.labels.max().item() + 1
        
    def init_models(self):
        """Initialize teacher and student models."""
        config = self.config
        
        # Teacher: GCN
        self.teacher = GCN(
            nfeat=self.nfeat, nhid=config['hidden'],
            nclass=self.nclass, dropout=config['dropout']
        ).to(self.device)
        
        # Student: MLP
        self.student = MLP(
            nfeat=self.nfeat, nhid=config['hidden'],
            nclass=self.nclass, dropout=config['dropout']
        ).to(self.device)
        
        # Student optimizer
        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
    def init_losses(self):
        """Initialize loss functions."""
        # Hard loss (task loss)
        self.criterion_task = nn.CrossEntropyLoss()
        
        # Soft loss (KD loss)
        self.criterion_kd = SoftTarget(T=self.args.temperature)
        
        # Structure loss (RKD loss) - with adaptive sampling for large graphs
        self.criterion_struct = AdaptiveRKDLoss(max_samples=self.args.max_samples)
        
    def train_teacher(self):
        """Train teacher GCN model."""
        print('\n--- Training Teacher GCN ---')
        config = self.config
        
        optimizer = optim.Adam(
            self.teacher.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.teacher.train()
            optimizer.zero_grad()
            
            output = self.teacher(self.features, self.adj)
            loss = self.criterion_task(output[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            # Validation
            self.teacher.eval()
            with torch.no_grad():
                output = self.teacher(self.features, self.adj)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.teacher.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f'  Early stopping at epoch {epoch + 1}')
                break
        
        # Load best teacher
        self.teacher.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Test teacher
        with torch.no_grad():
            output = self.teacher(self.features, self.adj)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test])
        
        print(f'  Teacher Test Accuracy: {test_acc.item() * 100:.2f}%')
        return test_acc.item() * 100
        
    def train_student_with_distillation(self):
        """Train student MLP with knowledge distillation."""
        print('\n--- Training Student MLP with Distillation ---')
        print(f'  Loss weights: alpha={self.args.alpha}, beta={self.args.beta}, gamma={self.args.gamma}')
        
        config = self.config
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.student.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            student_output = self.student(self.features, self.adj)
            
            with torch.no_grad():
                teacher_output = self.teacher(self.features, self.adj)
            
            # 1. Task Loss (Hard Loss) - only on training nodes
            loss_task = self.criterion_task(
                student_output[self.idx_train],
                self.labels[self.idx_train]
            )
            
            # 2. KD Loss (Soft Loss) - on all nodes or training nodes
            loss_kd = self.criterion_kd(student_output, teacher_output)
            
            # 3. Structure Loss (RKD) - on training nodes
            # Training nodes have labels and are most relevant for structure learning
            loss_struct = self.criterion_struct(
                student_output, teacher_output,
                mask=self.idx_train
            )
            
            # Combined loss
            loss = (self.args.alpha * loss_task + 
                    self.args.beta * loss_kd + 
                    self.args.gamma * loss_struct)
            
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.student.eval()
            with torch.no_grad():
                output = self.student(self.features, self.adj)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                print(f'  Epoch {epoch + 1}: loss={loss.item():.4f}, '
                      f'task={loss_task.item():.4f}, kd={loss_kd.item():.4f}, '
                      f'struct={loss_struct.item():.4f}, val_acc={val_acc.item() * 100:.2f}%')
            
            if patience_counter >= config['patience']:
                print(f'  Early stopping at epoch {epoch + 1}')
                break
        
        # Load best student
        self.student.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        # Test student
        self.student.eval()
        with torch.no_grad():
            output = self.student(self.features, self.adj)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test])
        
        print(f'  Student Test Accuracy: {test_acc.item() * 100:.2f}%')
        return test_acc.item() * 100


def run_distillation(args):
    """Run distillation experiment with multiple seeds."""
    config = DATASET_CONFIGS.get(args.data, DATASET_CONFIGS['cora']).copy()
    
    teacher_accs = []
    student_accs = []
    
    for seed in range(args.num_runs):
        print(f'\n{"="*60}')
        print(f'Run {seed + 1}/{args.num_runs} (seed={seed})')
        print(f'{"="*60}')
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        
        trainer = DistillationTrainer(args, config)
        
        # Train teacher
        teacher_acc = trainer.train_teacher()
        teacher_accs.append(teacher_acc)
        
        # Train student with distillation
        student_acc = trainer.train_student_with_distillation()
        student_accs.append(student_acc)
    
    # Summary
    print(f'\n{"="*60}')
    print('DISTILLATION RESULTS SUMMARY')
    print(f'{"="*60}')
    print(f'Dataset: {args.data}')
    print(f'Loss weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}')
    print(f'Temperature: {args.temperature}')
    print(f'-' * 60)
    print(f'Teacher GCN: {np.mean(teacher_accs):.2f} ± {np.std(teacher_accs):.2f}%')
    print(f'Student MLP (Distilled): {np.mean(student_accs):.2f} ± {np.std(student_accs):.2f}%')
    print(f'{"="*60}')
    
    return teacher_accs, student_accs


def main():
    parser = argparse.ArgumentParser(description='Structure-Aware Knowledge Distillation')
    
    # Dataset
    parser.add_argument('--data', type=str, default='cora',
                        help='Dataset: cora, citeseer, pubmed, amazon-computers, amazon-photo')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--num_runs', type=int, default=1)
    
    # Loss weights (Trinity)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for task loss (CrossEntropy)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for KD loss (Soft Target)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Weight for structure loss (RKD)')
    
    # KD parameters
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for soft target')
    
    # Memory management
    parser.add_argument('--max_samples', type=int, default=2048,
                        help='Max samples for RKD computation (memory efficiency)')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    print(f'Using device: {"CUDA" if args.cuda else "CPU"}')
    
    run_distillation(args)


if __name__ == '__main__':
    main()
