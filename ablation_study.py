"""
Ablation Study: Prove RKD's Independent Contribution

Compare:
1. MLP Baseline (no distillation)
2. GLNN-style (gamma=0, only soft target KD)
3. Ours (gamma=optimal, with structure loss)

This proves the value of the RKD structure loss.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models import GCN, MLPBatchNorm
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


DATASET_CONFIGS = {
    'cora': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5, 'epochs': 300, 'patience': 100},
    'citeseer': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5, 'epochs': 300, 'patience': 100},
    'pubmed': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5, 'epochs': 300, 'patience': 100},
    'amazon-computers': {'hidden': 256, 'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0, 'epochs': 500, 'patience': 150},
    'amazon-photo': {'hidden': 256, 'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0, 'epochs': 500, 'patience': 150},
}


class AblationTrainer:
    def __init__(self, dataset, config, device):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.load_data()
        
    def load_data(self):
        adj, features, labels, *_, idx_train, idx_val, idx_test = load_data_new(self.dataset)
        
        features = preprocess_features(features)
        supports = preprocess_adj(adj)
        
        i = torch.from_numpy(features[0]).long().to(self.device)
        v = torch.from_numpy(features[1]).to(self.device)
        self.features = torch.sparse_coo_tensor(i.t(), v, features[2]).to(self.device)
        
        i = torch.from_numpy(supports[0]).long().to(self.device)
        v = torch.from_numpy(supports[1]).to(self.device)
        self.adj = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(self.device)
        
        self.labels = labels.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)
        self.idx_test = idx_test.to(self.device)
        
        self.nfeat = self.features.shape[1]
        self.nclass = self.labels.max().item() + 1
        
    def train_teacher(self):
        config = self.config
        teacher = GCN(self.nfeat, config['hidden'], self.nclass, 0.5).to(self.device)
        optimizer = optim.Adam(teacher.parameters(), lr=config['lr'], weight_decay=config['wd_teacher'])
        criterion = nn.CrossEntropyLoss()
        
        best_val, best_state, patience = 0, None, 0
        for epoch in range(config['epochs']):
            teacher.train()
            optimizer.zero_grad()
            out = teacher(self.features, self.adj)
            loss = criterion(out[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            teacher.eval()
            with torch.no_grad():
                val_acc = accuracy(teacher(self.features, self.adj)[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config['patience']:
                    break
        
        teacher.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
            
        with torch.no_grad():
            test_acc = accuracy(teacher(self.features, self.adj)[self.idx_test], self.labels[self.idx_test]).item() * 100
        return teacher, test_acc
    
    def train_mlp_baseline(self):
        """Train MLP without any distillation."""
        config = self.config
        mlp = MLPBatchNorm(self.nfeat, config['hidden'], self.nclass, 0.5).to(self.device)
        optimizer = optim.Adam(mlp.parameters(), lr=config['lr'], weight_decay=config['wd_student'])
        criterion = nn.CrossEntropyLoss()
        
        best_val, best_state, patience = 0, None, 0
        for epoch in range(config['epochs']):
            mlp.train()
            optimizer.zero_grad()
            out = mlp(self.features, self.adj)
            loss = criterion(out[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            mlp.eval()
            with torch.no_grad():
                val_acc = accuracy(mlp(self.features, self.adj)[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config['patience']:
                    break
        
        mlp.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        mlp.eval()
        with torch.no_grad():
            test_acc = accuracy(mlp(self.features, self.adj)[self.idx_test], self.labels[self.idx_test]).item() * 100
        return test_acc
    
    def train_student_distill(self, teacher, gamma=0.0, alpha=1.0, beta=1.0, temperature=4.0):
        """Train student with distillation. gamma=0 is GLNN-style, gamma>0 is Ours."""
        config = self.config
        student = MLPBatchNorm(self.nfeat, config['hidden'], self.nclass, 0.5).to(self.device)
        optimizer = optim.Adam(student.parameters(), lr=config['lr'], weight_decay=config['wd_student'])
        
        criterion_task = nn.CrossEntropyLoss()
        criterion_kd = SoftTarget(T=temperature)
        criterion_struct = AdaptiveRKDLoss(max_samples=2048)
        
        best_val, best_state, patience = 0, None, 0
        for epoch in range(config['epochs']):
            student.train()
            optimizer.zero_grad()
            
            student_out = student(self.features, self.adj)
            with torch.no_grad():
                teacher_out = teacher(self.features, self.adj)
            
            loss_task = criterion_task(student_out[self.idx_train], self.labels[self.idx_train])
            loss_kd = criterion_kd(student_out, teacher_out)
            loss_struct = criterion_struct(student_out, teacher_out, mask=self.idx_train)
            
            loss = alpha * loss_task + beta * loss_kd + gamma * loss_struct
            loss.backward()
            optimizer.step()
            
            student.eval()
            with torch.no_grad():
                val_acc = accuracy(student(self.features, self.adj)[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config['patience']:
                    break
        
        student.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        student.eval()
        with torch.no_grad():
            test_acc = accuracy(student(self.features, self.adj)[self.idx_test], self.labels[self.idx_test]).item() * 100
        return test_acc


def run_ablation(dataset, num_runs=10, device='cuda'):
    config = DATASET_CONFIGS[dataset]
    
    results = {
        'teacher': [], 'baseline': [], 'glnn': [], 'ours': []
    }
    
    for seed in range(num_runs):
        print(f"  Run {seed+1}/{num_runs}", end=" ")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        trainer = AblationTrainer(dataset, config, device)
        
        # Teacher
        teacher, teacher_acc = trainer.train_teacher()
        results['teacher'].append(teacher_acc)
        
        # MLP Baseline
        baseline_acc = trainer.train_mlp_baseline()
        results['baseline'].append(baseline_acc)
        
        # GLNN (gamma=0)
        glnn_acc = trainer.train_student_distill(teacher, gamma=0.0)
        results['glnn'].append(glnn_acc)
        
        # Ours (gamma=1.0)
        ours_acc = trainer.train_student_distill(teacher, gamma=1.0)
        results['ours'].append(ours_acc)
        
        print(f"T={teacher_acc:.1f} B={baseline_acc:.1f} G={glnn_acc:.1f} O={ours_acc:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data', type=str, default='cora')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
    else:
        datasets = [args.data]
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY: {dataset.upper()}")
        print(f"{'='*60}")
        
        results = run_ablation(dataset, args.num_runs, device)
        all_results[dataset] = {
            'teacher': {'mean': np.mean(results['teacher']), 'std': np.std(results['teacher'])},
            'baseline': {'mean': np.mean(results['baseline']), 'std': np.std(results['baseline'])},
            'glnn': {'mean': np.mean(results['glnn']), 'std': np.std(results['glnn'])},
            'ours': {'mean': np.mean(results['ours']), 'std': np.std(results['ours'])},
        }
    
    # Print summary table
    print("\n" + "="*90)
    print("ABLATION STUDY SUMMARY")
    print("="*90)
    print(f"{'Dataset':<18} {'Teacher':<14} {'MLP Baseline':<14} {'GLNN (γ=0)':<14} {'Ours (γ=1)':<14}")
    print("-"*90)
    
    for dataset, r in all_results.items():
        print(f"{dataset:<18} "
              f"{r['teacher']['mean']:.2f}±{r['teacher']['std']:.2f}  "
              f"{r['baseline']['mean']:.2f}±{r['baseline']['std']:.2f}  "
              f"{r['glnn']['mean']:.2f}±{r['glnn']['std']:.2f}  "
              f"{r['ours']['mean']:.2f}±{r['ours']['std']:.2f}")
    
    print("="*90)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_study.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/ablation_study.json")


if __name__ == '__main__':
    main()
