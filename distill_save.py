"""
Structure-Aware Knowledge Distillation with Model Saving

Saves:
1. Model checkpoints (teacher & student)
2. Training history (losses, accuracies per epoch)
3. Final results in JSON format for plotting
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
    'cora': {'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5, 'hidden': 64, 'dropout': 0.5, 'epochs': 300, 'patience': 100},
    'citeseer': {'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5, 'hidden': 64, 'dropout': 0.5, 'epochs': 300, 'patience': 100},
    'pubmed': {'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5, 'hidden': 64, 'dropout': 0.5, 'epochs': 300, 'patience': 100},
    'amazon-computers': {'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0, 'hidden': 256, 'dropout': 0.5, 'epochs': 500, 'patience': 150},
    'amazon-photo': {'lr': 0.01, 'wd_teacher': 0, 'wd_student': 0, 'hidden': 256, 'dropout': 0.5, 'epochs': 500, 'patience': 150},
}


class DistillationTrainerWithSave:
    def __init__(self, args, config, seed):
        self.args = args
        self.config = config
        self.seed = seed
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        # Create save directories
        self.save_dir = f'checkpoints/{args.data}'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'teacher': {'train_loss': [], 'val_acc': [], 'test_acc': None},
            'student': {'train_loss': [], 'task_loss': [], 'kd_loss': [], 'struct_loss': [], 
                       'val_acc': [], 'test_acc': None}
        }
        
        self.load_data()
        self.init_models()
        self.init_losses()
        
    def load_data(self):
        self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, \
            self.train_mask, self.val_mask, self.test_mask, \
            self.idx_train, self.idx_val, self.idx_test = load_data_new(self.args.data)
        
        self.features = preprocess_features(self.features)
        self.supports = preprocess_adj(self.adj)
        
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
        config = self.config
        self.teacher = GCN(self.nfeat, config['hidden'], self.nclass, config['dropout']).to(self.device)
        self.student = MLPBatchNorm(self.nfeat, config['hidden'], self.nclass, config['dropout']).to(self.device)
        self.optimizer = optim.Adam(self.student.parameters(), lr=config['lr'], weight_decay=config['wd_student'])
        
    def init_losses(self):
        self.criterion_task = nn.CrossEntropyLoss()
        self.criterion_kd = SoftTarget(T=self.args.temperature)
        self.criterion_struct = AdaptiveRKDLoss(max_samples=self.args.max_samples)
        
    def train_teacher(self):
        print(f'\n--- Training Teacher GCN (seed={self.seed}) ---')
        config = self.config
        optimizer = optim.Adam(self.teacher.parameters(), lr=config['lr'], weight_decay=config['wd_teacher'])
        
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
            
            self.teacher.eval()
            with torch.no_grad():
                output = self.teacher(self.features, self.adj)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val]).item()
            
            self.history['teacher']['train_loss'].append(loss.item())
            self.history['teacher']['val_acc'].append(val_acc * 100)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.teacher.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        self.teacher.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            output = self.teacher(self.features, self.adj)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test]).item() * 100
        
        self.history['teacher']['test_acc'] = test_acc
        
        # Save teacher model
        torch.save(best_state, f'{self.save_dir}/teacher_seed{self.seed}.pt')
        print(f'  Teacher Test Accuracy: {test_acc:.2f}%')
        return test_acc
        
    def train_student_with_distillation(self):
        print(f'\n--- Training Student MLP with Distillation (seed={self.seed}) ---')
        config = self.config
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.student.train()
            self.optimizer.zero_grad()
            
            student_output = self.student(self.features, self.adj)
            with torch.no_grad():
                teacher_output = self.teacher(self.features, self.adj)
            
            loss_task = self.criterion_task(student_output[self.idx_train], self.labels[self.idx_train])
            loss_kd = self.criterion_kd(student_output, teacher_output)
            loss_struct = self.criterion_struct(student_output, teacher_output, mask=self.idx_train)
            
            loss = self.args.alpha * loss_task + self.args.beta * loss_kd + self.args.gamma * loss_struct
            loss.backward()
            self.optimizer.step()
            
            self.student.eval()
            with torch.no_grad():
                output = self.student(self.features, self.adj)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val]).item()
            
            # Record history
            self.history['student']['train_loss'].append(loss.item())
            self.history['student']['task_loss'].append(loss_task.item())
            self.history['student']['kd_loss'].append(loss_kd.item())
            self.history['student']['struct_loss'].append(loss_struct.item())
            self.history['student']['val_acc'].append(val_acc * 100)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        self.student.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.student.eval()
        
        with torch.no_grad():
            output = self.student(self.features, self.adj)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test]).item() * 100
        
        self.history['student']['test_acc'] = test_acc
        
        # Save student model
        torch.save(best_state, f'{self.save_dir}/student_seed{self.seed}.pt')
        print(f'  Student Test Accuracy: {test_acc:.2f}%')
        return test_acc
    
    def save_history(self):
        """Save training history to JSON."""
        history_file = f'{self.save_dir}/history_seed{self.seed}.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f'  History saved to {history_file}')


def run_distillation_with_save(args):
    config = DATASET_CONFIGS.get(args.data, DATASET_CONFIGS['cora']).copy()
    
    all_results = {
        'dataset': args.data,
        'config': {
            'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma,
            'temperature': args.temperature, **config
        },
        'runs': []
    }
    
    for seed in range(args.num_runs):
        print(f'\n{"="*60}')
        print(f'Run {seed + 1}/{args.num_runs} (seed={seed})')
        print(f'{"="*60}')
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        
        trainer = DistillationTrainerWithSave(args, config, seed)
        teacher_acc = trainer.train_teacher()
        student_acc = trainer.train_student_with_distillation()
        trainer.save_history()
        
        all_results['runs'].append({
            'seed': seed,
            'teacher_acc': teacher_acc,
            'student_acc': student_acc
        })
    
    # Save summary results
    results_file = f'checkpoints/{args.data}/results_summary.json'
    
    teacher_accs = [r['teacher_acc'] for r in all_results['runs']]
    student_accs = [r['student_acc'] for r in all_results['runs']]
    all_results['summary'] = {
        'teacher_mean': np.mean(teacher_accs),
        'teacher_std': np.std(teacher_accs),
        'student_mean': np.mean(student_accs),
        'student_std': np.std(student_accs)
    }
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\n{"="*60}')
    print('RESULTS SUMMARY')
    print(f'{"="*60}')
    print(f'Dataset: {args.data}')
    print(f'Teacher GCN: {np.mean(teacher_accs):.2f} ± {np.std(teacher_accs):.2f}%')
    print(f'Student MLP: {np.mean(student_accs):.2f} ± {np.std(student_accs):.2f}%')
    print(f'Results saved to {results_file}')
    
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--max_samples', type=int, default=2048)
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    print(f'Using device: {"CUDA" if args.cuda else "CPU"}')
    
    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
        for dataset in datasets:
            args.data = dataset
            run_distillation_with_save(args)
    else:
        run_distillation_with_save(args)


if __name__ == '__main__':
    main()
