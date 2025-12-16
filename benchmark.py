"""
Benchmark script for Step 1: Run baseline experiments
- Teacher GCN: 200 epochs with CrossEntropy Loss
- Student MLP: 200 epochs with CrossEntropy Loss (no KD)

Datasets: Cora, Citeseer, PubMed, Amazon-Computers, Amazon-Photo
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import defaultdict

from models import GCN, MLP
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj


class BenchmarkTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.load_data()
        
    def load_data(self):
        """Load and preprocess dataset."""
        self.adj, self.features, self.labels, self.y_train, self.y_val, self.y_test, \
            self.train_mask, self.val_mask, self.test_mask, \
            self.idx_train, self.idx_val, self.idx_test = load_data_new(self.args.data)
        
        print(f'Dataset: {self.args.data}')
        print(f'  Nodes: {self.features.shape[0]}')
        print(f'  Features: {self.features.shape[1]}')
        print(f'  Classes: {self.labels.max().item() + 1}')
        print(f'  Train/Val/Test: {len(self.idx_train)}/{len(self.idx_val)}/{len(self.idx_test)}')
        
        # Preprocess features and adjacency
        self.features = preprocess_features(self.features)
        self.supports = preprocess_adj(self.adj)
        
        # Convert to torch tensors
        i = torch.from_numpy(self.features[0]).long().to(self.device)
        v = torch.from_numpy(self.features[1]).to(self.device)
        self.features = torch.sparse.FloatTensor(i.t(), v, self.features[2]).to(self.device)
        
        i = torch.from_numpy(self.supports[0]).long().to(self.device)
        v = torch.from_numpy(self.supports[1]).to(self.device)
        self.adj = torch.sparse.FloatTensor(i.t(), v, self.supports[2]).float().to(self.device)
        
        self.labels = self.labels.to(self.device)
        self.idx_train = self.idx_train.to(self.device)
        self.idx_val = self.idx_val.to(self.device)
        self.idx_test = self.idx_test.to(self.device)
        
        # Store dimensions
        self.nfeat = self.features.shape[1]
        self.nclass = self.labels.max().item() + 1
        
    def train_model(self, model_type='gcn'):
        """Train a single model (GCN or MLP)."""
        # Initialize model
        if model_type == 'gcn':
            model = GCN(nfeat=self.nfeat, nhid=self.args.hidden, 
                       nclass=self.nclass, dropout=self.args.dropout)
        else:
            model = MLP(nfeat=self.nfeat, nhid=self.args.hidden,
                       nclass=self.nclass, dropout=self.args.dropout)
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, 
                              weight_decay=self.args.weight_decay)
        
        best_val_acc = 0
        best_state = None
        
        for epoch in range(self.args.epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            output = model(self.features, self.adj)
            loss = criterion(output[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(self.features, self.adj)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val])
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
        
        # Load best model and test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            output = model(self.features, self.adj)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test])
        
        return test_acc.item() * 100  # Return percentage


def run_benchmark(args):
    """Run benchmark for a single dataset with multiple seeds."""
    results = {'gcn': [], 'mlp': []}
    
    for seed in range(args.num_runs):
        print(f'\n--- Run {seed + 1}/{args.num_runs} (seed={seed}) ---')
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        
        trainer = BenchmarkTrainer(args)
        
        # Train GCN (Teacher)
        gcn_acc = trainer.train_model('gcn')
        results['gcn'].append(gcn_acc)
        print(f'  GCN Test Accuracy: {gcn_acc:.2f}%')
        
        # Train MLP (Student baseline)
        mlp_acc = trainer.train_model('mlp')
        results['mlp'].append(mlp_acc)
        print(f'  MLP Test Accuracy: {mlp_acc:.2f}%')
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora',
                       help='Dataset: cora, citeseer, pubmed, amazon-computers, amazon-photo')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs with different seeds')
    parser.add_argument('--all', action='store_true', default=False,
                       help='Run on all datasets')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    print(f'Using device: {"CUDA" if args.cuda else "CPU"}')
    
    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
    else:
        datasets = [args.data]
    
    all_results = {}
    
    for dataset in datasets:
        print(f'\n{"="*60}')
        print(f'DATASET: {dataset.upper()}')
        print(f'{"="*60}')
        
        args.data = dataset
        results = run_benchmark(args)
        all_results[dataset] = results
    
    # Print summary table
    print('\n' + '='*80)
    print('BENCHMARK RESULTS SUMMARY (Test Accuracy %)')
    print('='*80)
    print(f'{"Dataset":<20} {"GCN (Teacher)":<25} {"MLP (Student)":<25}')
    print('-'*80)
    
    for dataset, results in all_results.items():
        gcn_mean = np.mean(results['gcn'])
        gcn_std = np.std(results['gcn'])
        mlp_mean = np.mean(results['mlp'])
        mlp_std = np.std(results['mlp'])
        print(f'{dataset:<20} {gcn_mean:.2f} ± {gcn_std:.2f}{"":>10} {mlp_mean:.2f} ± {mlp_std:.2f}')
    
    print('='*80)
    print(f'\nSettings: epochs={args.epochs}, hidden={args.hidden}, '
          f'lr={args.lr}, dropout={args.dropout}, runs={args.num_runs}')


if __name__ == '__main__':
    main()
