"""
Benchmark script for Step 1: Run baseline experiments
- Teacher GCN: with CrossEntropy Loss
- Student MLP: with CrossEntropy Loss (no KD)

Datasets: Cora, Citeseer, PubMed, Amazon-Computers, Amazon-Photo
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models import GCN, MLP
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj


# Dataset-specific hyperparameters
DATASET_CONFIGS = {
    'cora': {
        'lr': 0.01,
        'weight_decay': 5e-4,
        'hidden': 64,
        'dropout': 0.5,
        'epochs': 200,
        'patience': 50,
    },
    'citeseer': {
        'lr': 0.01,
        'weight_decay': 5e-4,
        'hidden': 64,
        'dropout': 0.5,
        'epochs': 200,
        'patience': 50,
    },
    'pubmed': {
        'lr': 0.01,
        'weight_decay': 5e-4,
        'hidden': 64,
        'dropout': 0.5,
        'epochs': 200,
        'patience': 50,
    },
    'amazon-computers': {
        'lr': 0.01,
        'weight_decay': 0,  # Critical: no weight decay for Amazon
        'hidden': 256,
        'dropout': 0.5,
        'epochs': 500,
        'patience': 100,
    },
    'amazon-photo': {
        'lr': 0.01,
        'weight_decay': 0,  # Critical: no weight decay for Amazon
        'hidden': 256,
        'dropout': 0.5,
        'epochs': 500,
        'patience': 100,
    },
}


class BenchmarkTrainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
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
        print(f'  Config: hidden={self.config["hidden"]}, lr={self.config["lr"]}, '
              f'wd={self.config["weight_decay"]}, epochs={self.config["epochs"]}')

        # Preprocess features and adjacency
        self.features = preprocess_features(self.features)
        self.supports = preprocess_adj(self.adj)

        # Convert to torch tensors
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

        # Store dimensions
        self.nfeat = self.features.shape[1]
        self.nclass = self.labels.max().item() + 1

    def train_model(self, model_type='gcn'):
        """Train a single model (GCN or MLP) with early stopping."""
        config = self.config

        # Initialize model
        if model_type == 'gcn':
            model = GCN(nfeat=self.nfeat, nhid=config['hidden'],
                        nclass=self.nclass, dropout=config['dropout'])
        else:
            model = MLP(nfeat=self.nfeat, nhid=config['hidden'],
                        nclass=self.nclass, dropout=config['dropout'])

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=config['weight_decay'])

        best_val_acc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(config['epochs']):
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
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config['patience']:
                break

        # Load best model and test
        model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            output = model(self.features, self.adj)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test])

        return test_acc.item() * 100  # Return percentage


def run_benchmark(args, config):
    """Run benchmark for a single dataset with multiple seeds."""
    results = {'gcn': [], 'mlp': []}

    for seed in range(args.num_runs):
        print(f'\n--- Run {seed + 1}/{args.num_runs} (seed={seed}) ---')

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)

        trainer = BenchmarkTrainer(args, config)

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
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs with different seeds')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Run on all datasets')
    # Override options
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--hidden', type=int, default=None, help='Override hidden dim')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Override weight decay')
    parser.add_argument('--dropout', type=float, default=None, help='Override dropout')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(f'Using device: {"CUDA" if args.cuda else "CPU"}')

    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
    else:
        datasets = [args.data]

    all_results = {}

    for dataset in datasets:
        print(f'\n{"=" * 60}')
        print(f'DATASET: {dataset.upper()}')
        print(f'{"=" * 60}')

        args.data = dataset

        # Get dataset-specific config
        config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['cora']).copy()

        # Apply overrides if specified
        if args.epochs is not None:
            config['epochs'] = args.epochs
        if args.hidden is not None:
            config['hidden'] = args.hidden
        if args.lr is not None:
            config['lr'] = args.lr
        if args.weight_decay is not None:
            config['weight_decay'] = args.weight_decay
        if args.dropout is not None:
            config['dropout'] = args.dropout

        results = run_benchmark(args, config)
        all_results[dataset] = {'results': results, 'config': config}

    # Print summary table
    print('\n' + '=' * 80)
    print('BENCHMARK RESULTS SUMMARY (Test Accuracy %)')
    print('=' * 80)
    print(f'{"Dataset":<20} {"GCN (Teacher)":<25} {"MLP (Student)":<25}')
    print('-' * 80)

    for dataset, data in all_results.items():
        results = data['results']
        gcn_mean = np.mean(results['gcn'])
        gcn_std = np.std(results['gcn'])
        mlp_mean = np.mean(results['mlp'])
        mlp_std = np.std(results['mlp'])
        print(f'{dataset:<20} {gcn_mean:.2f} ± {gcn_std:.2f}{"":>10} {mlp_mean:.2f} ± {mlp_std:.2f}')

    print('=' * 80)

    # Print configs used
    print('\nConfigs used:')
    for dataset, data in all_results.items():
        c = data['config']
        print(f'  {dataset}: hidden={c["hidden"]}, lr={c["lr"]}, wd={c["weight_decay"]}, '
              f'epochs={c["epochs"]}, patience={c["patience"]}')


if __name__ == '__main__':
    main()
