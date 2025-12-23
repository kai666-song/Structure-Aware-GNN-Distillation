"""
Phase 1: Establish the True Bar - Run GloGNN++ on Actor and Squirrel
Using Geom-GCN standard splits (10 folds) for fair comparison

This script runs GloGNN++ (MLP_NORM model) on heterophilic datasets
to establish the baseline performance we need to beat.

Target Performance:
- Actor: > 36.5%
- Squirrel: > 38%
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter
import math
import json
from collections import defaultdict
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MLP_NORM(nn.Module):
    """GloGNN++ Model - MLP with Global Homophily Normalization"""
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta, 
                 norm_func_id, norm_layers, orders, orders_func_id, cuda):
        super(MLP_NORM, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nnodes, nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)
        self.delta = torch.tensor(delta)
        self.norm_layers = norm_layers
        self.orders = orders
        self.class_eye = torch.eye(nclass)
        self.nodes_eye = torch.eye(nnodes)

        if cuda:
            self.orders_weight = Parameter(
                torch.ones(orders, 1) / orders, requires_grad=True
            ).to('cuda')
            self.orders_weight_matrix = Parameter(
                torch.DoubleTensor(nclass, orders), requires_grad=True
            ).to('cuda')
            self.orders_weight_matrix2 = Parameter(
                torch.DoubleTensor(orders, orders), requires_grad=True
            ).to('cuda')
            self.diag_weight = Parameter(
                torch.ones(nclass, 1) / nclass, requires_grad=True
            ).to('cuda')
            self.alpha = self.alpha.cuda()
            self.beta = self.beta.cuda()
            self.gamma = self.gamma.cuda()
            self.delta = self.delta.cuda()
            self.class_eye = self.class_eye.cuda()
            self.nodes_eye = self.nodes_eye.cuda()
        else:
            self.orders_weight = Parameter(
                torch.ones(orders, 1) / orders, requires_grad=True
            )
            self.orders_weight_matrix = Parameter(
                torch.DoubleTensor(nclass, orders), requires_grad=True
            )
            self.orders_weight_matrix2 = Parameter(
                torch.DoubleTensor(orders, orders), requires_grad=True
            )
            self.diag_weight = Parameter(
                torch.ones(nclass, 1) / nclass, requires_grad=True
            )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def forward(self, x, adj):
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)
        xA = self.fc3(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        return F.log_softmax(x, dim=1)

    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        tmp_orders = torch.spmm(adj, res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders


def load_geom_gcn_data(dataset_name, split_idx=0):
    """
    Load heterophilic dataset with Geom-GCN standard splits.
    Uses PyTorch Geometric's built-in datasets.
    """
    from torch_geometric.datasets import Actor, WikipediaNetwork
    import scipy.sparse as sp
    
    if dataset_name == 'actor':
        dataset = Actor(root='./data/actor')
    elif dataset_name == 'squirrel':
        dataset = WikipediaNetwork(root='./data/heterophilic', name='Squirrel', 
                                   geom_gcn_preprocess=True)
    elif dataset_name == 'chameleon':
        dataset = WikipediaNetwork(root='./data/heterophilic', name='Chameleon',
                                   geom_gcn_preprocess=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data = dataset[0]
    
    # Get features and labels
    features = data.x.numpy()
    labels = data.y.numpy()
    num_nodes = features.shape[0]
    
    # Build adjacency matrix
    edge_index = data.edge_index.numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    
    # Get Geom-GCN splits (10 folds available)
    if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        split_idx = split_idx % data.train_mask.shape[1]
        train_mask = data.train_mask[:, split_idx].numpy()
        val_mask = data.val_mask[:, split_idx].numpy()
        test_mask = data.test_mask[:, split_idx].numpy()
    else:
        # Fallback for single split
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()
    
    idx_train = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    idx_test = np.where(test_mask)[0]
    
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    import scipy.sparse as sp
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    import scipy.sparse as sp
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def run_glognn_experiment(dataset_name, config, num_splits=10, use_cuda=False):
    """
    Run GloGNN++ on a dataset with multiple splits.
    
    Args:
        dataset_name: 'actor' or 'squirrel'
        config: hyperparameter configuration dict
        num_splits: number of Geom-GCN splits to use (default 10)
        use_cuda: whether to use GPU
    
    Returns:
        mean_acc, std_acc, all_results
    """
    import scipy.sparse as sp
    
    results = []
    
    for split_idx in range(num_splits):
        print(f"\n--- Split {split_idx} ---")
        
        # Load data with specific split
        adj, features, labels, idx_train, idx_val, idx_test = load_geom_gcn_data(
            dataset_name, split_idx
        )
        
        # Normalize adjacency and features
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        features = normalize(sp.csr_matrix(features))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
        # Convert to float64 for GloGNN
        features = features.to(torch.float64)
        adj = adj.to(torch.float64)
        
        if use_cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
        
        # Initialize model
        model = MLP_NORM(
            nnodes=adj.shape[0],
            nfeat=features.shape[1],
            nhid=config['hidden'],
            nclass=int(labels.max().item()) + 1,
            dropout=config['dropout'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            delta=config['delta'],
            norm_func_id=config['norm_func_id'],
            norm_layers=config['norm_layers'],
            orders=config['orders'],
            orders_func_id=config['orders_func_id'],
            cuda=use_cuda
        )
        
        if use_cuda:
            model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'])
        
        # Training
        best_val_acc = 0
        best_test_acc = 0
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            output = model(features, adj)
            val_acc = accuracy(output[idx_val], labels[idx_val]).item()
            test_acc = accuracy(output[idx_test], labels[idx_test]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['early_stopping']:
                break
        
        print(f"Split {split_idx}: Test Acc = {best_test_acc*100:.2f}%")
        results.append(best_test_acc)
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    
    return mean_acc, std_acc, results


# GloGNN++ hyperparameters from the paper
GLOGNN_CONFIGS = {
    'actor': {  # film in GloGNN paper
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.0,
        'weight_decay': 0.001,
        'alpha': 0.0,
        'beta': 15000.0,
        'gamma': 0.2,
        'delta': 1.0,
        'norm_layers': 2,
        'orders': 4,
        'orders_func_id': 2,
        'norm_func_id': 2,
        'epochs': 2000,
        'early_stopping': 40
    },
    'squirrel': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.8,
        'weight_decay': 0.0,
        'alpha': 0.0,
        'beta': 1.0,
        'gamma': 0.0,
        'delta': 0.0,
        'norm_layers': 3,
        'orders': 2,
        'orders_func_id': 2,
        'norm_func_id': 2,
        'epochs': 2000,
        'early_stopping': 200
    },
    'chameleon': {
        'hidden': 64,
        'lr': 0.01,
        'dropout': 0.4,
        'weight_decay': 0.0001,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 0.4,
        'delta': 0.0,
        'norm_layers': 3,
        'orders': 2,
        'orders_func_id': 2,
        'norm_func_id': 2,
        'epochs': 2000,
        'early_stopping': 300
    }
}


def main():
    parser = argparse.ArgumentParser(description='Run GloGNN++ Baseline')
    parser.add_argument('--dataset', type=str, default='actor', 
                       choices=['actor', 'squirrel', 'chameleon'])
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()
    
    print("="*60)
    print(f"Running GloGNN++ Baseline on {args.dataset.upper()}")
    print(f"Using Geom-GCN standard splits ({args.num_splits} folds)")
    print("="*60)
    
    config = GLOGNN_CONFIGS[args.dataset]
    print(f"\nConfig: {config}")
    
    mean_acc, std_acc, all_results = run_glognn_experiment(
        args.dataset, config, args.num_splits, args.cuda
    )
    
    print("\n" + "="*60)
    print(f"RESULTS: {args.dataset.upper()}")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"All splits: {[f'{r*100:.2f}' for r in all_results]}")
    print("="*60)
    
    # Save results
    if args.save_results:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'glognn_baseline_{args.dataset}.json')
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': args.dataset,
                'model': 'GloGNN++',
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'all_results': all_results,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
