"""
Phase 4 Task 1: Save Teacher's Penultimate Layer Features
==========================================================

Extract and save the 64-dim features from GloGNN++ (before the final classifier).
These features will be used for Feature-based RKD.

Usage:
    python baselines/save_teacher_features.py --dataset actor --device cuda
    python baselines/save_teacher_features.py --all --device cuda
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader_v2 import load_data_with_glognn_splits
from configs.experiment_config import GLOGNN_CONFIGS, NUM_SPLITS


class MLP_NORM_WithFeatures(nn.Module):
    """
    GloGNN++ Model modified to return penultimate features.
    """
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, 
                 gamma, delta, norm_func_id, norm_layers, orders, 
                 orders_func_id, device='cpu'):
        super().__init__()
        
        self.fc1 = nn.Linear(nfeat, nhid).double()
        self.fc2 = nn.Linear(nhid, nclass).double()
        self.fc3 = nn.Linear(nnodes, nhid).double()
        self.nclass = nclass
        self.nhid = nhid
        self.dropout = dropout
        self.alpha = torch.tensor(alpha, dtype=torch.float64)
        self.beta = torch.tensor(beta, dtype=torch.float64)
        self.gamma = torch.tensor(gamma, dtype=torch.float64)
        self.delta = torch.tensor(delta, dtype=torch.float64)
        self.norm_layers = norm_layers
        self.orders = orders
        self.class_eye = torch.eye(nclass, dtype=torch.float64)
        self.nodes_eye = torch.eye(nnodes, dtype=torch.float64)
        self.device = device

        self.orders_weight = Parameter(
            torch.ones(orders, 1, dtype=torch.float64) / orders, requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.zeros(nclass, orders, dtype=torch.float64), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.zeros(orders, orders, dtype=torch.float64), requires_grad=True
        )
        self.diag_weight = Parameter(
            torch.ones(nclass, 1, dtype=torch.float64) / nclass, requires_grad=True
        )
        
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if norm_func_id == 1:
            self.norm = self._norm_func1
        else:
            self.norm = self._norm_func2

        if orders_func_id == 1:
            self.order_func = self._order_func1
        elif orders_func_id == 2:
            self.order_func = self._order_func2
        else:
            self.order_func = self._order_func3

    def to(self, device):
        super().to(device)
        self.device = device
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.gamma = self.gamma.to(device)
        self.delta = self.delta.to(device)
        self.class_eye = self.class_eye.to(device)
        self.nodes_eye = self.nodes_eye.to(device)
        return self

    def forward(self, x, adj, return_features=False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input features
            adj: Adjacency matrix
            return_features: If True, return (logits, penultimate_features)
        """
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)
        xA = self.fc3(adj)
        
        # This is the penultimate layer feature (nhid dimensional)
        penultimate = F.relu(self.delta * xX + (1 - self.delta) * xA)
        
        x = F.dropout(penultimate, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        
        if return_features:
            return x, penultimate
        return x

    def _norm_func1(self, x, h0, adj):
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

    def _norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def _order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def _order_func2(self, x, res, adj):
        tmp_orders = torch.spmm(adj, res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def _order_func3(self, x, res, adj):
        orders_para = torch.mm(
            torch.relu(torch.mm(x, self.orders_weight_matrix)),
            self.orders_weight_matrix2
        )
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train_and_save_features(dataset, split_idx, config, device):
    """Train GloGNN++ and save both logits and penultimate features."""
    
    data = load_data_with_glognn_splits(dataset, split_idx)
    
    features = data['features'].to(torch.float64).to(device)
    adj = data['adj'].to(torch.float64).to(device)
    labels = data['labels'].to(device)
    idx_train = data['idx_train'].to(device)
    idx_val = data['idx_val'].to(device)
    idx_test = data['idx_test'].to(device)
    
    model = MLP_NORM_WithFeatures(
        nnodes=data['num_nodes'],
        nfeat=data['num_features'],
        nhid=config['hidden'],
        nclass=data['num_classes'],
        dropout=config['dropout'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        delta=config['delta'],
        norm_func_id=config['norm_func_id'],
        norm_layers=config['norm_layers'],
        orders=config['orders'],
        orders_func_id=config['orders_func_id'],
        device=device,
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    best_val_acc = 0
    best_test_acc = 0
    best_logits = None
    best_features = None
    cost_val = []
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        output = model(features, adj)
        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            output, penultimate = model(features, adj, return_features=True)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_test = accuracy(output[idx_test], labels[idx_test])
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_test_acc = acc_test
            best_logits = output.detach().cpu().float()
            best_features = penultimate.detach().cpu().float()
        
        cost_val.append(loss_val.item())
        if epoch > config['early_stopping']:
            if cost_val[-1] > np.mean(cost_val[-(config['early_stopping']+1):-1]):
                break
    
    return best_test_acc.item(), best_logits, best_features


def save_teacher_features(dataset, device='cuda'):
    """Save teacher features for all splits."""
    
    config = GLOGNN_CONFIGS[dataset]
    
    print(f"\n{'='*70}")
    print(f"Extracting Teacher Features for {dataset.upper()}")
    print(f"Hidden dim: {config['hidden']}")
    print(f"{'='*70}")
    
    save_dir = os.path.join('checkpoints', f'glognn_teacher_{dataset}')
    os.makedirs(save_dir, exist_ok=True)
    
    all_accs = []
    
    for split_idx in range(NUM_SPLITS):
        print(f"\nSplit {split_idx}...", end=" ")
        
        test_acc, logits, features = train_and_save_features(
            dataset, split_idx, config, device
        )
        all_accs.append(test_acc * 100)
        
        # Save to split directory
        split_dir = os.path.join(save_dir, f'split_{split_idx}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Save logits (overwrite existing)
        torch.save({
            'logits': logits,
            'config': config,
            'test_acc': test_acc * 100,
        }, os.path.join(split_dir, 'teacher_logits.pt'))
        
        # Save features (NEW!)
        torch.save({
            'features': features,
            'feature_dim': features.shape[1],
            'num_nodes': features.shape[0],
        }, os.path.join(split_dir, 'teacher_features.pt'))
        
        print(f"Test: {test_acc*100:.2f}%, Features: {features.shape}")
    
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    
    print(f"\n{'='*70}")
    print(f"TEACHER FEATURE EXTRACTION COMPLETE")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Feature dim: {config['hidden']}")
    print(f"Saved to: {save_dir}")
    print(f"{'='*70}")
    
    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'chameleon', 'squirrel'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    datasets = ['actor', 'squirrel'] if args.all else [args.dataset]
    
    for dataset in datasets:
        save_teacher_features(dataset, device)


if __name__ == '__main__':
    main()
