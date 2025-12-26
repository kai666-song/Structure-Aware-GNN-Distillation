"""
Main Experiment Script for Gated AFD-KD
=======================================

This is the primary script to reproduce all main results in the paper.

Usage:
    python scripts/run_main_experiments.py --experiment all
    python scripts/run_main_experiments.py --experiment filtered_datasets
    python scripts/run_main_experiments.py --experiment mechanism_analysis
    python scripts/run_main_experiments.py --experiment homophily_breakdown
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kd_losses.adaptive_kd import GatedAFDLoss, AFDLoss


# ============================================================================
# Data Loading
# ============================================================================

def load_filtered_data(dataset, split_idx, device):
    """Load filtered dataset from Platonov et al. (ICLR 2023)."""
    data_dir = '../heterophilous-graphs/data'
    filename = f'{dataset}_filtered.npz'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}\n"
                               f"Please clone: git clone https://github.com/yandex-research/heterophilous-graphs")
    
    data = np.load(filepath)
    
    features = torch.FloatTensor(data['node_features'])
    labels = torch.LongTensor(data['node_labels'])
    edges = data['edges']
    
    train_mask = torch.BoolTensor(data['train_masks'][split_idx])
    val_mask = torch.BoolTensor(data['val_masks'][split_idx])
    test_mask = torch.BoolTensor(data['test_masks'][split_idx])
    
    num_nodes = features.shape[0]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    
    adj_raw = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    adj_raw = adj_raw.tocsr()
    
    adj = adj_raw + sp.eye(num_nodes)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    adj_coo = adj_norm.tocoo()
    indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
    values = torch.FloatTensor(adj_coo.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(adj_coo.shape))
    
    row_sum = features.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum
    
    num_classes = len(torch.unique(labels))
    
    return {
        'features': features.to(device),
        'labels': labels.to(device),
        'adj': adj_tensor.to(device),
        'adj_raw': adj_raw,
        'train_mask': train_mask.to(device),
        'val_mask': val_mask.to(device),
        'test_mask': test_mask.to(device),
        'num_classes': num_classes,
        'num_features': features.shape[1],
        'num_nodes': num_nodes,
    }


# ============================================================================
# Models
# ============================================================================

class SimpleMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(nhid)
        self.norm2 = nn.LayerNorm(nhid)
    
    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.fc3(x)


class GCNWithSkip(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(nfeat, nhid)
        self.input_norm = nn.LayerNorm(nhid)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(nhid, nhid))
            self.norms.append(nn.LayerNorm(nhid))
        self.output_proj = nn.Linear(nhid, nclass)
        self.dropout = dropout
    
    def forward(self, x, adj):
        x = F.gelu(self.input_norm(self.input_proj(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        for layer, norm in zip(self.layers, self.norms):
            h = torch.spmm(adj, x)
            h = F.gelu(norm(layer(h)))
            h = F.dropout(h, self.dropout, training=self.training)
            x = x + h
        return self.output_proj(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_teacher(data, device, epochs=500):
    model = GCNWithSkip(data['num_features'], 256, data['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    best_val_acc, best_test_acc, best_logits = 0, 0, None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data['features'], data['adj'])
        loss = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data['features'], data['adj'])
            preds = logits.argmax(dim=1)
            val_acc = (preds[data['val_mask']] == data['labels'][data['val_mask']]).float().mean().item()
            test_acc = (preds[data['test_mask']] == data['labels'][data['test_mask']]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_logits = logits.detach()
                patience = 0
            else:
                patience += 1
        if patience >= 100:
            break
    
    return best_test_acc, best_logits


def train_simple_kd(data, teacher_logits, device, T=4.0, epochs=500):
    model = SimpleMLP(data['num_features'], 256, data['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data['features'])
        loss_ce = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        loss = loss_ce + loss_kd
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data['features'])
            preds = logits.argmax(dim=1)
            val_acc = (preds[data['val_mask']] == data['labels'][data['val_mask']]).float().mean().item()
            test_acc = (preds[data['test_mask']] == data['labels'][data['test_mask']]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
    return best_test_acc


def train_gated_afd(data, teacher_logits, device, K=5, gate_threshold=0.3, epochs=500):
    model = SimpleMLP(data['num_features'], 256, data['num_classes']).to(device)
    gated_loss = GatedAFDLoss(
        adj=data['adj_raw'], K=K, temperature=4.0, lambda_kd=1.0,
        gate_init_threshold=gate_threshold, gate_sharpness=5.0,
        learnable_gate=True, device=device
    )
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': gated_loss.parameters(), 'lr': 0.005}
    ])
    
    best_val_acc, best_test_acc = 0, 0
    
    for epoch in range(epochs):
        model.train()
        gated_loss.train()
        optimizer.zero_grad()
        logits = model(data['features'])
        loss, _ = gated_loss(logits, teacher_logits, data['labels'], data['train_mask'])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data['features'])
            preds = logits.argmax(dim=1)
            val_acc = (preds[data['val_mask']] == data['labels'][data['val_mask']]).float().mean().item()
            test_acc = (preds[data['test_mask']] == data['labels'][data['test_mask']]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
    return best_test_acc


# ============================================================================
# Main Experiments
# ============================================================================

def run_filtered_datasets(device, num_splits=5):
    """Main experiment on filtered datasets."""
    print("\n" + "="*80)
    print("MAIN EXPERIMENT: FILTERED DATASETS")
    print("="*80)
    
    results = {}
    
    for dataset in ['squirrel', 'chameleon']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}_filtered")
        print(f"{'='*60}")
        
        results[dataset] = {'teacher': [], 'simple_kd': [], 'gated_afd': []}
        
        for split_idx in range(num_splits):
            print(f"\n  Split {split_idx}:")
            data = load_filtered_data(dataset, split_idx, device)
            
            teacher_acc, teacher_logits = train_teacher(data, device)
            results[dataset]['teacher'].append(teacher_acc * 100)
            print(f"    Teacher: {teacher_acc*100:.2f}%")
            
            simple_acc = train_simple_kd(data, teacher_logits, device)
            results[dataset]['simple_kd'].append(simple_acc * 100)
            print(f"    Simple KD: {simple_acc*100:.2f}%")
            
            gated_acc = train_gated_afd(data, teacher_logits, device)
            results[dataset]['gated_afd'].append(gated_acc * 100)
            print(f"    Gated AFD: {gated_acc*100:.2f}%")
        
        # Summary
        print(f"\n  Summary:")
        for method in ['teacher', 'simple_kd', 'gated_afd']:
            accs = results[dataset][method]
            print(f"    {method}: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'filtered_datasets'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_splits', type=int, default=5)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.experiment in ['all', 'filtered_datasets']:
        results = run_filtered_datasets(device, args.num_splits)
        
        # Save results
        os.makedirs('../results/main', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'../results/main/results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
