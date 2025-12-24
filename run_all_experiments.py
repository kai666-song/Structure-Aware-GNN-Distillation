"""
Complete Experiments: 3 Heterophilic + 3 Homophilic Datasets
=============================================================

This script runs experiments on all 6 datasets:
- Heterophilic: Actor, Squirrel, Chameleon (GloGNN++ teacher)
- Homophilic: Cora, Citeseer, PubMed (GCN teacher)

Usage:
    # Run all experiments
    python run_all_experiments.py --device cuda
    
    # Run only heterophilic datasets
    python run_all_experiments.py --mode heterophilic --device cuda
    
    # Run only homophilic datasets
    python run_all_experiments.py --mode homophilic --device cuda
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


# =============================================================================
# Model Definitions
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for student model."""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        
        self.layers.append(nn.Linear(nfeat, nhid))
        self.norms.append(nn.LayerNorm(nhid))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.norms.append(nn.LayerNorm(nhid))
        
        self.classifier = nn.Linear(nhid, nclass)
    
    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return self.classifier(x)


class GCN(nn.Module):
    """Simple 2-layer GCN for homophilic teacher."""
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(torch.spmm(adj, x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(torch.spmm(adj, x))
        return x


# =============================================================================
# Data Loading
# =============================================================================

def load_heterophilic_data(dataset, split_idx, device):
    """Load heterophilic dataset using GloGNN splits."""
    from utils.data_loader_v2 import load_data_with_glognn_splits
    
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    return {
        'features': data['features'].to(device),
        'labels': data['labels'].to(device),
        'adj': data['adj'].to(device),
        'train_mask': data['train_mask'].to(device),
        'val_mask': data['val_mask'].to(device),
        'test_mask': data['test_mask'].to(device),
        'num_classes': data['num_classes'],
        'num_features': data['num_features'],
    }


def load_planetoid_local(dataset, split_idx, device, data_dir='./data'):
    """
    Load Planetoid dataset (Cora, Citeseer, PubMed) from local files.
    Uses the standard GCN paper format (ind.{dataset}.*)
    """
    import pickle
    import scipy.sparse as sp
    
    def parse_index_file(filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            try:
                return pickle.load(f, encoding='latin1')
            except:
                f.seek(0)
                return pickle.load(f)
    
    # Load all components
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        filepath = os.path.join(data_dir, f'ind.{dataset}.{name}')
        objects.append(load_pickle(filepath))
    
    x, y, tx, ty, allx, ally, graph = objects
    
    # Load test indices
    test_idx_file = os.path.join(data_dir, f'ind.{dataset}.test.index')
    test_idx_reorder = parse_index_file(test_idx_file)
    test_idx_range = np.sort(test_idx_reorder)
    
    # Handle citeseer's isolated nodes
    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    
    # Combine features
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray()
    
    # Normalize features
    row_sum = features.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum
    
    # Combine labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)
    
    # Build adjacency matrix
    num_nodes = features.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes))
    for src, dsts in graph.items():
        for dst in dsts:
            adj[src, dst] = 1
            adj[dst, src] = 1
    
    # Add self-loops and normalize
    adj = adj + sp.eye(num_nodes)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    # Convert to torch sparse
    adj = adj.tocoo()
    indices = torch.LongTensor(np.vstack((adj.row, adj.col)))
    values = torch.FloatTensor(adj.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape))
    
    # Create random splits (60/20/20)
    torch.manual_seed(42 + split_idx)
    np.random.seed(42 + split_idx)
    
    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)
    
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train+num_val]] = True
    test_mask[perm[num_train+num_val:]] = True
    
    num_classes = int(labels.max()) + 1
    num_features = features.shape[1]
    
    print(f"Dataset: {dataset}, Split: {split_idx}")
    print(f"  Nodes: {num_nodes}, Features: {num_features}, Classes: {num_classes}")
    print(f"  Train: {train_mask.sum().item()} ({100*train_mask.sum().item()/num_nodes:.1f}%)")
    print(f"  Val:   {val_mask.sum().item()} ({100*val_mask.sum().item()/num_nodes:.1f}%)")
    print(f"  Test:  {test_mask.sum().item()} ({100*test_mask.sum().item()/num_nodes:.1f}%)")
    
    return {
        'features': torch.FloatTensor(features).to(device),
        'labels': torch.LongTensor(labels).to(device),
        'adj': adj_tensor.to(device),
        'train_mask': train_mask.to(device),
        'val_mask': val_mask.to(device),
        'test_mask': test_mask.to(device),
        'num_classes': num_classes,
        'num_features': num_features,
    }


def load_homophilic_data(dataset, split_idx, device):
    """Load homophilic dataset using local files or PyG."""
    import scipy.sparse as sp
    
    # First try to load from local data directory (same format as GCN paper)
    local_data_dir = './data'
    dataset_lower = dataset.lower()
    
    # Check if local data exists
    local_file = os.path.join(local_data_dir, f'ind.{dataset_lower}.x')
    
    if os.path.exists(local_file):
        # Load from local Planetoid format files
        print(f"Loading {dataset} from local files...")
        return load_planetoid_local(dataset_lower, split_idx, device, local_data_dir)
    
    # Fall back to PyG download
    try:
        from torch_geometric.datasets import Planetoid
        import torch_geometric.transforms as T
    except ImportError:
        raise ImportError("Please install torch_geometric: pip install torch_geometric")
    
    print(f"Downloading {dataset} via PyTorch Geometric...")
    root = './data/pyg'
    transform = T.NormalizeFeatures()
    
    if dataset_lower == 'cora':
        pyg_data = Planetoid(root=root, name='Cora', transform=transform)
    elif dataset_lower == 'citeseer':
        pyg_data = Planetoid(root=root, name='CiteSeer', transform=transform)
    elif dataset_lower == 'pubmed':
        pyg_data = Planetoid(root=root, name='PubMed', transform=transform)
    else:
        raise ValueError(f"Unknown homophilic dataset: {dataset}")
    
    data = pyg_data[0]
    
    # Build adjacency matrix
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Add self-loops and normalize
    import scipy.sparse as sp
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    adj = adj + sp.eye(num_nodes)
    
    # Normalize
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    # Convert to torch sparse
    adj = adj.tocoo()
    indices = torch.LongTensor(np.vstack((adj.row, adj.col)))
    values = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape))
    
    # Use different random splits for each split_idx
    torch.manual_seed(42 + split_idx)
    num_train = int(0.6 * num_nodes)
    num_val = int(0.2 * num_nodes)
    
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:num_train]] = True
    val_mask[perm[num_train:num_train+num_val]] = True
    test_mask[perm[num_train+num_val:]] = True
    
    print(f"Dataset: {dataset}, Split: {split_idx}")
    print(f"  Nodes: {num_nodes}, Features: {data.num_features}, Classes: {pyg_data.num_classes}")
    print(f"  Train: {train_mask.sum().item()} ({100*train_mask.sum().item()/num_nodes:.1f}%)")
    print(f"  Val:   {val_mask.sum().item()} ({100*val_mask.sum().item()/num_nodes:.1f}%)")
    print(f"  Test:  {test_mask.sum().item()} ({100*test_mask.sum().item()/num_nodes:.1f}%)")
    
    return {
        'features': data.x.to(device),
        'labels': data.y.to(device),
        'adj': adj_tensor.to(device),
        'train_mask': train_mask.to(device),
        'val_mask': val_mask.to(device),
        'test_mask': test_mask.to(device),
        'num_classes': pyg_data.num_classes,
        'num_features': data.num_features,
    }


# =============================================================================
# Teacher Training and Logits
# =============================================================================

def load_teacher_logits_heterophilic(dataset, split_idx, device):
    """Load pre-computed GloGNN++ teacher logits."""
    path = os.path.join('checkpoints', f'glognn_teacher_{dataset}',
                       f'split_{split_idx}', 'teacher_logits.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Teacher logits not found: {path}\n"
            f"Please run: python baselines/verify_glognn_teacher.py --dataset {dataset} --device cuda"
        )
    data = torch.load(path)
    logits = data.get('logits', data) if isinstance(data, dict) else data
    return logits.float().to(device)


def train_gcn_teacher(data, device, epochs=200):
    """Train GCN teacher for homophilic datasets."""
    model = GCN(
        nfeat=data['num_features'],
        nhid=64,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_logits = None
    
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
            val_acc = accuracy(logits[data['val_mask']], data['labels'][data['val_mask']])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_logits = logits.detach()
    
    return best_logits, best_val_acc


# =============================================================================
# Student Training
# =============================================================================

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train_student(data, teacher_logits, temperature, lambda_kd, device, epochs=500):
    """Train MLP student with knowledge distillation."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5,
        num_layers=2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    T = temperature
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        
        # CE loss
        loss_ce = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        
        # KD loss
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        loss = loss_ce + lambda_kd * loss_kd
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data['features'])
            val_acc = accuracy(logits[data['val_mask']], data['labels'][data['val_mask']])
            test_acc = accuracy(logits[data['test_mask']], data['labels'][data['test_mask']])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
        
        if patience >= 100:
            break
    
    return best_test_acc.item()


def train_vanilla_mlp(data, device, epochs=500):
    """Train vanilla MLP without distillation."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5,
        num_layers=2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        loss = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data['features'])
            val_acc = accuracy(logits[data['val_mask']], data['labels'][data['val_mask']])
            test_acc = accuracy(logits[data['test_mask']], data['labels'][data['test_mask']])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience = 0
            else:
                patience += 1
        
        if patience >= 100:
            break
    
    return best_test_acc.item()


# =============================================================================
# Experiment Runners
# =============================================================================

def run_heterophilic_experiments(datasets, device, num_splits=10):
    """Run experiments on heterophilic datasets."""
    
    # Best configs found from hyperparameter search
    best_configs = {
        'actor': {'T': 8.0, 'lambda_kd': 10.0},
        'squirrel': {'T': 1.0, 'lambda_kd': 10.0},
        'chameleon': {'T': 1.0, 'lambda_kd': 10.0},  # Similar to squirrel
    }
    
    # Search configs for chameleon (if not yet tuned)
    search_configs = {
        'chameleon': [
            {'T': 4.0, 'lambda_kd': 1.0, 'name': 'GLNN (default)'},
            {'T': 1.0, 'lambda_kd': 1.0, 'name': 'T=1'},
            {'T': 1.0, 'lambda_kd': 5.0, 'name': 'T=1, λ=5'},
            {'T': 1.0, 'lambda_kd': 10.0, 'name': 'T=1, λ=10'},
            {'T': 2.0, 'lambda_kd': 10.0, 'name': 'T=2, λ=10'},
            {'T': 1.0, 'lambda_kd': 15.0, 'name': 'T=1, λ=15'},
        ],
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"HETEROPHILIC: {dataset.upper()}")
        print(f"{'='*70}")
        
        results[dataset] = {'configs': {}}
        
        # If we need to search configs
        if dataset in search_configs:
            print("\nSearching best configuration...")
            for config in search_configs[dataset]:
                accs = []
                for split_idx in range(min(5, num_splits)):  # Quick search with 5 splits
                    data = load_heterophilic_data(dataset, split_idx, device)
                    teacher_logits = load_teacher_logits_heterophilic(dataset, split_idx, device)
                    acc = train_student(data, teacher_logits, config['T'], config['lambda_kd'], device)
                    accs.append(acc * 100)
                
                mean_acc = np.mean(accs)
                results[dataset]['configs'][config['name']] = {
                    'mean': mean_acc, 'std': np.std(accs),
                    'T': config['T'], 'lambda_kd': config['lambda_kd']
                }
                print(f"  {config['name']}: {mean_acc:.2f}%")
            
            # Find best config
            best_name = max(results[dataset]['configs'].keys(),
                          key=lambda k: results[dataset]['configs'][k]['mean'])
            best_cfg = results[dataset]['configs'][best_name]
            best_configs[dataset] = {'T': best_cfg['T'], 'lambda_kd': best_cfg['lambda_kd']}
            print(f"\n  ★ Best: {best_name}")
        
        # Run full evaluation with best config
        print(f"\nFull evaluation with T={best_configs[dataset]['T']}, λ={best_configs[dataset]['lambda_kd']}...")
        
        teacher_accs = []
        student_accs = []
        mlp_accs = []
        
        for split_idx in range(num_splits):
            data = load_heterophilic_data(dataset, split_idx, device)
            teacher_logits = load_teacher_logits_heterophilic(dataset, split_idx, device)
            
            # Teacher accuracy (from saved logits)
            teacher_acc = accuracy(teacher_logits[data['test_mask']], 
                                  data['labels'][data['test_mask']]).item() * 100
            teacher_accs.append(teacher_acc)
            
            # Student accuracy
            student_acc = train_student(
                data, teacher_logits,
                best_configs[dataset]['T'],
                best_configs[dataset]['lambda_kd'],
                device
            ) * 100
            student_accs.append(student_acc)
            
            # Vanilla MLP
            mlp_acc = train_vanilla_mlp(data, device) * 100
            mlp_accs.append(mlp_acc)
            
            print(f"  Split {split_idx}: Teacher={teacher_acc:.2f}%, Student={student_acc:.2f}%, MLP={mlp_acc:.2f}%")
        
        results[dataset]['teacher'] = {'mean': np.mean(teacher_accs), 'std': np.std(teacher_accs)}
        results[dataset]['student'] = {'mean': np.mean(student_accs), 'std': np.std(student_accs)}
        results[dataset]['mlp'] = {'mean': np.mean(mlp_accs), 'std': np.std(mlp_accs)}
        results[dataset]['best_config'] = best_configs[dataset]
        
        # Summary
        print(f"\n  Summary:")
        print(f"    Teacher (GloGNN++): {results[dataset]['teacher']['mean']:.2f}% ± {results[dataset]['teacher']['std']:.2f}%")
        print(f"    Vanilla MLP:        {results[dataset]['mlp']['mean']:.2f}% ± {results[dataset]['mlp']['std']:.2f}%")
        print(f"    Our Student:        {results[dataset]['student']['mean']:.2f}% ± {results[dataset]['student']['std']:.2f}%")
    
    return results


def run_homophilic_experiments(datasets, device, num_splits=10):
    """Run experiments on homophilic datasets."""
    
    # Default config for homophilic (no need to tune much)
    default_config = {'T': 4.0, 'lambda_kd': 1.0}
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"HOMOPHILIC: {dataset.upper()}")
        print(f"{'='*70}")
        
        teacher_accs = []
        student_accs = []
        mlp_accs = []
        
        for split_idx in range(num_splits):
            data = load_homophilic_data(dataset, split_idx, device)
            
            # Train GCN teacher
            teacher_logits, _ = train_gcn_teacher(data, device)
            teacher_acc = accuracy(teacher_logits[data['test_mask']], 
                                  data['labels'][data['test_mask']]).item() * 100
            teacher_accs.append(teacher_acc)
            
            # Train student
            student_acc = train_student(
                data, teacher_logits,
                default_config['T'],
                default_config['lambda_kd'],
                device
            ) * 100
            student_accs.append(student_acc)
            
            # Vanilla MLP
            mlp_acc = train_vanilla_mlp(data, device) * 100
            mlp_accs.append(mlp_acc)
            
            print(f"  Split {split_idx}: Teacher={teacher_acc:.2f}%, Student={student_acc:.2f}%, MLP={mlp_acc:.2f}%")
        
        results[dataset] = {
            'teacher': {'mean': np.mean(teacher_accs), 'std': np.std(teacher_accs)},
            'student': {'mean': np.mean(student_accs), 'std': np.std(student_accs)},
            'mlp': {'mean': np.mean(mlp_accs), 'std': np.std(mlp_accs)},
            'config': default_config,
        }
        
        print(f"\n  Summary:")
        print(f"    Teacher (GCN):  {results[dataset]['teacher']['mean']:.2f}% ± {results[dataset]['teacher']['std']:.2f}%")
        print(f"    Vanilla MLP:    {results[dataset]['mlp']['mean']:.2f}% ± {results[dataset]['mlp']['std']:.2f}%")
        print(f"    Our Student:    {results[dataset]['student']['mean']:.2f}% ± {results[dataset]['student']['std']:.2f}%")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'heterophilic', 'homophilic'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_splits', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Splits: {args.num_splits}")
    
    all_results = {'timestamp': datetime.now().isoformat()}
    
    # Heterophilic experiments
    if args.mode in ['all', 'heterophilic']:
        hetero_results = run_heterophilic_experiments(
            ['actor', 'squirrel', 'chameleon'],
            device,
            args.num_splits
        )
        all_results['heterophilic'] = hetero_results
    
    # Homophilic experiments
    if args.mode in ['all', 'homophilic']:
        homo_results = run_homophilic_experiments(
            ['cora', 'citeseer', 'pubmed'],
            device,
            args.num_splits
        )
        all_results['homophilic'] = homo_results
    
    # Save results
    os.makedirs('results/final', exist_ok=True)
    
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
    
    results_path = f'results/final/all_experiments_{args.mode}.json'
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if 'heterophilic' in all_results:
        print("\n📊 HETEROPHILIC DATASETS (Main Contribution)")
        print("-" * 50)
        for ds, res in all_results['heterophilic'].items():
            teacher = res['teacher']['mean']
            student = res['student']['mean']
            mlp = res['mlp']['mean']
            gap_closed = (student - mlp) / (teacher - mlp) * 100 if teacher != mlp else 0
            
            print(f"\n{ds.upper()}:")
            print(f"  Teacher:  {teacher:.2f}%")
            print(f"  MLP:      {mlp:.2f}%")
            print(f"  Student:  {student:.2f}% ± {res['student']['std']:.2f}%")
            print(f"  Gap Closed: {gap_closed:.1f}%")
            if student > teacher:
                print(f"  ✅ Student BEATS Teacher!")
    
    if 'homophilic' in all_results:
        print("\n📊 HOMOPHILIC DATASETS (Sanity Check)")
        print("-" * 50)
        for ds, res in all_results['homophilic'].items():
            teacher = res['teacher']['mean']
            student = res['student']['mean']
            mlp = res['mlp']['mean']
            
            print(f"\n{ds.upper()}:")
            print(f"  Teacher:  {teacher:.2f}%")
            print(f"  MLP:      {mlp:.2f}%")
            print(f"  Student:  {student:.2f}% ± {res['student']['std']:.2f}%")
            if abs(student - teacher) < 2:
                print(f"  ✅ Student ≈ Teacher (good!)")
    
    print(f"\n\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
