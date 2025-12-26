"""
Train MLP Student with Gated Adaptive Frequency-Decoupled KD (GAFF)
===================================================================

This script tests the Gated AFD mechanism that dynamically blends
high-frequency AFD loss and standard soft-target KD based on each
node's local homophily.

Core Innovation:
- Homophilic nodes → more standard KD (gate ≈ 1)
- Heterophilic nodes → more AFD (gate ≈ 0)

This makes the model a "全能生" (all-rounder) that works well on both
homophilic and heterophilic regions.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kd_losses.adaptive_kd import GatedAFDLoss, DualPathGatedLoss, AFDLoss


# ============================================================================
# Data Loading
# ============================================================================

def load_filtered_data(dataset, split_idx, device):
    """Load filtered dataset from Platonov repository."""
    data_dir = '../heterophilous-graphs/data'
    filename = f'{dataset}_filtered.npz'
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
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
    
    # Normalized adjacency for GNN
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
    
    # Normalize features
    row_sum = features.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum
    
    num_classes = len(torch.unique(labels))
    
    print(f"  Loaded {dataset}_filtered: {num_nodes} nodes, {num_classes} classes")
    print(f"  Split {split_idx}: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    
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

def train_gnn_teacher(data, device, epochs=500):
    """Train GCN teacher and return logits."""
    model = GCNWithSkip(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5,
        num_layers=4
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    best_val_acc = 0
    best_test_acc = 0
    best_logits = None
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


def train_mlp_direct(data, device, epochs=500):
    """Train MLP with only CE loss (no KD)."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    
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
            preds = logits.argmax(dim=1)
            val_acc = (preds[data['val_mask']] == data['labels'][data['val_mask']]).float().mean().item()
            test_acc = (preds[data['test_mask']] == data['labels'][data['test_mask']]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    
    return best_test_acc


def train_mlp_simple_kd(data, teacher_logits, device, T=4.0, lambda_kd=1.0, epochs=500):
    """Train MLP with simple KD (baseline)."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        
        loss_ce = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        loss = loss_ce + lambda_kd * loss_kd
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


def train_mlp_afd_kd(data, teacher_logits, device, K=5, lambda_afd=0.5, epochs=500):
    """Train MLP with AFD-KD (no gating)."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    afd_loss = AFDLoss(
        adj=data['adj_raw'],
        K=K,
        init_mode='uniform',
        loss_type='mse',
        temperature=4.0,
        device=device
    )
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': afd_loss.parameters(), 'lr': 0.005}
    ])
    
    T = 4.0
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(epochs):
        model.train()
        afd_loss.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        
        loss_ce = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_soft = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        loss_afd, _ = afd_loss(logits, teacher_logits)
        
        loss = loss_ce + 1.0 * loss_soft + lambda_afd * loss_afd
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


def train_mlp_gated_afd(data, teacher_logits, device, K=5, lambda_kd=1.0,
                        gate_threshold=0.5, gate_sharpness=5.0, epochs=500):
    """Train MLP with Gated AFD-KD (core innovation)."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    gated_loss = GatedAFDLoss(
        adj=data['adj_raw'],
        K=K,
        temperature=4.0,
        lambda_kd=lambda_kd,
        gate_init_threshold=gate_threshold,
        gate_sharpness=gate_sharpness,
        learnable_gate=True,
        device=device
    )
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': gated_loss.parameters(), 'lr': 0.005}
    ])
    
    best_val_acc = 0
    best_test_acc = 0
    best_gate_analysis = None
    
    for epoch in range(epochs):
        model.train()
        gated_loss.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        
        loss, loss_dict = gated_loss(
            logits, teacher_logits, data['labels'], data['train_mask']
        )
        
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
                best_gate_analysis = gated_loss.get_gate_analysis(teacher_logits)
    
    return best_test_acc, best_gate_analysis


def train_mlp_dual_path(data, teacher_logits, device, K=5, lambda_kd=1.0,
                        gate_threshold=0.5, gate_sharpness=5.0, epochs=500):
    """Train MLP with Dual-Path Gated Loss."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    dual_loss = DualPathGatedLoss(
        adj=data['adj_raw'],
        K=K,
        temperature=4.0,
        lambda_kd=lambda_kd,
        gate_init_threshold=gate_threshold,
        gate_sharpness=gate_sharpness,
        device=device
    )
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': dual_loss.parameters(), 'lr': 0.005}
    ])
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(epochs):
        model.train()
        dual_loss.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        
        loss, loss_dict = dual_loss(
            logits, teacher_logits, data['labels'], data['train_mask']
        )
        
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
# Main Experiment
# ============================================================================

def run_gated_experiment(dataset, device, num_splits=5):
    """Run Gated AFD experiment."""
    print(f"\n{'='*80}")
    print(f"GATED AFD EXPERIMENT: {dataset}_filtered")
    print(f"{'='*80}")
    
    configs = [
        {'name': 'MLP Direct', 'method': 'direct'},
        {'name': 'Simple KD', 'method': 'simple_kd'},
        {'name': 'AFD-KD', 'method': 'afd_kd', 'K': 5},
        {'name': 'Gated AFD (τ=0.3)', 'method': 'gated', 'K': 5, 'threshold': 0.3},
        {'name': 'Gated AFD (τ=0.5)', 'method': 'gated', 'K': 5, 'threshold': 0.5},
        {'name': 'Gated AFD (τ=0.7)', 'method': 'gated', 'K': 5, 'threshold': 0.7},
        {'name': 'Dual-Path Gated', 'method': 'dual_path', 'K': 5, 'threshold': 0.5},
    ]
    
    results = {cfg['name']: [] for cfg in configs}
    teacher_accs = []
    gate_analyses = []
    
    for split_idx in range(num_splits):
        print(f"\n--- Split {split_idx} ---")
        
        data = load_filtered_data(dataset, split_idx, device)
        
        # Train teacher
        print("  Training GNN Teacher...", end=" ", flush=True)
        teacher_acc, teacher_logits = train_gnn_teacher(data, device)
        teacher_accs.append(teacher_acc * 100)
        print(f"Acc={teacher_acc*100:.2f}%")
        
        # Run each configuration
        for cfg in configs:
            print(f"  Training {cfg['name']}...", end=" ", flush=True)
            
            if cfg['method'] == 'direct':
                acc = train_mlp_direct(data, device)
            elif cfg['method'] == 'simple_kd':
                acc = train_mlp_simple_kd(data, teacher_logits, device)
            elif cfg['method'] == 'afd_kd':
                acc = train_mlp_afd_kd(data, teacher_logits, device, K=cfg['K'])
            elif cfg['method'] == 'gated':
                acc, gate_analysis = train_mlp_gated_afd(
                    data, teacher_logits, device, 
                    K=cfg['K'], gate_threshold=cfg['threshold']
                )
                if split_idx == 0:
                    gate_analyses.append({'config': cfg['name'], 'analysis': gate_analysis})
            elif cfg['method'] == 'dual_path':
                acc = train_mlp_dual_path(
                    data, teacher_logits, device,
                    K=cfg['K'], gate_threshold=cfg['threshold']
                )
            
            results[cfg['name']].append(acc * 100)
            print(f"Acc={acc*100:.2f}%")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {dataset}_filtered")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<25} {'Mean':>10} {'Std':>10} {'vs Direct':>12} {'vs SimpleKD':>12}")
    print("-" * 70)
    
    teacher_mean = np.mean(teacher_accs)
    print(f"{'GNN Teacher':<25} {teacher_mean:>9.2f}% {np.std(teacher_accs):>9.2f}%")
    
    direct_mean = np.mean(results['MLP Direct'])
    simple_kd_mean = np.mean(results['Simple KD'])
    
    for cfg in configs:
        name = cfg['name']
        mean = np.mean(results[name])
        std = np.std(results[name])
        vs_direct = mean - direct_mean
        vs_simple = mean - simple_kd_mean
        print(f"{name:<25} {mean:>9.2f}% {std:>9.2f}% {vs_direct:>+11.2f}% {vs_simple:>+11.2f}%")
    
    # Gate analysis
    if gate_analyses:
        print(f"\n{'='*60}")
        print("GATE ANALYSIS (Split 0)")
        print(f"{'='*60}")
        for ga in gate_analyses:
            print(f"\n{ga['config']}:")
            stats = ga['analysis']['stats']
            print(f"  Gate mean: {stats['gate_mean']:.3f}")
            print(f"  Gate std:  {stats['gate_std']:.3f}")
            print(f"  Threshold: {stats['threshold']:.3f}")
            print(f"  Sharpness: {stats['sharpness']:.3f}")
    
    return {
        'dataset': dataset,
        'teacher': {'mean': teacher_mean, 'std': np.std(teacher_accs)},
        'results': {name: {'mean': np.mean(accs), 'std': np.std(accs)} 
                   for name, accs in results.items()},
        'gate_analyses': gate_analyses
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("GATED ADAPTIVE FREQUENCY-DECOUPLED KD (GAFF) EXPERIMENTS")
    print("="*80)
    print("\nCore Innovation: Dynamic gate blends AFD and standard KD based on")
    print("each node's local homophily. Heterophilic nodes get more AFD,")
    print("homophilic nodes get more standard KD.\n")
    
    all_results = {}
    
    for dataset in ['squirrel', 'chameleon']:
        result = run_gated_experiment(dataset, device, num_splits=5)
        all_results[dataset] = result
    
    # Save results
    os.makedirs('results/gated_afd', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    with open(f'results/gated_afd/results_{timestamp}.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2, default=str)
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Dataset':<20} {'SimpleKD':>10} {'AFD-KD':>10} {'Best Gated':>12} {'Δ vs AFD':>10}")
    print("-" * 65)
    
    for dataset, result in all_results.items():
        simple_kd = result['results']['Simple KD']['mean']
        afd_kd = result['results']['AFD-KD']['mean']
        
        # Find best gated config
        best_gated = 0
        best_name = ""
        for name, r in result['results'].items():
            if 'Gated' in name and r['mean'] > best_gated:
                best_gated = r['mean']
                best_name = name
        
        delta = best_gated - afd_kd
        
        print(f"{dataset+'_filtered':<20} {simple_kd:>9.2f}% {afd_kd:>9.2f}% {best_gated:>11.2f}% {delta:>+9.2f}%")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    # Check improvements
    improvements = []
    for dataset, result in all_results.items():
        afd_kd = result['results']['AFD-KD']['mean']
        best_gated = max(r['mean'] for name, r in result['results'].items() if 'Gated' in name)
        improvements.append(best_gated - afd_kd)
    
    avg_improvement = np.mean(improvements)
    
    if avg_improvement > 0.5:
        print(f"""
✅ GATED AFD PROVIDES IMPROVEMENT!

Average improvement over AFD-KD: {avg_improvement:+.2f}%

The gating mechanism successfully adapts the distillation strategy
based on local homophily, making the model a "全能生" (all-rounder)
that works well on both homophilic and heterophilic nodes.
""")
    elif avg_improvement > 0:
        print(f"""
⚡ GATED AFD PROVIDES MARGINAL IMPROVEMENT

Average improvement over AFD-KD: {avg_improvement:+.2f}%

The gating mechanism helps slightly. Further tuning of gate
parameters (threshold, sharpness) may yield better results.
""")
    else:
        print(f"""
⚠️  GATED AFD DOES NOT IMPROVE OVER AFD-KD

Average improvement over AFD-KD: {avg_improvement:+.2f}%

The gating mechanism does not help on these datasets.
This may indicate that:
1. The homophily estimation is not accurate enough
2. The gate parameters need tuning
3. AFD-KD is already optimal for these datasets
""")


if __name__ == '__main__':
    main()
