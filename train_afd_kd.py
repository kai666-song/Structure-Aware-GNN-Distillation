"""
Train MLP Student with Adaptive Frequency-Decoupled KD (AFD-KD)
===============================================================

This script trains MLP students using the upgraded AFD-KD loss with
learnable Bernstein polynomial spectral filters.

Key Features:
1. Learnable spectral filter θ_k optimized jointly with MLP
2. Automatic discovery of optimal frequency bands for KD
3. Support for single-band and multi-band configurations

Usage:
    python train_afd_kd.py --dataset squirrel --K 10 --num_bands 3
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
import scipy.sparse as sp
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kd_losses.adaptive_kd import HybridAFDLoss, AFDLoss, MultiBandAFDLoss


# =============================================================================
# Data Loading
# =============================================================================

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
    
    # Build adjacency matrix (raw, without self-loops, for AFD)
    num_nodes = features.shape[0]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    
    adj_raw = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    adj_raw = adj_raw.tocsr()
    
    # Normalized adjacency for GNN (with self-loops)
    adj = adj_raw + sp.eye(num_nodes)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    # Convert to torch sparse
    adj_coo = adj.tocoo()
    indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
    values = torch.FloatTensor(adj_coo.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(adj_coo.shape))
    
    # Normalize features
    row_sum = features.sum(dim=1, keepdim=True)
    row_sum[row_sum == 0] = 1
    features = features / row_sum
    
    num_classes = len(torch.unique(labels))
    
    return {
        'features': features.to(device),
        'labels': labels.to(device),
        'adj': adj_tensor.to(device),
        'adj_raw': adj_raw,  # For AFD loss
        'train_mask': train_mask.to(device),
        'val_mask': val_mask.to(device),
        'test_mask': test_mask.to(device),
        'num_classes': num_classes,
        'num_features': features.shape[1],
        'num_nodes': num_nodes,
    }


# =============================================================================
# Models
# =============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP Student."""
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
    """GCN Teacher with skip connections."""
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


# =============================================================================
# Training Functions
# =============================================================================

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


def train_mlp_simple_kd(data, teacher_logits, device, temperature=4.0, lambda_kd=1.0, epochs=500):
    """Train MLP with simple KD (baseline)."""
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    T = temperature
    
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


def train_mlp_afd_kd(data, teacher_logits, device, 
                     K=10, num_bands=1, temperature=4.0,
                     lambda_soft=1.0, lambda_afd=0.5,
                     use_multi_band=False, epochs=500):
    """
    Train MLP with AFD-KD loss.
    
    Key: Both MLP parameters and AFD filter parameters (θ) are optimized jointly.
    """
    model = SimpleMLP(
        nfeat=data['num_features'],
        nhid=256,
        nclass=data['num_classes'],
        dropout=0.5
    ).to(device)
    
    # Create AFD loss with learnable parameters
    afd_loss_fn = HybridAFDLoss(
        adj=data['adj_raw'],
        K=K,
        num_bands=num_bands,
        temperature=temperature,
        lambda_soft=lambda_soft,
        lambda_afd=lambda_afd,
        use_multi_band=use_multi_band,
        device=device
    )
    
    # IMPORTANT: Optimizer includes both model params AND AFD loss params (θ)
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': afd_loss_fn.parameters(), 'lr': 0.005}  # Slower LR for filter params
    ])
    
    best_val_acc = 0
    best_test_acc = 0
    best_freq_response = None
    
    for epoch in range(epochs):
        model.train()
        afd_loss_fn.train()
        optimizer.zero_grad()
        
        logits = model(data['features'])
        
        # Compute hybrid loss (CE + soft KD + AFD)
        loss, loss_dict = afd_loss_fn(
            logits, teacher_logits, data['labels'], data['train_mask']
        )
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        afd_loss_fn.eval()
        with torch.no_grad():
            logits = model(data['features'])
            preds = logits.argmax(dim=1)
            val_acc = (preds[data['val_mask']] == data['labels'][data['val_mask']]).float().mean().item()
            test_acc = (preds[data['test_mask']] == data['labels'][data['test_mask']]).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_freq_response = afd_loss_fn.get_frequency_response()
    
    return best_test_acc, best_freq_response


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


# =============================================================================
# Visualization
# =============================================================================

def plot_frequency_response(freq_response, save_path=None):
    """Plot learned frequency response."""
    plt.figure(figsize=(10, 6))
    
    if isinstance(freq_response, tuple):
        # Single filter
        lambdas, response = freq_response
        plt.plot(lambdas, response, 'b-', linewidth=2, label='Learned Filter')
        plt.fill_between(lambdas, 0, response, alpha=0.3)
    else:
        # Multi-band
        for band_info in freq_response:
            lambdas = band_info['lambdas']
            response = band_info['response']
            weight = band_info['weight']
            plt.plot(lambdas, response, linewidth=2, 
                    label=f"Band {band_info['band']} (w={weight:.3f})")
    
    plt.xlabel('Normalized Frequency (λ)', fontsize=12)
    plt.ylabel('Filter Response h(λ)', fontsize=12)
    plt.title('Learned Spectral Filter Response', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, None])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved frequency response plot to {save_path}")
    
    plt.close()


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(dataset, device, num_splits=5, K=10, num_bands=3, 
                   lambda_soft=1.0, lambda_afd=0.5):
    """Run complete AFD-KD experiment."""
    print(f"\n{'='*80}")
    print(f"AFD-KD EXPERIMENT: {dataset}_filtered")
    print(f"K={K}, num_bands={num_bands}, λ_soft={lambda_soft}, λ_afd={lambda_afd}")
    print(f"{'='*80}")
    
    results = {
        'teacher': [],
        'mlp_direct': [],
        'simple_kd': [],
        'afd_kd_single': [],
        'afd_kd_multi': [],
    }
    
    all_freq_responses = []
    
    for split_idx in range(num_splits):
        print(f"\n--- Split {split_idx} ---")
        
        data = load_filtered_data(dataset, split_idx, device)
        print(f"  Loaded: {data['num_nodes']} nodes, {data['num_classes']} classes")
        
        # Train teacher
        print("  Training GNN Teacher...", end=" ", flush=True)
        teacher_acc, teacher_logits = train_gnn_teacher(data, device)
        results['teacher'].append(teacher_acc * 100)
        print(f"Acc={teacher_acc*100:.2f}%")
        
        # MLP Direct
        print("  Training MLP Direct...", end=" ", flush=True)
        direct_acc = train_mlp_direct(data, device)
        results['mlp_direct'].append(direct_acc * 100)
        print(f"Acc={direct_acc*100:.2f}%")
        
        # Simple KD
        print("  Training Simple KD...", end=" ", flush=True)
        simple_acc = train_mlp_simple_kd(data, teacher_logits, device)
        results['simple_kd'].append(simple_acc * 100)
        print(f"Acc={simple_acc*100:.2f}%")
        
        # AFD-KD (single band)
        print("  Training AFD-KD (single)...", end=" ", flush=True)
        afd_single_acc, freq_resp_single = train_mlp_afd_kd(
            data, teacher_logits, device,
            K=K, num_bands=1, lambda_soft=lambda_soft, lambda_afd=lambda_afd,
            use_multi_band=False
        )
        results['afd_kd_single'].append(afd_single_acc * 100)
        print(f"Acc={afd_single_acc*100:.2f}%")
        
        # AFD-KD (multi-band)
        print("  Training AFD-KD (multi)...", end=" ", flush=True)
        afd_multi_acc, freq_resp_multi = train_mlp_afd_kd(
            data, teacher_logits, device,
            K=K, num_bands=num_bands, lambda_soft=lambda_soft, lambda_afd=lambda_afd,
            use_multi_band=True
        )
        results['afd_kd_multi'].append(afd_multi_acc * 100)
        all_freq_responses.append(freq_resp_multi)
        print(f"Acc={afd_multi_acc*100:.2f}%")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {dataset}_filtered")
    print(f"{'='*80}")
    
    print(f"\n{'Method':<20} {'Mean':>10} {'Std':>10} {'vs Direct':>12} {'vs SimpleKD':>12}")
    print("-" * 65)
    
    direct_mean = np.mean(results['mlp_direct'])
    simple_mean = np.mean(results['simple_kd'])
    
    for method, accs in results.items():
        mean = np.mean(accs)
        std = np.std(accs)
        vs_direct = mean - direct_mean
        vs_simple = mean - simple_mean
        print(f"{method:<20} {mean:>9.2f}% {std:>9.2f}% {vs_direct:>+11.2f}% {vs_simple:>+11.2f}%")
    
    return results, all_freq_responses


def main():
    parser = argparse.ArgumentParser(description='AFD-KD Training')
    parser.add_argument('--dataset', type=str, default='squirrel',
                       choices=['squirrel', 'chameleon'])
    parser.add_argument('--K', type=int, default=10,
                       help='Bernstein polynomial order')
    parser.add_argument('--num_bands', type=int, default=3,
                       help='Number of frequency bands for multi-band AFD')
    parser.add_argument('--lambda_soft', type=float, default=1.0,
                       help='Weight for soft-target KD loss')
    parser.add_argument('--lambda_afd', type=float, default=0.5,
                       help='Weight for AFD loss')
    parser.add_argument('--num_splits', type=int, default=5,
                       help='Number of data splits to run')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save frequency response plots')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("ADAPTIVE FREQUENCY-DECOUPLED KNOWLEDGE DISTILLATION (AFD-KD)")
    print("="*80)
    print("\nThis experiment tests the upgraded AFD-KD with learnable Bernstein")
    print("polynomial spectral filters on filtered (clean) datasets.\n")
    
    # Run experiments
    all_results = {}
    all_freq_responses = {}
    
    for dataset in [args.dataset]:
        results, freq_responses = run_experiment(
            dataset, device, 
            num_splits=args.num_splits,
            K=args.K,
            num_bands=args.num_bands,
            lambda_soft=args.lambda_soft,
            lambda_afd=args.lambda_afd
        )
        all_results[dataset] = results
        all_freq_responses[dataset] = freq_responses
        
        # Save frequency response plot
        if args.save_plots and freq_responses:
            os.makedirs('figures/afd_kd', exist_ok=True)
            plot_frequency_response(
                freq_responses[-1],  # Last split
                f'figures/afd_kd/{dataset}_freq_response.png'
            )
    
    # Save results
    os.makedirs('results/afd_kd', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy to python types for JSON
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
    
    results_path = f'results/afd_kd/experiment_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset}_filtered:")
        
        teacher = np.mean(results['teacher'])
        direct = np.mean(results['mlp_direct'])
        simple = np.mean(results['simple_kd'])
        afd_single = np.mean(results['afd_kd_single'])
        afd_multi = np.mean(results['afd_kd_multi'])
        
        best_afd = max(afd_single, afd_multi)
        improvement = best_afd - simple
        
        print(f"  Teacher:     {teacher:.2f}%")
        print(f"  MLP Direct:  {direct:.2f}%")
        print(f"  Simple KD:   {simple:.2f}%")
        print(f"  AFD-KD:      {best_afd:.2f}% ({improvement:+.2f}% vs Simple KD)")
        
        if improvement > 0.5:
            print(f"  ✅ AFD-KD improves over Simple KD!")
        else:
            print(f"  ⚠️  AFD-KD shows marginal improvement")


if __name__ == '__main__':
    main()
