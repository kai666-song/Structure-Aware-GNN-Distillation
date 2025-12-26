"""
Verify Gated AFD on Homophily Breakdown
=======================================

This script verifies if Gated AFD solves the "homophilic node performance drop"
observed in Task 3 (the 4.68% drop on high-homophily nodes).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kd_losses.adaptive_kd import GatedAFDLoss, AFDLoss


# ============================================================================
# Data Loading (same as before)
# ============================================================================

def load_filtered_data(dataset, split_idx, device):
    data_dir = '../heterophilous-graphs/data'
    filename = f'{dataset}_filtered.npz'
    filepath = os.path.join(data_dir, filename)
    
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


def compute_node_homophily(adj, labels):
    """Compute per-node homophily."""
    adj_coo = adj.tocoo()
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    num_nodes = len(labels_np)
    
    match_count = np.zeros(num_nodes, dtype=np.float32)
    degree = np.zeros(num_nodes, dtype=np.float32)
    
    for r, c in zip(adj_coo.row, adj_coo.col):
        degree[r] += 1
        if labels_np[r] == labels_np[c]:
            match_count[r] += 1
    
    degree[degree == 0] = 1
    return match_count / degree


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
    
    best_val_acc, best_logits = 0, None
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
            val_acc = (logits.argmax(1)[data['val_mask']] == data['labels'][data['val_mask']]).float().mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_logits = logits.detach()
    
    return best_logits


def train_and_get_preds(data, teacher_logits, device, method='simple_kd', epochs=500):
    """Train model and return predictions."""
    model = SimpleMLP(data['num_features'], 256, data['num_classes']).to(device)
    
    if method == 'gated':
        loss_fn = GatedAFDLoss(data['adj_raw'], K=5, temperature=4.0, lambda_kd=1.0,
                               gate_init_threshold=0.3, device=device)
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
            {'params': loss_fn.parameters(), 'lr': 0.005}
        ])
    elif method == 'afd':
        loss_fn = AFDLoss(data['adj_raw'], K=5, device=device)
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
            {'params': loss_fn.parameters(), 'lr': 0.005}
        ])
    else:
        loss_fn = None
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    T = 4.0
    best_val_acc, best_preds = 0, None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data['features'])
        
        loss_ce = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        if method == 'gated':
            loss_fn.train()
            loss, _ = loss_fn(logits, teacher_logits, data['labels'], data['train_mask'])
        elif method == 'afd':
            loss_fn.train()
            loss_afd, _ = loss_fn(logits, teacher_logits)
            loss = loss_ce + loss_kd + 0.5 * loss_afd
        else:
            loss = loss_ce + loss_kd
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(data['features'])
            preds = logits.argmax(1)
            val_acc = (preds[data['val_mask']] == data['labels'][data['val_mask']]).float().mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_preds = preds.cpu().numpy()
    
    return best_preds


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("HOMOPHILY BREAKDOWN ANALYSIS: Gated AFD vs Simple KD vs AFD-KD")
    print("="*80)
    print("\nGoal: Verify if Gated AFD solves the 'homophilic node performance drop'")
    
    for dataset in ['chameleon']:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}_filtered")
        print(f"{'='*70}")
        
        # Collect results across splits
        bins = np.linspace(0, 1, 6)
        results = {method: {i: [] for i in range(5)} for method in ['simple_kd', 'afd', 'gated']}
        
        for split_idx in range(5):
            print(f"\n  Split {split_idx}:")
            data = load_filtered_data(dataset, split_idx, device)
            node_homophily = compute_node_homophily(data['adj_raw'], data['labels'])
            
            # Train teacher
            teacher_logits = train_teacher(data, device)
            
            # Train each method and get predictions
            for method in ['simple_kd', 'afd', 'gated']:
                preds = train_and_get_preds(data, teacher_logits, device, method)
                labels_np = data['labels'].cpu().numpy()
                test_mask_np = data['test_mask'].cpu().numpy()
                
                # Analyze by homophily bins
                for i in range(5):
                    low, high = bins[i], bins[i+1]
                    if i == 4:
                        bin_mask = (node_homophily >= low) & (node_homophily <= high) & test_mask_np
                    else:
                        bin_mask = (node_homophily >= low) & (node_homophily < high) & test_mask_np
                    
                    if bin_mask.sum() > 0:
                        acc = (preds[bin_mask] == labels_np[bin_mask]).mean() * 100
                        results[method][i].append(acc)
        
        # Summary table
        print(f"\n{'='*70}")
        print(f"ACCURACY BY LOCAL HOMOPHILY: {dataset}_filtered")
        print(f"{'='*70}")
        print(f"\n{'Homophily Bin':<15} {'Simple KD':>12} {'AFD-KD':>12} {'Gated AFD':>12} {'Gated-Simple':>14}")
        print("-" * 70)
        
        for i in range(5):
            bin_label = f"[{bins[i]:.1f}, {bins[i+1]:.1f})"
            simple = np.mean(results['simple_kd'][i]) if results['simple_kd'][i] else 0
            afd = np.mean(results['afd'][i]) if results['afd'][i] else 0
            gated = np.mean(results['gated'][i]) if results['gated'][i] else 0
            delta = gated - simple
            
            marker = "✅" if delta > 0 else "❌" if delta < -1 else "  "
            print(f"{bin_label:<15} {simple:>11.2f}% {afd:>11.2f}% {gated:>11.2f}% {delta:>+13.2f}% {marker}")
        
        # Key insight
        low_homo = np.mean([np.mean(results['gated'][i]) - np.mean(results['simple_kd'][i]) 
                          for i in range(2) if results['gated'][i]])
        high_homo = np.mean([np.mean(results['gated'][i]) - np.mean(results['simple_kd'][i]) 
                           for i in range(3, 5) if results['gated'][i]])
        
        print(f"\n  Key Insight:")
        print(f"    Low homophily (h < 0.4) Gated vs Simple: {low_homo:+.2f}%")
        print(f"    High homophily (h >= 0.6) Gated vs Simple: {high_homo:+.2f}%")
        
        if high_homo > 0:
            print(f"    ✅ Gated AFD SOLVES the homophilic node performance drop!")
        else:
            print(f"    ⚠️  Gated AFD still has issues on homophilic nodes")


if __name__ == '__main__':
    main()
