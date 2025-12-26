"""
Verify KD Results on Filtered (Clean) Datasets
==============================================

This is the CRITICAL experiment to address the reviewer's concern about data leakage.

We train a GNN Teacher on the filtered (clean) Squirrel/Chameleon datasets,
then verify if KD still helps the MLP Student.

If KD still works on filtered data: Our contribution is valid
If KD fails on filtered data: Our results were due to data leakage
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    
    # Build adjacency matrix
    num_nodes = features.shape[0]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    
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
    values = torch.FloatTensor(adj.data.astype(np.float32))
    adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape))
    
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
        'train_mask': train_mask.to(device),
        'val_mask': val_mask.to(device),
        'test_mask': test_mask.to(device),
        'num_classes': num_classes,
        'num_features': features.shape[1],
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
    """GCN with skip connections (strong baseline from Platonov paper)."""
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
            x = x + h  # Skip connection
        
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
    """Train MLP with only CE loss."""
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


def train_mlp_kd(data, teacher_logits, device, temperature=4.0, lambda_kd=1.0, epochs=500):
    """Train MLP with KD loss."""
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


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\n" + "="*90)
    print("VERIFICATION ON FILTERED (CLEAN) DATASETS")
    print("="*90)
    print("\nThis experiment addresses the reviewer's concern about data leakage.")
    print("We use the filtered versions of Squirrel and Chameleon that have")
    print("duplicate nodes removed.\n")
    
    datasets = ['squirrel', 'chameleon']
    num_splits = 5
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset}_filtered")
        print(f"{'='*70}")
        
        teacher_accs = []
        mlp_direct_accs = []
        mlp_kd_accs = []
        
        for split_idx in range(num_splits):
            print(f"\n  Split {split_idx}:")
            
            data = load_filtered_data(dataset, split_idx, device)
            
            # Train GNN Teacher
            print(f"    Training GNN Teacher...", end=" ", flush=True)
            teacher_acc, teacher_logits = train_gnn_teacher(data, device)
            teacher_accs.append(teacher_acc * 100)
            print(f"Acc={teacher_acc*100:.2f}%")
            
            # Train MLP Direct
            print(f"    Training MLP Direct...", end=" ", flush=True)
            mlp_direct_acc = train_mlp_direct(data, device)
            mlp_direct_accs.append(mlp_direct_acc * 100)
            print(f"Acc={mlp_direct_acc*100:.2f}%")
            
            # Train MLP with KD
            print(f"    Training MLP + KD...", end=" ", flush=True)
            mlp_kd_acc = train_mlp_kd(data, teacher_logits, device)
            mlp_kd_accs.append(mlp_kd_acc * 100)
            print(f"Acc={mlp_kd_acc*100:.2f}%")
        
        all_results[dataset] = {
            'teacher': (np.mean(teacher_accs), np.std(teacher_accs)),
            'mlp_direct': (np.mean(mlp_direct_accs), np.std(mlp_direct_accs)),
            'mlp_kd': (np.mean(mlp_kd_accs), np.std(mlp_kd_accs)),
        }
        
        print(f"\n  Summary for {dataset}_filtered:")
        print(f"    GNN Teacher:  {all_results[dataset]['teacher'][0]:.2f}% ± {all_results[dataset]['teacher'][1]:.2f}%")
        print(f"    MLP Direct:   {all_results[dataset]['mlp_direct'][0]:.2f}% ± {all_results[dataset]['mlp_direct'][1]:.2f}%")
        print(f"    MLP + KD:     {all_results[dataset]['mlp_kd'][0]:.2f}% ± {all_results[dataset]['mlp_kd'][1]:.2f}%")
        
        kd_gain = all_results[dataset]['mlp_kd'][0] - all_results[dataset]['mlp_direct'][0]
        gap_closed = (all_results[dataset]['mlp_kd'][0] - all_results[dataset]['mlp_direct'][0]) / \
                     (all_results[dataset]['teacher'][0] - all_results[dataset]['mlp_direct'][0]) * 100 \
                     if all_results[dataset]['teacher'][0] != all_results[dataset]['mlp_direct'][0] else 0
        
        print(f"\n    KD Gain: {kd_gain:+.2f}%")
        print(f"    Gap Closed: {gap_closed:.1f}%")
    
    # Final summary
    print("\n" + "="*90)
    print("FINAL SUMMARY")
    print("="*90)
    print(f"\n{'Dataset':<20} {'Teacher':>12} {'MLP Direct':>12} {'MLP + KD':>12} {'KD Gain':>10} {'Gap Closed':>12}")
    print("-" * 80)
    
    for dataset, r in all_results.items():
        teacher = r['teacher'][0]
        direct = r['mlp_direct'][0]
        kd = r['mlp_kd'][0]
        gain = kd - direct
        gap = (kd - direct) / (teacher - direct) * 100 if teacher != direct else 0
        
        print(f"{dataset+'_filtered':<20} {teacher:>11.2f}% {direct:>11.2f}% {kd:>11.2f}% {gain:>+9.2f}% {gap:>11.1f}%")
    
    print("\n" + "="*90)
    print("CONCLUSION")
    print("="*90)
    
    # Check if KD still works
    avg_kd_gain = np.mean([all_results[d]['mlp_kd'][0] - all_results[d]['mlp_direct'][0] for d in datasets])
    
    if avg_kd_gain > 5:
        print("""
✅ KD STILL WORKS ON CLEAN DATA!

The knowledge distillation approach provides significant improvement even on
the filtered (clean) datasets where duplicate nodes have been removed.

This validates our core contribution: MLP can learn from GNN's soft predictions
to achieve competitive performance without graph structure at inference time.

The improvement is NOT due to data leakage.
""")
    elif avg_kd_gain > 1:
        print("""
⚡ KD PROVIDES MODERATE IMPROVEMENT ON CLEAN DATA

The knowledge distillation approach still helps on filtered datasets,
but the improvement is smaller than on the original datasets.

This suggests that some of the original improvement may have been due to
data leakage, but the core approach is still valid.
""")
    else:
        print("""
⚠️  KD PROVIDES MINIMAL IMPROVEMENT ON CLEAN DATA

The knowledge distillation approach does not significantly help on
the filtered (clean) datasets.

This suggests that the original results may have been largely due to
data leakage. The core contribution needs to be re-evaluated.
""")


if __name__ == '__main__':
    main()
