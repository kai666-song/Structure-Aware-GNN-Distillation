"""
从训练好的模型中提取数据并生成图表
===================================

这个脚本会：
1. 训练模型并提取θ_k系数
2. 提取特征用于t-SNE可视化
3. 计算Dirichlet Energy
4. 生成基于真实数据的图表

Usage:
    python scripts/extract_and_plot.py --dataset chameleon
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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kd_losses.adaptive_kd import GatedAFDLoss, AFDLoss

# 设置图表样式
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'paper')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_filtered_data(dataset, split_idx, device):
    """Load filtered dataset."""
    data_dir = '../../heterophilous-graphs/data'
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


# =============================================================================
# Models
# =============================================================================

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
        h = F.relu(self.norm2(self.fc2(x)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.fc3(h), h  # 返回logits和特征
    
    def get_features(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        h = F.relu(self.norm2(self.fc2(x)))
        return h


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
        return self.output_proj(x), x  # 返回logits和特征


# =============================================================================
# Training and Extraction
# =============================================================================

def train_and_extract(data, device, epochs=300):
    """训练模型并提取θ_k系数和特征"""
    
    # 1. 训练Teacher
    print("Training GNN Teacher...")
    teacher = GCNWithSkip(data['num_features'], 256, data['num_classes']).to(device)
    optimizer = optim.AdamW(teacher.parameters(), lr=0.01, weight_decay=1e-5)
    
    best_val_acc, best_logits, best_features_t = 0, None, None
    for epoch in range(epochs):
        teacher.train()
        optimizer.zero_grad()
        logits, features = teacher(data['features'], data['adj'])
        loss = F.cross_entropy(logits[data['train_mask']], data['labels'][data['train_mask']])
        loss.backward()
        optimizer.step()
        
        teacher.eval()
        with torch.no_grad():
            logits, features = teacher(data['features'], data['adj'])
            val_acc = (logits.argmax(1)[data['val_mask']] == data['labels'][data['val_mask']]).float().mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_logits = logits.detach()
                best_features_t = features.detach()
    
    teacher_logits = best_logits
    teacher_features = best_features_t
    print(f"  Teacher Val Acc: {best_val_acc*100:.2f}%")
    
    # 2. 训练Gated AFD Student
    print("Training Gated AFD Student...")
    student = SimpleMLP(data['num_features'], 256, data['num_classes']).to(device)
    gated_loss = GatedAFDLoss(
        adj=data['adj_raw'], K=5, temperature=4.0, lambda_kd=1.0,
        gate_init_threshold=0.3, gate_sharpness=5.0,
        learnable_gate=True, device=device
    )
    
    optimizer = optim.Adam([
        {'params': student.parameters(), 'lr': 0.01, 'weight_decay': 5e-4},
        {'params': gated_loss.parameters(), 'lr': 0.005}
    ])
    
    best_val_acc, best_features_s, best_theta = 0, None, None
    for epoch in range(epochs):
        student.train()
        gated_loss.train()
        optimizer.zero_grad()
        logits, features = student(data['features'])
        loss, _ = gated_loss(logits, teacher_logits, data['labels'], data['train_mask'])
        loss.backward()
        optimizer.step()
        
        student.eval()
        with torch.no_grad():
            logits, features = student(data['features'])
            val_acc = (logits.argmax(1)[data['val_mask']] == data['labels'][data['val_mask']]).float().mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_features_s = features.detach()
                # 提取θ_k
                best_theta = gated_loss.afd_loss.spectral_filter.theta.detach().cpu().numpy()
    
    print(f"  Student Val Acc: {best_val_acc*100:.2f}%")
    
    return {
        'teacher_features': teacher_features.cpu().numpy(),
        'student_features': best_features_s.cpu().numpy(),
        'theta_k': best_theta,
        'labels': data['labels'].cpu().numpy(),
        'test_mask': data['test_mask'].cpu().numpy(),
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_theta_k(theta_k, dataset_name, save_path):
    """绘制θ_k系数分布"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    K = len(theta_k)
    k_values = np.arange(K)
    
    # 归一化
    theta_norm = np.abs(theta_k) / np.abs(theta_k).sum()
    
    colors = ['#3498db' if i < K//2 else '#e74c3c' for i in range(K)]
    bars = ax.bar(k_values, theta_norm, color=colors, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Polynomial Order k')
    ax.set_ylabel('Normalized Coefficient |θ_k|')
    ax.set_title(f'Learned Spectral Filter Coefficients ({dataset_name})')
    ax.set_xticks(k_values)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Low-frequency'),
                      Patch(facecolor='#e74c3c', label='High-frequency')]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tsne_comparison(teacher_features, student_features, labels, test_mask, save_path):
    """绘制t-SNE对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 只使用测试集节点
    teacher_test = teacher_features[test_mask]
    student_test = student_features[test_mask]
    labels_test = labels[test_mask]
    
    # 降采样（如果节点太多）
    max_samples = 500
    if len(labels_test) > max_samples:
        idx = np.random.choice(len(labels_test), max_samples, replace=False)
        teacher_test = teacher_test[idx]
        student_test = student_test[idx]
        labels_test = labels_test[idx]
    
    # t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    teacher_2d = tsne.fit_transform(teacher_test)
    student_2d = tsne.fit_transform(student_test)
    
    # 绘图
    n_classes = len(np.unique(labels_test))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for ax, (data_2d, title) in zip(axes, [(teacher_2d, 'GNN Teacher'), (student_2d, 'Gated AFD Student')]):
        for i in range(n_classes):
            mask = labels_test == i
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1], c=[colors[i]], 
                      alpha=0.6, s=20, label=f'Class {i}')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    
    axes[0].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chameleon',
                       choices=['squirrel', 'chameleon'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"\n{'='*60}")
    print(f"Extracting data from {args.dataset}_filtered")
    print(f"{'='*60}\n")
    
    # 加载数据
    data = load_filtered_data(args.dataset, split_idx=0, device=device)
    print(f"Loaded: {data['num_nodes']} nodes, {data['num_classes']} classes")
    
    # 训练并提取
    results = train_and_extract(data, device)
    
    # 生成图表
    print("\nGenerating plots...")
    
    # θ_k分布图
    plot_theta_k(
        results['theta_k'], 
        args.dataset.capitalize(),
        os.path.join(OUTPUT_DIR, f'theta_k_{args.dataset}.pdf')
    )
    
    # t-SNE对比图
    plot_tsne_comparison(
        results['teacher_features'],
        results['student_features'],
        results['labels'],
        results['test_mask'],
        os.path.join(OUTPUT_DIR, f'tsne_{args.dataset}.pdf')
    )
    
    print(f"\n✓ All plots saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
