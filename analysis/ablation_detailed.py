"""
Granular Ablation Study (任务3)

进一步拆解各损失组件的贡献：
1. Only Logits KD (完全没有结构损失)
2. Logits + RKD
3. Logits + TCD (Topology Consistency)
4. Logits + RKD + TCD (Full)

不仅看最终 Accuracy，还要记录收敛速度（Epochs to converge）
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import GAT, MLPBatchNorm, convert_adj_to_edge_index
from utils import load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


class SimpleTopologyLoss(nn.Module):
    """简化的拓扑一致性损失"""
    def __init__(self, temperature=0.5, max_samples=2048):
        super().__init__()
        self.temperature = temperature
        self.max_samples = max_samples
    
    def forward(self, student_out, teacher_out, edge_index):
        src, dst = edge_index[0], edge_index[1]
        
        num_edges = len(src)
        if num_edges > self.max_samples:
            perm = torch.randperm(num_edges, device=src.device)[:self.max_samples]
            src, dst = src[perm], dst[perm]
        
        student_feat = F.normalize(F.softmax(student_out / self.temperature, dim=1), p=2, dim=1)
        teacher_feat = F.normalize(F.softmax(teacher_out / self.temperature, dim=1), p=2, dim=1)
        
        student_sim = (student_feat[src] * student_feat[dst]).sum(dim=1)
        teacher_sim = (teacher_feat[src] * teacher_feat[dst]).sum(dim=1)
        
        return F.mse_loss(student_sim, teacher_sim.detach())


def run_ablation_config(config_name, use_kd, use_rkd, use_tcd, 
                        dataset, features_sparse, adj_sparse, edge_index,
                        labels, idx_train, idx_val, idx_test,
                        teacher, nfeat, nclass, train_config, device):
    """运行单个消融配置"""
    
    student = MLPBatchNorm(nfeat, train_config['hidden'], nclass, dropout=0.5).to(device)
    
    criterion_task = nn.CrossEntropyLoss()
    criterion_kd = SoftTarget(T=4.0) if use_kd else None
    criterion_rkd = AdaptiveRKDLoss(max_samples=2048) if use_rkd else None
    criterion_tcd = SimpleTopologyLoss(max_samples=2048) if use_tcd else None
    
    optimizer = torch.optim.Adam(student.parameters(), lr=train_config['lr'],
                                 weight_decay=train_config['wd_student'])
    
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    converge_epoch = train_config['epochs']
    
    # 记录训练曲线
    train_accs = []
    val_accs = []
    
    for epoch in range(train_config['epochs']):
        student.train()
        optimizer.zero_grad()
        
        student_out = student(features_sparse, adj_sparse)
        with torch.no_grad():
            teacher_out = teacher(features_sparse, edge_index)
        
        # Task loss (always)
        loss = criterion_task(student_out[idx_train], labels[idx_train])
        
        # KD loss
        if criterion_kd:
            loss = loss + criterion_kd(student_out, teacher_out)
        
        # RKD loss
        if criterion_rkd:
            loss = loss + criterion_rkd(student_out, teacher_out, mask=idx_train)
        
        # TCD loss
        if criterion_tcd:
            loss = loss + criterion_tcd(student_out, teacher_out, edge_index)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        student.eval()
        with torch.no_grad():
            output = student(features_sparse, adj_sparse)
            train_acc = (output[idx_train].argmax(1) == labels[idx_train]).float().mean().item()
            val_acc = (output[idx_val].argmax(1) == labels[idx_val]).float().mean().item()
        
        train_accs.append(train_acc * 100)
        val_accs.append(val_acc * 100)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_counter = 0
            converge_epoch = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= train_config['patience']:
            break
    
    # Load best and test
    student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    student.eval()
    
    with torch.no_grad():
        output = student(features_sparse, adj_sparse)
        test_acc = (output[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
    
    return {
        'test_acc': test_acc,
        'converge_epoch': converge_epoch,
        'train_curve': train_accs,
        'val_curve': val_accs,
    }


def run_detailed_ablation(dataset='cora', num_runs=5, device='cuda'):
    """运行详细消融实验"""
    print(f"\n{'='*70}")
    print(f"DETAILED ABLATION STUDY: {dataset.upper()}")
    print(f"{'='*70}")
    
    # 加载数据
    adj, features, labels, *_, idx_train, idx_val, idx_test = load_data_new(dataset)
    
    features_processed = preprocess_features(features)
    supports = preprocess_adj(adj)
    
    i = torch.from_numpy(features_processed[0]).long().to(device)
    v = torch.from_numpy(features_processed[1]).to(device)
    features_sparse = torch.sparse_coo_tensor(i.t(), v, features_processed[2]).to(device)
    
    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    adj_sparse = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(device)
    
    edge_index = convert_adj_to_edge_index(adj_sparse).to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    
    nfeat = features_sparse.shape[1]
    nclass = labels.max().item() + 1
    
    train_config = {
        'hidden': 64 if dataset in ['cora', 'citeseer', 'pubmed'] else 256,
        'gat_heads': 8 if dataset in ['cora', 'citeseer', 'pubmed'] else 4,
        'epochs': 300,
        'patience': 100,
        'lr': 0.01,
        'wd_teacher': 5e-4 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
        'wd_student': 1e-5 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
    }
    
    # 消融配置
    configs = [
        ('Task Only', False, False, False),
        ('+ KD', True, False, False),
        ('+ KD + RKD', True, True, False),
        ('+ KD + TCD', True, False, True),
        ('+ KD + RKD + TCD (Full)', True, True, True),
    ]
    
    results = {name: {'accs': [], 'epochs': [], 'curves': []} for name, *_ in configs}
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        
        # 训练 Teacher
        teacher = GAT(nfeat, train_config['hidden'], nclass, dropout=0.6,
                      heads=train_config['gat_heads']).to(device)
        
        optimizer = torch.optim.Adam(teacher.parameters(), lr=train_config['lr'],
                                     weight_decay=train_config['wd_teacher'])
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(train_config['epochs']):
            teacher.train()
            optimizer.zero_grad()
            output = teacher(features_sparse, edge_index)
            loss = F.cross_entropy(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            
            teacher.eval()
            with torch.no_grad():
                output = teacher(features_sparse, edge_index)
                val_acc = (output[idx_val].argmax(1) == labels[idx_val]).float().mean()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= train_config['patience']:
                break
        
        teacher.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            teacher_out = teacher(features_sparse, edge_index)
            teacher_acc = (teacher_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
        print(f"  Teacher: {teacher_acc:.2f}%")
        
        # 运行各消融配置
        for name, use_kd, use_rkd, use_tcd in configs:
            result = run_ablation_config(
                name, use_kd, use_rkd, use_tcd,
                dataset, features_sparse, adj_sparse, edge_index,
                labels, idx_train, idx_val, idx_test,
                teacher, nfeat, nclass, train_config, device
            )
            
            results[name]['accs'].append(result['test_acc'])
            results[name]['epochs'].append(result['converge_epoch'])
            results[name]['curves'].append(result['val_curve'])
            
            print(f"  {name}: {result['test_acc']:.2f}% (converged at epoch {result['converge_epoch']})")
    
    # 汇总
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    summary = {'dataset': dataset, 'configs': {}}
    
    print(f"\n{'Config':<25} {'Accuracy':<20} {'Converge Epoch':<15}")
    print("-" * 60)
    
    for name, use_kd, use_rkd, use_tcd in configs:
        acc_mean = np.mean(results[name]['accs'])
        acc_std = np.std(results[name]['accs'])
        epoch_mean = np.mean(results[name]['epochs'])
        epoch_std = np.std(results[name]['epochs'])
        
        summary['configs'][name] = {
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'epoch_mean': epoch_mean,
            'epoch_std': epoch_std,
            'use_kd': use_kd,
            'use_rkd': use_rkd,
            'use_tcd': use_tcd,
        }
        
        print(f"{name:<25} {acc_mean:>6.2f} ± {acc_std:<6.2f}   {epoch_mean:>5.1f} ± {epoch_std:<5.1f}")
    
    # 保存
    os.makedirs('results', exist_ok=True)
    with open(f'results/ablation_detailed_{dataset}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 绘图
    plot_ablation(summary, results, dataset)
    
    return summary


def plot_ablation(summary, results, dataset):
    """绘制消融实验图"""
    configs = list(summary['configs'].keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 准确率对比
    ax1 = axes[0]
    x = np.arange(len(configs))
    means = [summary['configs'][c]['acc_mean'] for c in configs]
    stds = [summary['configs'][c]['acc_std'] for c in configs]
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']
    bars = ax1.bar(x, means, yerr=stds, capsize=4, color=colors)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title(f'Ablation Study - {dataset.upper()}', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(' + ', '\n+ ') for c in configs], fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for bar, mean in zip(bars, means):
        ax1.annotate(f'{mean:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    # 图2: 收敛速度对比
    ax2 = axes[1]
    epoch_means = [summary['configs'][c]['epoch_mean'] for c in configs]
    epoch_stds = [summary['configs'][c]['epoch_std'] for c in configs]
    
    bars2 = ax2.bar(x, epoch_means, yerr=epoch_stds, capsize=4, color=colors)
    
    ax2.set_ylabel('Epochs to Converge', fontsize=11)
    ax2.set_title(f'Convergence Speed - {dataset.upper()}', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace(' + ', '\n+ ') for c in configs], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, mean in zip(bars2, epoch_means):
        ax2.annotate(f'{mean:.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/ablation_detailed_{dataset}.png', dpi=150)
    plt.close()
    print(f"\nFigure saved to figures/ablation_detailed_{dataset}.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--num_runs', type=int, default=5)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    run_detailed_ablation(args.data, args.num_runs, device)
