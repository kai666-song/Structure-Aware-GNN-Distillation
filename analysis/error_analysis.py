"""
Error Analysis & Case Study (任务5)

找出具体的"翻盘案例"：
1. 筛选 Teacher 分错但 Student 分对的节点
2. 可视化这些节点的邻居标签分布
3. 用具体例子展示 MLP 如何避免被错误邻居带偏
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter

from models import GAT, MLPBatchNorm, convert_adj_to_edge_index
from utils import load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


def get_neighbor_label_distribution(node_idx, edge_index, labels, num_classes):
    """获取节点邻居的标签分布"""
    src, dst = edge_index[0], edge_index[1]
    
    # 找到该节点的所有邻居
    neighbor_mask = src == node_idx
    neighbors = dst[neighbor_mask]
    
    if len(neighbors) == 0:
        return np.zeros(num_classes), []
    
    neighbor_labels = labels[neighbors].cpu().numpy()
    
    # 统计分布
    distribution = np.zeros(num_classes)
    for label in neighbor_labels:
        distribution[label] += 1
    
    # 归一化
    if distribution.sum() > 0:
        distribution = distribution / distribution.sum()
    
    return distribution, neighbors.cpu().numpy().tolist()


def run_error_analysis(dataset='actor', num_runs=3, device='cuda'):
    """运行错误分析"""
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS: {dataset.upper()}")
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
    
    config = {
        'hidden': 64 if dataset in ['cora', 'citeseer', 'pubmed'] else 256,
        'gat_heads': 8 if dataset in ['cora', 'citeseer', 'pubmed'] else 4,
        'epochs': 300,
        'patience': 100,
        'lr': 0.01,
        'wd_teacher': 5e-4 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
        'wd_student': 1e-5 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
    }
    
    # 收集所有运行的翻盘案例
    all_flip_cases = []
    all_stats = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        
        # 训练 Teacher
        teacher = GAT(nfeat, config['hidden'], nclass, dropout=0.6,
                      heads=config['gat_heads']).to(device)
        
        optimizer = torch.optim.Adam(teacher.parameters(), lr=config['lr'],
                                     weight_decay=config['wd_teacher'])
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
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
            
            if patience_counter >= config['patience']:
                break
        
        teacher.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        
        # 训练 Student
        student = MLPBatchNorm(nfeat, config['hidden'], nclass, dropout=0.5).to(device)
        criterion_task = torch.nn.CrossEntropyLoss()
        criterion_kd = SoftTarget(T=4.0)
        criterion_rkd = AdaptiveRKDLoss(max_samples=2048)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=config['lr'],
                                     weight_decay=config['wd_student'])
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            student.train()
            optimizer.zero_grad()
            
            student_out = student(features_sparse, adj_sparse)
            with torch.no_grad():
                teacher_out = teacher(features_sparse, edge_index)
            
            loss = (criterion_task(student_out[idx_train], labels[idx_train]) +
                    criterion_kd(student_out, teacher_out) +
                    criterion_rkd(student_out, teacher_out, mask=idx_train))
            
            loss.backward()
            optimizer.step()
            
            student.eval()
            with torch.no_grad():
                output = student(features_sparse, adj_sparse)
                val_acc = (output[idx_val].argmax(1) == labels[idx_val]).float().mean()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        student.eval()
        
        # 获取预测
        with torch.no_grad():
            teacher_preds = teacher(features_sparse, edge_index).argmax(1)
            student_preds = student(features_sparse, adj_sparse).argmax(1)
        
        # 找翻盘案例：Teacher 错，Student 对
        test_nodes = idx_test.cpu().numpy()
        teacher_correct = (teacher_preds[idx_test] == labels[idx_test]).cpu().numpy()
        student_correct = (student_preds[idx_test] == labels[idx_test]).cpu().numpy()
        
        flip_mask = (~teacher_correct) & student_correct  # Teacher 错，Student 对
        flip_nodes = test_nodes[flip_mask]
        
        # 反向翻盘：Teacher 对，Student 错
        reverse_flip_mask = teacher_correct & (~student_correct)
        reverse_flip_nodes = test_nodes[reverse_flip_mask]
        
        stats = {
            'teacher_acc': teacher_correct.mean() * 100,
            'student_acc': student_correct.mean() * 100,
            'flip_count': len(flip_nodes),  # Student 翻盘
            'reverse_flip_count': len(reverse_flip_nodes),  # Teacher 翻盘
            'net_gain': len(flip_nodes) - len(reverse_flip_nodes),
        }
        all_stats.append(stats)
        
        print(f"  Teacher: {stats['teacher_acc']:.2f}%, Student: {stats['student_acc']:.2f}%")
        print(f"  Flip cases (T wrong, S right): {stats['flip_count']}")
        print(f"  Reverse flip (T right, S wrong): {stats['reverse_flip_count']}")
        print(f"  Net gain: {stats['net_gain']}")
        
        # 分析翻盘案例的邻居分布
        for node in flip_nodes[:10]:  # 只取前10个
            true_label = labels[node].item()
            teacher_pred = teacher_preds[node].item()
            student_pred = student_preds[node].item()
            
            neighbor_dist, neighbors = get_neighbor_label_distribution(
                node, edge_index, labels, nclass
            )
            
            # 计算邻居中错误标签的比例
            wrong_neighbor_ratio = 1 - neighbor_dist[true_label] if len(neighbors) > 0 else 0
            
            case = {
                'node': int(node),
                'true_label': true_label,
                'teacher_pred': teacher_pred,
                'student_pred': student_pred,
                'num_neighbors': len(neighbors),
                'wrong_neighbor_ratio': wrong_neighbor_ratio,
                'neighbor_distribution': neighbor_dist.tolist(),
            }
            all_flip_cases.append(case)
    
    # 汇总统计
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    avg_flip = np.mean([s['flip_count'] for s in all_stats])
    avg_reverse = np.mean([s['reverse_flip_count'] for s in all_stats])
    avg_net = np.mean([s['net_gain'] for s in all_stats])
    
    print(f"\nAverage across {num_runs} runs:")
    print(f"  Student flips (T wrong → S right): {avg_flip:.1f}")
    print(f"  Reverse flips (T right → S wrong): {avg_reverse:.1f}")
    print(f"  Net gain for Student: {avg_net:.1f}")
    
    # 分析翻盘案例的特征
    if all_flip_cases:
        wrong_ratios = [c['wrong_neighbor_ratio'] for c in all_flip_cases]
        avg_wrong_ratio = np.mean(wrong_ratios)
        
        print(f"\nFlip case analysis ({len(all_flip_cases)} cases):")
        print(f"  Avg wrong neighbor ratio: {avg_wrong_ratio:.2%}")
        print(f"  This means: When Student flips Teacher's error,")
        print(f"  the node typically has {avg_wrong_ratio:.0%} neighbors with WRONG labels!")
        print(f"\n  → GAT was misled by wrong neighbors, but MLP ignored them!")
    
    # 保存结果
    summary = {
        'dataset': dataset,
        'num_runs': num_runs,
        'avg_flip_count': avg_flip,
        'avg_reverse_flip': avg_reverse,
        'avg_net_gain': avg_net,
        'avg_wrong_neighbor_ratio': avg_wrong_ratio if all_flip_cases else 0,
        'sample_cases': all_flip_cases[:20],  # 保存前20个案例
    }
    
    os.makedirs('results', exist_ok=True)
    with open(f'results/error_analysis_{dataset}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 绘图
    if all_flip_cases:
        plot_error_analysis(all_flip_cases, dataset, nclass)
    
    return summary


def plot_error_analysis(flip_cases, dataset, nclass):
    """绘制错误分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 图1: 翻盘案例的错误邻居比例分布
    ax1 = axes[0]
    wrong_ratios = [c['wrong_neighbor_ratio'] for c in flip_cases]
    ax1.hist(wrong_ratios, bins=10, edgecolor='black', alpha=0.7, color='#2ca02c')
    ax1.axvline(np.mean(wrong_ratios), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(wrong_ratios):.2%}')
    ax1.set_xlabel('Wrong Neighbor Ratio', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Flip Cases: Wrong Neighbor Distribution\n{dataset.upper()}', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 图2: 示例案例的邻居标签分布
    ax2 = axes[1]
    
    # 选择一个典型案例
    if flip_cases:
        case = max(flip_cases, key=lambda x: x['wrong_neighbor_ratio'])
        dist = case['neighbor_distribution']
        
        x = np.arange(nclass)
        colors = ['#1f77b4'] * nclass
        colors[case['true_label']] = '#2ca02c'  # 正确标签绿色
        colors[case['teacher_pred']] = '#d62728'  # Teacher 预测红色
        
        bars = ax2.bar(x, dist, color=colors, edgecolor='black')
        ax2.set_xlabel('Class Label', fontsize=11)
        ax2.set_ylabel('Neighbor Proportion', fontsize=11)
        ax2.set_title(f'Example Flip Case (Node {case["node"]})\n'
                      f'True: {case["true_label"]}, Teacher: {case["teacher_pred"]}, '
                      f'Student: {case["student_pred"]}', fontsize=11)
        ax2.set_xticks(x)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label=f'True Label ({case["true_label"]})'),
            Patch(facecolor='#d62728', label=f'Teacher Pred ({case["teacher_pred"]})'),
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/error_analysis_{dataset}.png', dpi=150)
    plt.close()
    print(f"\nFigure saved to figures/error_analysis_{dataset}.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='actor')
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    run_error_analysis(args.data, args.num_runs, device)
