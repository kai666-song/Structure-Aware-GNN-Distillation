"""
Node-level Homophily Analysis (任务4)

验证假设："Student 在异配区域修正了 Teacher 的错误"

分析内容：
1. 计算每个测试节点的局部同配率 (Local Homophily Ratio)
2. 将节点按同配率分组 (0.0-0.2, 0.2-0.4, ..., 0.8-1.0)
3. 对比 Teacher 和 Student 在各区间的准确率
4. 可视化结果

预期：在低同配率区间 (0.0-0.2)，Student 准确率应显著高于 Teacher
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from models import GAT, MLPBatchNorm, convert_adj_to_edge_index
from utils import load_data_new, preprocess_features, preprocess_adj


def compute_local_homophily(edge_index, labels, num_nodes):
    """
    计算每个节点的局部同配率
    
    Local Homophily = 同类邻居数 / 总邻居数
    
    Returns:
        homophily: [N] tensor, 每个节点的局部同配率
    """
    src, dst = edge_index[0], edge_index[1]
    
    # 统计每个节点的邻居数和同类邻居数
    neighbor_count = torch.zeros(num_nodes, device=edge_index.device)
    same_label_count = torch.zeros(num_nodes, device=edge_index.device)
    
    # 对于每条边 (src -> dst)，检查 src 和 dst 是否同类
    same_label = (labels[src] == labels[dst]).float()
    
    # 累加到 src 节点
    neighbor_count.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
    same_label_count.scatter_add_(0, src, same_label)
    
    # 计算同配率 (避免除零)
    homophily = same_label_count / (neighbor_count + 1e-8)
    homophily[neighbor_count == 0] = 0.5  # 孤立节点设为中性值
    
    return homophily


def analyze_by_homophily_bins(homophily, predictions, labels, mask, bins=5):
    """
    按同配率区间分析准确率
    
    Args:
        homophily: [N] 每个节点的同配率
        predictions: [N] 预测标签
        labels: [N] 真实标签
        mask: 测试集 mask
        bins: 区间数量
    
    Returns:
        bin_edges: 区间边界
        accuracies: 每个区间的准确率
        counts: 每个区间的节点数
    """
    # 只看测试集
    test_homophily = homophily[mask].cpu().numpy()
    test_preds = predictions[mask].cpu().numpy()
    test_labels = labels[mask].cpu().numpy()
    
    # 定义区间
    bin_edges = np.linspace(0, 1, bins + 1)
    accuracies = []
    counts = []
    
    for i in range(bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        
        # 找到该区间的节点
        if i == bins - 1:  # 最后一个区间包含右边界
            in_bin = (test_homophily >= low) & (test_homophily <= high)
        else:
            in_bin = (test_homophily >= low) & (test_homophily < high)
        
        count = in_bin.sum()
        counts.append(count)
        
        if count > 0:
            correct = (test_preds[in_bin] == test_labels[in_bin]).sum()
            acc = correct / count * 100
        else:
            acc = 0
        
        accuracies.append(acc)
    
    return bin_edges, accuracies, counts


def run_homophily_analysis(dataset='actor', num_runs=5, device='cuda'):
    """
    运行完整的同配性分析
    """
    print(f"\n{'='*70}")
    print(f"HOMOPHILY ANALYSIS: {dataset.upper()}")
    print(f"{'='*70}")
    
    # 加载数据
    adj, features, labels, *_, idx_train, idx_val, idx_test = load_data_new(dataset)
    
    # 预处理
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
    idx_test = idx_test.to(device)
    
    nfeat = features_sparse.shape[1]
    nclass = labels.max().item() + 1
    num_nodes = features_sparse.shape[0]
    
    # 计算全局和局部同配率
    local_homophily = compute_local_homophily(edge_index, labels, num_nodes)
    global_homophily = local_homophily.mean().item()
    
    print(f"\nDataset Statistics:")
    print(f"  Nodes: {num_nodes}, Edges: {edge_index.shape[1]}")
    print(f"  Global Homophily: {global_homophily:.4f}")
    print(f"  Test nodes: {len(idx_test)}")
    
    # 多次运行收集结果
    all_teacher_accs = defaultdict(list)
    all_student_accs = defaultdict(list)
    
    config = {
        'hidden': 64 if dataset in ['cora', 'citeseer', 'pubmed'] else 256,
        'gat_heads': 8 if dataset in ['cora', 'citeseer', 'pubmed'] else 4,
        'epochs': 300,
        'patience': 100,
        'lr': 0.01,
        'wd_teacher': 5e-4 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
        'wd_student': 1e-5 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
    }
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        
        # 初始化模型
        teacher = GAT(nfeat, config['hidden'], nclass, dropout=0.6, 
                      heads=config['gat_heads']).to(device)
        student = MLPBatchNorm(nfeat, config['hidden'], nclass, dropout=0.5).to(device)
        
        # 训练 Teacher
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
        
        # 训练 Student (with KD)
        from kd_losses import SoftTarget, AdaptiveRKDLoss
        criterion_task = torch.nn.CrossEntropyLoss()
        criterion_kd = SoftTarget(T=4.0)
        criterion_rkd = AdaptiveRKDLoss(max_samples=2048)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=config['lr'],
                                     weight_decay=config['wd_student'])
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for param in teacher.parameters():
            param.requires_grad = False
        
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
        
        # 获取预测结果
        with torch.no_grad():
            teacher_preds = teacher(features_sparse, edge_index).argmax(1)
            student_preds = student(features_sparse, adj_sparse).argmax(1)
        
        # 按同配率区间分析
        bins = 5
        _, teacher_accs, counts = analyze_by_homophily_bins(
            local_homophily, teacher_preds, labels, idx_test, bins=bins
        )
        _, student_accs, _ = analyze_by_homophily_bins(
            local_homophily, student_preds, labels, idx_test, bins=bins
        )
        
        for i in range(bins):
            all_teacher_accs[i].append(teacher_accs[i])
            all_student_accs[i].append(student_accs[i])
        
        # 打印本次运行结果
        teacher_total = (teacher_preds[idx_test] == labels[idx_test]).float().mean() * 100
        student_total = (student_preds[idx_test] == labels[idx_test]).float().mean() * 100
        print(f"  Teacher: {teacher_total:.2f}%, Student: {student_total:.2f}%")
    
    # 汇总结果
    print(f"\n{'='*70}")
    print("HOMOPHILY ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    bin_labels = ['0.0-0.2\n(Heterophilic)', '0.2-0.4', '0.4-0.6', 
                  '0.6-0.8', '0.8-1.0\n(Homophilic)']
    
    results = {
        'dataset': dataset,
        'global_homophily': global_homophily,
        'bins': [],
    }
    
    print(f"\n{'Bin':<20} {'Teacher':<15} {'Student':<15} {'Gap':<10} {'Count':<10}")
    print("-" * 70)
    
    for i in range(bins):
        t_mean = np.mean(all_teacher_accs[i])
        t_std = np.std(all_teacher_accs[i])
        s_mean = np.mean(all_student_accs[i])
        s_std = np.std(all_student_accs[i])
        gap = s_mean - t_mean
        
        results['bins'].append({
            'range': f'{i*0.2:.1f}-{(i+1)*0.2:.1f}',
            'teacher_mean': t_mean,
            'teacher_std': t_std,
            'student_mean': s_mean,
            'student_std': s_std,
            'gap': gap,
            'count': int(counts[i])
        })
        
        gap_str = f"{gap:+.2f}%" if gap != 0 else "0.00%"
        winner = "✨" if gap > 2 else ""
        print(f"{bin_labels[i]:<20} {t_mean:>6.2f}±{t_std:<6.2f} {s_mean:>6.2f}±{s_std:<6.2f} {gap_str:<10} {counts[i]:<10} {winner}")
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    with open(f'results/homophily_analysis_{dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘图
    plot_homophily_analysis(results, dataset)
    
    return results


def plot_homophily_analysis(results, dataset):
    """绘制同配性分析图"""
    bins = results['bins']
    x = np.arange(len(bins))
    width = 0.35
    
    teacher_means = [b['teacher_mean'] for b in bins]
    student_means = [b['student_mean'] for b in bins]
    teacher_stds = [b['teacher_std'] for b in bins]
    student_stds = [b['student_std'] for b in bins]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, teacher_means, width, yerr=teacher_stds,
                   label='Teacher (GAT)', color='#ff7f0e', capsize=3)
    bars2 = ax.bar(x + width/2, student_means, width, yerr=student_stds,
                   label='Student (MLP)', color='#1f77b4', capsize=3)
    
    ax.set_xlabel('Local Homophily Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy by Local Homophily - {dataset.upper()}\n'
                 f'(Global Homophily: {results["global_homophily"]:.3f})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([b['range'] for b in bins])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 标注 gap
    for i, (t, s) in enumerate(zip(teacher_means, student_means)):
        gap = s - t
        if abs(gap) > 1:
            color = 'green' if gap > 0 else 'red'
            ax.annotate(f'{gap:+.1f}%', xy=(i, max(t, s) + 3),
                       ha='center', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/homophily_analysis_{dataset}.png', dpi=150)
    plt.close()
    print(f"\nFigure saved to figures/homophily_analysis_{dataset}.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='actor')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--all', action='store_true', help='Run on all heterophilic datasets')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if args.all:
        datasets = ['actor', 'squirrel', 'chameleon', 'cora', 'citeseer']
        for dataset in datasets:
            try:
                run_homophily_analysis(dataset, args.num_runs, device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    else:
        run_homophily_analysis(args.data, args.num_runs, device)
