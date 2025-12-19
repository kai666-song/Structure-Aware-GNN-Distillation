"""
Robustness/Perturbation Study (任务2)

证明 MLP 在推理阶段不需要图结构是一个巨大的优势

实验设计：
1. 在测试集的图结构中加入随机噪声（随机增删 10%-50% 的边）
2. 对比 Teacher GNN 和 Student MLP 的性能变化
3. 预期：Teacher 性能大幅下降，Student 保持不变

这将是论文中非常有力的一张对比图！
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import GAT, MLPBatchNorm, convert_adj_to_edge_index
from utils import load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


def perturb_edges(edge_index, num_nodes, perturbation_ratio, mode='both'):
    """
    对图结构进行扰动
    
    Args:
        edge_index: [2, E] 原始边
        num_nodes: 节点数
        perturbation_ratio: 扰动比例 (0.0-1.0)
        mode: 'add' (只加边), 'remove' (只删边), 'both' (同时加删)
    
    Returns:
        perturbed_edge_index: 扰动后的边
    """
    device = edge_index.device
    num_edges = edge_index.shape[1]
    num_perturb = int(num_edges * perturbation_ratio)
    
    if num_perturb == 0:
        return edge_index
    
    # 转换为集合便于操作
    edge_set = set()
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_set.add((src, dst))
    
    if mode in ['remove', 'both']:
        # 随机删除边
        edges_to_remove = num_perturb // 2 if mode == 'both' else num_perturb
        edges_list = list(edge_set)
        remove_indices = np.random.choice(len(edges_list), 
                                          min(edges_to_remove, len(edges_list)), 
                                          replace=False)
        for idx in remove_indices:
            edge_set.discard(edges_list[idx])
    
    if mode in ['add', 'both']:
        # 随机添加边
        edges_to_add = num_perturb // 2 if mode == 'both' else num_perturb
        added = 0
        max_attempts = edges_to_add * 10
        attempts = 0
        
        while added < edges_to_add and attempts < max_attempts:
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst and (src, dst) not in edge_set:
                edge_set.add((src, dst))
                added += 1
            attempts += 1
    
    # 转换回 tensor
    if len(edge_set) == 0:
        return edge_index
    
    edges = list(edge_set)
    src = torch.tensor([e[0] for e in edges], device=device)
    dst = torch.tensor([e[1] for e in edges], device=device)
    
    return torch.stack([src, dst], dim=0)


def run_robustness_study(dataset='cora', num_runs=5, device='cuda'):
    """
    运行鲁棒性实验
    """
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS STUDY: {dataset.upper()}")
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
    
    edge_index_clean = convert_adj_to_edge_index(adj_sparse).to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    
    nfeat = features_sparse.shape[1]
    nclass = labels.max().item() + 1
    num_nodes = features_sparse.shape[0]
    
    print(f"\nDataset: {dataset}")
    print(f"  Nodes: {num_nodes}, Edges: {edge_index_clean.shape[1]}")
    
    # 配置
    config = {
        'hidden': 64 if dataset in ['cora', 'citeseer', 'pubmed'] else 256,
        'gat_heads': 8 if dataset in ['cora', 'citeseer', 'pubmed'] else 4,
        'epochs': 300,
        'patience': 100,
        'lr': 0.01,
        'wd_teacher': 5e-4 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
        'wd_student': 1e-5 if dataset in ['cora', 'citeseer', 'pubmed'] else 0,
    }
    
    # 扰动比例
    perturbation_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # 存储结果
    results = {
        'dataset': dataset,
        'perturbation_ratios': perturbation_ratios,
        'teacher_accs': {r: [] for r in perturbation_ratios},
        'student_accs': {r: [] for r in perturbation_ratios},
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
        
        # ========== 训练 Teacher (在干净图上) ==========
        optimizer = torch.optim.Adam(teacher.parameters(), lr=config['lr'],
                                     weight_decay=config['wd_teacher'])
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            teacher.train()
            optimizer.zero_grad()
            output = teacher(features_sparse, edge_index_clean)
            loss = F.cross_entropy(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            
            teacher.eval()
            with torch.no_grad():
                output = teacher(features_sparse, edge_index_clean)
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
        
        # ========== 训练 Student (with KD) ==========
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
                teacher_out = teacher(features_sparse, edge_index_clean)
            
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
        
        # ========== 测试不同扰动比例 ==========
        for ratio in perturbation_ratios:
            # 扰动图结构
            if ratio > 0:
                edge_index_perturbed = perturb_edges(edge_index_clean, num_nodes, ratio, mode='both')
            else:
                edge_index_perturbed = edge_index_clean
            
            with torch.no_grad():
                # Teacher 使用扰动后的图
                teacher_out = teacher(features_sparse, edge_index_perturbed)
                teacher_acc = (teacher_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
                
                # Student 完全不使用图结构！
                student_out = student(features_sparse, None)  # adj=None
                student_acc = (student_out[idx_test].argmax(1) == labels[idx_test]).float().mean().item() * 100
            
            results['teacher_accs'][ratio].append(teacher_acc)
            results['student_accs'][ratio].append(student_acc)
        
        # 打印本次结果
        print(f"  Clean graph - Teacher: {results['teacher_accs'][0.0][-1]:.2f}%, "
              f"Student: {results['student_accs'][0.0][-1]:.2f}%")
        print(f"  50% perturb - Teacher: {results['teacher_accs'][0.5][-1]:.2f}%, "
              f"Student: {results['student_accs'][0.5][-1]:.2f}%")
    
    # ========== 汇总结果 ==========
    print(f"\n{'='*70}")
    print("ROBUSTNESS STUDY RESULTS")
    print(f"{'='*70}")
    
    summary = {
        'dataset': dataset,
        'perturbation_ratios': perturbation_ratios,
        'teacher': {},
        'student': {},
    }
    
    print(f"\n{'Perturbation':<15} {'Teacher':<20} {'Student':<20} {'Gap':<10}")
    print("-" * 65)
    
    for ratio in perturbation_ratios:
        t_mean = np.mean(results['teacher_accs'][ratio])
        t_std = np.std(results['teacher_accs'][ratio])
        s_mean = np.mean(results['student_accs'][ratio])
        s_std = np.std(results['student_accs'][ratio])
        
        summary['teacher'][str(ratio)] = {'mean': t_mean, 'std': t_std}
        summary['student'][str(ratio)] = {'mean': s_mean, 'std': s_std}
        
        gap = s_mean - t_mean
        print(f"{ratio*100:>5.0f}%          {t_mean:>6.2f} ± {t_std:<6.2f}   "
              f"{s_mean:>6.2f} ± {s_std:<6.2f}   {gap:+.2f}%")
    
    # 计算性能下降
    teacher_drop = summary['teacher']['0.0']['mean'] - summary['teacher']['0.5']['mean']
    student_drop = summary['student']['0.0']['mean'] - summary['student']['0.5']['mean']
    
    print(f"\nPerformance Drop (0% → 50% perturbation):")
    print(f"  Teacher (GAT): -{teacher_drop:.2f}%")
    print(f"  Student (MLP): -{student_drop:.2f}%")
    print(f"\n✨ Student is {teacher_drop - student_drop:.2f}% MORE ROBUST than Teacher!")
    
    summary['teacher_drop'] = teacher_drop
    summary['student_drop'] = student_drop
    summary['robustness_gain'] = teacher_drop - student_drop
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    with open(f'results/robustness_{dataset}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 绘图
    plot_robustness(summary, dataset)
    
    return summary


def plot_robustness(results, dataset):
    """绘制鲁棒性对比图"""
    ratios = results['perturbation_ratios']
    x = [r * 100 for r in ratios]
    
    teacher_means = [results['teacher'][str(r)]['mean'] for r in ratios]
    teacher_stds = [results['teacher'][str(r)]['std'] for r in ratios]
    student_means = [results['student'][str(r)]['mean'] for r in ratios]
    student_stds = [results['student'][str(r)]['std'] for r in ratios]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(x, teacher_means, yerr=teacher_stds, marker='o', markersize=8,
                linewidth=2, capsize=4, label='Teacher (GAT) - Uses Graph', color='#ff7f0e')
    ax.errorbar(x, student_means, yerr=student_stds, marker='s', markersize=8,
                linewidth=2, capsize=4, label='Student (MLP) - Graph-Free', color='#1f77b4')
    
    ax.set_xlabel('Edge Perturbation Ratio (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Robustness to Graph Perturbation - {dataset.upper()}\n'
                 f'(Teacher drops {results["teacher_drop"]:.1f}%, '
                 f'Student drops {results["student_drop"]:.1f}%)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    
    # 添加注释
    ax.annotate(f'MLP is {results["robustness_gain"]:.1f}%\nmore robust!',
                xy=(40, (teacher_means[-2] + student_means[-2]) / 2),
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/robustness_{dataset}.png', dpi=150)
    plt.close()
    print(f"\nFigure saved to figures/robustness_{dataset}.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'actor']
        for dataset in datasets:
            try:
                run_robustness_study(dataset, args.num_runs, device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    else:
        run_robustness_study(args.data, args.num_runs, device)
