"""
Micro-Level Error Analysis: Accuracy by Homophily Ratio
=======================================================

This script analyzes model performance across different homophily bins
to prove that our method excels on heterophilic (low-homophily) nodes.

Key Question: Does our Student outperform Teacher on difficult nodes?

Output:
- Accuracy breakdown by homophily bins
- Visualization (bar chart)
- Statistical comparison

Usage:
    python analysis/homophily_breakdown.py --dataset actor
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EnhancedMLP
from utils.data_utils import load_data_new


def compute_node_homophily(adj, labels):
    """
    Compute per-node homophily based on ground truth labels.
    
    h_i = fraction of neighbors with same label as node i
    """
    adj_coo = adj.tocoo()
    row, col = adj_coo.row, adj_coo.col
    
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    num_nodes = len(labels_np)
    
    match_count = np.zeros(num_nodes, dtype=np.float32)
    degree = np.zeros(num_nodes, dtype=np.float32)
    
    for r, c in zip(row, col):
        degree[r] += 1
        if labels_np[r] == labels_np[c]:
            match_count[r] += 1
    
    degree[degree == 0] = 1
    homophily = match_count / degree
    
    return homophily


def load_model_predictions(dataset, model_type='student', split_idx=0):
    """
    Load or compute model predictions.
    
    For Teacher: use saved logits
    For Student: load trained model and compute
    """
    if model_type == 'teacher':
        # Load teacher logits
        path = f'./data/teacher_logits_{dataset}.pt'
        data = torch.load(path)
        logits = data.get('logits', data) if isinstance(data, dict) else data
        return logits.argmax(dim=1).numpy()
    
    else:  # student
        # Load data
        adj, features, labels, *_, train_mask, val_mask, test_mask, *_ = \
            load_data_new(dataset, split_idx=split_idx, use_pe=False)
        
        # Add PE
        pe_path = f'./data/pe_rw_{dataset}.pt'
        if os.path.exists(pe_path):
            pe = torch.load(pe_path)['pe']
            if hasattr(features, 'todense'):
                features = torch.FloatTensor(np.array(features.todense()))
            features = torch.cat([features, pe], dim=1)
        
        # Load model
        model_path = f'./checkpoints/{dataset}_split{split_idx}_best.pt'
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return None
        
        num_classes = int(labels.max().item()) + 1
        model = EnhancedMLP(
            nfeat=features.shape[1],
            nhid=256,
            nclass=num_classes,
            dropout=0.5,
            num_layers=3
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        with torch.no_grad():
            logits = model(features)
            preds = logits.argmax(dim=1).numpy()
        
        return preds


def analyze_by_homophily(dataset, num_bins=5, split_idx=0):
    """
    Analyze accuracy by homophily bins.
    """
    # Load data
    adj, features, labels, *_, train_mask, val_mask, test_mask, *_ = \
        load_data_new(dataset, split_idx=split_idx, use_pe=False)
    
    labels_np = labels.numpy()
    test_mask_np = test_mask if isinstance(test_mask, np.ndarray) else test_mask.numpy()
    
    # Compute node homophily
    node_homophily = compute_node_homophily(adj, labels)
    
    # Load predictions
    teacher_preds = load_model_predictions(dataset, 'teacher', split_idx)
    student_preds = load_model_predictions(dataset, 'student', split_idx)
    
    if student_preds is None:
        print("Student model not found. Running with teacher only.")
        student_preds = teacher_preds
    
    # Define bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_labels = [f'[{bins[i]:.1f}, {bins[i+1]:.1f})' for i in range(num_bins)]
    
    # Analyze each bin
    results = {
        'bins': bin_labels,
        'teacher_acc': [],
        'student_acc': [],
        'node_counts': [],
        'homophily_means': []
    }
    
    for i in range(num_bins):
        low, high = bins[i], bins[i+1]
        
        # Find test nodes in this bin
        if i == num_bins - 1:  # Last bin includes upper bound
            bin_mask = (node_homophily >= low) & (node_homophily <= high) & test_mask_np
        else:
            bin_mask = (node_homophily >= low) & (node_homophily < high) & test_mask_np
        
        bin_indices = np.where(bin_mask)[0]
        
        if len(bin_indices) == 0:
            results['teacher_acc'].append(0)
            results['student_acc'].append(0)
            results['node_counts'].append(0)
            results['homophily_means'].append((low + high) / 2)
            continue
        
        # Compute accuracy
        teacher_correct = (teacher_preds[bin_indices] == labels_np[bin_indices]).sum()
        student_correct = (student_preds[bin_indices] == labels_np[bin_indices]).sum()
        
        teacher_acc = teacher_correct / len(bin_indices) * 100
        student_acc = student_correct / len(bin_indices) * 100
        
        results['teacher_acc'].append(teacher_acc)
        results['student_acc'].append(student_acc)
        results['node_counts'].append(len(bin_indices))
        results['homophily_means'].append(node_homophily[bin_indices].mean())
    
    return results


def plot_homophily_analysis(results, dataset, save_path=None):
    """Create bar chart comparing Teacher vs Student by homophily."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results['bins']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, results['teacher_acc'], width, 
                   label='GloGNN++ (Teacher)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, results['student_acc'], width,
                   label='Our Method (Student)', color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for bar, count in zip(bars1, results['node_counts']):
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Node Homophily Ratio', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Accuracy by Homophily Ratio - {dataset.upper()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(results['bins'], rotation=15)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add node counts as secondary info
    ax2 = ax.twinx()
    ax2.plot(x, results['node_counts'], 'g--', marker='o', alpha=0.5, label='Node Count')
    ax2.set_ylabel('Node Count', color='green', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.close()
    
    return fig


def run_multi_split_analysis(dataset, num_splits=10, num_bins=5):
    """Run analysis across multiple splits and aggregate."""
    
    all_results = []
    
    for split_idx in range(num_splits):
        print(f"Analyzing split {split_idx}...")
        results = analyze_by_homophily(dataset, num_bins, split_idx)
        all_results.append(results)
    
    # Aggregate results
    aggregated = {
        'bins': all_results[0]['bins'],
        'teacher_acc_mean': [],
        'teacher_acc_std': [],
        'student_acc_mean': [],
        'student_acc_std': [],
        'node_counts': all_results[0]['node_counts'],  # Same across splits
        'improvements': []
    }
    
    for i in range(num_bins):
        teacher_accs = [r['teacher_acc'][i] for r in all_results]
        student_accs = [r['student_acc'][i] for r in all_results]
        
        aggregated['teacher_acc_mean'].append(np.mean(teacher_accs))
        aggregated['teacher_acc_std'].append(np.std(teacher_accs))
        aggregated['student_acc_mean'].append(np.mean(student_accs))
        aggregated['student_acc_std'].append(np.std(student_accs))
        aggregated['improvements'].append(np.mean(student_accs) - np.mean(teacher_accs))
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Homophily Breakdown Analysis')
    parser.add_argument('--dataset', type=str, default='actor')
    parser.add_argument('--num_bins', type=int, default=5)
    parser.add_argument('--num_splits', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./figures')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Homophily Breakdown Analysis - {args.dataset.upper()}")
    print(f"{'='*60}")
    
    if args.num_splits > 1:
        results = run_multi_split_analysis(args.dataset, args.num_splits, args.num_bins)
    else:
        results = analyze_by_homophily(args.dataset, args.num_bins, split_idx=0)
        # Convert to aggregated format
        results = {
            'bins': results['bins'],
            'teacher_acc_mean': results['teacher_acc'],
            'student_acc_mean': results['student_acc'],
            'node_counts': results['node_counts'],
            'improvements': [s - t for s, t in zip(results['student_acc'], results['teacher_acc'])]
        }
    
    # Print results
    print("\nResults by Homophily Bin:")
    print("-" * 70)
    print(f"{'Bin':<15} {'Teacher':<12} {'Student':<12} {'Δ':<10} {'Nodes':<10}")
    print("-" * 70)
    
    for i, bin_label in enumerate(results['bins']):
        t_acc = results['teacher_acc_mean'][i]
        s_acc = results['student_acc_mean'][i]
        delta = results['improvements'][i]
        nodes = results['node_counts'][i]
        
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        print(f"{bin_label:<15} {t_acc:.1f}%{'':<6} {s_acc:.1f}%{'':<6} {delta_str:<10} {nodes:<10}")
    
    print("-" * 70)
    
    # Key insight
    low_homo_improvement = np.mean(results['improvements'][:2])  # First 2 bins
    high_homo_improvement = np.mean(results['improvements'][-2:])  # Last 2 bins
    
    print(f"\nKey Insight:")
    print(f"  Low homophily (h < 0.4) improvement: {low_homo_improvement:+.1f}%")
    print(f"  High homophily (h > 0.6) improvement: {high_homo_improvement:+.1f}%")
    
    # Save results
    save_path = f'./results/homophily_breakdown_{args.dataset}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    # Plot
    plot_results = {
        'bins': results['bins'],
        'teacher_acc': results['teacher_acc_mean'],
        'student_acc': results['student_acc_mean'],
        'node_counts': results['node_counts']
    }
    fig_path = os.path.join(args.save_dir, f'homophily_breakdown_{args.dataset}.png')
    plot_homophily_analysis(plot_results, args.dataset, fig_path)


if __name__ == '__main__':
    main()
