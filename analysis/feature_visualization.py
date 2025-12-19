"""
Feature Space Visualization (Task 6)

Compute and visualize:
1. t-SNE of Teacher vs Student embeddings
2. Intra-class / Inter-class distance ratio
3. Davies-Bouldin Index (lower = better clustering)
4. Silhouette Score

Hypothesis: Student's feature space is more compact than Teacher's,
explaining better generalization.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from collections import defaultdict

from models import GAT, MLPBatchNorm, convert_adj_to_edge_index
from utils import load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


def compute_cluster_metrics(embeddings, labels):
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: [N, D] feature embeddings
        labels: [N] class labels
    
    Returns:
        dict with metrics
    """
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Davies-Bouldin Index (lower = better)
    try:
        db_index = davies_bouldin_score(embeddings_np, labels_np)
    except:
        db_index = float('nan')
    
    # Silhouette Score (higher = better, range [-1, 1])
    try:
        silhouette = silhouette_score(embeddings_np, labels_np)
    except:
        silhouette = float('nan')
    
    # Compute intra-class and inter-class distances manually
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    # Class centroids
    centroids = []
    for c in unique_labels:
        mask = labels == c
        centroid = embeddings[mask].mean(dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    
    # Intra-class distance (average distance to centroid within class)
    intra_distances = []
    for i, c in enumerate(unique_labels):
        mask = labels == c
        class_embeddings = embeddings[mask]
        distances = torch.norm(class_embeddings - centroids[i], dim=1)
        intra_distances.append(distances.mean().item())
    avg_intra = np.mean(intra_distances)
    
    # Inter-class distance (average distance between centroids)
    inter_distances = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = torch.norm(centroids[i] - centroids[j]).item()
            inter_distances.append(dist)
    avg_inter = np.mean(inter_distances) if inter_distances else 0
    
    # Compactness ratio (lower = more compact, better separation)
    compactness_ratio = avg_intra / (avg_inter + 1e-8)
    
    return {
        'davies_bouldin': db_index,
        'silhouette': silhouette,
        'avg_intra_distance': avg_intra,
        'avg_inter_distance': avg_inter,
        'compactness_ratio': compactness_ratio,
    }


def get_embeddings(model, features, adj_or_edge_index, model_type='mlp'):
    """Extract embeddings from model's penultimate layer."""
    model.eval()
    
    with torch.no_grad():
        if model_type == 'mlp':
            # For MLP, get output before final layer
            if features.is_sparse:
                x = features.to_dense()
            x = features.float()
            x = model.fc_in(x)
            x = model.bn_in(x)
            x = F.relu(x)
            return x
        else:
            # For GAT, get output after first layer
            if features.is_sparse:
                x = features.to_dense()
            x = features.float()
            x = F.dropout(x, p=model.dropout, training=False)
            x = model.gat1(x, adj_or_edge_index)
            x = F.elu(x)
            return x


def run_feature_analysis(dataset='actor', num_runs=3, device='cuda'):
    """Run feature space analysis."""
    print(f"\n{'='*70}")
    print(f"FEATURE SPACE ANALYSIS: {dataset.upper()}")
    print(f"{'='*70}")
    
    # Load data
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
        'hidden': 64,
        'epochs': 300,
        'patience': 100,
        'lr': 0.01,
        'wd_teacher': 5e-4,
        'wd_student': 1e-5,
    }
    
    # Collect metrics across runs
    all_metrics = {'teacher': [], 'student': []}
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        
        # Train Teacher
        teacher = GAT(nfeat, config['hidden'], nclass, dropout=0.6, heads=4).to(device)
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
        for p in teacher.parameters():
            p.requires_grad = False
        
        # Train Student
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
        
        # Get embeddings
        teacher_emb = get_embeddings(teacher, features_sparse, edge_index, 'gat')
        student_emb = get_embeddings(student, features_sparse, adj_sparse, 'mlp')
        
        # Compute metrics on test set
        test_labels = labels[idx_test]
        teacher_test_emb = teacher_emb[idx_test]
        student_test_emb = student_emb[idx_test]
        
        teacher_metrics = compute_cluster_metrics(teacher_test_emb, test_labels)
        student_metrics = compute_cluster_metrics(student_test_emb, test_labels)
        
        all_metrics['teacher'].append(teacher_metrics)
        all_metrics['student'].append(student_metrics)
        
        print(f"  Teacher - DB: {teacher_metrics['davies_bouldin']:.3f}, "
              f"Silhouette: {teacher_metrics['silhouette']:.3f}, "
              f"Compactness: {teacher_metrics['compactness_ratio']:.3f}")
        print(f"  Student - DB: {student_metrics['davies_bouldin']:.3f}, "
              f"Silhouette: {student_metrics['silhouette']:.3f}, "
              f"Compactness: {student_metrics['compactness_ratio']:.3f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("FEATURE SPACE ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    summary = {'dataset': dataset, 'num_runs': num_runs}
    
    for model_name in ['teacher', 'student']:
        metrics_list = all_metrics[model_name]
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        summary[model_name] = avg_metrics
    
    print(f"\n{'Metric':<25} {'Teacher':<20} {'Student':<20} {'Better':<10}")
    print("-" * 75)
    
    for metric in ['davies_bouldin', 'silhouette', 'compactness_ratio']:
        t_mean = summary['teacher'][metric]['mean']
        t_std = summary['teacher'][metric]['std']
        s_mean = summary['student'][metric]['mean']
        s_std = summary['student'][metric]['std']
        
        # Determine which is better
        if metric in ['davies_bouldin', 'compactness_ratio']:
            better = 'Student' if s_mean < t_mean else 'Teacher'
        else:
            better = 'Student' if s_mean > t_mean else 'Teacher'
        
        marker = '✨' if better == 'Student' else ''
        
        print(f"{metric:<25} {t_mean:>6.3f}±{t_std:<6.3f}   {s_mean:>6.3f}±{s_std:<6.3f}   {better:<10} {marker}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    t_db = summary['teacher']['davies_bouldin']['mean']
    s_db = summary['student']['davies_bouldin']['mean']
    t_compact = summary['teacher']['compactness_ratio']['mean']
    s_compact = summary['student']['compactness_ratio']['mean']
    
    if s_db < t_db:
        print(f"✓ Student has LOWER Davies-Bouldin Index ({s_db:.3f} vs {t_db:.3f})")
        print("  → Student's clusters are more separated!")
    
    if s_compact < t_compact:
        print(f"✓ Student has LOWER Compactness Ratio ({s_compact:.3f} vs {t_compact:.3f})")
        print("  → Student's feature space is more compact!")
        print("  → This explains better generalization!")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open(f'results/feature_analysis_{dataset}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualization
    generate_tsne_comparison(teacher_emb, student_emb, labels, idx_test, dataset)
    
    return summary


def generate_tsne_comparison(teacher_emb, student_emb, labels, idx_test, dataset):
    """Generate t-SNE comparison figure."""
    print("\nGenerating t-SNE visualization...")
    
    # Use test set only
    teacher_test = teacher_emb[idx_test].cpu().numpy()
    student_test = student_emb[idx_test].cpu().numpy()
    labels_test = labels[idx_test].cpu().numpy()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    teacher_2d = tsne.fit_transform(teacher_test)
    student_2d = tsne.fit_transform(student_test)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_labels = np.unique(labels_test)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for ax, emb_2d, title in [(axes[0], teacher_2d, 'Teacher (GAT)'),
                               (axes[1], student_2d, 'Student (MLP)')]:
        for i, label in enumerate(unique_labels):
            mask = labels_test == label
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                      c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle(f'Feature Space Comparison - {dataset.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/feature_tsne_{dataset}.png', dpi=150)
    plt.close()
    
    print(f"  Saved to figures/feature_tsne_{dataset}.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='actor')
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    run_feature_analysis(args.data, args.num_runs, device)
