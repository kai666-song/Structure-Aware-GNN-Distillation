"""
Visualization Script: t-SNE Feature Visualization

Creates visualizations comparing:
1. Teacher GCN embeddings
2. Student MLP (Distilled) embeddings
3. MLP Baseline (without distillation) embeddings
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

from models import GCN, MLP
from utils import load_data_new, preprocess_features, preprocess_adj


def get_embeddings(model, features, adj):
    """Extract model output embeddings."""
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
    return output.cpu().numpy()


def plot_tsne(embeddings, labels, title, save_path, figsize=(8, 6)):
    """Create t-SNE visualization."""
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='tab10',
        alpha=0.7,
        s=10
    )
    plt.colorbar(scatter, label='Class')
    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_comparison(teacher_emb, student_emb, labels, dataset, save_dir):
    """Create side-by-side comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # t-SNE for both
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    
    teacher_2d = tsne.fit_transform(teacher_emb)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    student_2d = tsne.fit_transform(student_emb)
    
    # Teacher plot
    scatter1 = axes[0].scatter(
        teacher_2d[:, 0], teacher_2d[:, 1],
        c=labels, cmap='tab10', alpha=0.7, s=10
    )
    axes[0].set_title(f'Teacher GCN - {dataset.upper()}', fontsize=14)
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    
    # Student plot
    scatter2 = axes[1].scatter(
        student_2d[:, 0], student_2d[:, 1],
        c=labels, cmap='tab10', alpha=0.7, s=10
    )
    axes[1].set_title(f'Student MLP (Distilled) - {dataset.upper()}', fontsize=14)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    save_path = f'{save_dir}/tsne_comparison_{dataset}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_training_curves(dataset, save_dir):
    """Plot training loss and accuracy curves."""
    import json
    
    history_file = f'checkpoints/{dataset}/history_seed0.json'
    if not os.path.exists(history_file):
        print(f'History file not found: {history_file}')
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs_t = range(1, len(history['teacher']['train_loss']) + 1)
    epochs_s = range(1, len(history['student']['train_loss']) + 1)
    
    axes[0].plot(epochs_t, history['teacher']['train_loss'], label='Teacher Loss', alpha=0.8)
    axes[0].plot(epochs_s, history['student']['train_loss'], label='Student Total Loss', alpha=0.8)
    axes[0].plot(epochs_s, history['student']['task_loss'], label='Task Loss', alpha=0.6, linestyle='--')
    axes[0].plot(epochs_s, history['student']['kd_loss'], label='KD Loss', alpha=0.6, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss - {dataset.upper()}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs_t, history['teacher']['val_acc'], label='Teacher Val Acc', alpha=0.8)
    axes[1].plot(epochs_s, history['student']['val_acc'], label='Student Val Acc', alpha=0.8)
    axes[1].axhline(y=history['teacher']['test_acc'], color='blue', linestyle=':', 
                    label=f"Teacher Test: {history['teacher']['test_acc']:.1f}%")
    axes[1].axhline(y=history['student']['test_acc'], color='orange', linestyle=':', 
                    label=f"Student Test: {history['student']['test_acc']:.1f}%")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Validation Accuracy - {dataset.upper()}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/training_curves_{dataset}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', help='Visualize all datasets')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Create output directory
    save_dir = 'figures'
    os.makedirs(save_dir, exist_ok=True)
    
    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
    else:
        datasets = [args.data]
    
    for dataset in datasets:
        print(f'\n{"="*50}')
        print(f'Visualizing {dataset.upper()}')
        print(f'{"="*50}')
        
        # Load data
        adj, features, labels, *_ = load_data_new(dataset)
        num_nodes = features.shape[0]
        nfeat = features.shape[1]
        nclass = labels.max().item() + 1
        labels_np = labels.numpy()
        
        # Preprocess
        features = preprocess_features(features)
        supports = preprocess_adj(adj)
        
        i = torch.from_numpy(features[0]).long().to(device)
        v = torch.from_numpy(features[1]).to(device)
        features = torch.sparse_coo_tensor(i.t(), v, features[2]).to(device)
        
        i = torch.from_numpy(supports[0]).long().to(device)
        v = torch.from_numpy(supports[1]).to(device)
        adj = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(device)
        
        # Get hidden dim
        hidden = 256 if 'amazon' in dataset else 64
        
        # Load models
        checkpoint_dir = f'checkpoints/{dataset}'
        
        if not os.path.exists(f'{checkpoint_dir}/teacher_seed0.pt'):
            print(f'Checkpoints not found for {dataset}, skipping...')
            continue
        
        # Teacher
        teacher = GCN(nfeat, hidden, nclass, dropout=0.5).to(device)
        teacher.load_state_dict(torch.load(f'{checkpoint_dir}/teacher_seed0.pt', map_location=device))
        
        # Student (Distilled)
        student = MLP(nfeat, hidden, nclass, dropout=0.5).to(device)
        student.load_state_dict(torch.load(f'{checkpoint_dir}/student_seed0.pt', map_location=device))
        
        # Get embeddings
        teacher_emb = get_embeddings(teacher, features, adj)
        student_emb = get_embeddings(student, features, adj)
        
        # For large datasets, subsample for visualization
        if num_nodes > 5000:
            np.random.seed(42)
            idx = np.random.choice(num_nodes, 5000, replace=False)
            teacher_emb = teacher_emb[idx]
            student_emb = student_emb[idx]
            labels_np = labels_np[idx]
            print(f'Subsampled to 5000 nodes for visualization')
        
        # Create visualizations
        plot_tsne(teacher_emb, labels_np, f'Teacher GCN - {dataset.upper()}', 
                  f'{save_dir}/tsne_teacher_{dataset}.png')
        plot_tsne(student_emb, labels_np, f'Student MLP (Distilled) - {dataset.upper()}', 
                  f'{save_dir}/tsne_student_{dataset}.png')
        plot_comparison(teacher_emb, student_emb, labels_np, dataset, save_dir)
        plot_training_curves(dataset, save_dir)
    
    print(f'\nAll visualizations saved to {save_dir}/')


if __name__ == '__main__':
    main()
