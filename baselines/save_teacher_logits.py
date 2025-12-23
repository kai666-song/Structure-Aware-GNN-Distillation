"""
Save Teacher Model Logits for Knowledge Distillation

This script trains the best teacher model (GloGNN++) and saves:
1. Model weights
2. Soft logits (pre-softmax outputs)
3. Soft labels (post-softmax probabilities)

These will be used in Phase 2 for knowledge distillation.
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_glognn_baseline import (
    MLP_NORM, load_geom_gcn_data, normalize, sparse_mx_to_torch_sparse_tensor,
    GLOGNN_CONFIGS, accuracy
)
import torch.optim as optim
import scipy.sparse as sp


def train_and_save_teacher(dataset_name, num_splits=10, use_cuda=False):
    """
    Train GloGNN++ teacher and save logits for each split.
    
    Args:
        dataset_name: 'actor' or 'squirrel'
        num_splits: number of splits to train
        use_cuda: whether to use GPU
    """
    config = GLOGNN_CONFIGS[dataset_name]
    
    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'checkpoints',
        f'glognn_teacher_{dataset_name}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for split_idx in range(num_splits):
        print(f"\n{'='*50}")
        print(f"Training Teacher for Split {split_idx}")
        print(f"{'='*50}")
        
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_geom_gcn_data(
            dataset_name, split_idx
        )
        
        # Normalize
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
        features = normalize(sp.csr_matrix(features))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        
        # Convert to float64 for GloGNN
        features = features.to(torch.float64)
        adj = adj.to(torch.float64)
        
        if use_cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
        
        # Initialize model
        model = MLP_NORM(
            nnodes=adj.shape[0],
            nfeat=features.shape[1],
            nhid=config['hidden'],
            nclass=int(labels.max().item()) + 1,
            dropout=config['dropout'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            delta=config['delta'],
            norm_func_id=config['norm_func_id'],
            norm_layers=config['norm_layers'],
            orders=config['orders'],
            orders_func_id=config['orders_func_id'],
            cuda=use_cuda
        )
        
        if use_cuda:
            model.cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'])
        
        # Training
        best_val_acc = 0
        best_test_acc = 0
        best_model_state = None
        best_logits = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                output = model(features, adj)
            val_acc = accuracy(output[idx_val], labels[idx_val]).item()
            test_acc = accuracy(output[idx_test], labels[idx_test]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                # Save model state (convert to float32 for storage)
                best_model_state = {k: v.float().cpu().clone() for k, v in model.state_dict().items()}
                best_logits = output.float().detach().cpu()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['early_stopping']:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Split {split_idx}: Val Acc = {best_val_acc*100:.2f}%, Test Acc = {best_test_acc*100:.2f}%")
        results.append(best_test_acc)
        
        # Save to split directory
        split_dir = os.path.join(output_dir, f'split_{split_idx}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Save model weights
        torch.save(best_model_state, os.path.join(split_dir, 'teacher_model.pth'))
        
        # Save logits (pre-softmax)
        torch.save(best_logits, os.path.join(split_dir, 'teacher_logits.pt'))
        
        # Save soft labels (post-softmax probabilities)
        soft_labels = F.softmax(best_logits, dim=1)
        torch.save(soft_labels, os.path.join(split_dir, 'teacher_soft_labels.pt'))
        
        # Save split info
        split_info = {
            'split_idx': split_idx,
            'val_acc': best_val_acc,
            'test_acc': best_test_acc,
            'num_train': len(idx_train),
            'num_val': len(idx_val),
            'num_test': len(idx_test),
            'config': config
        }
        with open(os.path.join(split_dir, 'info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Saved teacher outputs to {split_dir}")
    
    # Save summary
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    
    summary = {
        'dataset': dataset_name,
        'model': 'GloGNN++',
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'all_results': [r * 100 for r in results],
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"TEACHER TRAINING COMPLETE")
    print(f"Dataset: {dataset_name}")
    print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Saved to: {output_dir}")
    print(f"{'='*50}")
    
    return mean_acc, std_acc, results


def save_teacher_logits(dataset_name, save_dir='./data', use_cuda=False):
    """
    Train GloGNN++ teacher on split 0 and save logits for homophily computation.
    
    This is a simplified version that just saves the logits for feature generation.
    
    Args:
        dataset_name: 'actor', 'squirrel', or 'chameleon'
        save_dir: Directory to save logits
        use_cuda: Whether to use GPU
    
    Returns:
        logits: torch.Tensor of shape (N, C)
    """
    config = GLOGNN_CONFIGS[dataset_name]
    
    print(f"\n{'='*50}")
    print(f"Training GloGNN++ Teacher for {dataset_name}")
    print(f"{'='*50}")
    
    # Load data (split 0)
    adj, features, labels, idx_train, idx_val, idx_test = load_geom_gcn_data(
        dataset_name, split_idx=0
    )
    
    # Normalize
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    features = normalize(sp.csr_matrix(features))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # Convert to float64 for GloGNN
    features = features.to(torch.float64)
    adj = adj.to(torch.float64)
    
    if use_cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    
    # Initialize model
    model = MLP_NORM(
        nnodes=adj.shape[0],
        nfeat=features.shape[1],
        nhid=config['hidden'],
        nclass=int(labels.max().item()) + 1,
        dropout=config['dropout'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        delta=config['delta'],
        norm_func_id=config['norm_func_id'],
        norm_layers=config['norm_layers'],
        orders=config['orders'],
        orders_func_id=config['orders_func_id'],
        cuda=use_cuda
    )
    
    if use_cuda:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    
    # Training
    best_val_acc = 0
    best_logits = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
        val_acc = accuracy(output[idx_val], labels[idx_val]).item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_logits = output.float().detach().cpu()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save logits
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'teacher_logits_{dataset_name}.pt')
    torch.save({
        'logits': best_logits,
        'soft_labels': F.softmax(best_logits, dim=1),
        'dataset': dataset_name,
        'val_acc': best_val_acc
    }, save_path)
    
    print(f"Saved teacher logits to: {save_path}")
    print(f"Logits shape: {best_logits.shape}")
    print(f"Val accuracy: {best_val_acc*100:.2f}%")
    
    return best_logits


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Save Teacher Logits')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'squirrel', 'chameleon'])
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: only save logits for split 0')
    args = parser.parse_args()
    
    if args.quick:
        save_teacher_logits(args.dataset, use_cuda=args.cuda)
    else:
        train_and_save_teacher(args.dataset, args.num_splits, args.cuda)
