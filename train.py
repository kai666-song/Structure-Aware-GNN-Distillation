"""
SOTA Spectral-Decoupled Knowledge Distillation Trainer
======================================================

This is the main training script for Phase 3.

Key Features:
1. NO GNN forward pass during training - uses pre-saved Teacher logits
2. Spectral-decoupled adaptive loss
3. Enhanced MLP with residual connections
4. Positional encoding support

Usage:
    # Basic run on Actor
    python run_sota.py --dataset actor --num_runs 10
    
    # With custom hyperparameters
    python run_sota.py --dataset actor --lambda_spectral 1.0 --alpha_high 2.0
    
    # Quick test (1 run)
    python run_sota.py --dataset actor --num_runs 1 --epochs 200

Author: Structure-Aware GNN KD Project
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

# Local imports
from models import EnhancedMLP, ResMLP, MLPBatchNorm
from utils.data_utils import load_data_new
from kd_losses.adaptive_kd import HybridAdaptiveLoss, AdaptiveSpectralKDLoss


def load_teacher_logits(dataset, split_idx, data_dir='./data'):
    """
    Load pre-saved teacher logits.
    
    First tries split-specific logits, then falls back to general logits.
    """
    # Try split-specific path first
    split_path = os.path.join(
        'checkpoints', f'glognn_teacher_{dataset}', 
        f'split_{split_idx}', 'teacher_logits.pt'
    )
    
    if os.path.exists(split_path):
        logits = torch.load(split_path)
        if isinstance(logits, dict):
            logits = logits.get('logits', logits.get('soft_labels', list(logits.values())[0]))
        print(f"Loaded split-specific teacher logits from {split_path}")
        return logits
    
    # Fall back to general logits
    general_path = os.path.join(data_dir, f'teacher_logits_{dataset}.pt')
    if os.path.exists(general_path):
        data = torch.load(general_path)
        logits = data.get('logits', data) if isinstance(data, dict) else data
        print(f"Loaded general teacher logits from {general_path}")
        return logits
    
    raise FileNotFoundError(
        f"Teacher logits not found for {dataset}!\n"
        f"Run: python baselines/save_teacher_logits.py --dataset {dataset}"
    )


def load_homophily_weights(dataset, data_dir='./data'):
    """Load pre-computed homophily weights."""
    path = os.path.join(data_dir, f'homophily_weights_{dataset}.pt')
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Homophily weights not found: {path}\n"
            f"Run: python features/generate_homophily.py --dataset {dataset} --hard"
        )
    
    data = torch.load(path)
    return data['homophily']


def load_positional_encoding(dataset, data_dir='./data'):
    """Load pre-computed positional encoding."""
    path = os.path.join(data_dir, f'pe_rw_{dataset}.pt')
    
    if not os.path.exists(path):
        print(f"Warning: PE not found at {path}, running without PE")
        return None
    
    data = torch.load(path)
    return data['pe']


def accuracy(output, labels):
    """Compute accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def train_epoch(model, optimizer, features, labels, teacher_logits,
                loss_fn, train_mask, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (NO graph structure needed!)
    logits = model(features)
    
    # Compute hybrid loss
    loss, loss_dict = loss_fn(
        logits, teacher_logits, labels, 
        train_mask=train_mask, compute_all=True
    )
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Training accuracy
    with torch.no_grad():
        train_acc = accuracy(logits[train_mask], labels[train_mask])
    
    return loss.item(), train_acc.item(), loss_dict


@torch.no_grad()
def evaluate(model, features, labels, mask):
    """Evaluate model."""
    model.eval()
    logits = model(features)
    acc = accuracy(logits[mask], labels[mask])
    return acc.item()


def run_single_experiment(args, split_idx=0):
    """
    Run a single experiment on one split.
    
    Returns:
        test_acc: Test accuracy
        best_val_acc: Best validation accuracy
        history: Training history
    """
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Split {split_idx} | Device: {device}")
    print(f"{'='*60}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    
    # Load graph data
    adj, features, labels, y_train, y_val, y_test, \
        train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = \
        load_data_new(args.dataset, split_idx=split_idx, use_pe=False)
    
    # Convert features to dense tensor
    if hasattr(features, 'todense'):
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(features)
    
    # Load and concatenate positional encoding
    if args.use_pe:
        pe = load_positional_encoding(args.dataset)
        if pe is not None:
            print(f"Original features: {features.shape}")
            features = torch.cat([features, pe], dim=1)
            print(f"With PE: {features.shape}")
    
    # Load teacher logits
    teacher_logits = load_teacher_logits(args.dataset, split_idx)
    
    # Load homophily weights
    homophily = load_homophily_weights(args.dataset)
    
    # Convert masks to tensors
    if isinstance(train_mask, np.ndarray):
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    # Move to device
    features = features.to(device)
    labels = labels.to(device)
    teacher_logits = teacher_logits.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    
    num_features = features.shape[1]
    num_classes = int(labels.max().item()) + 1
    
    print(f"Features: {features.shape}")
    print(f"Classes: {num_classes}")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    # =========================================================================
    # 2. Initialize Model
    # =========================================================================
    
    if args.model == 'enhanced':
        model = EnhancedMLP(
            nfeat=num_features,
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout,
            num_layers=args.num_layers,
            use_residual=True,
            norm_type='layer'
        )
    elif args.model == 'resmlp':
        model = ResMLP(
            nfeat=num_features,
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout,
            num_layers=args.num_layers
        )
    else:  # 'mlp'
        model = MLPBatchNorm(
            nfeat=num_features,
            nhid=args.hidden,
            nclass=num_classes,
            dropout=args.dropout,
            num_layers=args.num_layers
        )
    
    model = model.to(device)
    print(f"Model: {args.model} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # =========================================================================
    # 3. Initialize Loss Function
    # =========================================================================
    
    loss_fn = HybridAdaptiveLoss(
        adj=adj,
        homophily_weights=homophily,
        lambda_spectral=args.lambda_spectral,
        lambda_soft=args.lambda_soft,
        temperature=args.temperature,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        high_freq_scale=args.high_freq_scale,
        device=device
    )
    
    # =========================================================================
    # 4. Training Loop
    # =========================================================================
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}
    
    for epoch in range(args.epochs):
        # Train
        loss, train_acc, loss_dict = train_epoch(
            model, optimizer, features, labels, teacher_logits,
            loss_fn, train_mask, device
        )
        
        # Evaluate
        val_acc = evaluate(model, features, labels, val_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        
        # Record history
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            if args.save_model:
                save_path = os.path.join(
                    args.checkpoint_dir, 
                    f'{args.dataset}_split{split_idx}_best.pt'
                )
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        
        # Logging
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | "
                  f"Test: {test_acc*100:.2f}% | Best: {best_test_acc*100:.2f}%")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nBest: Epoch {best_epoch} | Val: {best_val_acc*100:.2f}% | Test: {best_test_acc*100:.2f}%")
    
    return best_test_acc, best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='SOTA Spectral KD Training')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'squirrel', 'chameleon', 'cora', 'citeseer', 'pubmed'])
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs (splits for heterophilic datasets)')
    
    # Model
    parser.add_argument('--model', type=str, default='enhanced',
                       choices=['enhanced', 'resmlp', 'mlp'],
                       help='Student model architecture')
    parser.add_argument('--hidden', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Features
    parser.add_argument('--use_pe', action='store_true', default=True,
                       help='Use positional encoding')
    parser.add_argument('--no_pe', action='store_true',
                       help='Disable positional encoding')
    
    # Loss weights
    parser.add_argument('--lambda_spectral', type=float, default=1.0,
                       help='Weight for spectral KD loss')
    parser.add_argument('--lambda_soft', type=float, default=0.5,
                       help='Weight for standard soft target loss')
    parser.add_argument('--alpha_low', type=float, default=1.0,
                       help='Weight for low-frequency loss')
    parser.add_argument('--alpha_high', type=float, default=1.5,
                       help='Weight for high-frequency loss')
    parser.add_argument('--high_freq_scale', type=float, default=2.0,
                       help='Scaling factor for high-freq loss')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='KD temperature')
    
    # Training
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=100,
                       help='Early stopping patience')
    parser.add_argument('--log_every', type=int, default=50,
                       help='Log every N epochs')
    
    # System
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_model', action='store_true',
                       help='Save best model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Handle PE flag
    if args.no_pe:
        args.use_pe = False
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Print configuration
    print("="*70)
    print("SOTA Spectral-Decoupled Knowledge Distillation")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} (hidden={args.hidden}, layers={args.num_layers})")
    print(f"Use PE: {args.use_pe}")
    print(f"Lambda Spectral: {args.lambda_spectral}, Lambda Soft: {args.lambda_soft}")
    print(f"Alpha Low: {args.alpha_low}, Alpha High: {args.alpha_high}")
    print(f"Temperature: {args.temperature}")
    print("="*70)
    
    # Run experiments
    all_test_accs = []
    all_val_accs = []
    
    for run_idx in range(args.num_runs):
        test_acc, val_acc, history = run_single_experiment(args, split_idx=run_idx)
        all_test_accs.append(test_acc)
        all_val_accs.append(val_acc)
    
    # Summary
    mean_test = np.mean(all_test_accs) * 100
    std_test = np.std(all_test_accs) * 100
    mean_val = np.mean(all_val_accs) * 100
    std_val = np.std(all_val_accs) * 100
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Test Accuracy: {mean_test:.2f}% ± {std_test:.2f}%")
    print(f"Val Accuracy:  {mean_val:.2f}% ± {std_val:.2f}%")
    print(f"All Test Accs: {[f'{a*100:.2f}' for a in all_test_accs]}")
    print("="*70)
    
    # Save results
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'use_pe': args.use_pe,
        'mean_test_acc': mean_test,
        'std_test_acc': std_test,
        'mean_val_acc': mean_val,
        'std_val_acc': std_val,
        'all_test_accs': [a * 100 for a in all_test_accs],
        'all_val_accs': [a * 100 for a in all_val_accs],
        'config': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(
        args.results_dir, 
        f'sota_{args.dataset}_{args.model}_pe{args.use_pe}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
