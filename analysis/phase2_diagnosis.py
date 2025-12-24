"""
Phase 2: Core Pathology Diagnosis
=================================

This script performs 4 diagnostic tasks to identify why Spectral KD 
underperforms GLNN baseline:

1. Spectral Learnability Verification
2. Loss Weight Balancing Test  
3. Component Ablation (Low vs High frequency)
4. Homophily Weight Quality Check

Usage:
    python analysis/phase2_diagnosis.py --task all --device cuda
    python analysis/phase2_diagnosis.py --task spectral --device cuda
    python analysis/phase2_diagnosis.py --task balance --device cuda
    python analysis/phase2_diagnosis.py --task ablation --device cuda
    python analysis/phase2_diagnosis.py --task homophily --device cuda
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader_v2 import load_data_with_glognn_splits
from models import EnhancedMLP, MLPBatchNorm
from configs.experiment_config import NUM_SPLITS
import scipy.sparse as sp


def load_teacher_logits(dataset, split_idx):
    """Load teacher logits."""
    path = os.path.join(
        'checkpoints', f'glognn_teacher_{dataset}',
        f'split_{split_idx}', 'teacher_logits.pt'
    )
    data = torch.load(path)
    logits = data.get('logits', data) if isinstance(data, dict) else data
    return logits.float()


def load_homophily_weights(dataset):
    """Load homophily weights."""
    path = os.path.join('data', f'homophily_weights_{dataset}.pt')
    data = torch.load(path)
    return data['homophily']


def load_pe(dataset):
    """Load positional encoding."""
    path = os.path.join('data', f'pe_rw_{dataset}.pt')
    if os.path.exists(path):
        data = torch.load(path)
        return data['pe']
    return None


def compute_normalized_adj(adj_raw):
    """Compute D^{-1}A (row-normalized adjacency)."""
    if not sp.isspmatrix_csr(adj_raw):
        adj_raw = adj_raw.tocsr()
    
    degree = np.array(adj_raw.sum(axis=1)).flatten()
    degree[degree == 0] = 1
    d_inv = 1.0 / degree
    d_inv_diag = sp.diags(d_inv, format='csr')
    A_norm = d_inv_diag @ adj_raw
    
    # Convert to torch sparse
    A_coo = A_norm.tocoo()
    indices = torch.LongTensor(np.vstack([A_coo.row, A_coo.col]))
    values = torch.FloatTensor(A_coo.data)
    return torch.sparse_coo_tensor(indices, values, A_coo.shape)


def compute_spectral_errors(Z_student, Z_teacher, A_norm, device):
    """
    Compute Low-Frequency and High-Frequency errors.
    
    LF-Error: MSE(A_norm @ Z_S, A_norm @ Z_T)
    HF-Error: MSE((I - A_norm) @ Z_S, (I - A_norm) @ Z_T)
    """
    A_norm = A_norm.to(device)
    Z_s = Z_student.to(device)
    Z_t = Z_teacher.to(device)
    
    # Low-frequency: A_norm @ Z
    Z_s_low = torch.sparse.mm(A_norm, Z_s)
    Z_t_low = torch.sparse.mm(A_norm, Z_t)
    
    # High-frequency: Z - A_norm @ Z = (I - A_norm) @ Z
    Z_s_high = Z_s - Z_s_low
    Z_t_high = Z_t - Z_t_low
    
    # MSE errors
    lf_error = F.mse_loss(Z_s_low, Z_t_low).item()
    hf_error = F.mse_loss(Z_s_high, Z_t_high).item()
    
    return lf_error, hf_error


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)



# =============================================================================
# TASK 1: Spectral Learnability Verification
# =============================================================================

def train_model_and_get_logits(dataset, split_idx, model_type, config, device):
    """
    Train a model and return final logits on test set.
    
    model_type: 'glnn' or 'spectral'
    """
    # Load data
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    # Add PE for spectral model
    if model_type == 'spectral':
        pe = load_pe(dataset)
        if pe is not None:
            features = torch.cat([features, pe.to(device)], dim=1)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    
    # Initialize model
    model = EnhancedMLP(
        nfeat=features.shape[1],
        nhid=config['hidden'],
        nclass=data['num_classes'],
        dropout=config['dropout'],
        num_layers=config['num_layers'],
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                          weight_decay=config['weight_decay'])
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    T = config['temperature']
    
    best_val_acc = 0
    best_logits = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        # CE loss
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        # KD loss (soft target)
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        if model_type == 'spectral':
            # Add spectral loss
            A_norm = compute_normalized_adj(data['adj_raw']).to(device)
            Z_s_low = torch.sparse.mm(A_norm, logits)
            Z_t_low = torch.sparse.mm(A_norm, teacher_logits)
            Z_s_high = logits - Z_s_low
            Z_t_high = teacher_logits - Z_t_low
            
            homophily = load_homophily_weights(dataset).squeeze().to(device)
            
            # Low-freq KL
            p_s_low = F.log_softmax(Z_s_low / T, dim=1)
            p_t_low = F.softmax(Z_t_low / T, dim=1)
            loss_low = F.kl_div(p_s_low, p_t_low, reduction='none').sum(dim=1) * (T * T)
            
            # High-freq MSE
            loss_high = ((Z_s_high - Z_t_high) ** 2).mean(dim=1) * 2.0
            
            loss_spectral = (homophily * loss_low + (1 - homophily) * loss_high).mean()
            
            loss = loss_ce + config['lambda_soft'] * loss_kd + config['lambda_spectral'] * loss_spectral
        else:
            loss = loss_ce + config['lambda_kd'] * loss_kd
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_logits = logits.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    return best_logits, data


def task1_spectral_learnability(datasets, device):
    """
    Task 1: Verify if MLP can learn high-frequency signals.
    """
    print("\n" + "=" * 70)
    print("TASK 1: Spectral Learnability Verification")
    print("=" * 70)
    
    glnn_config = {
        'hidden': 256, 'num_layers': 2, 'dropout': 0.5,
        'lr': 0.01, 'weight_decay': 5e-4, 'temperature': 4.0,
        'lambda_kd': 1.0, 'epochs': 500, 'patience': 100,
    }
    
    spectral_config = {
        'hidden': 256, 'num_layers': 3, 'dropout': 0.5,
        'lr': 0.01, 'weight_decay': 5e-4, 'temperature': 4.0,
        'lambda_soft': 0.5, 'lambda_spectral': 1.0,
        'epochs': 500, 'patience': 100,
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        
        glnn_lf_errors, glnn_hf_errors = [], []
        spec_lf_errors, spec_hf_errors = [], []
        glnn_accs, spec_accs = [], []
        
        for split_idx in range(min(3, NUM_SPLITS)):  # Quick test with 3 splits
            print(f"  Split {split_idx}...")
            
            # Train GLNN
            glnn_logits, data = train_model_and_get_logits(
                dataset, split_idx, 'glnn', glnn_config, device
            )
            
            # Train Spectral KD
            spec_logits, _ = train_model_and_get_logits(
                dataset, split_idx, 'spectral', spectral_config, device
            )
            
            # Load teacher logits
            teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
            
            # Compute normalized adjacency
            A_norm = compute_normalized_adj(data['adj_raw'])
            
            # Compute spectral errors on TEST SET
            test_mask = data['test_mask']
            
            glnn_lf, glnn_hf = compute_spectral_errors(
                glnn_logits[test_mask], teacher_logits[test_mask], 
                A_norm[test_mask][:, test_mask] if False else A_norm,  # Use full graph
                device
            )
            spec_lf, spec_hf = compute_spectral_errors(
                spec_logits[test_mask], teacher_logits[test_mask],
                A_norm, device
            )
            
            glnn_lf_errors.append(glnn_lf)
            glnn_hf_errors.append(glnn_hf)
            spec_lf_errors.append(spec_lf)
            spec_hf_errors.append(spec_hf)
            
            # Accuracy
            labels = data['labels'].to(device)
            glnn_acc = accuracy(glnn_logits[test_mask], labels[test_mask]).item()
            spec_acc = accuracy(spec_logits[test_mask], labels[test_mask]).item()
            glnn_accs.append(glnn_acc)
            spec_accs.append(spec_acc)
        
        results[dataset] = {
            'glnn': {
                'lf_error': np.mean(glnn_lf_errors),
                'hf_error': np.mean(glnn_hf_errors),
                'accuracy': np.mean(glnn_accs) * 100,
            },
            'spectral': {
                'lf_error': np.mean(spec_lf_errors),
                'hf_error': np.mean(spec_hf_errors),
                'accuracy': np.mean(spec_accs) * 100,
            }
        }
        
        print(f"\n  Results for {dataset}:")
        print(f"  {'Method':<12} {'LF-Error':<12} {'HF-Error':<12} {'Accuracy':<10}")
        print(f"  {'-'*46}")
        print(f"  {'GLNN':<12} {results[dataset]['glnn']['lf_error']:<12.4f} "
              f"{results[dataset]['glnn']['hf_error']:<12.4f} "
              f"{results[dataset]['glnn']['accuracy']:<10.2f}%")
        print(f"  {'Spectral':<12} {results[dataset]['spectral']['lf_error']:<12.4f} "
              f"{results[dataset]['spectral']['hf_error']:<12.4f} "
              f"{results[dataset]['spectral']['accuracy']:<10.2f}%")
    
    return results



# =============================================================================
# TASK 2: Loss Weight Balancing Test
# =============================================================================

def train_with_config(dataset, split_idx, lambda_soft, lambda_spectral, device):
    """Train with specific loss weights and return test accuracy."""
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    # Add PE
    pe = load_pe(dataset)
    if pe is not None:
        features = torch.cat([features, pe.to(device)], dim=1)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    homophily = load_homophily_weights(dataset).squeeze().to(device)
    A_norm = compute_normalized_adj(data['adj_raw']).to(device)
    
    model = EnhancedMLP(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=3,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    T = 4.0
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        # CE loss
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        # Soft target KD loss
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_soft = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        # Spectral loss
        if lambda_spectral > 0:
            Z_s_low = torch.sparse.mm(A_norm, logits)
            Z_t_low = torch.sparse.mm(A_norm, teacher_logits)
            Z_s_high = logits - Z_s_low
            Z_t_high = teacher_logits - Z_t_low
            
            p_s_low = F.log_softmax(Z_s_low / T, dim=1)
            p_t_low = F.softmax(Z_t_low / T, dim=1)
            loss_low = F.kl_div(p_s_low, p_t_low, reduction='none').sum(dim=1) * (T * T)
            loss_high = ((Z_s_high - Z_t_high) ** 2).mean(dim=1) * 2.0
            
            loss_spectral = (homophily * loss_low + (1 - homophily) * loss_high).mean()
        else:
            loss_spectral = 0
        
        loss = loss_ce + lambda_soft * loss_soft + lambda_spectral * loss_spectral
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
        
        if patience_counter >= 100:
            break
    
    return best_test_acc.item()


def task2_loss_balancing(datasets, device):
    """
    Task 2: Grid search for optimal loss weights.
    """
    print("\n" + "=" * 70)
    print("TASK 2: Loss Weight Balancing Test")
    print("=" * 70)
    print("Fixed: lambda_soft = 1.0")
    print("Varying: lambda_spectral = [0.0, 0.1, 0.5, 1.0]")
    
    lambda_spectral_values = [0.0, 0.1, 0.5, 1.0]
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        results[dataset] = {}
        
        for lam_spec in lambda_spectral_values:
            accs = []
            for split_idx in range(min(3, NUM_SPLITS)):
                acc = train_with_config(dataset, split_idx, 
                                       lambda_soft=1.0, 
                                       lambda_spectral=lam_spec, 
                                       device=device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[dataset][lam_spec] = {'mean': mean_acc, 'std': std_acc}
            print(f"  lambda_spectral={lam_spec}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Analyze trend
        accs_list = [results[dataset][lam]['mean'] for lam in lambda_spectral_values]
        if accs_list[-1] < accs_list[0]:
            if any(accs_list[i] > accs_list[0] for i in range(1, len(accs_list)-1)):
                trend = "Inverted-U (optimal at intermediate value)"
            else:
                trend = "Monotonic DECREASE (Spectral Loss is HARMFUL)"
        else:
            trend = "Monotonic INCREASE (Spectral Loss helps)"
        
        print(f"  Trend: {trend}")
        results[dataset]['trend'] = trend
    
    return results



# =============================================================================
# TASK 3: Component Ablation (Low vs High frequency)
# =============================================================================

def train_with_ablation(dataset, split_idx, variant, device):
    """
    Train with ablated spectral loss.
    
    variant: 'only_low', 'only_high', 'both'
    """
    data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
    
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)
    
    pe = load_pe(dataset)
    if pe is not None:
        features = torch.cat([features, pe.to(device)], dim=1)
    
    teacher_logits = load_teacher_logits(dataset, split_idx).to(device)
    homophily = load_homophily_weights(dataset).squeeze().to(device)
    A_norm = compute_normalized_adj(data['adj_raw']).to(device)
    
    model = EnhancedMLP(
        nfeat=features.shape[1], nhid=256, nclass=data['num_classes'],
        dropout=0.5, num_layers=3,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    T = 4.0
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        logits = model(features)
        
        # CE loss
        loss_ce = ce_loss(logits[train_mask], labels[train_mask])
        
        # Soft target KD loss
        p_s = F.log_softmax(logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        loss_soft = F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)
        
        # Spectral components
        Z_s_low = torch.sparse.mm(A_norm, logits)
        Z_t_low = torch.sparse.mm(A_norm, teacher_logits)
        Z_s_high = logits - Z_s_low
        Z_t_high = teacher_logits - Z_t_low
        
        p_s_low = F.log_softmax(Z_s_low / T, dim=1)
        p_t_low = F.softmax(Z_t_low / T, dim=1)
        loss_low = F.kl_div(p_s_low, p_t_low, reduction='none').sum(dim=1) * (T * T)
        loss_high = ((Z_s_high - Z_t_high) ** 2).mean(dim=1) * 2.0
        
        # Ablation variants
        if variant == 'only_low':
            loss_spectral = (homophily * loss_low).mean()
        elif variant == 'only_high':
            loss_spectral = ((1 - homophily) * loss_high).mean()
        else:  # 'both'
            loss_spectral = (homophily * loss_low + (1 - homophily) * loss_high).mean()
        
        loss = loss_ce + 1.0 * loss_soft + 0.5 * loss_spectral
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(features)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
        
        if patience_counter >= 100:
            break
    
    return best_test_acc.item()


def task3_component_ablation(datasets, device):
    """
    Task 3: Ablation study on Low vs High frequency components.
    """
    print("\n" + "=" * 70)
    print("TASK 3: Component Ablation (Low vs High Frequency)")
    print("=" * 70)
    print("Base: lambda_soft = 1.0, lambda_spectral = 0.5")
    
    variants = ['only_low', 'only_high', 'both']
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        results[dataset] = {}
        
        for variant in variants:
            accs = []
            for split_idx in range(min(3, NUM_SPLITS)):
                acc = train_with_ablation(dataset, split_idx, variant, device)
                accs.append(acc * 100)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            results[dataset][variant] = {'mean': mean_acc, 'std': std_acc}
            print(f"  {variant:<12}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        # Analysis
        low_acc = results[dataset]['only_low']['mean']
        high_acc = results[dataset]['only_high']['mean']
        both_acc = results[dataset]['both']['mean']
        
        if high_acc < low_acc and high_acc < both_acc:
            analysis = "HIGH-FREQ IS HARMFUL - MLP cannot learn sharp signals"
        elif high_acc > low_acc:
            analysis = "HIGH-FREQ HELPS - heterophilic benefit"
        else:
            analysis = "MIXED - both components contribute"
        
        print(f"  Analysis: {analysis}")
        results[dataset]['analysis'] = analysis
    
    return results



# =============================================================================
# TASK 4: Homophily Weight Quality Check
# =============================================================================

def task4_homophily_quality(datasets, device):
    """
    Task 4: Check homophily weight quality and per-group performance.
    """
    print("\n" + "=" * 70)
    print("TASK 4: Homophily Weight Quality Check")
    print("=" * 70)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} ---")
        
        # Load homophily weights
        homophily = load_homophily_weights(dataset).squeeze()
        
        # Statistics
        h_mean = homophily.mean().item()
        h_std = homophily.std().item()
        
        # Distribution
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        distribution = {}
        for low, high in bins:
            mask = (homophily >= low) & (homophily < high)
            count = mask.sum().item()
            pct = count / len(homophily) * 100
            distribution[f'{low}-{high}'] = {'count': count, 'pct': pct}
        
        print(f"  Homophily stats: mean={h_mean:.4f}, std={h_std:.4f}")
        print(f"  Distribution:")
        for bin_name, stats in distribution.items():
            print(f"    h ∈ [{bin_name}): {stats['count']:5d} nodes ({stats['pct']:5.1f}%)")
        
        # Strong heterophilic nodes (h < 0.2)
        hetero_mask = homophily < 0.2
        hetero_count = hetero_mask.sum().item()
        hetero_pct = hetero_count / len(homophily) * 100
        
        print(f"\n  Strong heterophilic nodes (h < 0.2): {hetero_count} ({hetero_pct:.1f}%)")
        
        # Per-group accuracy analysis
        print(f"\n  Per-group accuracy analysis (3 splits):")
        
        group_results = defaultdict(lambda: {'glnn': [], 'spectral': []})
        
        for split_idx in range(min(3, NUM_SPLITS)):
            data = load_data_with_glognn_splits(dataset, split_idx, dtype=torch.float32)
            features = data['features'].to(device)
            labels = data['labels'].to(device)
            test_mask = data['test_mask']
            
            # Train GLNN
            glnn_config = {
                'hidden': 256, 'num_layers': 2, 'dropout': 0.5,
                'lr': 0.01, 'weight_decay': 5e-4, 'temperature': 4.0,
                'lambda_kd': 1.0, 'epochs': 500, 'patience': 100,
            }
            glnn_logits, _ = train_model_and_get_logits(
                dataset, split_idx, 'glnn', glnn_config, device
            )
            
            # Train Spectral
            spectral_config = {
                'hidden': 256, 'num_layers': 3, 'dropout': 0.5,
                'lr': 0.01, 'weight_decay': 5e-4, 'temperature': 4.0,
                'lambda_soft': 0.5, 'lambda_spectral': 1.0,
                'epochs': 500, 'patience': 100,
            }
            spec_logits, _ = train_model_and_get_logits(
                dataset, split_idx, 'spectral', spectral_config, device
            )
            
            # Per-group accuracy
            for bin_name, (low, high) in zip(
                ['h<0.2', 'h:0.2-0.5', 'h>=0.5'],
                [(0, 0.2), (0.2, 0.5), (0.5, 1.0)]
            ):
                h_mask = (homophily >= low) & (homophily < high)
                combined_mask = test_mask & h_mask.to(test_mask.device)
                
                if combined_mask.sum() > 0:
                    glnn_acc = accuracy(
                        glnn_logits[combined_mask], 
                        labels[combined_mask]
                    ).item()
                    spec_acc = accuracy(
                        spec_logits[combined_mask], 
                        labels[combined_mask]
                    ).item()
                    
                    group_results[bin_name]['glnn'].append(glnn_acc)
                    group_results[bin_name]['spectral'].append(spec_acc)
        
        # Print per-group results
        print(f"  {'Group':<12} {'GLNN':<15} {'Spectral':<15} {'Winner':<10}")
        print(f"  {'-'*52}")
        
        results[dataset] = {
            'homophily_mean': h_mean,
            'homophily_std': h_std,
            'distribution': distribution,
            'hetero_pct': hetero_pct,
            'per_group': {},
        }
        
        for group_name in ['h<0.2', 'h:0.2-0.5', 'h>=0.5']:
            if group_results[group_name]['glnn']:
                glnn_mean = np.mean(group_results[group_name]['glnn']) * 100
                spec_mean = np.mean(group_results[group_name]['spectral']) * 100
                winner = 'GLNN' if glnn_mean > spec_mean else 'Spectral'
                diff = spec_mean - glnn_mean
                
                print(f"  {group_name:<12} {glnn_mean:>6.2f}%{'':>7} {spec_mean:>6.2f}%{'':>7} "
                      f"{winner} ({diff:+.2f}%)")
                
                results[dataset]['per_group'][group_name] = {
                    'glnn': glnn_mean,
                    'spectral': spec_mean,
                    'winner': winner,
                    'diff': diff,
                }
    
    return results



# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Core Pathology Diagnosis')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'spectral', 'balance', 'ablation', 'homophily'],
                       help='Which diagnostic task to run')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['actor', 'squirrel'],
                       help='Datasets to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    all_results = {}
    
    if args.task in ['all', 'spectral']:
        all_results['task1_spectral'] = task1_spectral_learnability(args.datasets, device)
    
    if args.task in ['all', 'balance']:
        all_results['task2_balance'] = task2_loss_balancing(args.datasets, device)
    
    if args.task in ['all', 'ablation']:
        all_results['task3_ablation'] = task3_component_ablation(args.datasets, device)
    
    if args.task in ['all', 'homophily']:
        all_results['task4_homophily'] = task4_homophily_quality(args.datasets, device)
    
    # Save results
    os.makedirs('results/phase2_diagnosis', exist_ok=True)
    results_path = f'results/phase2_diagnosis/diagnosis_{args.task}.json'
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    if 'task1_spectral' in all_results:
        print("\n📊 Task 1 (Spectral Learnability):")
        for ds, res in all_results['task1_spectral'].items():
            hf_ratio = res['spectral']['hf_error'] / (res['glnn']['hf_error'] + 1e-8)
            print(f"  {ds}: Spectral HF-Error is {hf_ratio:.2f}x of GLNN")
            if hf_ratio > 1.5:
                print(f"    → MLP struggles to learn high-frequency signals")
    
    if 'task2_balance' in all_results:
        print("\n📊 Task 2 (Loss Balancing):")
        for ds, res in all_results['task2_balance'].items():
            if 'trend' in res:
                print(f"  {ds}: {res['trend']}")
    
    if 'task3_ablation' in all_results:
        print("\n📊 Task 3 (Component Ablation):")
        for ds, res in all_results['task3_ablation'].items():
            if 'analysis' in res:
                print(f"  {ds}: {res['analysis']}")
    
    if 'task4_homophily' in all_results:
        print("\n📊 Task 4 (Homophily Quality):")
        for ds, res in all_results['task4_homophily'].items():
            hetero_pct = res.get('hetero_pct', 0)
            print(f"  {ds}: {hetero_pct:.1f}% nodes are strongly heterophilic (h<0.2)")
            if 'per_group' in res and 'h<0.2' in res['per_group']:
                diff = res['per_group']['h<0.2']['diff']
                winner = res['per_group']['h<0.2']['winner']
                print(f"    → On h<0.2 nodes: {winner} wins by {abs(diff):.2f}%")


if __name__ == '__main__':
    main()
