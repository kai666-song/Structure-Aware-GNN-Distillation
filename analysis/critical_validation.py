"""
Critical Validation Experiments for Paper Defense
==================================================
This module addresses three potential fatal flaws identified in red-team review:

1. Vanilla MLP Baseline - Is distillation actually helping?
2. Dirichlet Energy Analysis - Does Student preserve high-frequency info on heterophilic graphs?
3. Gamma Sensitivity - Is TCD loss actually beneficial (gamma > 0)?

Reference: These are the "killer questions" reviewers will ask at NeurIPS/ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os
import sys
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, MLPBatchNorm, GAT, GCN, GCNII
from utils.data_utils import load_data_new
from kd_losses.st import SoftTarget
from kd_losses.rkd import RKDLoss
from kd_losses.topology_kd import ContrastiveTopologyLoss


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Experiment 1: Vanilla MLP Baseline
# =============================================================================

def train_vanilla_mlp(features, labels, idx_train, idx_val, idx_test, 
                      n_features, n_classes, device, epochs=500, patience=50):
    """Train a vanilla MLP without any distillation."""
    model = MLPBatchNorm(n_features, 256, n_classes, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    features_tensor = torch.FloatTensor(features).to(device)
    labels_tensor = labels.to(device)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(features_tensor)
        loss = F.cross_entropy(output[idx_train], labels_tensor[idx_train])
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            output = model(features_tensor)
            val_pred = output[idx_val].argmax(dim=1)
            val_acc = (val_pred == labels_tensor[idx_val]).float().mean().item()
            
            test_pred = output[idx_test].argmax(dim=1)
            test_acc = (test_pred == labels_tensor[idx_test]).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_test_acc * 100


def run_vanilla_mlp_experiment(dataset='actor', num_runs=10, device='cuda'):
    """
    Run Vanilla MLP baseline experiment.
    
    Critical Question: Is Student (35.3%) > Vanilla MLP?
    If not, distillation provides no benefit over simple supervised learning.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Vanilla MLP Baseline")
    print("=" * 70)
    print(f"Dataset: {dataset}, Runs: {num_runs}")
    print()
    
    vanilla_accs = []
    
    for run in range(num_runs):
        set_seed(42 + run)
        
        # Load data with Geom-GCN split
        adj, features, labels, y_train, y_val, y_test, \
            train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = \
            load_data_new(dataset, split_idx=run % 10)
        
        # Convert features
        if hasattr(features, 'toarray'):
            features = features.toarray()
        elif hasattr(features, 'todense'):
            features = np.array(features.todense())
        
        n_features = features.shape[1]
        n_classes = int(labels.max()) + 1
        
        # Train vanilla MLP
        acc = train_vanilla_mlp(
            features, labels, idx_train, idx_val, idx_test,
            n_features, n_classes, device
        )
        vanilla_accs.append(acc)
        print(f"  Run {run+1}: {acc:.2f}%")
    
    mean_acc = np.mean(vanilla_accs)
    std_acc = np.std(vanilla_accs)
    
    print()
    print(f"Vanilla MLP: {mean_acc:.2f} ± {std_acc:.2f}%")
    
    return {
        'dataset': dataset,
        'num_runs': num_runs,
        'vanilla_mlp_mean': mean_acc,
        'vanilla_mlp_std': std_acc,
        'all_accs': vanilla_accs
    }


# =============================================================================
# Experiment 2: Dirichlet Energy Analysis
# =============================================================================

def compute_dirichlet_energy(features, adj, device='cuda'):
    """
    Compute Dirichlet Energy: E(X) = trace(X^T L X)
    
    Lower energy = smoother features (more similar across edges)
    Higher energy = sharper features (preserves high-frequency info)
    
    For heterophilic graphs, we WANT higher energy (Student should NOT oversmooth).
    """
    import scipy.sparse as sp
    
    # Convert to dense if needed
    if hasattr(features, 'toarray'):
        X = torch.FloatTensor(features.toarray()).to(device)
    elif isinstance(features, np.ndarray):
        X = torch.FloatTensor(features).to(device)
    else:
        X = features.to(device)
    
    # Normalize features
    X = F.normalize(X, p=2, dim=1)
    
    # Build Laplacian: L = D - A
    if sp.issparse(adj):
        adj_dense = adj.toarray()
    else:
        adj_dense = adj
    
    # Degree matrix
    D = np.diag(adj_dense.sum(axis=1))
    L = D - adj_dense
    L_tensor = torch.FloatTensor(L).to(device)
    
    # Dirichlet Energy = trace(X^T L X)
    # Efficient computation: sum of (x_i - x_j)^2 for all edges
    energy = torch.trace(X.T @ L_tensor @ X)
    
    # Normalize by number of nodes
    energy = energy / X.shape[0]
    
    return energy.item()


def compute_dirichlet_energy_from_model(model, features, adj, device='cuda'):
    """Compute Dirichlet Energy of model's learned representations."""
    model.eval()
    
    if hasattr(features, 'toarray'):
        features_tensor = torch.FloatTensor(features.toarray()).to(device)
    else:
        features_tensor = torch.FloatTensor(features).to(device)
    
    with torch.no_grad():
        # Get hidden representations (before final layer)
        if hasattr(model, 'get_embedding'):
            embeddings = model.get_embedding(features_tensor)
        else:
            # For MLP, get output of first layer
            x = F.relu(model.fc1(features_tensor))
            if hasattr(model, 'bn1'):
                x = model.bn1(x)
            embeddings = x
    
    return compute_dirichlet_energy(embeddings.cpu().numpy(), adj, device)


def run_dirichlet_energy_experiment(dataset='actor', num_runs=5, device='cuda'):
    """
    Compute Dirichlet Energy for Teacher vs Student.
    
    Expected Results:
    - Homophilic (Cora): Student Energy ≈ Teacher Energy (both smooth)
    - Heterophilic (Actor): Student Energy > Teacher Energy (Student preserves sharpness)
    """
    print("=" * 70)
    print("EXPERIMENT 2: Dirichlet Energy Analysis")
    print("=" * 70)
    print(f"Dataset: {dataset}, Runs: {num_runs}")
    print()
    
    teacher_energies = []
    student_energies = []
    input_energies = []
    
    for run in range(num_runs):
        set_seed(42 + run)
        
        # Load data
        adj, features, labels, y_train, y_val, y_test, \
            train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = \
            load_data_new(dataset, split_idx=run % 10)
        
        if hasattr(features, 'toarray'):
            features_np = features.toarray()
        else:
            features_np = np.array(features.todense()) if hasattr(features, 'todense') else features
        
        n_features = features_np.shape[1]
        n_classes = int(labels.max()) + 1
        
        # Compute input feature energy
        input_energy = compute_dirichlet_energy(features_np, adj, device)
        input_energies.append(input_energy)
        
        # Train Teacher (GAT)
        import scipy.sparse as sp
        edge_index = torch.LongTensor(np.array(sp.coo_matrix(adj).nonzero())).to(device)
        features_tensor = torch.FloatTensor(features_np).to(device)
        labels_tensor = labels.to(device)
        
        teacher = GAT(n_features, 8, n_classes, dropout=0.6, alpha=0.2, nheads=8).to(device)
        optimizer = optim.Adam(teacher.parameters(), lr=0.005, weight_decay=5e-4)
        
        # Train teacher
        for epoch in range(200):
            teacher.train()
            optimizer.zero_grad()
            output = teacher(features_tensor, edge_index)
            loss = F.cross_entropy(output[idx_train], labels_tensor[idx_train])
            loss.backward()
            optimizer.step()
        
        # Get teacher embeddings and compute energy
        teacher.eval()
        with torch.no_grad():
            # Get attention-weighted embeddings
            teacher_out = teacher(features_tensor, edge_index)
        teacher_energy = compute_dirichlet_energy(teacher_out.cpu().numpy(), adj, device)
        teacher_energies.append(teacher_energy)
        
        # Train Student (MLP with distillation)
        student = MLPBatchNorm(n_features, 256, n_classes, dropout=0.5).to(device)
        optimizer = optim.Adam(student.parameters(), lr=0.01, weight_decay=5e-4)
        kd_loss_fn = SoftTarget(T=4.0)
        
        for epoch in range(300):
            student.train()
            teacher.eval()
            optimizer.zero_grad()
            
            student_out = student(features_tensor)
            with torch.no_grad():
                teacher_out = teacher(features_tensor, edge_index)
            
            loss_task = F.cross_entropy(student_out[idx_train], labels_tensor[idx_train])
            loss_kd = kd_loss_fn(student_out, teacher_out)
            loss = loss_task + loss_kd
            
            loss.backward()
            optimizer.step()
        
        # Get student embeddings and compute energy
        student.eval()
        with torch.no_grad():
            student_out = student(features_tensor)
        student_energy = compute_dirichlet_energy(student_out.cpu().numpy(), adj, device)
        student_energies.append(student_energy)
        
        print(f"  Run {run+1}: Input={input_energy:.4f}, Teacher={teacher_energy:.4f}, Student={student_energy:.4f}")
    
    print()
    print(f"Input Features Energy:  {np.mean(input_energies):.4f} ± {np.std(input_energies):.4f}")
    print(f"Teacher (GAT) Energy:   {np.mean(teacher_energies):.4f} ± {np.std(teacher_energies):.4f}")
    print(f"Student (MLP) Energy:   {np.mean(student_energies):.4f} ± {np.std(student_energies):.4f}")
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(student_energies, teacher_energies)
    print()
    print(f"Paired t-test (Student vs Teacher): t={t_stat:.3f}, p={p_value:.4f}")
    
    if np.mean(student_energies) > np.mean(teacher_energies):
        print("✓ Student has HIGHER energy (preserves high-frequency info)")
    else:
        print("⚠ Student has LOWER energy (may be oversmoothing)")
    
    return {
        'dataset': dataset,
        'num_runs': num_runs,
        'input_energy': {'mean': np.mean(input_energies), 'std': np.std(input_energies)},
        'teacher_energy': {'mean': np.mean(teacher_energies), 'std': np.std(teacher_energies)},
        'student_energy': {'mean': np.mean(student_energies), 'std': np.std(student_energies)},
        'ttest_pvalue': p_value,
        'student_higher': bool(np.mean(student_energies) > np.mean(teacher_energies))
    }


# =============================================================================
# Experiment 3: Gamma (TCD Weight) Sensitivity Analysis
# =============================================================================

def train_with_gamma(features, labels, adj, idx_train, idx_val, idx_test,
                     n_features, n_classes, gamma, device='cuda'):
    """Train student with specific gamma (TCD weight)."""
    import scipy.sparse as sp
    
    edge_index = torch.LongTensor(np.array(sp.coo_matrix(adj).nonzero())).to(device)
    features_tensor = torch.FloatTensor(features).to(device)
    labels_tensor = labels.to(device)
    
    # Train Teacher
    teacher = GAT(n_features, 8, n_classes, dropout=0.6, alpha=0.2, nheads=8).to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.005, weight_decay=5e-4)
    
    for epoch in range(200):
        teacher.train()
        optimizer.zero_grad()
        output = teacher(features_tensor, edge_index)
        loss = F.cross_entropy(output[idx_train], labels_tensor[idx_train])
        loss.backward()
        optimizer.step()
    
    # Train Student with specified gamma
    student = MLPBatchNorm(n_features, 256, n_classes, dropout=0.5).to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.01, weight_decay=5e-4)
    
    kd_loss_fn = SoftTarget(T=4.0)
    rkd_loss_fn = RKDLoss()
    tcd_loss_fn = ContrastiveTopologyLoss(margin=0.5)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(300):
        student.train()
        teacher.eval()
        optimizer.zero_grad()
        
        student_out = student(features_tensor)
        with torch.no_grad():
            teacher_out = teacher(features_tensor, edge_index)
        
        # Task loss
        loss_task = F.cross_entropy(student_out[idx_train], labels_tensor[idx_train])
        
        # KD loss
        loss_kd = kd_loss_fn(student_out, teacher_out)
        
        # RKD loss (always included)
        loss_rkd = rkd_loss_fn(student_out, teacher_out)
        
        # TCD loss (weighted by gamma)
        if gamma > 0:
            loss_tcd = tcd_loss_fn(student_out, teacher_out, edge_index)
        else:
            loss_tcd = 0
        
        loss = loss_task + loss_kd + loss_rkd + gamma * loss_tcd
        
        loss.backward()
        optimizer.step()
        
        # Validation
        student.eval()
        with torch.no_grad():
            output = student(features_tensor)
            val_pred = output[idx_val].argmax(dim=1)
            val_acc = (val_pred == labels_tensor[idx_val]).float().mean().item()
            
            test_pred = output[idx_test].argmax(dim=1)
            test_acc = (test_pred == labels_tensor[idx_test]).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    
    return best_test_acc * 100


def run_gamma_sensitivity_experiment(dataset='actor', num_runs=5, device='cuda'):
    """
    Test sensitivity to gamma (TCD weight).
    
    Critical Question: Is gamma > 0 actually beneficial?
    If gamma=0 is optimal, the entire TCD methodology is invalidated.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Gamma (TCD Weight) Sensitivity")
    print("=" * 70)
    print(f"Dataset: {dataset}, Runs: {num_runs}")
    print()
    
    gamma_values = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    results = {gamma: [] for gamma in gamma_values}
    
    for run in range(num_runs):
        set_seed(42 + run)
        print(f"Run {run+1}/{num_runs}:")
        
        # Load data
        adj, features, labels, y_train, y_val, y_test, \
            train_mask, val_mask, test_mask, idx_train, idx_val, idx_test = \
            load_data_new(dataset, split_idx=run % 10)
        
        if hasattr(features, 'toarray'):
            features = features.toarray()
        elif hasattr(features, 'todense'):
            features = np.array(features.todense())
        
        n_features = features.shape[1]
        n_classes = int(labels.max()) + 1
        
        for gamma in gamma_values:
            acc = train_with_gamma(
                features, labels, adj, idx_train, idx_val, idx_test,
                n_features, n_classes, gamma, device
            )
            results[gamma].append(acc)
            print(f"  gamma={gamma}: {acc:.2f}%")
    
    print()
    print("Summary:")
    print("-" * 40)
    
    best_gamma = None
    best_mean = 0
    
    for gamma in gamma_values:
        mean_acc = np.mean(results[gamma])
        std_acc = np.std(results[gamma])
        print(f"  gamma={gamma}: {mean_acc:.2f} ± {std_acc:.2f}%")
        
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_gamma = gamma
    
    print()
    print(f"Best gamma: {best_gamma} ({best_mean:.2f}%)")
    
    if best_gamma > 0:
        print("✓ TCD loss (gamma > 0) is beneficial!")
    else:
        print("⚠ WARNING: gamma=0 is optimal - TCD may not be helping!")
    
    return {
        'dataset': dataset,
        'num_runs': num_runs,
        'gamma_values': gamma_values,
        'results': {str(g): {'mean': np.mean(results[g]), 'std': np.std(results[g]), 'all': results[g]} 
                   for g in gamma_values},
        'best_gamma': best_gamma,
        'best_accuracy': best_mean,
        'tcd_beneficial': best_gamma > 0
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_critical_validations(datasets=['actor', 'cora'], num_runs=5, device='cuda'):
    """Run all critical validation experiments."""
    
    os.makedirs('results', exist_ok=True)
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*70}\n")
        
        dataset_results = {}
        
        # Experiment 1: Vanilla MLP
        print("\n" + "="*70)
        vanilla_results = run_vanilla_mlp_experiment(dataset, num_runs, device)
        dataset_results['vanilla_mlp'] = vanilla_results
        
        # Experiment 2: Dirichlet Energy
        print("\n" + "="*70)
        energy_results = run_dirichlet_energy_experiment(dataset, num_runs, device)
        dataset_results['dirichlet_energy'] = energy_results
        
        # Experiment 3: Gamma Sensitivity
        print("\n" + "="*70)
        gamma_results = run_gamma_sensitivity_experiment(dataset, num_runs, device)
        dataset_results['gamma_sensitivity'] = gamma_results
        
        all_results[dataset] = dataset_results
    
    # Save results
    output_path = 'results/critical_validation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("CRITICAL VALIDATION SUMMARY")
    print("="*70)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        
        # Vanilla MLP comparison
        vanilla_acc = results['vanilla_mlp']['vanilla_mlp_mean']
        print(f"  1. Vanilla MLP: {vanilla_acc:.2f}%")
        
        # Dirichlet Energy
        teacher_e = results['dirichlet_energy']['teacher_energy']['mean']
        student_e = results['dirichlet_energy']['student_energy']['mean']
        print(f"  2. Dirichlet Energy: Teacher={teacher_e:.4f}, Student={student_e:.4f}")
        
        # Gamma sensitivity
        best_gamma = results['gamma_sensitivity']['best_gamma']
        best_acc = results['gamma_sensitivity']['best_accuracy']
        print(f"  3. Best gamma: {best_gamma} ({best_acc:.2f}%)")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Critical Validation Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['vanilla', 'energy', 'gamma', 'all'],
                       help='Which experiment to run')
    parser.add_argument('--dataset', type=str, default='actor',
                       help='Dataset to use')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.experiment == 'vanilla':
        results = run_vanilla_mlp_experiment(args.dataset, args.num_runs, device)
    elif args.experiment == 'energy':
        results = run_dirichlet_energy_experiment(args.dataset, args.num_runs, device)
    elif args.experiment == 'gamma':
        results = run_gamma_sensitivity_experiment(args.dataset, args.num_runs, device)
    else:
        results = run_all_critical_validations([args.dataset, 'cora'], args.num_runs, device)
    
    # Save individual experiment results
    if args.experiment != 'all':
        output_path = f'results/critical_{args.experiment}_{args.dataset}.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
