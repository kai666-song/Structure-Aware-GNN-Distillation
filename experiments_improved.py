"""
Improved Experiments Script

Implements three key improvements suggested by reviewers:
1. Heterophilic graph experiments (Chameleon, Squirrel, Actor)
2. Citeseer-specific optimization (degree-aware topology loss)
3. Statistical significance testing (paired t-test)

Usage:
    python experiments_improved.py --experiment heterophilic
    python experiments_improved.py --experiment citeseer_optimize
    python experiments_improved.py --experiment significance_test
    python experiments_improved.py --experiment all
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from scipy import stats

from models import GCN, GAT, MLPBatchNorm, convert_adj_to_edge_index
from utils import accuracy, load_data_new, preprocess_features, preprocess_adj
from kd_losses import SoftTarget, AdaptiveRKDLoss


# =============================================================================
# Degree-Aware Topology Loss (for Citeseer optimization)
# =============================================================================

class DegreeAwareTopologyLoss(nn.Module):
    """
    Topology loss that filters out low-degree nodes.
    
    Key insight: Low-degree nodes in Citeseer are often isolated/noisy.
    Filtering them improves signal-to-noise ratio for topology learning.
    """
    def __init__(self, min_degree=2, temperature=0.5, max_samples=2048):
        super().__init__()
        self.min_degree = min_degree
        self.temperature = temperature
        self.max_samples = max_samples
        
    def forward(self, student_out, teacher_out, edge_index, node_degrees, mask=None):
        """
        Args:
            node_degrees: Degree of each node [N]
        """
        # Filter edges where both endpoints have sufficient degree
        src, dst = edge_index[0], edge_index[1]
        
        # Degree mask: only keep edges between high-degree nodes
        degree_mask = (node_degrees[src] >= self.min_degree) & \
                      (node_degrees[dst] >= self.min_degree)
        
        if degree_mask.sum() == 0:
            return torch.tensor(0.0, device=student_out.device)
        
        src = src[degree_mask]
        dst = dst[degree_mask]
        
        # Sample if too many edges
        num_edges = len(src)
        if num_edges > self.max_samples:
            perm = torch.randperm(num_edges, device=src.device)[:self.max_samples]
            src, dst = src[perm], dst[perm]
        
        # Compute similarity alignment
        student_feat = F.normalize(F.softmax(student_out / self.temperature, dim=1), p=2, dim=1)
        teacher_feat = F.normalize(F.softmax(teacher_out / self.temperature, dim=1), p=2, dim=1)
        
        student_sim = (student_feat[src] * student_feat[dst]).sum(dim=1)
        teacher_sim = (teacher_feat[src] * teacher_feat[dst]).sum(dim=1)
        
        loss = F.mse_loss(student_sim, teacher_sim.detach())
        return loss


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    # Homophilic (existing)
    'cora': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
             'epochs': 300, 'patience': 100, 'gat_heads': 8, 'lambda_topo': 1.0},
    'citeseer': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
                 'epochs': 300, 'patience': 100, 'gat_heads': 8, 'lambda_topo': 0.3},  # Reduced!
    'pubmed': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
               'epochs': 300, 'patience': 100, 'gat_heads': 8, 'lambda_topo': 1.0},
    
    # Heterophilic (NEW - potential killer feature!)
    'chameleon': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
                  'epochs': 500, 'patience': 150, 'gat_heads': 4, 'lambda_topo': 0.5},
    'squirrel': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
                 'epochs': 500, 'patience': 150, 'gat_heads': 4, 'lambda_topo': 0.5},
    'actor': {'hidden': 64, 'lr': 0.01, 'wd_teacher': 5e-4, 'wd_student': 1e-5,
              'epochs': 500, 'patience': 150, 'gat_heads': 4, 'lambda_topo': 0.3},
}


# =============================================================================
# Unified Trainer with Improvements
# =============================================================================

class ImprovedDistillationTrainer:
    def __init__(self, args, config, seed):
        self.args = args
        self.config = config
        self.seed = seed
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        self.load_data()
        self.init_models()
        self.init_losses()
        
    def load_data(self):
        """Load and preprocess dataset."""
        self.adj, self.features, self.labels, *_, \
            self.idx_train, self.idx_val, self.idx_test = load_data_new(self.args.data)
        
        # Preprocess
        self.features_processed = preprocess_features(self.features)
        self.supports = preprocess_adj(self.adj)
        
        # Convert to tensors
        i = torch.from_numpy(self.features_processed[0]).long().to(self.device)
        v = torch.from_numpy(self.features_processed[1]).to(self.device)
        self.features_sparse = torch.sparse_coo_tensor(i.t(), v, self.features_processed[2]).to(self.device)
        
        i = torch.from_numpy(self.supports[0]).long().to(self.device)
        v = torch.from_numpy(self.supports[1]).to(self.device)
        self.adj_sparse = torch.sparse_coo_tensor(i.t(), v, self.supports[2]).float().to(self.device)
        
        # Edge index for GAT
        self.edge_index = convert_adj_to_edge_index(self.adj_sparse).to(self.device)
        
        # Compute node degrees (for degree-aware loss)
        self.node_degrees = torch.zeros(self.features_sparse.shape[0], device=self.device)
        src, dst = self.edge_index[0], self.edge_index[1]
        self.node_degrees.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        
        self.labels = self.labels.to(self.device)
        self.idx_train = self.idx_train.to(self.device)
        self.idx_val = self.idx_val.to(self.device)
        self.idx_test = self.idx_test.to(self.device)
        
        self.nfeat = self.features_sparse.shape[1]
        self.nclass = self.labels.max().item() + 1
        
        # Print dataset stats
        print(f"  Nodes: {self.features_sparse.shape[0]}, Edges: {self.edge_index.shape[1]}")
        print(f"  Features: {self.nfeat}, Classes: {self.nclass}")
        print(f"  Avg Degree: {self.node_degrees.mean():.2f}")
        
    def init_models(self):
        """Initialize teacher and student models."""
        config = self.config
        
        # Teacher: GAT
        self.teacher = GAT(
            self.nfeat, config['hidden'], self.nclass,
            dropout=0.6, heads=config['gat_heads']
        ).to(self.device)
        
        # Student: MLPBatchNorm
        self.student = MLPBatchNorm(
            self.nfeat, config['hidden'], self.nclass, dropout=0.5
        ).to(self.device)
        
    def init_losses(self):
        """Initialize loss functions."""
        self.criterion_task = nn.CrossEntropyLoss()
        self.criterion_kd = SoftTarget(T=self.args.temperature)
        self.criterion_rkd = AdaptiveRKDLoss(max_samples=2048)
        
        # Use degree-aware topology loss for Citeseer
        if self.args.data == 'citeseer' and self.args.use_degree_aware:
            self.criterion_topo = DegreeAwareTopologyLoss(
                min_degree=self.args.min_degree,
                temperature=0.5,
                max_samples=2048
            )
            self.use_degree_aware = True
        else:
            self.criterion_topo = None
            self.use_degree_aware = False
    
    def train_teacher(self):
        """Train teacher GAT."""
        config = self.config
        optimizer = optim.Adam(self.teacher.parameters(), lr=config['lr'], 
                               weight_decay=config['wd_teacher'])
        
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.teacher.train()
            optimizer.zero_grad()
            
            output = self.teacher(self.features_sparse, self.edge_index)
            loss = self.criterion_task(output[self.idx_train], self.labels[self.idx_train])
            loss.backward()
            optimizer.step()
            
            # Validation
            self.teacher.eval()
            with torch.no_grad():
                output = self.teacher(self.features_sparse, self.edge_index)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.teacher.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        # Load best and freeze
        self.teacher.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Test
        with torch.no_grad():
            output = self.teacher(self.features_sparse, self.edge_index)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test]).item() * 100
        
        return test_acc
    
    def train_student(self):
        """Train student with distillation."""
        config = self.config
        lambda_topo = config.get('lambda_topo', 1.0)
        
        optimizer = optim.Adam(self.student.parameters(), lr=config['lr'],
                               weight_decay=config['wd_student'])
        
        best_val_acc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            self.student.train()
            optimizer.zero_grad()
            
            student_out = self.student(self.features_sparse, self.adj_sparse)
            with torch.no_grad():
                teacher_out = self.teacher(self.features_sparse, self.edge_index)
            
            # Task loss
            loss_task = self.criterion_task(
                student_out[self.idx_train], self.labels[self.idx_train]
            )
            
            # KD loss
            loss_kd = self.criterion_kd(student_out, teacher_out)
            
            # RKD loss
            loss_rkd = self.criterion_rkd(student_out, teacher_out, mask=self.idx_train)
            
            # Topology loss (degree-aware for Citeseer)
            if self.use_degree_aware:
                loss_topo = self.criterion_topo(
                    student_out, teacher_out, self.edge_index, 
                    self.node_degrees, mask=self.idx_train
                )
            else:
                loss_topo = self._simple_topo_loss(student_out, teacher_out)
            
            # Combined loss
            loss = (self.args.alpha * loss_task + 
                    self.args.beta * loss_kd + 
                    self.args.gamma * loss_rkd +
                    lambda_topo * loss_topo)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            self.student.eval()
            with torch.no_grad():
                output = self.student(self.features_sparse, self.adj_sparse)
                val_acc = accuracy(output[self.idx_val], self.labels[self.idx_val]).item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                break
        
        # Load best
        self.student.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.student.eval()
        
        # Test
        with torch.no_grad():
            output = self.student(self.features_sparse, self.adj_sparse)
            test_acc = accuracy(output[self.idx_test], self.labels[self.idx_test]).item() * 100
        
        return test_acc
    
    def _simple_topo_loss(self, student_out, teacher_out, max_samples=2048):
        """Simple topology loss for non-Citeseer datasets."""
        src, dst = self.edge_index[0], self.edge_index[1]
        
        num_edges = len(src)
        if num_edges > max_samples:
            perm = torch.randperm(num_edges, device=src.device)[:max_samples]
            src, dst = src[perm], dst[perm]
        
        student_feat = F.normalize(F.softmax(student_out / 0.5, dim=1), p=2, dim=1)
        teacher_feat = F.normalize(F.softmax(teacher_out / 0.5, dim=1), p=2, dim=1)
        
        student_sim = (student_feat[src] * student_feat[dst]).sum(dim=1)
        teacher_sim = (teacher_feat[src] * teacher_feat[dst]).sum(dim=1)
        
        return F.mse_loss(student_sim, teacher_sim.detach())


# =============================================================================
# Experiment Functions
# =============================================================================

def run_single_dataset(args, dataset):
    """Run experiment on a single dataset."""
    args.data = dataset
    config = DATASET_CONFIGS.get(dataset, DATASET_CONFIGS['cora']).copy()
    
    results = {'teacher_accs': [], 'student_accs': []}
    
    for seed in range(args.num_runs):
        print(f"\n--- {dataset.upper()} Run {seed+1}/{args.num_runs} ---")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.cuda:
            torch.cuda.manual_seed(seed)
        
        trainer = ImprovedDistillationTrainer(args, config, seed)
        
        teacher_acc = trainer.train_teacher()
        results['teacher_accs'].append(teacher_acc)
        print(f"  Teacher (GAT): {teacher_acc:.2f}%")
        
        student_acc = trainer.train_student()
        results['student_accs'].append(student_acc)
        print(f"  Student (MLP): {student_acc:.2f}%")
        
        if args.cuda:
            torch.cuda.empty_cache()
    
    return results


def run_heterophilic_experiments(args):
    """
    Experiment 1: Heterophilic Graphs
    
    Key hypothesis: On heterophilic graphs, GNN performance degrades because
    neighbors have different labels. MLP (which ignores neighbors) may actually
    perform better. Our distillation should preserve MLP's advantage while
    still learning useful knowledge from the teacher.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: HETEROPHILIC GRAPHS")
    print("="*70)
    print("Hypothesis: Student MLP may significantly outperform Teacher GAT")
    print("on heterophilic graphs where neighbor aggregation hurts performance.")
    print("="*70)
    
    datasets = ['chameleon', 'squirrel', 'actor']
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*50}")
        
        try:
            results = run_single_dataset(args, dataset)
            all_results[dataset] = results
            
            teacher_mean = np.mean(results['teacher_accs'])
            teacher_std = np.std(results['teacher_accs'])
            student_mean = np.mean(results['student_accs'])
            student_std = np.std(results['student_accs'])
            gap = student_mean - teacher_mean
            
            print(f"\nResults for {dataset}:")
            print(f"  Teacher (GAT): {teacher_mean:.2f} ± {teacher_std:.2f}%")
            print(f"  Student (MLP): {student_mean:.2f} ± {student_std:.2f}%")
            print(f"  Gap: {gap:+.2f}% {'✨ Student wins!' if gap > 0 else ''}")
            
        except Exception as e:
            print(f"Error on {dataset}: {e}")
            all_results[dataset] = None
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/heterophilic_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def run_citeseer_optimization(args):
    """
    Experiment 2: Citeseer Optimization
    
    Problem: Citeseer has many isolated nodes, making topology loss noisy.
    Solution: Use degree-aware topology loss that filters low-degree nodes.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: CITESEER OPTIMIZATION")
    print("="*70)
    print("Testing degree-aware topology loss to handle isolated nodes.")
    print("="*70)
    
    args.data = 'citeseer'
    config = DATASET_CONFIGS['citeseer'].copy()
    
    # Test different configurations
    configs_to_test = [
        {'lambda_topo': 1.0, 'use_degree_aware': False, 'name': 'baseline'},
        {'lambda_topo': 0.3, 'use_degree_aware': False, 'name': 'reduced_topo'},
        {'lambda_topo': 0.5, 'use_degree_aware': True, 'min_degree': 2, 'name': 'degree_aware_d2'},
        {'lambda_topo': 0.5, 'use_degree_aware': True, 'min_degree': 3, 'name': 'degree_aware_d3'},
    ]
    
    all_results = {}
    
    for cfg in configs_to_test:
        print(f"\n--- Config: {cfg['name']} ---")
        
        # Update args
        args.use_degree_aware = cfg.get('use_degree_aware', False)
        args.min_degree = cfg.get('min_degree', 2)
        config['lambda_topo'] = cfg['lambda_topo']
        
        results = {'teacher_accs': [], 'student_accs': []}
        
        for seed in range(args.num_runs):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)
            
            trainer = ImprovedDistillationTrainer(args, config, seed)
            teacher_acc = trainer.train_teacher()
            student_acc = trainer.train_student()
            
            results['teacher_accs'].append(teacher_acc)
            results['student_accs'].append(student_acc)
            
            if args.cuda:
                torch.cuda.empty_cache()
        
        all_results[cfg['name']] = results
        
        print(f"  Teacher: {np.mean(results['teacher_accs']):.2f} ± {np.std(results['teacher_accs']):.2f}%")
        print(f"  Student: {np.mean(results['student_accs']):.2f} ± {np.std(results['student_accs']):.2f}%")
    
    # Save results
    with open('results/citeseer_optimization.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def run_significance_tests(args):
    """
    Experiment 3: Statistical Significance Testing
    
    Perform paired t-tests to determine if improvements are statistically significant.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    
    # Load existing results or run new experiments
    results_files = {
        'cora': 'results/gat_distill_cora.json',
        'pubmed': 'results/gat_distill_pubmed.json',
        'amazon-photo': 'results/gat_distill_amazon-photo.json',
    }
    
    significance_results = {}
    
    for dataset, filepath in results_files.items():
        print(f"\n--- {dataset.upper()} ---")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            teacher_accs = data['runs']['teacher_accs']
            student_accs = data['runs']['student_accs']
        else:
            print(f"  Results file not found. Running experiment...")
            results = run_single_dataset(args, dataset)
            teacher_accs = results['teacher_accs']
            student_accs = results['student_accs']
        
        # Paired t-test (Student vs Teacher)
        t_stat, p_value = stats.ttest_rel(student_accs, teacher_accs)
        
        teacher_mean = np.mean(teacher_accs)
        student_mean = np.mean(student_accs)
        gap = student_mean - teacher_mean
        
        # Determine significance
        if p_value < 0.01:
            sig_level = "*** (p < 0.01)"
        elif p_value < 0.05:
            sig_level = "** (p < 0.05)"
        elif p_value < 0.1:
            sig_level = "* (p < 0.1)"
        else:
            sig_level = "n.s."
        
        significance_results[dataset] = {
            'teacher_mean': float(teacher_mean),
            'student_mean': float(student_mean),
            'gap': float(gap),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significance': sig_level,
            'student_wins': bool(gap > 0 and p_value < 0.05)
        }
        
        print(f"  Teacher: {teacher_mean:.2f}%")
        print(f"  Student: {student_mean:.2f}%")
        print(f"  Gap: {gap:+.2f}%")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significance: {sig_level}")
        
        if gap > 0 and p_value < 0.05:
            print(f"  ✨ STATISTICALLY SIGNIFICANT IMPROVEMENT!")
    
    # Summary table
    print("\n" + "="*70)
    print("SIGNIFICANCE TEST SUMMARY")
    print("="*70)
    print(f"{'Dataset':<15} {'Teacher':<12} {'Student':<12} {'Gap':<10} {'p-value':<10} {'Sig.':<15}")
    print("-"*70)
    for dataset, res in significance_results.items():
        print(f"{dataset:<15} {res['teacher_mean']:<12.2f} {res['student_mean']:<12.2f} "
              f"{res['gap']:+<10.2f} {res['p_value']:<10.4f} {res['significance']:<15}")
    
    # Save results
    with open('results/significance_tests.json', 'w') as f:
        json.dump(significance_results, f, indent=2)
    
    return significance_results


def main():
    parser = argparse.ArgumentParser(description='Improved Experiments')
    
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['heterophilic', 'citeseer_optimize', 'significance_test', 'all'])
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--num_runs', type=int, default=10)
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=4.0)
    
    # Citeseer optimization
    parser.add_argument('--use_degree_aware', action='store_true', default=False)
    parser.add_argument('--min_degree', type=int, default=2)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    print(f"Device: {'CUDA' if args.cuda else 'CPU'}")
    
    if args.experiment == 'heterophilic' or args.experiment == 'all':
        run_heterophilic_experiments(args)
    
    if args.experiment == 'citeseer_optimize' or args.experiment == 'all':
        run_citeseer_optimization(args)
    
    if args.experiment == 'significance_test' or args.experiment == 'all':
        run_significance_tests(args)


if __name__ == '__main__':
    main()
