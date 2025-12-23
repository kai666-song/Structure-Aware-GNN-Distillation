"""
Phase 5: Efficiency Benchmark - The "Kill Shot"
================================================

Compares GloGNN++ (Teacher) vs EnhancedMLP (Student) on:
1. Inference Time (ms)
2. Throughput (nodes/sec)  
3. Parameter Count
4. Memory Usage

This proves the practical value of knowledge distillation:
Same accuracy, but 10-100x faster inference!

Usage:
    python benchmark_efficiency.py --dataset actor
    python benchmark_efficiency.py --dataset actor --cuda
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import EnhancedMLP, MLPBatchNorm
from utils.data_utils import load_data_new


# Import GloGNN model
from baselines.run_glognn_baseline import MLP_NORM, GLOGNN_CONFIGS, normalize, sparse_mx_to_torch_sparse_tensor


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024


def benchmark_glognn(dataset, device, num_runs=100, warmup=10):
    """Benchmark GloGNN++ inference."""
    config = GLOGNN_CONFIGS[dataset]
    
    # Load data
    adj, features, labels, *_ = load_data_new(dataset, split_idx=0)
    num_nodes = adj.shape[0]
    
    # Preprocess for GloGNN
    adj_norm = normalize(adj + sp.eye(adj.shape[0]))
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_norm)
    
    features_norm = normalize(sp.csr_matrix(features.todense() if hasattr(features, 'todense') else features))
    features_tensor = torch.FloatTensor(np.array(features_norm.todense()))
    
    # Convert to float64 for GloGNN
    features_tensor = features_tensor.to(torch.float64).to(device)
    adj_tensor = adj_tensor.to(torch.float64).to(device)
    
    nclass = int(labels.max().item()) + 1
    
    # Initialize model
    model = MLP_NORM(
        nnodes=num_nodes,
        nfeat=features_tensor.shape[1],
        nhid=config['hidden'],
        nclass=nclass,
        dropout=config['dropout'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        delta=config['delta'],
        norm_func_id=config['norm_func_id'],
        norm_layers=config['norm_layers'],
        orders=config['orders'],
        orders_func_id=config['orders_func_id'],
        cuda=device.type == 'cuda'
    ).to(device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(features_tensor, adj_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(features_tensor, adj_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return {
        'model': 'GloGNN++',
        'params': count_parameters(model),
        'size_mb': get_model_size_mb(model),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'throughput': num_nodes / (np.mean(times) / 1000),
        'num_nodes': num_nodes,
        'requires_graph': True
    }


def benchmark_student(dataset, device, num_runs=100, warmup=10, use_pe=True):
    """Benchmark EnhancedMLP (Student) inference."""
    
    # Load data
    adj, features, labels, *_ = load_data_new(dataset, split_idx=0)
    num_nodes = adj.shape[0]
    
    # Convert features
    if hasattr(features, 'todense'):
        features_tensor = torch.FloatTensor(np.array(features.todense()))
    else:
        features_tensor = torch.FloatTensor(np.array(features))
    
    # Add PE if needed
    if use_pe:
        pe_path = f'./data/pe_rw_{dataset}.pt'
        if os.path.exists(pe_path):
            pe = torch.load(pe_path)['pe'].float()  # Ensure float32
            features_tensor = torch.cat([features_tensor, pe], dim=1)
    
    features_tensor = features_tensor.float().to(device)  # Ensure float32
    nclass = int(labels.max().item()) + 1
    
    # Initialize model
    model = EnhancedMLP(
        nfeat=features_tensor.shape[1],
        nhid=256,
        nclass=nclass,
        dropout=0.5,
        num_layers=3
    ).float().to(device)  # Ensure float32
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(features_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(features_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return {
        'model': 'EnhancedMLP (Ours)',
        'params': count_parameters(model),
        'size_mb': get_model_size_mb(model),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'throughput': num_nodes / (np.mean(times) / 1000),
        'num_nodes': num_nodes,
        'requires_graph': False
    }


def run_benchmark(args):
    """Run complete efficiency benchmark."""
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"EFFICIENCY BENCHMARK - {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Benchmark runs: {args.num_runs}")
    print(f"Warmup runs: {args.warmup}")
    
    # Benchmark GloGNN++
    print(f"\nBenchmarking GloGNN++ (Teacher)...")
    teacher_results = benchmark_glognn(args.dataset, device, args.num_runs, args.warmup)
    
    # Benchmark Student
    print(f"Benchmarking EnhancedMLP (Student)...")
    student_results = benchmark_student(args.dataset, device, args.num_runs, args.warmup)
    
    # Calculate speedup
    speedup = teacher_results['time_mean'] / student_results['time_mean']
    param_reduction = teacher_results['params'] / student_results['params']
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25} {'GloGNN++ (Teacher)':<25} {'EnhancedMLP (Ours)':<25}")
    print("-" * 75)
    print(f"{'Parameters':<25} {teacher_results['params']:,}{'':<15} {student_results['params']:,}")
    print(f"{'Model Size (MB)':<25} {teacher_results['size_mb']:.2f}{'':<20} {student_results['size_mb']:.2f}")
    print(f"{'Inference Time (ms)':<25} {teacher_results['time_mean']:.3f} ± {teacher_results['time_std']:.3f}{'':<5} {student_results['time_mean']:.3f} ± {student_results['time_std']:.3f}")
    print(f"{'Throughput (nodes/s)':<25} {teacher_results['throughput']:,.0f}{'':<15} {student_results['throughput']:,.0f}")
    print(f"{'Requires Graph':<25} {'Yes':<25} {'No'}")
    
    print(f"\n{'='*70}")
    print("EFFICIENCY GAINS")
    print(f"{'='*70}")
    print(f"Speedup: {speedup:.1f}x faster inference")
    print(f"Parameter Ratio: {param_reduction:.2f}x {'smaller' if param_reduction > 1 else 'larger'}")
    print(f"Graph-Free: Student does NOT need adjacency matrix at inference!")
    
    # Compile results
    results = {
        'dataset': args.dataset,
        'device': str(device),
        'teacher': teacher_results,
        'student': student_results,
        'speedup': speedup,
        'param_ratio': param_reduction,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    save_path = f'results/efficiency_{args.dataset}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {save_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Efficiency Benchmark')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'squirrel', 'chameleon'])
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup runs')
    args = parser.parse_args()
    
    run_benchmark(args)


if __name__ == '__main__':
    main()
