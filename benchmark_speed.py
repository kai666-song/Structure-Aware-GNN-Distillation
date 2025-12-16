"""
Efficiency Benchmark: Inference Speed & Parameter Count

Compares GCN (Teacher) vs MLP (Student) on:
1. Inference Time (ms)
2. Throughput (nodes/sec)
3. Parameter Count
"""

import os
import time
import numpy as np
import torch
import argparse
from tabulate import tabulate

from models import GCN, MLP
from utils import load_data_new, preprocess_features, preprocess_adj


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_inference(model, features, adj, num_runs=100, warmup=10):
    """Benchmark inference time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(features, adj)
    
    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(features, adj)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)


def run_benchmark(args):
    """Run efficiency benchmark on a dataset."""
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # Load data
    adj, features, labels, *_ = load_data_new(args.data)
    num_nodes = features.shape[0]
    nfeat = features.shape[1]
    nclass = labels.max().item() + 1
    
    # Preprocess
    features = preprocess_features(features)
    supports = preprocess_adj(adj)
    
    i = torch.from_numpy(features[0]).long().to(device)
    v = torch.from_numpy(features[1]).to(device)
    features = torch.sparse_coo_tensor(i.t(), v, features[2]).to(device)
    
    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    adj = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(device)
    
    # Get hidden dim from config
    hidden = 256 if 'amazon' in args.data else 64
    
    # Initialize models
    gcn = GCN(nfeat, hidden, nclass, dropout=0.5).to(device)
    mlp = MLP(nfeat, hidden, nclass, dropout=0.5).to(device)
    
    # Load trained weights if available
    checkpoint_dir = f'checkpoints/{args.data}'
    if os.path.exists(f'{checkpoint_dir}/teacher_seed0.pt'):
        gcn.load_state_dict(torch.load(f'{checkpoint_dir}/teacher_seed0.pt', map_location=device))
        print(f'Loaded teacher weights from {checkpoint_dir}')
    if os.path.exists(f'{checkpoint_dir}/student_seed0.pt'):
        mlp.load_state_dict(torch.load(f'{checkpoint_dir}/student_seed0.pt', map_location=device))
        print(f'Loaded student weights from {checkpoint_dir}')
    
    # Count parameters
    gcn_params = count_parameters(gcn)
    mlp_params = count_parameters(mlp)
    
    # Benchmark inference
    gcn_time, gcn_std = benchmark_inference(gcn, features, adj, num_runs=args.num_runs)
    mlp_time, mlp_std = benchmark_inference(mlp, features, adj, num_runs=args.num_runs)
    
    # Calculate throughput
    gcn_throughput = num_nodes / (gcn_time / 1000)  # nodes/sec
    mlp_throughput = num_nodes / (mlp_time / 1000)
    
    # Speedup
    speedup = gcn_time / mlp_time
    
    return {
        'dataset': args.data,
        'num_nodes': num_nodes,
        'gcn_params': gcn_params,
        'mlp_params': mlp_params,
        'gcn_time': gcn_time,
        'gcn_std': gcn_std,
        'mlp_time': mlp_time,
        'mlp_std': mlp_std,
        'gcn_throughput': gcn_throughput,
        'mlp_throughput': mlp_throughput,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    print(f'Device: {"CUDA" if args.cuda else "CPU"}')
    print(f'Benchmark runs: {args.num_runs}\n')
    
    if args.all:
        datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
    else:
        datasets = [args.data]
    
    results = []
    for dataset in datasets:
        print(f'Benchmarking {dataset}...')
        args.data = dataset
        result = run_benchmark(args)
        results.append(result)
    
    # Print results table
    print('\n' + '='*90)
    print('EFFICIENCY BENCHMARK RESULTS')
    print('='*90)
    
    # Table 1: Inference Time
    headers = ['Dataset', 'Nodes', 'GCN (ms)', 'MLP (ms)', 'Speedup']
    table_data = []
    for r in results:
        table_data.append([
            r['dataset'],
            f"{r['num_nodes']:,}",
            f"{r['gcn_time']:.3f} ± {r['gcn_std']:.3f}",
            f"{r['mlp_time']:.3f} ± {r['mlp_std']:.3f}",
            f"{r['speedup']:.1f}x"
        ])
    print('\nInference Time (lower is better):')
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Table 2: Parameters
    headers = ['Dataset', 'GCN Params', 'MLP Params', 'Reduction']
    table_data = []
    for r in results:
        reduction = r['gcn_params'] / r['mlp_params'] if r['mlp_params'] > 0 else 0
        table_data.append([
            r['dataset'],
            f"{r['gcn_params']:,}",
            f"{r['mlp_params']:,}",
            f"{reduction:.2f}x" if reduction > 1 else f"{1/reduction:.2f}x more"
        ])
    print('\nParameter Count:')
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Table 3: Throughput
    headers = ['Dataset', 'GCN (nodes/s)', 'MLP (nodes/s)']
    table_data = []
    for r in results:
        table_data.append([
            r['dataset'],
            f"{r['gcn_throughput']:,.0f}",
            f"{r['mlp_throughput']:,.0f}"
        ])
    print('\nThroughput (higher is better):')
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Save results
    import json
    os.makedirs('results', exist_ok=True)
    with open('results/efficiency_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to results/efficiency_benchmark.json')


if __name__ == '__main__':
    main()
