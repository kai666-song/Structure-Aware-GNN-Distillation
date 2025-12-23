"""
Phase 1: Establish the True Bar
================================

This script runs all strong baselines on Actor and Squirrel datasets
using Geom-GCN standard splits (10 folds) to establish the performance
bar that our knowledge distillation method must exceed.

Target Performance (from literature):
- Actor:    GloGNN++ ~36.5%, ACM-GNN ~37.5%
- Squirrel: GloGNN++ ~38%, ACM-GNN ~40%

These are the "true" baselines we need to beat, NOT GAT's 27%.

Usage:
    python run_phase1_baselines.py --all
    python run_phase1_baselines.py --glognn --dataset actor
    python run_phase1_baselines.py --acmgnn --dataset squirrel --save_teacher
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

# Add baselines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'baselines'))


def run_glognn_baseline(dataset, num_splits=10, cuda=False, save_results=True):
    """Run GloGNN++ baseline"""
    print(f"\n{'='*60}")
    print(f"Running GloGNN++ on {dataset.upper()}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        os.path.join('baselines', 'run_glognn_baseline.py'),
        '--dataset', dataset,
        '--num_splits', str(num_splits)
    ]
    if cuda:
        cmd.append('--cuda')
    if save_results:
        cmd.append('--save_results')
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0


def run_acmgnn_baseline(dataset, num_splits=10, cuda=False, save_results=True, save_teacher=False):
    """Run ACM-GNN baseline"""
    print(f"\n{'='*60}")
    print(f"Running ACM-GNN on {dataset.upper()}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        os.path.join('baselines', 'run_acmgnn_baseline.py'),
        '--dataset', dataset,
        '--num_splits', str(num_splits)
    ]
    if cuda:
        cmd.append('--cuda')
    if save_results:
        cmd.append('--save_results')
    if save_teacher:
        cmd.append('--save_teacher')
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0


def generate_summary_report():
    """Generate a summary report of all baseline results"""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY: BASELINE PERFORMANCE")
    print("="*70)
    
    # Target performance from literature
    targets = {
        'actor': {'GloGNN++': 36.5, 'ACM-GNN': 37.5},
        'squirrel': {'GloGNN++': 38.0, 'ACM-GNN': 40.0}
    }
    
    summary = {}
    
    for dataset in ['actor', 'squirrel']:
        print(f"\n{dataset.upper()}:")
        print("-" * 40)
        
        summary[dataset] = {}
        
        # Load GloGNN results
        glognn_file = os.path.join(results_dir, f'glognn_baseline_{dataset}.json')
        if os.path.exists(glognn_file):
            with open(glognn_file) as f:
                data = json.load(f)
            print(f"  GloGNN++: {data['mean_acc']:.2f}% ± {data['std_acc']:.2f}%")
            print(f"    Target: {targets[dataset]['GloGNN++']}%")
            summary[dataset]['glognn'] = data
        else:
            print(f"  GloGNN++: Not run yet")
        
        # Load ACM-GNN results
        acmgnn_file = os.path.join(results_dir, f'acmgnn_baseline_{dataset}.json')
        if os.path.exists(acmgnn_file):
            with open(acmgnn_file) as f:
                data = json.load(f)
            print(f"  ACM-GNN:  {data['mean_acc']:.2f}% ± {data['std_acc']:.2f}%")
            print(f"    Target: {targets[dataset]['ACM-GNN']}%")
            summary[dataset]['acmgnn'] = data
        else:
            print(f"  ACM-GNN:  Not run yet")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. If baselines match literature targets:
   - Use ACM-GNN as the Teacher model for distillation
   - Save teacher logits for all 10 splits
   
2. If baselines underperform:
   - Tune hyperparameters
   - Check data loading is correct
   - Verify Geom-GCN splits are being used
   
3. For distillation:
   - Run: python run_phase1_baselines.py --acmgnn --dataset actor --save_teacher
   - This saves teacher model weights and soft logits for each split
""")
    
    # Save summary
    summary_file = os.path.join(results_dir, 'phase1_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': summary,
            'targets': targets,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Run Strong Baselines')
    parser.add_argument('--all', action='store_true', help='Run all baselines on all datasets')
    parser.add_argument('--glognn', action='store_true', help='Run GloGNN++ baseline')
    parser.add_argument('--acmgnn', action='store_true', help='Run ACM-GNN baseline')
    parser.add_argument('--dataset', type=str, default='actor', 
                       choices=['actor', 'squirrel', 'chameleon', 'all'])
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--save_teacher', action='store_true',
                       help='Save teacher model and logits for distillation')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')
    args = parser.parse_args()
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    datasets = ['actor', 'squirrel'] if args.dataset == 'all' else [args.dataset]
    
    if args.all:
        # Run all baselines on all datasets
        for dataset in datasets:
            run_glognn_baseline(dataset, args.num_splits, args.cuda)
            run_acmgnn_baseline(dataset, args.num_splits, args.cuda, 
                              save_teacher=args.save_teacher)
        generate_summary_report()
    else:
        if args.glognn:
            for dataset in datasets:
                run_glognn_baseline(dataset, args.num_splits, args.cuda)
        
        if args.acmgnn:
            for dataset in datasets:
                run_acmgnn_baseline(dataset, args.num_splits, args.cuda,
                                  save_teacher=args.save_teacher)
        
        if args.summary or (args.glognn or args.acmgnn):
            generate_summary_report()
    
    if not any([args.all, args.glognn, args.acmgnn, args.summary]):
        parser.print_help()
        print("\nExample usage:")
        print("  python run_phase1_baselines.py --all")
        print("  python run_phase1_baselines.py --glognn --dataset actor")
        print("  python run_phase1_baselines.py --acmgnn --dataset actor --save_teacher")


if __name__ == '__main__':
    main()
