#!/usr/bin/env python
"""
Structure-Aware GNN Knowledge Distillation - Main Entry Point

Usage:
    python main.py --mode distill --data cora --num_runs 10
    python main.py --mode benchmark --all
    python main.py --mode heterophilic --num_runs 10
    python main.py --mode ablation --num_runs 5
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Structure-Aware GNN Knowledge Distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode distill --data cora --num_runs 10
  python main.py --mode benchmark --all
  python main.py --mode heterophilic --num_runs 10
  python main.py --mode significance
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['distill', 'benchmark', 'heterophilic', 
                                 'ablation', 'significance', 'citeseer_opt'],
                        help='Experiment mode to run')
    parser.add_argument('--data', type=str, default='cora',
                        help='Dataset name')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of runs')
    parser.add_argument('--all', action='store_true',
                        help='Run on all datasets (for benchmark mode)')
    parser.add_argument('--teacher', type=str, default='gat',
                        choices=['gcn', 'gat'],
                        help='Teacher model type')
    
    args = parser.parse_args()
    
    if args.mode == 'distill':
        print(f"Running distillation on {args.data} with {args.teacher} teacher...")
        from distill_gat import main as distill_main
        sys.argv = ['distill_gat.py', '--data', args.data, 
                    '--teacher', args.teacher, '--num_runs', str(args.num_runs)]
        distill_main()
        
    elif args.mode == 'benchmark':
        print("Running baseline benchmark...")
        from benchmark import main as benchmark_main
        if args.all:
            sys.argv = ['benchmark.py', '--all', '--num_runs', str(args.num_runs)]
        else:
            sys.argv = ['benchmark.py', '--data', args.data, '--num_runs', str(args.num_runs)]
        benchmark_main()
        
    elif args.mode == 'heterophilic':
        print("Running heterophilic graph experiments...")
        from experiments_improved import run_heterophilic_experiments
        
        class Args:
            cuda = True
            num_runs = args.num_runs
            alpha = 1.0
            beta = 1.0
            gamma = 1.0
            temperature = 4.0
            use_degree_aware = False
            min_degree = 2
            data = 'cora'
        
        run_heterophilic_experiments(Args())
        
    elif args.mode == 'ablation':
        print("Running ablation study...")
        from ablation_study import main as ablation_main
        sys.argv = ['ablation_study.py', '--num_runs', str(args.num_runs)]
        ablation_main()
        
    elif args.mode == 'significance':
        print("Running statistical significance tests...")
        from experiments_improved import run_significance_tests
        
        class Args:
            cuda = True
            num_runs = args.num_runs
            alpha = 1.0
            beta = 1.0
            gamma = 1.0
            temperature = 4.0
            use_degree_aware = False
            min_degree = 2
            data = 'cora'
        
        run_significance_tests(Args())
        
    elif args.mode == 'citeseer_opt':
        print("Running Citeseer optimization...")
        from experiments_improved import run_citeseer_optimization
        
        class Args:
            cuda = True
            num_runs = args.num_runs
            alpha = 1.0
            beta = 1.0
            gamma = 1.0
            temperature = 4.0
            use_degree_aware = False
            min_degree = 2
            data = 'citeseer'
        
        run_citeseer_optimization(Args())


if __name__ == '__main__':
    main()
