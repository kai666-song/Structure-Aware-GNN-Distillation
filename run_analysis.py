#!/usr/bin/env python
"""
Run All Advanced Analysis Experiments

Usage:
    python run_analysis.py --all                    # Run all analyses
    python run_analysis.py --homophily --data actor # Run homophily analysis on actor
    python run_analysis.py --robustness --all_data  # Run robustness on all datasets
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description='Advanced Analysis Experiments')
    
    # Analysis types
    parser.add_argument('--homophily', action='store_true', help='Run homophily analysis')
    parser.add_argument('--robustness', action='store_true', help='Run robustness study')
    parser.add_argument('--ablation', action='store_true', help='Run detailed ablation')
    parser.add_argument('--error', action='store_true', help='Run error analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    # Data options
    parser.add_argument('--data', type=str, default='actor', help='Dataset name')
    parser.add_argument('--all_data', action='store_true', help='Run on all datasets')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Determine which datasets to run
    if args.all_data:
        datasets = ['actor', 'cora', 'citeseer', 'pubmed']
    else:
        datasets = [args.data]
    
    # Run analyses
    if args.homophily or args.all:
        print("\n" + "="*70)
        print("RUNNING HOMOPHILY ANALYSIS")
        print("="*70)
        from analysis.homophily_analysis import run_homophily_analysis
        for dataset in datasets:
            try:
                run_homophily_analysis(dataset, args.num_runs, device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    
    if args.robustness or args.all:
        print("\n" + "="*70)
        print("RUNNING ROBUSTNESS STUDY")
        print("="*70)
        from analysis.robustness_study import run_robustness_study
        for dataset in datasets:
            try:
                run_robustness_study(dataset, args.num_runs, device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    
    if args.ablation or args.all:
        print("\n" + "="*70)
        print("RUNNING DETAILED ABLATION")
        print("="*70)
        from analysis.ablation_detailed import run_detailed_ablation
        for dataset in ['cora', 'actor']:  # Only run on key datasets
            try:
                run_detailed_ablation(dataset, args.num_runs, device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    
    if args.error or args.all:
        print("\n" + "="*70)
        print("RUNNING ERROR ANALYSIS")
        print("="*70)
        from analysis.error_analysis import run_error_analysis
        for dataset in ['actor']:  # Focus on heterophilic
            try:
                run_error_analysis(dataset, min(args.num_runs, 3), device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE!")
    print("="*70)
    print("\nResults saved to: results/")
    print("Figures saved to: figures/")


if __name__ == '__main__':
    main()
