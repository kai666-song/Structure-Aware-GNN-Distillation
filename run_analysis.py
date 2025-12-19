#!/usr/bin/env python
"""
Run All Advanced Analysis Experiments

Usage:
    python run_analysis.py --all                    # Run all analyses
    python run_analysis.py --homophily --data actor # Run homophily analysis on actor
    python run_analysis.py --robustness --all_data  # Run robustness on all datasets
    python run_analysis.py --stronger_teacher       # Test GCNII as stronger teacher
    python run_analysis.py --figures                # Generate publication figures
    python run_analysis.py --feature                # Feature space analysis
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
    parser.add_argument('--stronger_teacher', action='store_true', help='Test GCNII as stronger teacher')
    parser.add_argument('--figures', action='store_true', help='Generate publication figures')
    parser.add_argument('--feature', action='store_true', help='Feature space analysis')
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
        for dataset in ['cora', 'actor']:
            try:
                run_detailed_ablation(dataset, args.num_runs, device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    
    if args.error or args.all:
        print("\n" + "="*70)
        print("RUNNING ERROR ANALYSIS")
        print("="*70)
        from analysis.error_analysis import run_error_analysis
        for dataset in ['actor']:
            try:
                run_error_analysis(dataset, min(args.num_runs, 3), device)
            except Exception as e:
                print(f"Error on {dataset}: {e}")
    
    if args.stronger_teacher or args.all:
        print("\n" + "="*70)
        print("RUNNING STRONGER TEACHER EXPERIMENT (GCNII)")
        print("="*70)
        from analysis.stronger_teacher import run_stronger_teacher_experiment
        try:
            run_stronger_teacher_experiment('actor', args.num_runs, device)
        except Exception as e:
            print(f"Error: {e}")
    
    if args.feature or args.all:
        print("\n" + "="*70)
        print("RUNNING FEATURE SPACE ANALYSIS")
        print("="*70)
        from analysis.feature_visualization import run_feature_analysis
        try:
            run_feature_analysis('actor', min(args.num_runs, 3), device)
        except Exception as e:
            print(f"Error: {e}")
    
    if args.figures or args.all:
        print("\n" + "="*70)
        print("GENERATING PUBLICATION FIGURES")
        print("="*70)
        from analysis.generate_figures import generate_all_figures
        try:
            generate_all_figures()
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE!")
    print("="*70)
    print("\nResults saved to: results/")
    print("Figures saved to: figures/")


if __name__ == '__main__':
    main()
