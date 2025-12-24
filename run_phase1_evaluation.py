"""
Phase 1: Unassailable Evaluation Protocol
==========================================

This script runs all Phase 1 tasks in sequence:
1. Verify data splits are using GloGNN's official 60/20/20 splits
2. Verify GloGNN++ teacher reproduces original paper results
3. Run GLNN baseline (the TRUE competitor)
4. Generate summary report

CRITICAL: Do NOT proceed to Phase 2 until all verifications pass!

Usage:
    python run_phase1_evaluation.py --all
    python run_phase1_evaluation.py --dataset actor
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.experiment_config import (
    HETEROPHILIC_DATASETS,
    GLOGNN_REPORTED_RESULTS,
    REPRODUCTION_TOLERANCE,
    NUM_SPLITS,
    RESULTS_DIR,
    print_config,
)


def run_task1_verify_splits(datasets):
    """Task 1: Verify data splits are correct."""
    print("\n" + "=" * 70)
    print("TASK 1: Verifying Data Splits")
    print("=" * 70)
    
    from utils.data_loader_v2 import load_data_with_glognn_splits
    
    all_passed = True
    results = {}
    
    for dataset in datasets:
        print(f"\nChecking {dataset}...")
        try:
            # Load split 0 to verify
            data = load_data_with_glognn_splits(dataset, split_idx=0)
            
            # Verify split ratio (should be ~48/32/20 for Geom-GCN standard)
            n = data['num_nodes']
            train_ratio = data['train_mask'].sum().item() / n
            val_ratio = data['val_mask'].sum().item() / n
            test_ratio = data['test_mask'].sum().item() / n
            
            # Check if close to 48/32/20 (Geom-GCN standard, used by GloGNN)
            expected_train, expected_val, expected_test = 0.48, 0.32, 0.20
            tolerance = 0.05  # 5% tolerance
            
            train_ok = abs(train_ratio - expected_train) < tolerance
            val_ok = abs(val_ratio - expected_val) < tolerance
            test_ok = abs(test_ratio - expected_test) < tolerance
            
            passed = train_ok and val_ok and test_ok
            
            results[dataset] = {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'passed': passed,
            }
            
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  Split ratio: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")
            print(f"  Expected:    48%/32%/20% (Geom-GCN standard)")
            print(f"  Status:      {status}")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results[dataset] = {'error': str(e), 'passed': False}
            all_passed = False
    
    return all_passed, results


def run_task2_verify_teacher(datasets, device='cpu'):
    """Task 2: Verify GloGNN++ teacher performance."""
    print("\n" + "=" * 70)
    print("TASK 2: Verifying GloGNN++ Teacher Performance")
    print("=" * 70)
    
    from baselines.verify_glognn_teacher import verify_teacher
    
    all_passed = True
    results = {}
    
    for dataset in datasets:
        mean_acc, std_acc, passed = verify_teacher(dataset, device=device)
        results[dataset] = {
            'mean': mean_acc,
            'std': std_acc,
            'target': GLOGNN_REPORTED_RESULTS[dataset],
            'passed': passed,
        }
        if not passed:
            all_passed = False
    
    return all_passed, results


def run_task3_glnn_baseline(datasets, device='cpu'):
    """Task 3: Run GLNN baseline."""
    print("\n" + "=" * 70)
    print("TASK 3: Running GLNN Baseline")
    print("=" * 70)
    
    from baselines.glnn_baseline import run_glnn_baseline
    
    results = {}
    
    for dataset in datasets:
        result = run_glnn_baseline(dataset, device=device)
        results[dataset] = {
            'glnn_mean': result['glnn_mean'],
            'glnn_std': result['glnn_std'],
            'mlp_mean': result['mlp_mean'],
            'mlp_std': result['mlp_std'],
            'teacher_target': result['teacher_target'],
        }
    
    return results


def generate_summary_report(task1_results, task2_results, task3_results, datasets):
    """Generate Phase 1 summary report."""
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY REPORT")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'datasets': datasets,
        'task1_splits': task1_results,
        'task2_teacher': task2_results,
        'task3_glnn': task3_results,
    }
    
    # Task 1 Summary
    print("\n📋 Task 1: Data Splits Verification")
    print("-" * 50)
    task1_passed = all(r.get('passed', False) for r in task1_results.values())
    for ds, r in task1_results.items():
        if 'error' in r:
            print(f"  ✗ {ds}: ERROR - {r['error']}")
        else:
            status = "✓" if r['passed'] else "✗"
            print(f"  {status} {ds}: {r['train_ratio']:.1%}/{r['val_ratio']:.1%}/{r['test_ratio']:.1%}")
    
    # Task 2 Summary
    print("\n📋 Task 2: Teacher Verification")
    print("-" * 50)
    task2_passed = all(r.get('passed', False) for r in task2_results.values())
    for ds, r in task2_results.items():
        status = "✓" if r['passed'] else "✗"
        diff = r['mean'] - r['target']
        print(f"  {status} {ds}: {r['mean']:.2f}% (target: {r['target']:.2f}%, diff: {diff:+.2f}%)")
    
    # Task 3 Summary
    print("\n📋 Task 3: GLNN Baseline Results")
    print("-" * 50)
    print(f"  {'Dataset':<12} {'Teacher':<10} {'GLNN':<18} {'MLP':<18}")
    print("  " + "-" * 58)
    for ds, r in task3_results.items():
        print(f"  {ds:<12} {r['teacher_target']:<10.2f} "
              f"{r['glnn_mean']:.2f}±{r['glnn_std']:.2f}{'':>4} "
              f"{r['mlp_mean']:.2f}±{r['mlp_std']:.2f}")
    
    # Overall Status
    print("\n" + "=" * 70)
    print("OVERALL STATUS")
    print("=" * 70)
    
    all_passed = task1_passed and task2_passed
    
    if all_passed:
        print("✓ All Phase 1 verifications PASSED!")
        print("  You may proceed to Phase 2 (method development).")
        print("\n  Your targets to beat:")
        for ds, r in task3_results.items():
            print(f"    {ds}: GLNN = {r['glnn_mean']:.2f}% ± {r['glnn_std']:.2f}%")
    else:
        print("✗ Some verifications FAILED!")
        if not task1_passed:
            print("  - Task 1 (Data Splits): FAILED")
        if not task2_passed:
            print("  - Task 2 (Teacher Verification): FAILED")
        print("\n  DO NOT proceed until all verifications pass.")
    
    report['all_passed'] = bool(all_passed)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    report = convert_to_serializable(report)
    
    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, 'phase1_summary.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Evaluation Protocol')
    parser.add_argument('--dataset', type=str, default='actor',
                       choices=['actor', 'chameleon', 'squirrel'])
    parser.add_argument('--all', action='store_true',
                       help='Run on all datasets')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    parser.add_argument('--skip_task1', action='store_true',
                       help='Skip Task 1 (data splits verification)')
    parser.add_argument('--skip_task2', action='store_true',
                       help='Skip Task 2 (teacher verification)')
    parser.add_argument('--skip_task3', action='store_true',
                       help='Skip Task 3 (GLNN baseline)')
    args = parser.parse_args()
    
    datasets = HETEROPHILIC_DATASETS if args.all else [args.dataset]
    
    # Print configuration
    print_config()
    
    # Run tasks
    task1_results = {}
    task2_results = {}
    task3_results = {}
    
    if not args.skip_task1:
        task1_passed, task1_results = run_task1_verify_splits(datasets)
        if not task1_passed:
            print("\n⚠️  Task 1 failed. Please fix data loading before proceeding.")
            if not args.skip_task2 and not args.skip_task3:
                print("    Continuing with remaining tasks for diagnostic purposes...")
    
    if not args.skip_task2:
        task2_passed, task2_results = run_task2_verify_teacher(datasets, args.device)
        if not task2_passed:
            print("\n⚠️  Task 2 failed. Teacher verification did not pass.")
            print("    Please check GloGNN++ hyperparameters and data loading.")
    
    if not args.skip_task3:
        # Task 3 requires Task 2 to have saved teacher logits
        if args.skip_task2:
            print("\n⚠️  Skipping Task 3 because Task 2 was skipped.")
            print("    GLNN baseline requires teacher logits from Task 2.")
        else:
            task3_results = run_task3_glnn_baseline(datasets, args.device)
    
    # Generate summary
    if task1_results or task2_results or task3_results:
        generate_summary_report(task1_results, task2_results, task3_results, datasets)


if __name__ == '__main__':
    main()
