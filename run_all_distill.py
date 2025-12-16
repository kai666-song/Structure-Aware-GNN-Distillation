"""
Run distillation experiments on all datasets
"""

import subprocess
import sys

datasets = ['cora', 'citeseer', 'pubmed', 'amazon-computers', 'amazon-photo']
num_runs = 10

print("="*70)
print("STRUCTURE-AWARE KNOWLEDGE DISTILLATION - FULL EXPERIMENT")
print("="*70)

for dataset in datasets:
    print(f"\n>>> Running {dataset.upper()}")
    cmd = [
        sys.executable, 'distill.py',
        '--data', dataset,
        '--alpha', '1.0',
        '--beta', '1.0', 
        '--gamma', '1.0',
        '--num_runs', str(num_runs)
    ]
    subprocess.run(cmd)

print("\n" + "="*70)
print("ALL EXPERIMENTS COMPLETED")
print("="*70)
