"""
Hyperparameter sweep for gamma (structure loss weight)
"""

import subprocess
import sys

gammas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
dataset = 'cora'
num_runs = 3

print("="*60)
print(f"Gamma Sweep on {dataset.upper()}")
print("="*60)

for gamma in gammas:
    print(f"\n>>> Testing gamma={gamma}")
    cmd = [
        sys.executable, 'distill.py',
        '--data', dataset,
        '--alpha', '1.0',
        '--beta', '1.0', 
        '--gamma', str(gamma),
        '--num_runs', str(num_runs)
    ]
    subprocess.run(cmd)
