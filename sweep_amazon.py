"""
Parameter sweep for Amazon-Computers dataset
Try different alpha/gamma combinations to improve distillation
"""

import subprocess
import sys

# Amazon-Computers seems to need different hyperparameters
# Try: higher alpha (task loss), lower gamma (structure loss)

configs = [
    {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.0},   # Pure KD, no structure
    {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.1},   # Very low structure
    {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5},   # Low structure
    {'alpha': 2.0, 'beta': 1.0, 'gamma': 0.0},   # Higher task weight
    {'alpha': 2.0, 'beta': 1.0, 'gamma': 0.1},   # Higher task + low structure
    {'alpha': 1.0, 'beta': 2.0, 'gamma': 0.0},   # Higher KD weight
]

print("="*70)
print("AMAZON-COMPUTERS PARAMETER SWEEP")
print("="*70)

for config in configs:
    print(f"\n>>> Testing alpha={config['alpha']}, beta={config['beta']}, gamma={config['gamma']}")
    cmd = [
        sys.executable, 'distill.py',
        '--data', 'amazon-computers',
        '--alpha', str(config['alpha']),
        '--beta', str(config['beta']),
        '--gamma', str(config['gamma']),
        '--num_runs', '3'
    ]
    subprocess.run(cmd)
