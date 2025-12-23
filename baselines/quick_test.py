"""
Quick test script to verify baseline implementations
Runs only 1 split for fast verification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_glognn_baseline import run_glognn_experiment, GLOGNN_CONFIGS
from run_acmgnn_baseline import run_acmgnn_experiment, ACMGNN_CONFIGS

def main():
    print("="*60)
    print("QUICK BASELINE TEST (1 split each)")
    print("="*60)
    
    # Test GloGNN++ on Actor
    print("\n[1/2] Testing GloGNN++ on Actor...")
    mean_acc, std_acc, _ = run_glognn_experiment('actor', GLOGNN_CONFIGS['actor'], num_splits=1)
    print(f"GloGNN++ Actor: {mean_acc:.2f}%")
    
    # Test ACM-GNN on Actor
    print("\n[2/2] Testing ACM-GNN on Actor...")
    mean_acc, std_acc, _ = run_acmgnn_experiment('actor', ACMGNN_CONFIGS['actor'], num_splits=1)
    print(f"ACM-GNN Actor: {mean_acc:.2f}%")
    
    print("\n" + "="*60)
    print("Quick test completed!")
    print("="*60)

if __name__ == '__main__':
    main()
