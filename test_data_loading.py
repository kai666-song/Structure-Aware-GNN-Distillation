"""
Quick test script to verify data loading works correctly.

Usage:
    python test_data_loading.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test that data loading works for all datasets."""
    print("Testing data loading with GloGNN splits...")
    print("=" * 60)
    
    from utils.data_loader_v2 import load_data_with_glognn_splits
    
    datasets = ['actor', 'chameleon', 'squirrel']
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Testing {dataset}...")
        print("=" * 60)
        
        try:
            data = load_data_with_glognn_splits(dataset, split_idx=0)
            
            print(f"\n✓ Successfully loaded {dataset}")
            print(f"  - Nodes: {data['num_nodes']}")
            print(f"  - Features: {data['num_features']}")
            print(f"  - Classes: {data['num_classes']}")
            print(f"  - Train: {data['train_mask'].sum().item()}")
            print(f"  - Val: {data['val_mask'].sum().item()}")
            print(f"  - Test: {data['test_mask'].sum().item()}")
            print(f"  - Adj shape: {data['adj'].shape}")
            print(f"  - Features shape: {data['features'].shape}")
            
            # Verify split ratio
            n = data['num_nodes']
            train_ratio = data['train_mask'].sum().item() / n
            val_ratio = data['val_mask'].sum().item() / n
            test_ratio = data['test_mask'].sum().item() / n
            
            print(f"\n  Split ratio: {train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%}")
            
            # Check if close to 48/32/20 (Geom-GCN standard, used by GloGNN)
            if abs(train_ratio - 0.48) < 0.05 and abs(val_ratio - 0.32) < 0.05:
                print(f"  ✓ Split ratio matches Geom-GCN standard (48/32/20)")
            else:
                print(f"  ⚠ Split ratio does NOT match expected 48/32/20!")
                
        except Exception as e:
            print(f"\n✗ Failed to load {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Data loading test complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_data_loading()
