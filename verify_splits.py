"""
Verify that Geom-GCN standard splits are being used for heterophilic datasets.
This is critical for academic rigor and fair comparison with published baselines.
"""
import torch
from torch_geometric.datasets import Actor, WikipediaNetwork

def verify_actor_splits():
    """Verify Actor dataset uses Geom-GCN standard splits."""
    print("=" * 60)
    print("Actor Dataset Split Verification")
    print("=" * 60)
    
    dataset = Actor(root='./data/actor')
    data = dataset[0]
    
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of features: {data.x.shape[1]}")
    print(f"Number of classes: {data.y.max().item() + 1}")
    print()
    
    if hasattr(data, 'train_mask'):
        print(f"train_mask shape: {data.train_mask.shape}")
        print(f"val_mask shape: {data.val_mask.shape}")
        print(f"test_mask shape: {data.test_mask.shape}")
        
        if data.train_mask.dim() == 2:
            print()
            print("✓ Geom-GCN standard splits detected (10 fixed splits)")
            print(f"  Number of splits available: {data.train_mask.shape[1]}")
            
            total = data.x.shape[0]
            print()
            print("Split ratio verification (Geom-GCN standard: 48/32/20):")
            for i in range(min(3, data.train_mask.shape[1])):
                train_pct = 100 * data.train_mask[:, i].sum().item() / total
                val_pct = 100 * data.val_mask[:, i].sum().item() / total
                test_pct = 100 * data.test_mask[:, i].sum().item() / total
                print(f"  Split {i}: {train_pct:.1f}% / {val_pct:.1f}% / {test_pct:.1f}%")
            
            return True
        else:
            print("⚠ Only 1D mask found (not Geom-GCN standard)")
            return False
    else:
        print("⚠ No train_mask found")
        return False


def verify_wikipedia_splits(name):
    """Verify WikipediaNetwork dataset uses Geom-GCN standard splits."""
    print()
    print("=" * 60)
    print(f"{name.capitalize()} Dataset Split Verification")
    print("=" * 60)
    
    dataset = WikipediaNetwork(root='./data/heterophilic', name=name, 
                               geom_gcn_preprocess=True)
    data = dataset[0]
    
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of features: {data.x.shape[1]}")
    print(f"Number of classes: {data.y.max().item() + 1}")
    print()
    
    if hasattr(data, 'train_mask'):
        print(f"train_mask shape: {data.train_mask.shape}")
        
        if data.train_mask.dim() == 2:
            print()
            print("✓ Geom-GCN standard splits detected (10 fixed splits)")
            print(f"  Number of splits available: {data.train_mask.shape[1]}")
            
            total = data.x.shape[0]
            print()
            print("Split ratio verification (Geom-GCN standard: 48/32/20):")
            for i in range(min(3, data.train_mask.shape[1])):
                train_pct = 100 * data.train_mask[:, i].sum().item() / total
                val_pct = 100 * data.val_mask[:, i].sum().item() / total
                test_pct = 100 * data.test_mask[:, i].sum().item() / total
                print(f"  Split {i}: {train_pct:.1f}% / {val_pct:.1f}% / {test_pct:.1f}%")
            
            return True
        else:
            print("⚠ Only 1D mask found")
            return False
    return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GEOM-GCN STANDARD SPLITS VERIFICATION")
    print("Reference: Pei et al., ICLR 2020")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Verify Actor
    results['actor'] = verify_actor_splits()
    
    # Verify Chameleon
    results['chameleon'] = verify_wikipedia_splits('chameleon')
    
    # Verify Squirrel
    results['squirrel'] = verify_wikipedia_splits('squirrel')
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    for dataset, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {dataset.capitalize()}: {status}")
    
    if all(results.values()):
        print("\n✓ All heterophilic datasets use Geom-GCN standard splits!")
        print("  This ensures fair comparison with published baselines.")
    else:
        print("\n⚠ Some datasets may not use standard splits.")
        print("  Please verify data loading code.")
