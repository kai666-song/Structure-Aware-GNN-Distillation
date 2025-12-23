"""
Strong Baselines for Heterophilic Graph Learning

This module implements state-of-the-art baselines for fair comparison:

1. GLNN (Graph-Less Neural Network) - Zhang et al., 2021
   - MLP distilled from GNN, the direct competitor to our method
   
2. GloGNN++ - Li et al., 2022
   - Global homophily-aware GNN, SOTA on heterophilic graphs

All models use Geom-GCN standard splits (48/32/20) for fair comparison.
"""

# Only import what exists
try:
    from .glnn import GLNN, train_glnn
except ImportError:
    pass

# GloGNN imports are in run_glognn_baseline.py
