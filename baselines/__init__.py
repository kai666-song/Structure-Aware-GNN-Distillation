"""
Strong Baselines for Heterophilic Graph Learning

This module implements state-of-the-art baselines for fair comparison:

1. GLNN (Graph-Less Neural Network) - Zhang et al., 2021
   - MLP distilled from GNN, the direct competitor to our method
   
2. NOSMOG - Tian et al., 2022
   - Structure-distilled MLP with virtual node augmentation
   
3. GloGNN++ - Li et al., 2022
   - Global homophily-aware GNN, SOTA on heterophilic graphs
   
4. ACM-GNN - Luan et al., 2022
   - Adaptive Channel Mixing GNN for heterophilic graphs

All models use Geom-GCN standard splits (48/32/20) for fair comparison.
"""

from .glnn import GLNN, train_glnn
from .strong_teachers import GloGNN, ACMGNN, train_strong_teacher
