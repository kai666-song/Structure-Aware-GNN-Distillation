# Structure-Aware GNN Knowledge Distillation

This repository contains the implementation for **Graph Neural Network Knowledge Distillation** research, focusing on transferring knowledge from GCN (Teacher) to MLP (Student) for node classification tasks.

## Project Overview

Knowledge Distillation (KD) enables lightweight MLP models to learn from powerful GNN models, achieving competitive performance without requiring graph structure during inference.

### Key Features
- **Teacher Models**: GCN and GAT (Graph Attention Network)
- **Student Model**: MLPBatchNorm (MLP with BatchNorm for better convergence)
- **Innovation**: Topology Consistency Distillation (TCD) - structure-aware loss that aligns with graph topology
- **Highlight**: Student MLP surpasses Teacher on 3/4 datasets with GAT!
- **Datasets**: Cora, Citeseer, PubMed, Amazon-Computers, Amazon-Photo

## Installation

```bash
# Clone the repository
git clone https://github.com/kai666-song/Structure-Aware-GNN-Distillation.git
cd Structure-Aware-GNN-Distillation

# Install dependencies
pip install torch numpy scipy networkx torch_geometric
```

## Usage

### Run Baseline Benchmark
```bash
python benchmark.py --all --num_runs 10
```

### Run Structure-Aware Distillation
```bash
# With GCN teacher
python distill.py --data cora --alpha 1.0 --beta 1.0 --gamma 1.0 --num_runs 10

# With GAT teacher + Topology Loss (SOTA)
python distill_gat.py --data cora --teacher gat --lambda_topo 1.0 --num_runs 10
```

## Results

### SOTA Results: GAT Teacher + Topology Consistency Loss

| Dataset | GAT Teacher | Student MLP | Gap |
|---------|-------------|-------------|-----|
| Cora | 82.74±0.74 | **82.99±1.22** | +0.25% ✨ |
| Citeseer | 71.39±0.89 | 71.08±1.06 | -0.31% |
| PubMed | 78.00±0.40 | **79.51±0.84** | +1.51% ✨ |
| Amazon-Photo | 94.27±0.46 | **94.48±0.76** | +0.21% ✨ |

### GCN Teacher Results

| Dataset | GCN Teacher | Student MLP | Gap |
|---------|-------------|-------------|-----|
| Cora | 82.04±0.48 | **82.45±0.79** | +0.41% ✨ |
| Citeseer | 71.63±0.46 | 71.80±0.60 | +0.17% |
| PubMed | 79.12±0.39 | **80.00±0.44** | +0.88% ✨ |
| Amazon-Computers | 89.93±0.30 | 81.21±3.28 | -8.72% |
| Amazon-Photo | 93.95±0.52 | 93.56±0.98 | -0.39% |

### Ablation Study: Structure Loss Contribution

| Dataset | MLP Baseline | GLNN (γ=0) | Ours (γ=1) | Gain |
|---------|--------------|------------|------------|------|
| Cora | 45.69±1.33 | 81.82±0.73 | **82.31±0.63** | +0.49% |
| Citeseer | 42.64±2.00 | 71.75±0.55 | 71.58±0.70 | -0.17% |
| PubMed | 66.69±1.10 | 80.09±0.61 | 80.02±0.52 | -0.07% |
| Amazon-Computers | 41.25±5.68 | 81.47±3.88 | **83.15±3.11** | +1.68% |
| Amazon-Photo | 89.92±0.69 | 92.85±1.22 | **93.52±0.85** | +0.67% |

### Key Findings
- **Student > Teacher**: MLP surpasses Teacher on 3/4 datasets with GAT + Topology Loss!
- **Best Result**: 94.48% on Amazon-Photo (Student > GAT Teacher)
- **Topology Loss**: Explicitly aligns student features with graph structure
- **Massive Improvement**: MLP baseline ~45% → Distilled MLP ~83% on Cora (+38%)
- **Speedup**: MLP is 4-10x faster than GNN at inference time

## Method

### Loss Function
The distillation loss combines four components:

```
L_total = α * L_task + β * L_kd + γ * L_rkd + λ * L_topo
```

- **L_task**: CrossEntropy with ground truth labels
- **L_kd**: KL divergence with teacher soft labels (temperature=4.0)
- **L_rkd**: Relational Knowledge Distillation (global pairwise similarity)
- **L_topo**: Topology Consistency Loss (edge-masked similarity alignment)

### Innovation: Topology Consistency Distillation (TCD)
Unlike vanilla RKD which ignores graph structure, TCD explicitly aligns student's feature similarity with the graph adjacency matrix:
- Only computes loss for connected node pairs (edges)
- Forces MLP to learn: "if nodes i,j are neighbors, their features should be similar"
- Transfers topological knowledge without requiring graph at inference

## Project Structure

```
├── benchmark.py          # Baseline benchmark script
├── distill.py            # Structure-aware distillation (GCN teacher)
├── distill_gat.py        # SOTA distillation (GAT teacher + Topology Loss)
├── distill_save.py       # Distillation with model saving
├── ablation_study.py     # Ablation study script
├── models.py             # GCN, GAT, MLP, MLPBatchNorm definitions
├── layers.py             # Graph convolution layer
├── kd_losses/            # Knowledge distillation loss functions
│   ├── st.py             # Soft Target loss
│   ├── rkd.py            # RKD loss (with AdaptiveRKDLoss)
│   └── topology_kd.py    # Topology Consistency Loss (TCD)
├── utils/                # Utility functions
├── data/                 # Dataset files
├── checkpoints/          # Saved model weights
├── figures/              # Visualization outputs
└── results/              # Experiment results
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.4.0
- torch_geometric
- numpy, scipy, networkx

## References

- [Hinton et al. 2015] Distilling the Knowledge in a Neural Network
- [Kipf & Welling 2017] Semi-Supervised Classification with Graph Convolutional Networks
- [Park et al. 2019] Relational Knowledge Distillation

## License

MIT License
