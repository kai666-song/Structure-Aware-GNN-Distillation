# Structure-Aware GNN Knowledge Distillation

This repository contains the implementation for **Graph Neural Network Knowledge Distillation** research, focusing on transferring knowledge from GCN (Teacher) to MLP (Student) for node classification tasks.

## Project Overview

Knowledge Distillation (KD) enables lightweight MLP models to learn from powerful GNN models, achieving competitive performance without requiring graph structure during inference.

### Key Features
- **Teacher Model**: 2-layer Graph Convolutional Network (GCN)
- **Student Model**: MLPBatchNorm (MLP with BatchNorm for better convergence)
- **Innovation**: Structure-aware distillation with Relational Knowledge Distillation (RKD) loss
- **Highlight**: Student surpasses Teacher on Cora and PubMed datasets!
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
# Single dataset
python distill.py --data cora --alpha 1.0 --beta 1.0 --gamma 1.0 --num_runs 10

# All datasets
python run_all_distill.py
```

## Results

### Main Results (10 runs average)

| Dataset | Teacher GCN | Student MLP | Gap |
|---------|-------------|-------------|-----|
| Cora | 82.04±0.48 | **82.45±0.79** | +0.41% ✨ |
| Citeseer | 71.63±0.46 | 71.80±0.60 | +0.17% |
| PubMed | 79.12±0.39 | **80.00±0.44** | +0.88% ✨ |
| Amazon-Computers | 89.93±0.30 | 81.21±3.28 | -8.72% |
| Amazon-Photo | 93.95±0.52 | 93.56±0.98 | -0.39% |

### Ablation Study: RKD Contribution

| Dataset | MLP Baseline | GLNN (γ=0) | Ours (γ=1) | RKD Gain |
|---------|--------------|------------|------------|----------|
| Cora | 45.69±1.33 | 81.82±0.73 | **82.31±0.63** | +0.49% |
| Citeseer | 42.64±2.00 | 71.75±0.55 | 71.58±0.70 | -0.17% |
| PubMed | 66.69±1.10 | 80.09±0.61 | 80.02±0.52 | -0.07% |
| Amazon-Computers | 41.25±5.68 | 81.47±3.88 | **83.15±3.11** | +1.68% |
| Amazon-Photo | 89.92±0.69 | 92.85±1.22 | **93.52±0.85** | +0.67% |

### Key Findings
- **Student > Teacher**: On Cora and PubMed, distilled MLP surpasses Teacher GCN!
- **RKD Effectiveness**: Structure loss provides significant gains on Amazon datasets (+1.68% on Computers)
- **Massive Improvement**: MLP baseline ~45% → Distilled MLP ~82% on Cora (+37%)
- **Speedup**: MLP is 4-10x faster than GCN at inference time

## Method

The distillation loss combines three components:

```
L_total = α * L_task + β * L_kd + γ * L_struct
```

- **L_task**: CrossEntropy with ground truth labels
- **L_kd**: KL divergence with teacher soft labels (temperature=4.0)
- **L_struct**: Relational Knowledge Distillation loss (cosine similarity alignment)

## Project Structure

```
├── benchmark.py          # Baseline benchmark script
├── distill.py            # Structure-aware distillation training
├── distill_save.py       # Distillation with model saving
├── ablation_study.py     # Ablation study script
├── models.py             # GCN, MLP, MLPBatchNorm definitions
├── layers.py             # Graph convolution layer
├── kd_losses/            # Knowledge distillation loss functions
│   ├── st.py             # Soft Target loss
│   └── rkd.py            # RKD loss (with AdaptiveRKDLoss for large graphs)
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
