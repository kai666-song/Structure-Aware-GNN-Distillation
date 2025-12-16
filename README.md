# Structure-Aware GNN Knowledge Distillation

This repository contains the implementation for **Graph Neural Network Knowledge Distillation** research, focusing on transferring knowledge from GCN (Teacher) to MLP (Student) for node classification tasks.

## Project Overview

Knowledge Distillation (KD) enables lightweight MLP models to learn from powerful GNN models, achieving competitive performance without requiring graph structure during inference.

### Key Features
- **Teacher Model**: 2-layer Graph Convolutional Network (GCN)
- **Student Model**: 2-layer Multi-Layer Perceptron (MLP)
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

### Run Baseline Benchmark (Step 1)
```bash
# Single dataset
python benchmark.py --data cora --epochs 200 --num_runs 10

# All datasets
python benchmark.py --all --epochs 200 --num_runs 10
```

### Run Knowledge Distillation Training
```bash
# Cora with KD
python run.py --data cora --dropout 0.7 --T 4.0 --lambda_kd 0.7

# Citeseer with KD
python run.py --data citeseer --dropout 0.5 --T 3.0 --lambda_kd 0.7

# PubMed with KD
python run.py --data pubmed --dropout 0.7 --T 6.0 --lambda_kd 0.8
```

## Baseline Results (Step 1)

| Dataset | GCN (Teacher) | MLP (Student) | Gap |
|---------|---------------|---------------|-----|
| Cora | 81.98 ± 0.54 | 59.06 ± 0.78 | 22.92 |
| Citeseer | 71.39 ± 0.51 | 59.33 ± 1.00 | 12.06 |
| PubMed | 79.05 ± 0.36 | 73.51 ± 0.28 | 5.54 |
| Amazon-Computers | 89.93 ± 0.30 | 84.04 ± 0.47 | 5.89 |
| Amazon-Photo | 93.90 ± 0.50 | 90.25 ± 0.90 | 3.65 |

*Dataset-specific hyperparameters with early stopping, runs=10*

## Project Structure

```
├── benchmark.py          # Baseline benchmark script
├── train.py              # Training logic with KD
├── run.py                # Main entry point
├── models.py             # GCN and MLP model definitions
├── layers.py             # Graph convolution layer
├── params.py             # Hyperparameter settings
├── kd_losses/            # Knowledge distillation loss functions
│   ├── __init__.py
│   └── st.py             # Soft Target loss
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── data_utils.py     # Data loading (Planetoid + Amazon/Coauthor)
│   └── utils.py          # Helper functions
├── data/                 # Dataset files
└── results/              # Experiment results
    └── step1_baseline_results.md
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.4.0
- torch_geometric
- numpy
- scipy
- networkx

## References

- [Hinton et al. 2015] Distilling the Knowledge in a Neural Network
- [Kipf & Welling 2017] Semi-Supervised Classification with Graph Convolutional Networks

## License

MIT License
