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
| Cora | 81.09 ± 0.28 | 60.09 ± 0.44 | 21.00 |
| Citeseer | 71.30 ± 0.33 | 59.87 ± 0.41 | 11.43 |
| PubMed | 79.39 ± 0.23 | 73.28 ± 0.28 | 6.11 |
| Amazon-Computers | 63.35 ± 1.17 | 60.28 ± 2.08 | 3.07 |
| Amazon-Photo | 79.09 ± 2.10 | 76.10 ± 1.05 | 2.99 |

*Settings: epochs=200, hidden=64, lr=0.01, dropout=0.5, runs=10*

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
