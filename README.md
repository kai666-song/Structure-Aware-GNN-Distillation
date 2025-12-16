# Structure-Aware GNN Knowledge Distillation

This repository contains the implementation for **Graph Neural Network Knowledge Distillation** research, focusing on transferring knowledge from GCN (Teacher) to MLP (Student) for node classification tasks.

## Project Overview

Knowledge Distillation (KD) enables lightweight MLP models to learn from powerful GNN models, achieving competitive performance without requiring graph structure during inference.

### Key Features
- **Teacher Model**: 2-layer Graph Convolutional Network (GCN)
- **Student Model**: 2-layer Multi-Layer Perceptron (MLP)
- **Innovation**: Structure-aware distillation with Relational Knowledge Distillation (RKD) loss
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

### Baseline vs Distillation Comparison

| Dataset | Teacher GCN | MLP Baseline | MLP Distilled | Improvement |
|---------|-------------|--------------|---------------|-------------|
| Cora | 82.04 | 59.06 | **72.77** | +13.71 |
| Citeseer | 71.63 | 59.33 | **68.79** | +9.46 |
| PubMed | 79.12 | 73.51 | **77.87** | +4.36 |
| Amazon-Computers | 89.93 | 84.04 | 83.86 | -0.18 |
| Amazon-Photo | 93.95 | 90.25 | **92.90** | +2.65 |

### Key Findings
- **Cora**: MLP improved from 59% to 73% (+14%), closing 60% of the gap with Teacher
- **Citeseer**: MLP improved from 59% to 69% (+10%), closing 77% of the gap
- **PubMed**: MLP nearly matches Teacher (77.87% vs 79.12%)
- **Amazon-Photo**: MLP achieves 92.90%, only 1% below Teacher

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
├── run_all_distill.py    # Run distillation on all datasets
├── models.py             # GCN and MLP model definitions
├── layers.py             # Graph convolution layer
├── kd_losses/            # Knowledge distillation loss functions
│   ├── st.py             # Soft Target loss
│   └── rkd.py            # Relational Knowledge Distillation loss
├── utils/                # Utility functions
│   ├── data_utils.py     # Data loading
│   └── utils.py          # Helper functions
├── data/                 # Dataset files
└── results/              # Experiment results
    ├── step1_baseline_results.md
    └── step2_distillation_results.md
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
