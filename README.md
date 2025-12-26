# Gated Adaptive Frequency-Decoupled Knowledge Distillation (GAFF-KD)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **GNN-to-MLP Knowledge Distillation on Heterophilic Graphs with Adaptive Frequency-Decoupled Learning**

---

## Key Innovation

**Gated Adaptive Frequency-Decoupled KD (GAFF-KD)**: A node-level gating mechanism that dynamically blends high-frequency spectral alignment (AFD) and standard soft-target KD based on local homophily.

- **Heterophilic nodes** → More AFD (spectral filtering helps)
- **Homophilic nodes** → More standard KD (simple soft targets work well)

This makes the model an "all-rounder" that works well across different graph regions.

---

## Main Results

### Filtered Datasets (Platonov et al., ICLR 2023)

| Dataset | Teacher | Simple KD | AFD-KD | **Gated AFD** | Δ vs Simple KD |
|---------|---------|-----------|--------|---------------|----------------|
| Squirrel_filtered | 35.83% | 35.92% | 35.87% | **38.76%** | **+2.84%** |
| Chameleon_filtered | 33.96% | 34.16% | 34.29% | **36.74%** | **+2.58%** |
| Roman-empire | 79.74% | **69.20%** | 68.79% | 67.26% | -1.94% |

### Homophily Breakdown (Chameleon_filtered)

| Homophily | Simple KD | Gated AFD | Improvement |
|-----------|-----------|-----------|-------------|
| h < 0.4 | 32.57% | 32.41% | -0.17% |
| h ≥ 0.6 | 47.22% | **57.18%** | **+9.97%** |

**Key Finding**: Gated AFD solves the "homophilic node performance drop" observed in pure AFD methods.

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/GCN-with-Hinton-Knowledge-Distillation.git
cd GCN-with-Hinton-Knowledge-Distillation
pip install -r requirements.txt

# Clone heterophilous-graphs for filtered datasets
git clone https://github.com/yandex-research/heterophilous-graphs ../heterophilous-graphs
```

### Run Main Experiments

```bash
# Run on filtered datasets (Squirrel, Chameleon)
python scripts/run_main_experiments.py --experiment filtered_datasets

# Run Gated AFD experiments
python train_gated_afd.py

# Run mechanism analysis
python analysis/deep_mechanism_analysis.py
```

---

## Project Structure

```
├── scripts/
│   └── run_main_experiments.py   # Main reproducibility script
│
├── kd_losses/
│   ├── __init__.py               # Module exports
│   └── adaptive_kd.py            # Core: AFDLoss, GatedAFDLoss, DualPathGatedLoss
│
├── analysis/
│   └── verify_on_filtered.py     # Data leakage verification
│
├── utils/
│   ├── data_utils.py             # Data loading utilities
│   └── utils.py                  # General utilities
│
├── train_gated_afd.py            # Gated AFD training script
├── train_afd_kd.py               # AFD-KD training script
├── verify_gated_homophily.py     # Homophily breakdown verification
│
├── data/
│   └── heterophilic/             # Heterophilic graph datasets
│
├── checkpoints/                  # Saved model checkpoints
├── results/                      # Experiment results
└── figures/                      # Generated figures
```

---

## Method

### Loss Function

**Gated AFD-KD Loss**:

```
L_total = L_CE + λ * L_gated

L_gated = Σ_i [g_i * L_soft_i + (1 - g_i) * L_AFD_i]

where:
  g_i = σ(w * (h_i - τ) + b)  # Node-level gate
  h_i = local homophily of node i
  L_soft = KL(softmax(z_s/T) || softmax(z_t/T))  # Standard KD
  L_AFD = ||h(L) @ Z_T - h(L) @ Z_S||^2  # Spectral alignment
  h(L) = Σ_k θ_k * B_{k,K}(L)  # Learnable Bernstein filter
```

### Key Components

1. **Bernstein Polynomial Spectral Filter**: Learnable frequency response using stable Bernstein basis
2. **Node-level Homophily Computer**: Measures local feature similarity with neighbors
3. **Adaptive Gate**: Dynamically blends AFD and standard KD per node

---

## Ablation Studies

### K-order Sensitivity

| K | Chameleon | Squirrel |
|---|-----------|----------|
| 3 | 35.59% | 36.05% |
| **5** | **36.88%** | 35.96% |
| 10 | 36.02% | 35.92% |

### Gate Threshold Sensitivity

| τ | Squirrel | Chameleon |
|---|----------|-----------|
| **0.3** | **38.76%** | 35.46% |
| 0.5 | 35.69% | 34.65% |
| 0.7 | 36.77% | 34.61% |

---

## Negative Results

| Method | Dataset | Effect | Conclusion |
|--------|---------|--------|------------|
| AFD-KD | Roman-empire | -0.41% | Harmful on pseudo-heterophilic graphs |
| Gated AFD | Roman-empire | -1.94% | Simple KD is optimal |
| RKD | All | Collapse | Catastrophic failure |

**Lesson**: Method applicability depends on graph characteristics. Roman-empire, despite low edge homophily, benefits from standard KD.

---

## Citation

```bibtex
@article{gated_afd_kd_2025,
  title={Gated Adaptive Frequency-Decoupled Knowledge Distillation for Heterophilic Graphs},
  author={Your Name},
  year={2025}
}
```

---

## References

- Platonov et al. "A Critical Look at the Evaluation of GNNs under Heterophily" (ICLR 2023)
- He et al. "BernNet: Learning Arbitrary Graph Spectral Filters" (NeurIPS 2021)
- Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)

---

## License

MIT License - see [LICENSE](LICENSE) for details.
