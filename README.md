# Knowledge Distillation for Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **First systematic study of GNN-to-MLP knowledge distillation on heterophilic graphs. Simple Hinton KD with proper hyperparameter tuning enables MLP students to match or surpass GNN teachers.**

---

## Main Results

### Heterophilic Graphs (Main Contribution)

| Dataset | Teacher (GloGNN++) | Vanilla MLP | **Our Student** | Gap Closed |
|---------|-------------------|-------------|-----------------|------------|
| Actor | 37.74% | 35.22% | **37.53% ± 0.99%** | 91.7% |
| Squirrel | 60.11% | 30.49% | **60.37% ± 1.78%** | **100.9%** |
| Chameleon | 73.09% | 48.25% | **70.39% ± 1.53%** | 89.1% |

### Homophilic Graphs (Sanity Check)

| Dataset | Teacher (GCN) | Vanilla MLP | **Our Student** |
|---------|--------------|-------------|-----------------|
| Cora | 87.68% | 75.08% | **88.12% ± 1.64%** |
| CiteSeer | 76.85% | 72.15% | **77.34% ± 1.25%** |
| PubMed | 86.62% | 88.09% | **89.17% ± 0.39%** |

**Key Findings:**
- On Squirrel, Student **surpasses** Teacher (Gap Closed > 100%)
- On all homophilic datasets, Student **surpasses** Teacher
- Simple Hinton KD is sufficient - no need for complex methods
- **4/6 datasets: Student > Teacher**

---

## Best Configurations

| Dataset | Type | Temperature (T) | Lambda |
|---------|------|-----------------|--------|
| Actor | Heterophilic | 8.0 | 10.0 |
| Squirrel | Heterophilic | 1.0 | 10.0 |
| Chameleon | Heterophilic | 1.0 | 5.0 |
| Cora/CiteSeer/PubMed | Homophilic | 4.0 | 1.0 |

---

## Ablation Studies (Negative Results)

| Method | Effect | Conclusion |
|--------|--------|------------|
| Positional Encoding (RWPE) | -1.0% on Actor | Harmful |
| Relational KD (RKD) | Model collapse (~11%) | Catastrophic |
| Feature-based RKD | Model collapse (~11%) | Catastrophic |

**Conclusion:** Complex methods fail. Simple Hinton KD + hyperparameter tuning is the best approach.

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/GCN-with-Hinton-Knowledge-Distillation.git
cd GCN-with-Hinton-Knowledge-Distillation
pip install -r requirements.txt

# Clone GloGNN for heterophilic data and splits
git clone https://github.com/RecklessRonan/GloGNN.git ../GloGNN
```

### Dataset Information

| Dataset | Type | Nodes | Features | Classes | Source |
|---------|------|-------|----------|---------|--------|
| Actor | Heterophilic | 7,600 | 932 | 5 | GloGNN (Geom-GCN splits) |
| Squirrel | Heterophilic | 5,201 | 2,089 | 5 | GloGNN (Geom-GCN splits) |
| Chameleon | Heterophilic | 2,277 | 2,325 | 5 | GloGNN (Geom-GCN splits) |
| Cora | Homophilic | 2,708 | 1,433 | 7 | PyTorch Geometric |
| CiteSeer | Homophilic | 3,327 | 3,703 | 6 | PyTorch Geometric |
| PubMed | Homophilic | 19,717 | 500 | 3 | PyTorch Geometric |

**Data Download:**
- Heterophilic datasets (Actor, Squirrel, Chameleon): Automatically loaded from [GloGNN repository](https://github.com/RecklessRonan/GloGNN)
- Homophilic datasets (Cora, CiteSeer, PubMed): Automatically downloaded via PyTorch Geometric standard interfaces, or loaded from local `data/` directory

### Reproduce Results

```bash
# Run all 6 datasets
python run_all_experiments.py --mode all --device cuda

# Or run separately
python run_all_experiments.py --mode heterophilic --device cuda
python run_all_experiments.py --mode homophilic --device cuda
```

---

## Project Structure

```
├── run_all_experiments.py    # Main script for all 6 datasets
├── train_best_config.py      # Hyperparameter search
├── train_rkd.py              # RKD experiments (negative results)
├── models.py                 # Model definitions
├── layers.py                 # Layer definitions
│
├── baselines/
│   ├── verify_glognn_teacher.py  # Teacher verification
│   ├── save_teacher_logits.py    # Save teacher logits
│   └── glnn_baseline.py          # GLNN baseline
│
├── results/
│   ├── RESULTS.md                # Detailed results
│   └── final/                    # Final experiment results (JSON)
│
├── data/                         # Dataset files
│   ├── ind.cora.*                # Cora dataset
│   ├── ind.citeseer.*            # CiteSeer dataset
│   ├── ind.pubmed.*              # PubMed dataset
│   └── heterophilic/             # Heterophilic datasets
│
└── checkpoints/                  # Teacher logits
```

---

## Method

### Loss Function

Simple Hinton Knowledge Distillation:

```
L_total = L_CE(y, y_hat) + lambda_kd * T^2 * KL(softmax(z_s/T) || softmax(z_t/T))
```

Where:
- `L_CE`: Cross-entropy loss on hard labels
- `KL`: KL divergence between student and teacher soft labels
- `T`: Temperature (controls softness of probability distribution)
- `lambda_kd`: Weight for distillation loss

---

## Citation

```bibtex
@article{kd_heterophilic_graphs,
  title={Knowledge Distillation for Heterophilic Graphs},
  author={Your Name},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [GloGNN](https://github.com/RecklessRonan/GloGNN) for teacher model and data splits
- [GLNN](https://github.com/snap-stanford/graphless-neural-networks) for distillation framework
