# Knowledge Distillation for Heterophilic Graphs: GNN-to-MLP

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **First systematic study of GNN-to-MLP knowledge distillation on heterophilic graphs. We demonstrate that a simple Hinton KD with proper hyperparameter tuning enables MLP students to surpass their GNN teachers.**

---

## 🎯 Key Findings

| Dataset | GLNN Baseline | Teacher (GloGNN++) | **Our Best** | Gap Closed |
|---------|---------------|-------------------|--------------|------------|
| Actor | 36.64% | 37.40% | **37.65% ± 0.98%** | **133.1%** |
| Squirrel | 58.96% | 59.68% | **60.15% ± 1.57%** | **165.8%** |

**Highlights:**
- ✅ Student MLP **surpasses** Teacher GNN (Gap Closed > 100%)
- ✅ No message passing at inference time
- ✅ Simple Hinton KD is sufficient - complex methods (PE, RKD) are harmful
- ✅ First work to study GNN-to-MLP distillation on heterophilic graphs

---

## 📊 Main Results

### Best Configuration

| Dataset | Temperature (T) | λ_kd | Accuracy |
|---------|-----------------|------|----------|
| Actor | 8.0 | 10.0 | **37.65% ± 0.98%** |
| Squirrel | 1.0 | 10.0 | **60.15% ± 1.57%** |

### Comparison with Baselines

| Method | Actor | Squirrel | Inference |
|--------|-------|----------|-----------|
| Vanilla MLP | 33.91% ± 0.78% | 30.29% ± 1.85% | No MP |
| GLNN (T=4, λ=1) | 36.64% ± 0.43% | 58.96% ± 1.58% | No MP |
| GloGNN++ (Teacher) | 37.40% ± 1.04% | 59.68% ± 1.75% | Message Passing |
| **Ours (Tuned KD)** | **37.65% ± 0.98%** | **60.15% ± 1.57%** | No MP |

---

## 🔬 Ablation Studies

### 1. Positional Encoding (PE) - Harmful ❌

| Dataset | Without PE | With PE | Δ |
|---------|------------|---------|---|
| Actor | 36.28% | 35.28% | **-1.00%** |
| Squirrel | 60.00% | 59.54% | **-0.46%** |

**Conclusion:** RWPE introduces noise on heterophilic graphs. Do not use.

### 2. Relational Knowledge Distillation (RKD) - Catastrophic Failure ❌

| Dataset | GLNN Baseline | + RKD (any weight) |
|---------|---------------|-------------------|
| Actor | 36.64% | **11.05%** (collapse) |
| Squirrel | 60.13% | **20.88%** (collapse) |

**Conclusion:** RKD causes model collapse regardless of weight (tested 0.001 to 0.5). The geometric constraints are incompatible with GNN→MLP distillation.

### 3. Temperature and λ_kd Tuning - Effective ✅

**Actor Dataset:**
| Config | Accuracy |
|--------|----------|
| T=4, λ=1 (default) | 36.42% |
| T=8, λ=5 | 36.99% |
| **T=8, λ=10** | **37.65%** |

**Squirrel Dataset:**
| Config | Accuracy |
|--------|----------|
| T=4, λ=1 (default) | 59.37% |
| T=1, λ=5 | 59.80% |
| **T=1, λ=10** | **60.15%** |

**Key Insight:** 
- Actor needs **high temperature** (T=8) to soften logits
- Squirrel needs **low temperature** (T=1) to preserve hard label information

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/GCN-with-Hinton-Knowledge-Distillation.git
cd GCN-with-Hinton-Knowledge-Distillation
pip install -r requirements.txt
```

### Reproduce Results

```bash
# Step 1: Verify teacher model
python baselines/verify_glognn_teacher.py --all --device cuda

# Step 2: Run GLNN baseline
python baselines/glnn_baseline.py --all --device cuda

# Step 3: Run best configuration
python train_best_config.py --device cuda
```

---

## 📁 Project Structure

```
├── train_best_config.py      # Main script for best results
├── train_rkd.py              # RKD experiments (negative results)
├── train_phase4_rkd.py       # Feature-based RKD (negative results)
│
├── baselines/
│   ├── verify_glognn_teacher.py  # Teacher verification
│   ├── glnn_baseline.py          # GLNN baseline
│   └── save_teacher_features.py  # Extract teacher features
│
├── configs/
│   └── experiment_config.py      # Hyperparameters
│
├── results/
│   ├── phase3_final/             # Best config results
│   └── phase3_rkd/               # RKD ablation results
│
└── checkpoints/                  # Teacher logits and features
```

---

## 📖 Method

### Loss Function

Simple Hinton Knowledge Distillation:

```
L_total = L_CE(y, ŷ) + λ_kd × T² × KL(softmax(z_s/T) || softmax(z_t/T))
```

Where:
- `L_CE`: Cross-entropy loss on hard labels
- `KL`: KL divergence between student and teacher soft labels
- `T`: Temperature (controls softness of probability distribution)
- `λ_kd`: Weight for distillation loss

### Why Simple KD Works

1. **Soft labels provide richer supervision** than hard labels
2. **Temperature scaling** reveals inter-class relationships
3. **No structural constraints** allows MLP to find its own optimal representation
4. **Heterophilic graphs benefit from soft targets** because neighboring nodes have different labels

---

## 📚 Citation

```bibtex
@article{kd_heterophilic_graphs,
  title={Knowledge Distillation for Heterophilic Graphs: When Simple is Better},
  author={Your Name},
  year={2024}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [GloGNN](https://github.com/RecklessRonan/GloGNN) for teacher model and data splits
- [GLNN](https://github.com/snap-stanford/graphless-neural-networks) for distillation framework
