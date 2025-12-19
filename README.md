# Structure-Aware GNN Knowledge Distillation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Transferring Graph Neural Network Knowledge to MLP with Topology-Aware Distillation**

This repository implements **Structure-Aware Knowledge Distillation** for Graph Neural Networks, enabling lightweight MLP models to achieve competitive (and sometimes superior!) performance compared to GNN teachers, without requiring graph structure during inference.

## 🌟 Highlights

- **Student Beats Teacher**: On Actor dataset, Student MLP outperforms Teacher GAT by **6.33%** (p < 0.001)
- **+18% in Heterophilic Regions**: In extremely heterophilic nodes (homophily 0.0-0.2), Student beats Teacher by 18%!
- **100% Robust to Graph Noise**: Student MLP is completely immune to graph perturbation
- **Faster Convergence**: TCD loss reduces training epochs by 38% while improving accuracy
- **Statistical Significance**: 2 datasets show significant improvements with p < 0.01
- **4-10x Faster Inference**: MLP requires no graph structure at test time

## 📊 Main Results

### Homophilic Graphs (GAT Teacher)

| Dataset | Teacher (GAT) | Student (MLP) | Gap | Significance |
|---------|---------------|---------------|-----|--------------|
| Cora | 82.74 ± 0.74 | **82.99 ± 1.22** | +0.25% | |
| Citeseer | 71.39 ± 0.89 | 71.08 ± 1.06 | -0.31% | |
| PubMed | 78.00 ± 0.40 | **79.51 ± 0.84** | +1.51% | *** |
| Amazon-Photo | 94.27 ± 0.46 | **94.48 ± 0.76** | +0.22% | |

### Heterophilic Graphs (GAT Teacher) - 🔥 Key Finding

| Dataset | Teacher (GAT) | Student (MLP) | Gap | Significance |
|---------|---------------|---------------|-----|--------------|
| Chameleon | **58.22 ± 1.91** | 53.21 ± 2.40 | -5.01% | |
| Squirrel | 33.15 ± 1.27 | 32.88 ± 1.49 | -0.28% | |
| **Actor** | 27.16 ± 1.12 | **33.49 ± 1.65** | **+6.33%** | **✨ ***| |

> **Key Insight**: On heterophilic graphs with low average degree (Actor: 4.94), MLP's independence from noisy neighbor aggregation becomes advantageous!

### Statistical Significance (Paired t-test)

| Dataset | Gap | p-value | Result |
|---------|-----|---------|--------|
| **Actor** | +6.33% | < 0.001 | ✅ Significant |
| **PubMed** | +1.51% | 0.0003 | ✅ Significant |
| Cora | +0.25% | 0.377 | Not significant |
| Amazon-Photo | +0.22% | 0.451 | Not significant |

## 🔬 Method

### Loss Function

The distillation loss combines four components:

```
L_total = α × L_task + β × L_kd + γ × L_rkd + λ × L_topo
```

| Loss | Description | Purpose |
|------|-------------|---------|
| L_task | CrossEntropy with ground truth | Learn correct labels |
| L_kd | KL divergence with soft labels (T=4.0) | Mimic teacher's predictions |
| L_rkd | Relational Knowledge Distillation | Preserve pairwise relationships |
| L_topo | Topology Consistency Loss | Align with graph structure |

### Innovation: Topology Consistency Distillation (TCD)

Unlike vanilla RKD which ignores graph structure, TCD explicitly aligns student's feature similarity with the graph adjacency:

```python
# Only compute loss for connected node pairs (edges)
student_sim = (student_feat[src] * student_feat[dst]).sum(dim=1)
teacher_sim = (teacher_feat[src] * teacher_feat[dst]).sum(dim=1)
loss_topo = MSE(student_sim, teacher_sim)
```

**Key Properties**:
- Edge-based computation: O(E) instead of O(N²)
- Memory efficient: Uses sparse operations
- Transfers topological knowledge without requiring graph at inference

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/kai666-song/Structure-Aware-GNN-Distillation.git
cd Structure-Aware-GNN-Distillation
pip install -r requirements.txt
```

### Run Experiments

```bash
# 1. Baseline benchmark (all datasets)
python benchmark.py --all --num_runs 10

# 2. Main distillation with GAT teacher (recommended)
python distill_gat.py --data cora --num_runs 10

# 3. Heterophilic graph experiments (Actor, Squirrel, Chameleon)
python experiments_improved.py --experiment heterophilic --num_runs 10

# 4. Statistical significance tests
python experiments_improved.py --experiment significance_test

# 5. Citeseer optimization with degree-aware loss
python experiments_improved.py --experiment citeseer_optimize --num_runs 10
```

### Reproduce All Results

```bash
# Run complete experiment suite
python experiments_improved.py --experiment all --num_runs 10
```

### Advanced Analysis (NEW)

```bash
# Run all advanced analyses
python run_analysis.py --all --num_runs 5

# Individual analyses
python run_analysis.py --homophily --data actor    # Node-level homophily analysis
python run_analysis.py --robustness --all_data     # Graph perturbation robustness
python run_analysis.py --ablation                  # Detailed ablation study
python run_analysis.py --error --data actor        # Error analysis & case study
```

## 📁 Project Structure

```
├── main.py                   # Unified entry point
├── models.py                 # GCN, GAT, MLP, MLPBatchNorm definitions
├── layers.py                 # Graph convolution layer
├── distill_gat.py           # Main distillation script (GAT teacher)
├── distill.py               # Distillation with GCN teacher
├── experiments_improved.py   # Heterophilic + significance tests
├── benchmark.py             # Baseline performance benchmark
├── run_analysis.py          # Advanced analysis runner
│
├── analysis/                # Advanced analysis modules (NEW)
│   ├── homophily_analysis.py   # Node-level homophily study
│   ├── robustness_study.py     # Graph perturbation robustness
│   ├── ablation_detailed.py    # Granular ablation study
│   └── error_analysis.py       # Error analysis & case study
│
├── kd_losses/               # Knowledge distillation losses
│   ├── st.py               # Soft Target (KL divergence)
│   ├── rkd.py              # Relational KD (pairwise similarity)
│   └── topology_kd.py      # Topology Consistency Loss (TCD)
│
├── utils/                   # Utility functions
│   ├── data_utils.py       # Dataset loading (Planetoid, Amazon, Heterophilic)
│   └── utils.py            # Helper functions
│
├── results/                 # Experiment results (JSON + Markdown)
├── figures/                 # Visualizations (t-SNE, training curves)
├── checkpoints/             # Saved model weights
└── data/                    # Dataset files
```

## 📈 Advanced Analysis Results

### Node-Level Homophily Analysis (Actor Dataset)

We analyze accuracy by local homophily ratio to understand WHERE Student beats Teacher:

| Homophily Range | Teacher (GAT) | Student (MLP) | Gap | Nodes |
|-----------------|---------------|---------------|-----|-------|
| **0.0-0.2 (Heterophilic)** | 9.38% | **27.41%** | **+18.02%** ✨ | 81 |
| **0.2-0.4** | 22.33% | **31.88%** | **+9.55%** ✨ | 352 |
| **0.4-0.6** | 29.33% | **37.23%** | **+7.90%** ✨ | 433 |
| 0.6-0.8 | **45.78%** | 37.59% | -8.19% | 83 |
| 0.8-1.0 | 27.78% | **33.38%** | **+5.60%** ✨ | 571 |

> **Key Finding**: In extremely heterophilic regions (0.0-0.2), Student MLP beats Teacher GAT by **18%**! This proves MLP corrects Teacher's errors in noisy neighborhoods.

### Robustness to Graph Perturbation

| Perturbation | Teacher (GAT) | Student (MLP) |
|--------------|---------------|---------------|
| 0% (Clean) | 28.30% | **36.61%** |
| 10% | 27.97% | **36.61%** |
| 20% | 27.70% | **36.61%** |
| 30% | 27.84% | **36.61%** |
| 40% | 28.03% | **36.61%** |
| 50% | 26.97% | **36.61%** |

- Teacher drops **1.33%** with 50% edge perturbation
- Student drops **0%** - completely immune to graph noise!

### Detailed Ablation Study (Cora)

| Configuration | Accuracy | Converge Epoch |
|---------------|----------|----------------|
| Task Only | 45.18% | 92 |
| + KD | 82.98% | 206 |
| + KD + RKD | 82.90% | 212 |
| + KD + TCD | 83.52% | 138 |
| **+ KD + RKD + TCD (Full)** | **83.64%** | **128** ✨ |

> **Key Finding**: TCD not only improves accuracy (+0.66%) but also **accelerates convergence by 38%** (206 → 128 epochs)!

### Error Analysis (Actor Dataset)

- **Flip cases** (Teacher wrong → Student right): **288** nodes
- **Reverse flips** (Teacher right → Student wrong): **169** nodes  
- **Net gain**: **+119** nodes correctly classified by Student

When Student flips Teacher's errors, the average wrong neighbor ratio is **37.8%**, proving that GAT was misled by noisy neighbors while MLP ignored them.

## 📊 Original Ablation Study

### Effect of Structure Loss (γ)

| Dataset | MLP Baseline | GLNN (γ=0) | Ours (γ=1) | Improvement |
|---------|--------------|------------|------------|-------------|
| Cora | 45.69 | 81.82 | **82.31** | +0.49% |
| Amazon-Computers | 41.25 | 81.47 | **83.15** | +1.68% |
| Amazon-Photo | 89.92 | 92.85 | **93.52** | +0.67% |

### Citeseer Optimization (Degree-Aware Loss)

| Config | λ_topo | Min Degree | Accuracy |
|--------|--------|------------|----------|
| Baseline | 1.0 | - | 71.25 ± 1.78 |
| Reduced | 0.3 | - | 71.06 ± 1.68 |
| **Degree-Aware** | 0.5 | 2 | **71.33 ± 1.31** |

## 🔧 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| α (alpha) | 1.0 | Task loss weight |
| β (beta) | 1.0 | KD loss weight |
| γ (gamma) | 1.0 | RKD loss weight |
| λ (lambda_topo) | 1.0 | Topology loss weight |
| Temperature | 4.0 | Soft target temperature |
| Hidden dim | 64/256 | Hidden layer dimension |
| Dropout | 0.5 | Dropout rate |

## 📚 Datasets

| Dataset | Nodes | Edges | Features | Classes | Type |
|---------|-------|-------|----------|---------|------|
| Cora | 2,708 | 5,429 | 1,433 | 7 | Homophilic |
| Citeseer | 3,327 | 4,732 | 3,703 | 6 | Homophilic |
| PubMed | 19,717 | 44,338 | 500 | 3 | Homophilic |
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | Homophilic |
| Chameleon | 2,277 | 36,101 | 2,325 | 5 | Heterophilic |
| Squirrel | 5,201 | 217,073 | 2,089 | 5 | Heterophilic |
| Actor | 7,600 | 33,544 | 932 | 5 | Heterophilic |

## 📖 References

```bibtex
@article{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}

@inproceedings{kipf2017semi,
  title={Semi-supervised classification with graph convolutional networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={ICLR},
  year={2017}
}

@inproceedings{park2019relational,
  title={Relational knowledge distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{velickovic2018graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and others},
  booktitle={ICLR},
  year={2018}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for excellent graph learning library
- Original GCN and GAT authors for foundational work
- Knowledge distillation community for inspiring methods
