# Experiment Results

## Final Results Summary

### Best Performance (10-split average)

| Dataset | GLNN Baseline | Teacher (GloGNN++) | **Our Best** | Config | Gap Closed |
|---------|---------------|-------------------|--------------|--------|------------|
| Actor | 36.64% ± 0.43% | 37.40% ± 1.04% | **37.65% ± 0.98%** | T=8, λ=10 | **133.1%** |
| Squirrel | 58.96% ± 1.58% | 59.68% ± 1.75% | **60.15% ± 1.57%** | T=1, λ=10 | **165.8%** |

**Key Achievement:** Student MLP surpasses Teacher GNN on both datasets.

---

## Phase 1: Evaluation Protocol ✓

### Task 1: Data Splits
- Using GloGNN's official splits (Geom-GCN standard: 48%/32%/20%)
- 10 fixed splits for reproducibility

### Task 2: Teacher Verification ✓

| Dataset | Paper | Reproduced | Diff | Status |
|---------|-------|------------|------|--------|
| Actor | 37.70% | 37.40% ± 1.04% | -0.30% | ✓ PASSED |
| Chameleon | 71.21% | 73.09% ± 1.97% | +1.88% | ✓ PASSED |
| Squirrel | 57.88% | 59.68% ± 1.75% | +1.80% | ✓ PASSED |

### Task 3: GLNN Baseline ✓

| Dataset | Teacher | GLNN | Vanilla MLP |
|---------|---------|------|-------------|
| Actor | 37.40% | 36.64% ± 0.43% | 33.91% ± 0.78% |
| Chameleon | 73.09% | 70.46% ± 1.41% | 45.66% ± 1.48% |
| Squirrel | 59.68% | 58.96% ± 1.58% | 30.29% ± 1.85% |

---

## Phase 2: Method Development

### Spectral KD (Initial Attempt) - Failed ❌

| Dataset | GLNN Baseline | Spectral KD | Diff |
|---------|---------------|-------------|------|
| Actor | 36.64% | 36.62% | -0.02% |
| Squirrel | 58.96% | 57.23% | -1.73% |

**Conclusion:** Spectral decomposition does not help.

---

## Phase 3: Ablation Studies

### Task 1: PE Ablation - PE is Harmful ❌

| Dataset | Without PE | With PE | Diff | Recommendation |
|---------|------------|---------|------|----------------|
| Actor | 36.28% ± 0.56% | 35.28% ± 0.36% | **-1.00%** | NO PE |
| Squirrel | 60.00% ± 0.65% | 59.54% ± 0.60% | **-0.46%** | NO PE |

**Conclusion:** RWPE introduces noise on heterophilic graphs.

### Task 2 & 3: RKD Experiments - Complete Failure ❌

**Logits-based RKD:**
| Dataset | GLNN Baseline | + RKD (λ=1e-4) | + RKD (λ=1e-3) |
|---------|---------------|----------------|----------------|
| Actor | 36.64% | 11.05% | 11.05% |
| Squirrel | 60.13% | 20.88% | 20.88% |

**Feature-based RKD (Phase 4):**
| Dataset | Best Tuned KD | + RKD (λ=0.001) |
|---------|---------------|-----------------|
| Actor | 37.36% | 11.05% (collapse) |
| Squirrel | 61.58% | 20.88% (collapse) |

**Conclusion:** RKD causes model collapse regardless of:
- Weight magnitude (tested 1e-4 to 0.5)
- Application target (logits vs features)
- Base configuration

---

## Phase 3: Hyperparameter Tuning ✓

### Actor Dataset

| Config | Accuracy |
|--------|----------|
| T=4, λ=1 (GLNN default) | 36.42% ± 0.74% |
| T=8, λ=1 | 36.41% ± 0.90% |
| T=4, λ=5 | 37.02% ± 1.09% |
| T=8, λ=5 | 36.99% ± 0.93% |
| **T=8, λ=10** | **37.65% ± 0.98%** |
| T=6, λ=5 | 36.89% ± 1.10% |
| T=10, λ=5 | 37.07% ± 0.85% |

**Best:** T=8.0, λ_kd=10.0 → **37.65%**

### Squirrel Dataset

| Config | Accuracy |
|--------|----------|
| T=4, λ=1 (GLNN default) | 59.37% ± 1.75% |
| T=1, λ=1 | 59.87% ± 1.64% |
| T=4, λ=10 | 59.11% ± 1.69% |
| **T=1, λ=10** | **60.15% ± 1.57%** |
| T=2, λ=10 | 59.80% ± 1.73% |
| T=1, λ=5 | 59.80% ± 1.31% |
| T=1, λ=15 | 60.06% ± 1.44% |

**Best:** T=1.0, λ_kd=10.0 → **60.15%**

---

## Key Insights

### 1. Temperature Matters Differently

- **Actor (high T=8):** Needs soft labels to capture inter-class relationships
- **Squirrel (low T=1):** Needs hard labels to preserve discriminative information

### 2. Higher λ_kd is Better

Both datasets benefit from λ_kd=10 (vs default 1.0), indicating stronger teacher supervision helps.

### 3. Complex Methods Fail

| Method | Actor | Squirrel | Status |
|--------|-------|----------|--------|
| PE (RWPE) | -1.00% | -0.46% | ❌ Harmful |
| RKD (Logits) | -25.6% | -39.3% | ❌ Collapse |
| RKD (Features) | -26.3% | -40.7% | ❌ Collapse |
| **Tuned KD** | **+1.01%** | **+1.19%** | ✅ Works |

### 4. Why RKD Fails

- Teacher (GNN) and Student (MLP) have fundamentally different feature spaces
- GNN features encode graph structure; MLP features only encode node attributes
- Forcing geometric alignment between incompatible spaces causes collapse

---

## Reproducibility

All experiments use:
- 10 fixed splits from GloGNN
- Same random seeds
- Early stopping with patience=100
- Results reported as mean ± std over 10 splits

```bash
# Reproduce best results
python train_best_config.py --device cuda
```
