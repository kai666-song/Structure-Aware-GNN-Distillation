# Complete Experiment Results

## Final Results (6 Datasets)

### Heterophilic Graphs (Main Contribution)

| Dataset | Teacher (GloGNN++) | Vanilla MLP | **Our Student** | Best Config | Gap Closed |
|---------|-------------------|-------------|-----------------|-------------|------------|
| Actor | 37.74% ± 1.16% | 35.22% ± 1.46% | **37.53% ± 0.99%** | T=8, λ=10 | 91.7% |
| Squirrel | 60.11% ± 1.82% | 30.49% ± 1.46% | **60.37% ± 1.78%** | T=1, λ=10 | **100.9%** ✅ |
| Chameleon | 73.09% ± 1.97% | 48.25% ± 2.55% | **70.39% ± 1.53%** | T=1, λ=5 | 89.1% |

### Homophilic Graphs (Sanity Check)

| Dataset | Teacher (GCN) | Vanilla MLP | **Our Student** | Config |
|---------|--------------|-------------|-----------------|--------|
| Cora | 87.68% ± 1.80% | 75.08% ± 2.28% | **88.12% ± 1.64%** | T=4, λ=1 |
| CiteSeer | 76.85% ± 1.34% | 72.15% ± 1.86% | **77.34% ± 1.25%** | T=4, λ=1 |
| PubMed | 86.62% ± 0.47% | 88.09% ± 0.33% | **89.17% ± 0.39%** | T=4, λ=1 |

---

## Key Findings

### 1. Student Can Surpass Teacher

- **Squirrel**: Student (60.37%) > Teacher (60.11%) - Gap Closed 100.9%
- **All Homophilic**: Student consistently beats GCN Teacher

### 2. Temperature Matters

| Dataset Type | Optimal T | Reason |
|--------------|-----------|--------|
| Heterophilic (Actor) | High (T=8) | Needs soft labels to capture inter-class relationships |
| Heterophilic (Squirrel/Chameleon) | Low (T=1) | Needs hard labels to preserve discriminative info |
| Homophilic | Default (T=4) | Standard setting works well |

### 3. Complex Methods Fail

| Method | Actor | Squirrel | Status |
|--------|-------|----------|--------|
| + PE (RWPE) | -1.00% | -0.46% | ❌ Harmful |
| + RKD (Logits) | -25.6% | -39.3% | ❌ Collapse |
| + RKD (Features) | -26.3% | -40.7% | ❌ Collapse |
| **Tuned KD** | **+2.31%** | **+29.88%** | ✅ Works |

---

## Chameleon Hyperparameter Search

| Config | Accuracy (5 splits) |
|--------|---------------------|
| GLNN (default) T=4, λ=1 | 69.91% |
| T=1, λ=1 | 69.65% |
| **T=1, λ=5** | **70.00%** ★ |
| T=1, λ=10 | 69.82% |
| T=2, λ=10 | 69.65% |
| T=1, λ=15 | 69.47% |

---

## Reproducibility

All experiments use:
- 10 random splits (seeds 42-51)
- Early stopping with patience=100
- Results reported as mean ± std

```bash
# Reproduce all results
python run_all_experiments.py --mode all --device cuda
```

---

## Summary Table for Paper

| Dataset | Type | Teacher | Student | Δ |
|---------|------|---------|---------|---|
| Actor | Hetero | 37.74% | 37.53% | -0.21% |
| Squirrel | Hetero | 60.11% | **60.37%** | **+0.26%** |
| Chameleon | Hetero | 73.09% | 70.39% | -2.70% |
| Cora | Homo | 87.68% | **88.12%** | **+0.44%** |
| CiteSeer | Homo | 76.85% | **77.34%** | **+0.49%** |
| PubMed | Homo | 86.62% | **89.17%** | **+2.55%** |

**4/6 datasets: Student > Teacher** 🎉
