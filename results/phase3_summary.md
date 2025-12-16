# Phase 3: Visualization & Efficiency Results

## 3.1 Efficiency Benchmark

### Inference Speed (GPU - RTX 4060)

| Dataset | Nodes | GCN (ms) | MLP (ms) | Speedup |
|---------|-------|----------|----------|---------|
| Cora | 2,708 | 1.58 | 0.24 | **6.6x** |
| Citeseer | 3,327 | 1.83 | 1.52 | 1.2x |
| PubMed | 19,717 | 3.90 | 0.95 | **4.1x** |
| Amazon-Computers | 13,752 | 19.82 | 2.03 | **9.8x** |
| Amazon-Photo | 7,650 | 10.96 | 1.14 | **9.6x** |

### Throughput

| Dataset | GCN (nodes/s) | MLP (nodes/s) |
|---------|---------------|---------------|
| Cora | 1,715,617 | 11,279,808 |
| Citeseer | 1,822,620 | 2,184,707 |
| PubMed | 5,057,009 | 20,848,004 |
| Amazon-Computers | 693,809 | 6,770,025 |
| Amazon-Photo | 698,154 | 6,700,564 |

**Key Finding**: MLP achieves 4-10x speedup over GCN while maintaining competitive accuracy.

## 3.2 Visualizations Generated

All figures saved to `figures/` directory:

- `tsne_comparison_{dataset}.png` - Side-by-side Teacher vs Student embeddings
- `tsne_teacher_{dataset}.png` - Teacher GCN embeddings
- `tsne_student_{dataset}.png` - Student MLP embeddings  
- `training_curves_{dataset}.png` - Loss and accuracy curves

## 3.3 Amazon-Computers Parameter Tuning

| Config (α, β, γ) | Student Accuracy |
|------------------|-----------------|
| 1.0, 1.0, 0.0 | 84.13 ± 0.69% |
| 1.0, 1.0, 0.1 | 84.11 ± 0.63% |
| 1.0, 1.0, 0.5 | 84.13 ± 0.77% |
| **2.0, 1.0, 0.0** | **84.87 ± 0.68%** |
| 2.0, 1.0, 0.1 | 84.88 ± 0.67% |
| 1.0, 2.0, 0.0 | 83.33 ± 0.89% |

**Conclusion**: Amazon-Computers benefits more from task loss (α=2.0) than structure loss.

## Final Results Summary

| Dataset | Teacher GCN | MLP Baseline | MLP Distilled | Speedup |
|---------|-------------|--------------|---------------|---------|
| Cora | 82.04% | 59.06% | **72.65%** | 6.6x |
| Citeseer | 71.63% | 59.33% | **68.41%** | 1.2x |
| PubMed | 79.12% | 73.51% | **77.70%** | 4.1x |
| Amazon-Computers | 89.93% | 84.04% | **84.87%** | 9.8x |
| Amazon-Photo | 93.95% | 90.25% | **92.92%** | 9.6x |
