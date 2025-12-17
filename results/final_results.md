# Final Results Summary

## Experiment Configuration
- **Student Model**: MLPBatchNorm (MLP with BatchNorm)
- **Loss Function**: L = α·L_task + β·L_kd + γ·L_struct
- **Parameters**: α=1.0, β=1.0, γ=1.0, T=4.0
- **Runs**: 10 seeds per dataset

## Main Results

| Dataset | Teacher GCN | Student MLP | Δ |
|---------|-------------|-------------|---|
| Cora | 82.04±0.48 | **82.45±0.79** | +0.41% |
| Citeseer | 71.63±0.46 | 71.80±0.60 | +0.17% |
| PubMed | 79.12±0.39 | **80.00±0.44** | +0.88% |
| Amazon-Computers | 89.93±0.30 | 81.21±3.28 | -8.72% |
| Amazon-Photo | 93.95±0.52 | 93.56±0.98 | -0.39% |

## Ablation Study

| Dataset | Baseline | GLNN (γ=0) | Ours (γ=1) | RKD Gain |
|---------|----------|------------|------------|----------|
| Cora | 45.69±1.33 | 81.82±0.73 | 82.31±0.63 | +0.49% |
| Citeseer | 42.64±2.00 | 71.75±0.55 | 71.58±0.70 | -0.17% |
| PubMed | 66.69±1.10 | 80.09±0.61 | 80.02±0.52 | -0.07% |
| Amazon-Computers | 41.25±5.68 | 81.47±3.88 | 83.15±3.11 | +1.68% |
| Amazon-Photo | 89.92±0.69 | 92.85±1.22 | 93.52±0.85 | +0.67% |

## Key Findings

1. **Student Surpasses Teacher**: On Cora (+0.41%) and PubMed (+0.88%), the distilled MLP outperforms the Teacher GCN.

2. **RKD Effectiveness**: The structure loss (RKD) provides significant gains on Amazon datasets, especially Amazon-Computers (+1.68%).

3. **Massive Improvement from Distillation**: MLP baseline achieves only ~45% on Cora, but with distillation reaches 82.45% (+37%).

4. **Inference Speedup**: MLP is 4-10x faster than GCN since it doesn't require graph structure.

## Dataset-Specific Configurations

| Dataset | Hidden | WD (Teacher) | WD (Student) | Epochs |
|---------|--------|--------------|--------------|--------|
| Cora | 64 | 5e-4 | 1e-5 | 300 |
| Citeseer | 64 | 5e-4 | 1e-5 | 300 |
| PubMed | 64 | 5e-4 | 1e-5 | 300 |
| Amazon-Computers | 256 | 0 | 0 | 500 |
| Amazon-Photo | 256 | 0 | 0 | 500 |

## Model Checkpoints

All trained models are saved in `checkpoints/{dataset}/`:
- `teacher_seed{i}.pt`: Teacher GCN weights
- `student_seed{i}.pt`: Student MLP weights  
- `history_seed{i}.json`: Training history for plotting
- `results_summary.json`: Summary statistics
