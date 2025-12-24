# Experiment Results

## Phase 1: Evaluation Protocol Establishment ✓ COMPLETED

Phase 1 establishes an unassailable evaluation protocol with verified baselines.

### Task 1: Data Splits Standardization ✓

We use GloGNN's official splits (Geom-GCN standard: 48%/32%/20%) to ensure fair comparison.

| Dataset | Train | Val | Test | Source |
|---------|-------|-----|------|--------|
| Actor (Film) | 48% | 32% | 20% | GloGNN/Geom-GCN |
| Chameleon | 48% | 32% | 20% | GloGNN/Geom-GCN |
| Squirrel | 48% | 32% | 20% | GloGNN/Geom-GCN |

Note: GloGNN's split files are named `*_0.6_0.2_*.npz` but actual ratio is 48/32/20.

### Task 2: Teacher Verification ✓

GloGNN++ teacher successfully reproduces original paper results (tolerance: ±2.0%).

| Dataset | Target (Paper) | Achieved | Diff | Status |
|---------|----------------|----------|------|--------|
| Actor | 37.70% | 37.40% ± 1.04% | -0.30% | ✓ PASSED |
| Chameleon | 71.21% | 73.09% ± 1.97% | +1.88% | ✓ PASSED (Better!) |
| Squirrel | 57.88% | 59.68% ± 1.75% | +1.80% | ✓ PASSED (Better!) |

Teacher logits saved to `checkpoints/glognn_teacher_{dataset}/`.

### Task 3: GLNN Baseline ✓

GLNN (vanilla soft-label distillation) is the TRUE competitor we must beat.

| Dataset | Teacher (GloGNN++) | GLNN | Vanilla MLP | GLNN vs Teacher |
|---------|-------------------|------|-------------|-----------------|
| Actor | 37.40% | 36.64% ± 0.43% | 33.91% ± 0.78% | -0.76% |
| Chameleon | 73.09% | 70.46% ± 1.41% | 45.66% ± 1.48% | -2.63% |
| Squirrel | 59.68% | 58.96% ± 1.58% | 30.29% ± 1.85% | -0.72% |

Key observations:
- GLNN significantly outperforms Vanilla MLP (proves KD works)
- GLNN is slightly below teacher (expected for distillation)
- Chameleon/Squirrel show massive MLP→GLNN improvement (+25%/+28%)

### Task 4: "Graph-Free" Narrative ✓

Adopted **Strategy A (Defensive)**: "No message passing at inference"
- Our method requires pre-computed RWPE (uses graph structure)
- Graph is only needed during preprocessing, not online inference
- This is similar to many efficient GNN methods

---

## Phase 2: Method Development

### Current Results: Spectral KD vs GLNN Baseline

| Dataset | GLNN Baseline | Spectral KD (Ours) | Diff | Status |
|---------|---------------|-------------------|------|--------|
| Actor | 36.64% ± 0.43% | 36.62% ± 1.27% | -0.02% | ⚠️ On par |
| Chameleon | 70.46% ± 1.41% | 70.75% ± 2.57% | +0.29% | ⚠️ Slight improvement |
| Squirrel | 58.96% ± 1.58% | 57.23% ± 2.22% | -1.73% | ❌ Below baseline |

**Status**: Method needs improvement to consistently beat GLNN baseline.

---

## Efficiency Comparison (To Be Updated)

| Metric | GloGNN++ (Teacher) | MLP Student | Improvement |
|--------|-------------------|-------------|-------------|
| Parameters | ~100K | ~50K | ~2x smaller |
| Model Size | TBD | TBD | TBD |
| Inference Time | Requires graph | Graph-free | ✅ Faster |
| Message Passing | Required | Not Required* | ✅ |

*Requires pre-computed RWPE (one-time preprocessing using graph structure)

---

## Notes on "Graph-Free" Claims

**Clarification**: Our method does NOT require message passing during inference, but it DOES require:
1. Pre-computed Random Walk Positional Encoding (RWPE) - uses graph structure
2. Pre-computed homophily weights - uses graph structure and teacher predictions

The graph structure is only needed during preprocessing, not during online inference.

**Correct terminology**: "No message passing at inference" instead of "Graph-free inference"

---

## How to Reproduce

```bash
# Phase 1: Establish baselines (already completed)
python run_phase1_evaluation.py --all --device cuda

# Individual tasks:
python baselines/verify_glognn_teacher.py --all --device cuda
python baselines/glnn_baseline.py --all --device cuda

# Phase 2: Run our method (after Phase 1 passes)
python train.py --dataset actor --num_runs 10
```
