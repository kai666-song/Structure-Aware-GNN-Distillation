"""Run Chameleon experiment only."""

import os
import json
import numpy as np
import torch
from scipy import stats

# Import from experiments_improved
from experiments_improved import ImprovedDistillationTrainer, DATASET_CONFIGS

class Args:
    data = 'chameleon'
    cuda = torch.cuda.is_available()
    num_runs = 10
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    temperature = 4.0
    use_degree_aware = False
    min_degree = 2

args = Args()
config = DATASET_CONFIGS['chameleon']

print(f"Device: {'CUDA' if args.cuda else 'CPU'}")
print(f"Running Chameleon experiment with {args.num_runs} runs...")

results = {'teacher_accs': [], 'student_accs': []}

for seed in range(args.num_runs):
    print(f"\n--- Chameleon Run {seed+1}/{args.num_runs} ---")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    
    trainer = ImprovedDistillationTrainer(args, config, seed)
    
    teacher_acc = trainer.train_teacher()
    results['teacher_accs'].append(teacher_acc)
    print(f"  Teacher (GAT): {teacher_acc:.2f}%")
    
    student_acc = trainer.train_student()
    results['student_accs'].append(student_acc)
    print(f"  Student (MLP): {student_acc:.2f}%")
    
    if args.cuda:
        torch.cuda.empty_cache()

# Summary
teacher_mean = np.mean(results['teacher_accs'])
teacher_std = np.std(results['teacher_accs'])
student_mean = np.mean(results['student_accs'])
student_std = np.std(results['student_accs'])
gap = student_mean - teacher_mean

# Significance test
t_stat, p_value = stats.ttest_rel(results['student_accs'], results['teacher_accs'])

print(f"\n{'='*60}")
print("CHAMELEON RESULTS")
print(f"{'='*60}")
print(f"Teacher (GAT): {teacher_mean:.2f} ± {teacher_std:.2f}%")
print(f"Student (MLP): {student_mean:.2f} ± {student_std:.2f}%")
print(f"Gap: {gap:+.2f}%")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.01:
    print("Significance: *** (p < 0.01)")
elif p_value < 0.05:
    print("Significance: ** (p < 0.05)")
else:
    print("Significance: n.s.")

if gap > 0 and p_value < 0.05:
    print("\n✨ STATISTICALLY SIGNIFICANT IMPROVEMENT!")

# Save results
os.makedirs('results', exist_ok=True)
with open('results/chameleon_results.json', 'w') as f:
    json.dump({
        'teacher_accs': results['teacher_accs'],
        'student_accs': results['student_accs'],
        'teacher_mean': float(teacher_mean),
        'teacher_std': float(teacher_std),
        'student_mean': float(student_mean),
        'student_std': float(student_std),
        'gap': float(gap),
        't_statistic': float(t_stat),
        'p_value': float(p_value)
    }, f, indent=2)

print(f"\nResults saved to results/chameleon_results.json")
