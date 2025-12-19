"""
Publication-Quality Figures for Paper

Generates three key figures:
1. Figure 1: Homophily Analysis - Why Student beats Teacher
2. Figure 2: Robustness Study - Structure immunity advantage  
3. Figure 3: Stronger Teacher - SOTA validation

Style: Clean, professional, suitable for top-venue papers (ICLR/NeurIPS/KDD)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.grid': False,
    'grid.alpha': 0.3,
})

# Color scheme - professional and colorblind-friendly
COLORS = {
    'teacher': '#E74C3C',      # Red
    'student': '#3498DB',      # Blue
    'teacher_dark': '#C0392B', # Dark red
    'student_dark': '#2980B9', # Dark blue
    'highlight': '#F39C12',    # Orange for highlighting
    'gap_positive': '#27AE60', # Green for positive gap
    'gap_negative': '#E74C3C', # Red for negative gap
}


def load_results():
    """Load all experiment results."""
    results = {}
    
    # Homophily analysis
    with open('results/homophily_analysis_actor.json', 'r') as f:
        results['homophily'] = json.load(f)
    
    # Robustness study
    with open('results/robustness_actor.json', 'r') as f:
        results['robustness'] = json.load(f)
    
    # Stronger teacher
    with open('results/stronger_teacher_actor.json', 'r') as f:
        results['stronger_teacher'] = json.load(f)
    
    return results


def figure1_homophily(results, save_path='figures/paper_fig1_homophily.pdf'):
    """
    Figure 1: Homophily Analysis
    
    Shows accuracy breakdown by local homophily ratio.
    Highlights the dramatic improvement in heterophilic regions.
    """
    data = results['homophily']
    bins = data['bins']
    
    # Extract data
    ranges = [b['range'] for b in bins]
    teacher_means = [b['teacher_mean'] for b in bins]
    teacher_stds = [b['teacher_std'] for b in bins]
    student_means = [b['student_mean'] for b in bins]
    student_stds = [b['student_std'] for b in bins]
    counts = [b['count'] for b in bins]
    gaps = [b['gap'] for b in bins]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(ranges))
    width = 0.35
    
    # Bar plots
    bars_teacher = ax.bar(x - width/2, teacher_means, width, 
                          yerr=teacher_stds, capsize=3,
                          label='Teacher (GAT)', color=COLORS['teacher'],
                          edgecolor=COLORS['teacher_dark'], linewidth=1.2,
                          error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    bars_student = ax.bar(x + width/2, student_means, width,
                          yerr=student_stds, capsize=3,
                          label='Student (MLP)', color=COLORS['student'],
                          edgecolor=COLORS['student_dark'], linewidth=1.2,
                          error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Highlight the heterophilic region (0.0-0.2)
    highlight_rect = plt.Rectangle((-0.5, 0), 1, max(student_means) * 1.3,
                                    alpha=0.15, color=COLORS['highlight'],
                                    zorder=0)
    ax.add_patch(highlight_rect)
    
    # Add gap annotations
    for i, (gap, t_mean, s_mean) in enumerate(zip(gaps, teacher_means, student_means)):
        y_pos = max(t_mean, s_mean) + max(teacher_stds[i], student_stds[i]) + 3
        color = COLORS['gap_positive'] if gap > 0 else COLORS['gap_negative']
        sign = '+' if gap > 0 else ''
        
        # Special emphasis on first bar
        fontweight = 'bold' if i == 0 else 'normal'
        fontsize = 11 if i == 0 else 9
        
        ax.annotate(f'{sign}{gap:.1f}%', xy=(x[i], y_pos),
                   ha='center', va='bottom', fontsize=fontsize,
                   fontweight=fontweight, color=color)
    
    # Add node count annotations at bottom
    for i, count in enumerate(counts):
        ax.annotate(f'n={count}', xy=(x[i], 2),
                   ha='center', va='bottom', fontsize=8,
                   color='gray', style='italic')
    
    # Add special annotation for heterophilic region
    ax.annotate('Heterophilic\nRegion', xy=(0, 45), fontsize=10,
               ha='center', va='center', color=COLORS['highlight'],
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=COLORS['highlight'], alpha=0.9))
    
    # Labels and formatting
    ax.set_xlabel('Local Homophily Ratio', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Node-Level Accuracy by Homophily (Actor Dataset)', 
                fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(ranges)
    ax.set_ylim(0, 55)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    # Save in multiple formats
    plt.savefig(save_path, format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300)
    plt.close()
    
    print(f"✓ Figure 1 saved: {save_path}")


def figure2_robustness(results, save_path='figures/paper_fig2_robustness.pdf'):
    """
    Figure 2: Robustness to Graph Perturbation
    
    Shows Teacher's performance degradation vs Student's immunity.
    """
    data = results['robustness']
    
    ratios = [r * 100 for r in data['perturbation_ratios']]  # Convert to percentage
    
    teacher_means = [data['teacher'][str(r)]['mean'] for r in data['perturbation_ratios']]
    teacher_stds = [data['teacher'][str(r)]['std'] for r in data['perturbation_ratios']]
    student_means = [data['student'][str(r)]['mean'] for r in data['perturbation_ratios']]
    student_stds = [data['student'][str(r)]['std'] for r in data['perturbation_ratios']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot lines with error bands
    ax.fill_between(ratios, 
                    np.array(teacher_means) - np.array(teacher_stds),
                    np.array(teacher_means) + np.array(teacher_stds),
                    alpha=0.2, color=COLORS['teacher'])
    ax.plot(ratios, teacher_means, 'o-', color=COLORS['teacher'], 
            linewidth=2.5, markersize=8, label='Teacher (GAT)',
            markeredgecolor=COLORS['teacher_dark'], markeredgewidth=1.5)
    
    ax.fill_between(ratios,
                    np.array(student_means) - np.array(student_stds),
                    np.array(student_means) + np.array(student_stds),
                    alpha=0.2, color=COLORS['student'])
    ax.plot(ratios, student_means, 's-', color=COLORS['student'],
            linewidth=2.5, markersize=8, label='Student (MLP)',
            markeredgecolor=COLORS['student_dark'], markeredgewidth=1.5)
    
    # Add annotations
    # Teacher drop annotation
    ax.annotate('', xy=(50, teacher_means[-1]), xytext=(50, teacher_means[0]),
               arrowprops=dict(arrowstyle='<->', color=COLORS['teacher'], lw=1.5))
    ax.annotate(f'↓{data["teacher_drop"]:.1f}%', xy=(52, (teacher_means[0] + teacher_means[-1])/2),
               fontsize=10, color=COLORS['teacher'], fontweight='bold', va='center')
    
    # Student immunity annotation
    ax.annotate('Structure\nImmune!', xy=(25, student_means[0] + 2),
               fontsize=11, color=COLORS['student'], fontweight='bold',
               ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor=COLORS['student'], alpha=0.9))
    
    # Gap annotation
    mid_idx = len(ratios) // 2
    gap = student_means[mid_idx] - teacher_means[mid_idx]
    ax.annotate('', xy=(ratios[mid_idx], student_means[mid_idx]), 
               xytext=(ratios[mid_idx], teacher_means[mid_idx]),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, ls='--'))
    ax.annotate(f'+{gap:.1f}%', xy=(ratios[mid_idx] + 2, (student_means[mid_idx] + teacher_means[mid_idx])/2),
               fontsize=10, color='gray', fontweight='bold', va='center')
    
    # Labels and formatting
    ax.set_xlabel('Edge Perturbation Ratio (%)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Robustness to Graph Structure Noise (Actor Dataset)',
                fontweight='bold', pad=15)
    ax.set_xlim(-2, 55)
    ax.set_ylim(20, 42)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray')
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300)
    plt.close()
    
    print(f"✓ Figure 2 saved: {save_path}")


def figure3_stronger_teacher(results, save_path='figures/paper_fig3_stronger_teacher.pdf'):
    """
    Figure 3: Stronger Teacher Experiment
    
    Shows that Student improves with stronger Teacher (GCNII vs GAT).
    """
    data = results['stronger_teacher']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # === Left subplot: Absolute accuracy comparison ===
    teachers = ['GAT\n(2018)', 'GCNII\n(2020)']
    x = np.arange(len(teachers))
    width = 0.35
    
    teacher_accs = [data['gat']['teacher_mean'], data['gcnii']['teacher_mean']]
    teacher_stds = [data['gat']['teacher_std'], data['gcnii']['teacher_std']]
    student_accs = [data['gat']['student_mean'], data['gcnii']['student_mean']]
    student_stds = [data['gat']['student_std'], data['gcnii']['student_std']]
    
    bars1 = ax1.bar(x - width/2, teacher_accs, width, yerr=teacher_stds, capsize=4,
                    label='Teacher', color=COLORS['teacher'],
                    edgecolor=COLORS['teacher_dark'], linewidth=1.2,
                    error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax1.bar(x + width/2, student_accs, width, yerr=student_stds, capsize=4,
                    label='Student', color=COLORS['student'],
                    edgecolor=COLORS['student_dark'], linewidth=1.2,
                    error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    # Add value labels on bars
    for bar, val in zip(bars1, teacher_accs):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, student_accs):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=COLORS['student_dark'])
    
    # Add improvement arrow for GCNII
    ax1.annotate('', xy=(1 - width/2, data['gcnii']['teacher_mean']),
                xytext=(0 - width/2, data['gat']['teacher_mean']),
                arrowprops=dict(arrowstyle='->', color=COLORS['teacher'], lw=2))
    ax1.annotate('+6.2%', xy=(0.5 - width/2, 30.5), fontsize=9, 
                color=COLORS['teacher'], fontweight='bold', ha='center')
    
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('(a) Absolute Performance', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(teachers)
    ax1.set_ylim(0, 42)
    ax1.legend(loc='upper left', framealpha=0.95)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    
    # === Right subplot: Gap analysis ===
    gaps = [data['gat']['gap'], data['gcnii']['gap']]
    colors = [COLORS['gap_positive'], COLORS['gap_positive']]
    
    bars3 = ax2.bar(x, gaps, width * 1.5, color=colors,
                    edgecolor=[COLORS['student_dark'], COLORS['student_dark']], 
                    linewidth=1.5)
    
    # Add value labels
    for bar, gap in zip(bars3, gaps):
        ax2.annotate(f'+{gap:.2f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=COLORS['gap_positive'])
    
    # Add "Student > Teacher" annotation
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.annotate('Student > Teacher', xy=(0.5, 4), fontsize=10,
                ha='center', va='center', color=COLORS['gap_positive'],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLORS['gap_positive'], alpha=0.9))
    
    ax2.set_ylabel('Student - Teacher Gap (%)', fontweight='bold')
    ax2.set_title('(b) Performance Gap', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(teachers)
    ax2.set_ylim(-1, 8)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    
    # Main title
    fig.suptitle('Stronger Teacher Validation (Actor Dataset)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 3 saved: {save_path}")


def figure4_feature_space(save_path='figures/paper_fig4_feature_space.pdf'):
    """
    Figure 4 (Bonus): Feature Space Analysis
    
    Shows clustering quality metrics comparison.
    """
    # Load feature analysis results
    with open('results/feature_analysis_actor.json', 'r') as f:
        data = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    
    metrics = [
        ('davies_bouldin', 'Davies-Bouldin Index', '↓ Lower is Better', True),
        ('silhouette', 'Silhouette Score', '↑ Higher is Better', False),
        ('compactness_ratio', 'Compactness Ratio', '↓ Lower is Better', True),
    ]
    
    for ax, (metric, title, subtitle, lower_better) in zip(axes, metrics):
        teacher_mean = data['teacher'][metric]['mean']
        teacher_std = data['teacher'][metric]['std']
        student_mean = data['student'][metric]['mean']
        student_std = data['student'][metric]['std']
        
        x = [0, 1]
        means = [teacher_mean, student_mean]
        stds = [teacher_std, student_std]
        colors = [COLORS['teacher'], COLORS['student']]
        labels = ['Teacher\n(GAT)', 'Student\n(MLP)']
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                     edgecolor=[COLORS['teacher_dark'], COLORS['student_dark']],
                     linewidth=1.5, error_kw={'linewidth': 1.5, 'capthick': 1.5})
        
        # Highlight winner
        if lower_better:
            winner_idx = 0 if teacher_mean < student_mean else 1
        else:
            winner_idx = 0 if teacher_mean > student_mean else 1
        
        # Add star to winner
        winner_bar = bars[winner_idx]
        ax.annotate('*', xy=(winner_bar.get_x() + winner_bar.get_width()/2, 
                            winner_bar.get_height() + stds[winner_idx] + 0.3),
                   ha='center', va='bottom', fontsize=20, fontweight='bold',
                   color=COLORS['highlight'])
        
        # Calculate improvement
        if lower_better:
            improvement = (teacher_mean - student_mean) / teacher_mean * 100
        else:
            improvement = (student_mean - teacher_mean) / abs(teacher_mean) * 100
        
        if improvement > 0:
            ax.annotate(f'+{improvement:.1f}%', xy=(0.5, max(means) * 0.5),
                       ha='center', fontsize=10, color=COLORS['gap_positive'],
                       fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f'{title}\n({subtitle})', fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    fig.suptitle('Feature Space Quality Analysis (Actor Dataset)',
                fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 4 saved: {save_path}")


def generate_all_figures():
    """Generate all publication figures."""
    print("\n" + "="*60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*60 + "\n")
    
    os.makedirs('figures', exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Generate figures
    figure1_homophily(results)
    figure2_robustness(results)
    figure3_stronger_teacher(results)
    figure4_feature_space()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput files:")
    print("  - figures/paper_fig1_homophily.pdf/png")
    print("  - figures/paper_fig2_robustness.pdf/png")
    print("  - figures/paper_fig3_stronger_teacher.pdf/png")
    print("  - figures/paper_fig4_feature_space.pdf/png")
    print("\nThese figures are ready for paper submission!")


if __name__ == '__main__':
    generate_all_figures()
