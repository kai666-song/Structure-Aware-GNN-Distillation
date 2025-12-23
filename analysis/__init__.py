"""
Advanced Analysis Module

Contains:
- homophily_analysis.py: Node-level homophily analysis
- robustness_study.py: Graph perturbation robustness study
- ablation_detailed.py: Granular ablation study
- error_analysis.py: Error analysis and case study
- stronger_teacher.py: GCNII as stronger teacher experiment
- generate_figures.py: Publication-quality figure generation
- feature_visualization.py: Feature space analysis (Davies-Bouldin, t-SNE)
- critical_validation.py: Red Team Defense experiments (Vanilla MLP, Dirichlet Energy, Gamma)
"""

from .homophily_analysis import run_homophily_analysis
from .robustness_study import run_robustness_study
from .ablation_detailed import run_detailed_ablation
from .error_analysis import run_error_analysis
from .stronger_teacher import run_stronger_teacher_experiment
from .generate_figures import generate_all_figures
from .feature_visualization import run_feature_analysis
from .critical_validation import (
    run_vanilla_mlp_experiment,
    run_dirichlet_energy_experiment,
    run_gamma_sensitivity_experiment,
    run_all_critical_validations
)
