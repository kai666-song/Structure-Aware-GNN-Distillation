"""
Advanced Analysis Module

Contains:
- homophily_analysis.py: Node-level homophily analysis
- robustness_study.py: Graph perturbation robustness study
- ablation_detailed.py: Granular ablation study
- error_analysis.py: Error analysis and case study
"""

from .homophily_analysis import run_homophily_analysis
from .robustness_study import run_robustness_study
from .ablation_detailed import run_detailed_ablation
from .error_analysis import run_error_analysis
