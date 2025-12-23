"""
Feature Engineering Module for Knowledge Distillation
=====================================================

This module provides feature augmentation techniques to give MLP students
"positional awareness" and "structural sense" without requiring graph
structure at inference time.

Components:
- generate_pe.py: Random Walk Positional Encoding (RWPE)
- generate_homophily.py: Teacher-based Homophily Profiling
"""

from .generate_pe import generate_rwpe, load_pe
from .generate_homophily import generate_homophily_weights, load_homophily_weights
